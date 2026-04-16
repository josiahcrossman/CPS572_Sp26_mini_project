"""
Multi-task SFT training with per-task monitoring, validation, and resume.

Features over the original baseline:
  - Per-task loss tracking (gsm8k / tulu / code)
  - EMA-smoothed loss for readable training curves
  - Held-out validation split with periodic evaluation
  - Timing, throughput, and ETA reporting
  - Resume from a previously saved checkpoint
  - Data-coverage reporting (how much of the dataset is actually seen)

Usage:
    python -m evaluation.train_and_publish --base_model meta-llama/Llama-3.2-3B --no_publish
    python -m evaluation.train_and_publish --num_steps 500 --save_every 100 --val_fraction 0.05

Env:
    SFT_DATASETS_CACHE or HF_DATASETS_CACHE — Hugging Face datasets cache directory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

try:
    from . import sft_data
except ImportError:
    import sft_data

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(EVAL_DIR, "runs")
MANIFEST_NAME = "sft_manifest.jsonl"
TASKS = ("gsm8k", "tulu", "code")


def _append_manifest(entry: dict) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, MANIFEST_NAME)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def conversations_to_datums(
    conversations,
    renderer,
    max_length: int,
    source_labels: list[str] | None = None,
) -> tuple[list, list[str], dict]:
    """Convert chat conversations to Tinker datums, preserving source labels."""
    datums: list = []
    labels: list[str] = []
    skipped = 0
    for i, convo in enumerate(conversations):
        try:
            d = conversation_to_datum(
                convo,
                renderer,
                max_length=max_length,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            datums.append(d)
            labels.append(source_labels[i] if source_labels else "unknown")
        except Exception:
            skipped += 1
    return datums, labels, {"skipped_datum_errors": skipped, "n_datums": len(datums)}


# ── Loss helpers ──────────────────────────────────────────────────────


def _compute_loss(fwd_bwd_result, batch) -> float:
    logprobs = np.concatenate(
        [o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs]
    )
    weights = np.concatenate(
        [d.loss_fn_inputs["weights"].tolist() for d in batch]
    )
    return float(-np.dot(logprobs, weights) / max(weights.sum(), 1))


def _compute_per_datum_losses(fwd_bwd_result, batch) -> list[float]:
    losses = []
    for output, datum in zip(fwd_bwd_result.loss_fn_outputs, batch):
        lp = np.array(output["logprobs"].tolist())
        w = np.array(datum.loss_fn_inputs["weights"].tolist())
        losses.append(float(-np.dot(lp, w) / max(w.sum(), 1)))
    return losses


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Multi-task SFT on Tinker (LoRA)")

    # Model
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Base model id (e.g. meta-llama/Llama-3.2-3B)",
    )

    # Data
    parser.add_argument("--n_gsm8k", type=int, default=2500)
    parser.add_argument("--n_tulu", type=int, default=2500)
    parser.add_argument("--n_code", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_datums_seed", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=2048)

    # Training
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)

    # Checkpointing
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--checkpoint_name", type=str, default="sft-final")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--no_publish", action="store_true")

    # Monitoring & validation
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.05,
        help="Fraction of data held out for validation (0 disables)",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=50,
        help="Evaluate on validation set every N training steps",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Print smoothed training stats every N steps (1 = every step)",
    )

    # Resume
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint name to resume LoRA weights from (via load_state)",
    )
    parser.add_argument(
        "--resume_step",
        type=int,
        default=0,
        help="Step to resume from (adjusts data offset and counters)",
    )

    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    shuffle_d = (
        args.shuffle_datums_seed if args.shuffle_datums_seed is not None else args.seed
    )

    # ── Model & renderer ─────────────────────────────────────────────

    print(f"Model: {args.base_model}")
    tokenizer = get_tokenizer(args.base_model)
    renderer_name = model_info.get_recommended_renderer_name(args.base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # ── Data loading ─────────────────────────────────────────

    print("Loading mixed SFT data (train splits only)...")
    conversations, source_labels, data_stats = (
        sft_data.load_mixed_sft_conversations_tagged(
            args.n_gsm8k, args.n_tulu, args.n_code, args.seed,
        )
    )
    print(
        f"  Conversations: {len(conversations)} "
        f"(stats: {json.dumps(data_stats, indent=2)})"
    )

    print("Packing conversations into datums...")
    all_data, all_labels, pack_stats = conversations_to_datums(
        conversations, renderer, args.max_length, source_labels,
    )
    print(
        f"  Datums: {pack_stats['n_datums']}, "
        f"skipped (errors): {pack_stats['skipped_datum_errors']}"
    )
    if not all_data:
        raise SystemExit("No valid training datums; check max_length and dataset access.")

    # ── Shuffle ───────────────────────────────────────────────────────

    rng = np.random.default_rng(shuffle_d)
    order = rng.permutation(len(all_data))
    all_data = [all_data[i] for i in order]
    all_labels = [all_labels[i] for i in order]

    # ── Train / validation split ──────────────────────────────────────

    val_data: list = []
    val_labels: list[str] = []
    train_data, train_labels = all_data, all_labels

    if args.val_fraction > 0:
        n_val = max(1, int(len(all_data) * args.val_fraction))
        val_data = all_data[:n_val]
        val_labels = all_labels[:n_val]
        train_data = all_data[n_val:]
        train_labels = all_labels[n_val:]
        val_counts = {t: val_labels.count(t) for t in TASKS}
        train_counts = {t: train_labels.count(t) for t in TASKS}
        print(f"  Validation: {len(val_data)} datums {val_counts}")
        print(f"  Training:   {len(train_data)} datums {train_counts}")
    else:
        print("  Validation: disabled (--val_fraction 0)")

    # ── Training client ───────────────────────────────────────────────

    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    print("  Training client ready")

    if args.resume_from:
        print(f"Resuming weights from checkpoint '{args.resume_from}'...")
        tc.load_state(args.resume_from).result()
        print("  State loaded")

    # ── Training loop ─────────────────────────────────────────────────

    adam_params = types.AdamParams(
        learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    samples_total = args.num_steps * args.batch_size
    data_coverage = min(1.0, samples_total / len(train_data)) if train_data else 0
    print(
        f"\nTraining {args.num_steps} steps "
        f"(batch={args.batch_size}, lr={args.lr}, "
        f"train_datums={len(train_data)}, "
        f"coverage~{data_coverage:.0%})..."
    )
    if data_coverage < 0.5:
        print(
            f"  WARNING: only ~{data_coverage:.0%} of training data will be seen. "
            f"Consider increasing --num_steps or reducing dataset size."
        )

    ema_alpha = 0.05
    ema_loss: float | None = None
    ema_task: dict[str, float | None] = {t: None for t in TASKS}
    task_sample_counts: dict[str, int] = {t: 0 for t in TASKS}

    loss_history: list[dict] = []
    val_history: list[dict] = []
    saved_intermediate: list[dict] = []
    best_val_loss = float("inf")
    best_val_step = -1

    start_step = args.resume_step
    t0 = time.time()

    for step in range(start_step, args.num_steps):
        idx = (step * args.batch_size) % len(train_data)
        batch_indices = [(idx + j) % len(train_data) for j in range(args.batch_size)]
        batch = [train_data[i] for i in batch_indices]
        batch_labels = [train_labels[i] for i in batch_indices]

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = tc.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        # Overall loss
        loss = _compute_loss(fwd_bwd_result, batch)

        # Per-datum → per-task EMA
        datum_losses = _compute_per_datum_losses(fwd_bwd_result, batch)
        for dl, src in zip(datum_losses, batch_labels):
            prev = ema_task[src]
            ema_task[src] = dl if prev is None else ema_alpha * dl + (1 - ema_alpha) * prev
            task_sample_counts[src] += 1

        ema_loss = loss if ema_loss is None else ema_alpha * loss + (1 - ema_alpha) * ema_loss

        loss_entry: dict = {"step": step + 1, "loss": loss, "ema_loss": ema_loss}
        for t in TASKS:
            if ema_task[t] is not None:
                loss_entry[f"ema_{t}"] = ema_task[t]
        loss_history.append(loss_entry)

        # ── Logging ───────────────────────────────────────────────────

        steps_done = step - start_step + 1
        should_log = (
            steps_done == 1
            or (step + 1) % args.log_every == 0
            or (step + 1) == args.num_steps
        )
        if should_log:
            elapsed = time.time() - t0
            rate = steps_done / elapsed if elapsed > 0 else 0
            eta = (args.num_steps - step - 1) / rate if rate > 0 else 0

            parts = [
                f"Step {step + 1}/{args.num_steps}",
                f"loss={loss:.4f}",
                f"ema={ema_loss:.4f}",
            ]
            for t in TASKS:
                if ema_task[t] is not None:
                    parts.append(f"{t}={ema_task[t]:.4f}")
            parts.append(f"[{rate:.1f} step/s, eta {_fmt_time(eta)}]")
            print("  " + " | ".join(parts))

        # ── Validation ────────────────────────────────────────────────

        if val_data and (step + 1) % args.val_every == 0:
            val_losses_by_task: dict[str, list[float]] = defaultdict(list)
            all_val_losses: list[float] = []

            for vstart in range(0, len(val_data), args.batch_size):
                vbatch = val_data[vstart : vstart + args.batch_size]
                vlbls = val_labels[vstart : vstart + args.batch_size]
                if not vbatch:
                    break
                vfwd = tc.forward_backward(vbatch, loss_fn="cross_entropy")
                vresult = vfwd.result()
                for vl, vs in zip(
                    _compute_per_datum_losses(vresult, vbatch), vlbls
                ):
                    all_val_losses.append(vl)
                    val_losses_by_task[vs].append(vl)

            avg_val = float(np.mean(all_val_losses))
            val_entry: dict = {"step": step + 1, "val_loss": avg_val}
            parts = [f"  [Val @ step {step + 1}] loss={avg_val:.4f}"]
            for t in TASKS:
                if val_losses_by_task[t]:
                    t_avg = float(np.mean(val_losses_by_task[t]))
                    parts.append(f"{t}={t_avg:.4f}")
                    val_entry[f"val_{t}"] = t_avg
            print(" | ".join(parts))
            val_history.append(val_entry)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_val_step = step + 1

        # ── Intermediate checkpoint ───────────────────────────────────

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            name = f"{run_id}-step-{step + 1:06d}"
            ckpt = tc.save_weights_for_sampler(name=name).result()
            path = ckpt.path
            print(f"    [Checkpoint] {name} -> {path}")
            entry = {
                "run_id": run_id,
                "kind": "intermediate",
                "step": step + 1,
                "name": name,
                "checkpoint_path": path,
                "loss": float(loss),
                "ema_loss": float(ema_loss),
                "base_model": args.base_model,
            }
            saved_intermediate.append(entry)
            _append_manifest(entry)

    # ── Training summary ──────────────────────────────────────────────

    total_time = time.time() - t0
    total_steps = args.num_steps - start_step

    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"  Steps: {total_steps} in {_fmt_time(total_time)} "
        f"({total_steps / max(total_time, 1e-9):.1f} step/s)"
    )
    print(f"  Final loss (raw):  {loss:.4f}")
    print(f"  Final loss (EMA):  {ema_loss:.4f}")
    print(f"  Per-task final EMA loss:")
    for t in TASKS:
        v = ema_task[t]
        cnt = task_sample_counts[t]
        if v is not None:
            print(f"    {t:>8s}: {v:.4f}  ({cnt} samples seen)")
        else:
            print(f"    {t:>8s}: n/a")
    if val_history:
        print(f"  Best validation loss:  {best_val_loss:.4f} at step {best_val_step}")
        print(f"  Final validation loss: {val_history[-1]['val_loss']:.4f}")
    print(
        f"  Data: {len(train_data)} train datums, "
        f"~{data_coverage:.0%} coverage in {total_steps} steps"
    )
    print(f"{'=' * 60}")

    # ── Save final checkpoint ─────────────────────────────────────────

    print(f"\nSaving final checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Sampler checkpoint saved: {checkpoint_path}")

    train_ckpt = tc.save_state(args.checkpoint_name).result()
    training_weights_path = train_ckpt.path
    print(f"  Training weights saved (for GRPO / load_state): {training_weights_path}")

    if not args.no_publish:
        print("\nPublishing final checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published successfully!")
    else:
        print("\nSkipping publish (--no_publish).")

    # ── Persist metadata ──────────────────────────────────────────────

    training_meta = {
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lora_rank": args.rank,
        "max_length": args.max_length,
        "seed": args.seed,
        "shuffle_datums_seed": shuffle_d,
        "save_every": args.save_every,
        "run_id": run_id,
        "val_fraction": args.val_fraction,
        "resume_from": args.resume_from,
    }

    info = {
        "checkpoint_path": checkpoint_path,
        "training_weights_path": training_weights_path,
        "base_model": args.base_model,
        "renderer_name": renderer_name,
        "training": training_meta,
        "data": {
            **data_stats,
            **pack_stats,
            "n_conversations": len(conversations),
            "n_train": len(train_data),
            "n_val": len(val_data),
        },
        "published": not args.no_publish,
        "final_ema_loss": ema_loss,
        "final_task_ema": {t: ema_task[t] for t in TASKS},
        "best_val": (
            {"loss": best_val_loss, "step": best_val_step} if val_history else None
        ),
        "val_history": val_history,
        "intermediate_checkpoints": saved_intermediate,
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")

    final_entry = {
        "run_id": run_id,
        "kind": "final",
        "step": args.num_steps,
        "name": args.checkpoint_name,
        "checkpoint_path": checkpoint_path,
        "training_weights_path": training_weights_path,
        "base_model": args.base_model,
        "training": training_meta,
        "data": info["data"],
    }
    _append_manifest(final_entry)

    print("\nNext: evaluate with")
    print(
        f'  python -m evaluation.eval_all --checkpoint_path "{checkpoint_path}" '
        f"--base_model {args.base_model}"
    )


if __name__ == "__main__":
    main()
