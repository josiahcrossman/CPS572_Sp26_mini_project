"""
Train a model (minimal SFT), save checkpoint, and publish it.

NOTE: This is a TOY EXAMPLE that trains for a few steps on dummy data
to verify the full workflow end-to-end. You should replace the training
data and training logic with your own implementation.

TODO:
  - Replace DEMO_CONVERSATIONS with your task-specific training data
  - Tune hyperparameters (learning rate, batch size, number of steps, LoRA rank)
  - Add validation / early stopping as needed

Usage:
    python evaluation/train_and_publish.py --num_steps 500 --save_every 100
    python -m evaluation.train_and_publish --base_model meta-llama/Llama-3.2-3B --no_publish

Env:
    SFT_DATASETS_CACHE or HF_DATASETS_CACHE — Hugging Face datasets cache directory.
"""

from __future__ import annotations

import argparse
import json
import os
import time

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


def _append_manifest(entry: dict) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, MANIFEST_NAME)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def conversations_to_datums(
    conversations,
    renderer,
    max_length: int,
) -> tuple[list, dict]:
    """Convert chat conversations to Tinker datums; skip failures."""
    datums = []
    skipped = 0
    for convo in conversations:
        try:
            d = conversation_to_datum(
                convo,
                renderer,
                max_length=max_length,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            datums.append(d)
        except Exception:
            skipped += 1
    return datums, {"skipped_datum_errors": skipped, "n_datums": len(datums)}


def main():
    parser = argparse.ArgumentParser(description="Multi-task SFT on Tinker (LoRA)")
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Base model id (e.g. meta-llama/Llama-3.2-3B, meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument("--n_gsm8k", type=int, default=2500, help="GSM8K train samples to mix")
    parser.add_argument("--n_tulu", type=int, default=2500, help="Tulu-3 SFT train samples to mix")
    parser.add_argument("--n_code", type=int, default=2500, help="OpenCodeInstruct train samples to mix")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for dataset subsampling")
    parser.add_argument(
        "--shuffle_datums_seed",
        type=int,
        default=None,
        help="Seed for shuffling datums after packing (default: same as --seed)",
    )
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length for packing")
    parser.add_argument("--num_steps", type=int, default=500, help="Optimizer steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="If >0, save a sampler checkpoint every N steps (plus final)",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="sft-final",
        help="Name for the final sampler checkpoint",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run id for intermediate checkpoint names / manifest (default: timestamp)",
    )
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing the final checkpoint")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    shuffle_d = args.shuffle_datums_seed if args.shuffle_datums_seed is not None else args.seed

    print(f"Model: {args.base_model}")
    tokenizer = get_tokenizer(args.base_model)
    renderer_name = model_info.get_recommended_renderer_name(args.base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    print("Loading mixed SFT data (train splits only)...")
    conversations, data_stats = sft_data.load_mixed_sft_conversations(
        args.n_gsm8k,
        args.n_tulu,
        args.n_code,
        args.seed,
    )
    print(f"  Conversations: {len(conversations)} (stats: {json.dumps(data_stats, indent=2)})")

    print("Packing conversations into datums...")
    all_data, pack_stats = conversations_to_datums(conversations, renderer, args.max_length)
    print(f"  Datums: {pack_stats['n_datums']}, skipped (errors): {pack_stats['skipped_datum_errors']}")
    if not all_data:
        raise SystemExit("No valid training datums; check max_length and dataset access.")

    rng = np.random.default_rng(shuffle_d)
    order = rng.permutation(len(all_data))
    all_data = [all_data[i] for i in order]

    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    print("  Training client ready")

    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    print(
        f"\nTraining {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr}, "
        f"datums={len(all_data)})..."
    )

    saved_intermediate: list[dict] = []

    for step in range(args.num_steps):
        start = (step * args.batch_size) % len(all_data)
        batch = [all_data[(start + i) % len(all_data)] for i in range(args.batch_size)]

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        print(f"  Step {step + 1}/{args.num_steps} | Loss: {loss:.4f}")

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            name = f"{run_id}-step-{step + 1:06d}"
            ckpt = tc.save_weights_for_sampler(name=name).result()
            path = ckpt.path
            print(f"    Saved checkpoint: {name} -> {path}")
            entry = {
                "run_id": run_id,
                "kind": "intermediate",
                "step": step + 1,
                "name": name,
                "checkpoint_path": path,
                "loss": float(loss),
                "base_model": args.base_model,
            }
            saved_intermediate.append(entry)
            _append_manifest(entry)

    print(f"\nSaving final checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")

    if not args.no_publish:
        print("\nPublishing final checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published successfully!")
    else:
        print("\nSkipping publish (--no_publish).")

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
    }

    info = {
        "checkpoint_path": checkpoint_path,
        "base_model": args.base_model,
        "renderer_name": renderer_name,
        "training": training_meta,
        "data": {**data_stats, **pack_stats, "n_conversations": len(conversations)},
        "published": not args.no_publish,
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
