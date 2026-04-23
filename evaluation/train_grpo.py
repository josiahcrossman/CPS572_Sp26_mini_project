"""
Train a GSM8K-only GRPO stage from an existing SFT checkpoint.

This script intentionally leaves the existing SFT/eval pipeline unchanged.
It resumes LoRA weights from a training checkpoint (tinker://.../weights/... from
TrainingClient.save_state), not a sampler export (.../sampler_weights/...).

Reward design (dense, 4-component, benchmark-aligned):
  R = R_correct(+1.0) + R_partial(0..0.05) + R_close(0..0.03) + R_structure(0..0.02)
  - R_correct:   exact match on final numeric answer (benchmark-style extraction)
  - R_partial:   overlap between intermediate gold rationale numbers and completion
  - R_close:     relative numeric closeness when prediction is parseable but wrong
  - R_structure: reasoning quality + ANSWER: format compliance

The prompt template mirrors inspect_evals.gsm8k so training and evaluation see
the same input distribution. Answer extraction prioritises "ANSWER:" lines, then
falls back to last-number-from-end (matching the benchmark scorer).

Advantage normalization uses a 50/50 hybrid of z-score and rank-based to handle
clustered reward distributions more robustly.

Usage:
    python evaluation/train_grpo.py
    python -m evaluation.train_grpo --steps 2 --group_size 2 --prompts_per_step 2 --max_tokens 64

Safety:
    - Uses only GSM8K train for training and reward computation
    - Does not train on IFEval, GSM8K test, or HumanEval problems
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import torch
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum, datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer

try:
    from . import sft_data
except ImportError:
    import sft_data

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_INFO = os.path.join(EVAL_DIR, "checkpoint_info.json")
RUNS_DIR = os.path.join(EVAL_DIR, "runs")
MANIFEST_NAME = "grpo_manifest.jsonl"

R_CORRECT = 1.0
R_PARTIAL_MAX = 0.05
R_CLOSE_MAX = 0.03
R_STRUCTURE_MAX = 0.02
TRIVIAL_NUMBERS = frozenset({"0", "1"})

# Exact prompt template from inspect_evals.gsm8k (MATH_PROMPT_TEMPLATE).
# {question} is replaced with the raw GSM8K question text.
GSM8K_PROMPT_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your '
    'response should be of the form "ANSWER: $ANSWER" (without quotes) where '
    '$ANSWER is the answer to the problem.\n'
    '\n'
    '{question}\n'
    '\n'
    'Remember to put your answer on its own line at the end in the form '
    '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the '
    'problem, and you do not need to use a \\boxed command.\n'
    '\n'
    'Reasoning:'
)

_RE_ANSWER_LINE = re.compile(r"(?i)^\s*ANSWER\s*:\s*(.*)\s*$")


@dataclass
class GSM8KRecord:
    id: int
    question: str
    answer: str
    gold_final_answer: str
    gold_intermediate_numbers: frozenset


def _append_manifest(entry: dict[str, Any]) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, MANIFEST_NAME)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_checkpoint_info(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def canonicalize_number(text: str) -> str | None:
    cleaned = text.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        dec = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not dec.is_finite():
        return None
    try:
        if dec == 0:
            return "0"
        if dec == dec.to_integral():
            return str(dec.quantize(Decimal("1")))
        rendered = format(dec.normalize(), "f")
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        return rendered or "0"
    except (InvalidOperation, ValueError, ArithmeticError):
        return None


def extract_final_number(text: str) -> str | None:
    hash_match = re.findall(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if hash_match:
        return canonicalize_number(hash_match[-1])

    boxed_match = re.findall(r"\\boxed\{(-?\d[\d,]*(?:\.\d+)?)\}", text)
    if boxed_match:
        return canonicalize_number(boxed_match[-1])

    generic = re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", text)
    if not generic:
        return None
    return canonicalize_number(generic[-1])


def load_gsm8k_records(
    *,
    seed: int,
    train_size: int | None,
    dev_size: int,
    cache_dir: str | None,
) -> tuple[list[GSM8KRecord], list[GSM8KRecord]]:
    ds = load_dataset("openai/gsm8k", "main", split="train", cache_dir=cache_dir)
    ds = ds.shuffle(seed=seed)

    total_requested = (train_size or (len(ds) - dev_size)) + dev_size
    total_requested = min(total_requested, len(ds))
    subset = ds.select(range(total_requested))

    train_limit = total_requested - min(dev_size, total_requested)
    if train_size is not None:
        train_limit = min(train_size, train_limit)

    train_records: list[GSM8KRecord] = []
    dev_records: list[GSM8KRecord] = []

    for idx, row in enumerate(subset):
        gold = extract_final_number(str(row["answer"]))
        if gold is None:
            continue
        question_text = str(row["question"]).strip()
        answer_text = str(row["answer"]).strip()
        intermediate = compute_intermediate_numbers(question_text, answer_text, gold)
        rec = GSM8KRecord(
            id=idx,
            question=question_text,
            answer=answer_text,
            gold_final_answer=gold,
            gold_intermediate_numbers=intermediate,
        )
        if len(train_records) < train_limit:
            train_records.append(rec)
        else:
            dev_records.append(rec)

    if not train_records:
        raise SystemExit("No GSM8K train records loaded for GRPO.")

    return train_records, dev_records


def sample_batch(records: list[GSM8KRecord], start: int, batch_size: int) -> tuple[list[GSM8KRecord], int]:
    if not records:
        return [], start
    out = [records[(start + i) % len(records)] for i in range(batch_size)]
    return out, (start + batch_size) % len(records)


def decode_completion(tokenizer, tokens: list[int]) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=True)


def build_grpo_user_content(question: str) -> str:
    """Wrap the question in the inspect_evals GSM8K template so training matches benchmark prompts."""
    return GSM8K_PROMPT_TEMPLATE.format(question=question.strip())


def _extract_last_number_from_end(text: str) -> str | None:
    """Mimic the benchmark scorer: scan whitespace-separated tokens from the end,
    return the first one that looks numeric (after stripping commas/dollar signs)."""
    words = text.split()
    for word in reversed(words):
        cleaned = word.strip("$,;:()\"'").replace(",", "")
        if cleaned.replace(".", "", 1).replace("-", "", 1).isdigit():
            return canonicalize_number(cleaned)
    return None


def parse_math_prediction(completion_text: str) -> str | None:
    """Extract the predicted answer, aligned with the benchmark.

    Priority order:
      1. Last ``ANSWER:`` line (what the prompt template asks for)
      2. Last number from the end of the text (what the benchmark scorer does)
    """
    lines = completion_text.splitlines()
    answer_indices = [i for i, ln in enumerate(lines) if _RE_ANSWER_LINE.match(ln)]
    if answer_indices:
        m = _RE_ANSWER_LINE.match(lines[answer_indices[-1]])
        tail = m.group(1).strip() if m else ""
        if tail:
            nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", tail)
            if nums:
                return canonicalize_number(nums[-1])
    return _extract_last_number_from_end(completion_text)


def extract_all_numbers(text: str) -> set[str]:
    """Extract all canonicalized numbers from text."""
    matches = re.findall(r"(?<!\w)-?\d[\d,]*(?:\.\d+)?", text)
    result: set[str] = set()
    for m in matches:
        c = canonicalize_number(m)
        if c is not None:
            result.add(c)
    return result


def compute_intermediate_numbers(question: str, answer: str, gold_final: str) -> frozenset:
    """Numbers from the gold rationale that aren't in the question or the final answer."""
    question_nums = extract_all_numbers(question)
    answer_nums = extract_all_numbers(answer)
    intermediate = answer_nums - question_nums - {gold_final} - TRIVIAL_NUMBERS
    return frozenset(intermediate)


def numeric_closeness(pred: str, gold: str) -> float:
    """Relative closeness in [0, 1]: 1 when equal, 0 when far apart."""
    try:
        p, g = float(pred), float(gold)
    except (ValueError, OverflowError):
        return 0.0
    if g == p:
        return 1.0
    denom = max(abs(g), 1.0)
    return max(0.0, 1.0 - abs(p - g) / denom)


def structure_score(completion_text: str) -> float:
    """Quality of reasoning structure in [0, 1], aligned with benchmark expectations."""
    lines = [ln.strip() for ln in completion_text.splitlines() if ln.strip()]
    if not lines:
        return 0.0

    score = 0.0

    if len(lines) >= 3:
        score += 0.40
    elif len(lines) >= 2:
        score += 0.20

    if re.search(r"(?i)^\s*ANSWER\s*:", completion_text, re.MULTILINE):
        score += 0.60

    return min(score, 1.0)


def shaped_reward(
    record: GSM8KRecord,
    completion_text: str,
) -> tuple[float, dict[str, float]]:
    """
    Dense 4-component reward (benchmark-aligned):
      R = R_correct + R_partial + R_close + R_structure
    Correctness dominates (~1.0); shaping provides a small gradient nudge (~0.10 max).
    """
    pred = parse_math_prediction(completion_text)

    is_correct = pred is not None and pred == record.gold_final_answer
    c_correct = R_CORRECT if is_correct else 0.0

    c_partial = 0.0
    if record.gold_intermediate_numbers:
        completion_nums = extract_all_numbers(completion_text)
        overlap = len(record.gold_intermediate_numbers & completion_nums)
        c_partial = (overlap / len(record.gold_intermediate_numbers)) * R_PARTIAL_MAX

    c_close = 0.0
    if pred is not None and not is_correct:
        c_close = numeric_closeness(pred, record.gold_final_answer) * R_CLOSE_MAX

    c_structure = structure_score(completion_text) * R_STRUCTURE_MAX

    total = c_correct + c_partial + c_close + c_structure
    parts = {
        "r_correct": c_correct,
        "r_partial": c_partial,
        "r_close": c_close,
        "r_structure": c_structure,
        "r_total": total,
    }
    return total, parts


def format_reward_histogram(values: list[float], *, bins: tuple[float, ...]) -> str:
    """Compact histogram for logging (counts per bin, last bin inclusive of right edge)."""
    if not values:
        return "{}"
    arr = np.asarray(values, dtype=np.float64)
    counts, _ = np.histogram(arr, bins=list(bins))
    labels: list[str] = []
    for i in range(len(bins) - 1):
        labels.append(f"[{bins[i]:.1f},{bins[i + 1]:.1f})")
    parts = [f"{lab}:{int(c)}" for lab, c in zip(labels, counts, strict=True)]
    return "{" + ", ".join(parts) + "}"


def compute_old_logprob_sum(
    sampler,
    full_model_input: types.ModelInput,
    shifted_weights: list[float],
) -> tuple[float, int] | None:
    """Return (sum_of_logprobs, n_active_tokens) or None on failure."""
    prompt_logprobs = sampler.compute_logprobs(full_model_input).result()
    shifted_logprobs = prompt_logprobs[1 : len(shifted_weights) + 1]
    if len(shifted_logprobs) != len(shifted_weights):
        return None

    total = 0.0
    n_active = 0
    for logprob, weight in zip(shifted_logprobs, shifted_weights, strict=True):
        if weight <= 0:
            continue
        if logprob is None:
            return None
        total += float(logprob)
        n_active += 1
    if n_active == 0:
        return None
    return total, n_active


def make_grpo_loss(clip_eps: float, meta: list[tuple[float, float, int]]):
    """meta[i] = (advantage, old_logprob_sum, n_tokens) for data[i]."""

    def grpo_loss(data: list[types.Datum], logprobs_list: list[torch.Tensor]):
        losses: list[torch.Tensor] = []
        ratios: list[float] = []
        advantages: list[float] = []

        for datum, logprobs, (advantage, old_logprob_sum, n_tokens) in zip(
            data, logprobs_list, meta, strict=True
        ):
            mask = torch.tensor(
                datum.loss_fn_inputs["weights"].data,
                dtype=logprobs.dtype,
                device=logprobs.device,
            )
            if mask.shape != logprobs.shape:
                mask = mask.reshape(logprobs.shape)
            n_active = torch.count_nonzero(mask)
            if n_active == 0:
                continue

            current_logprob_sum = torch.sum(logprobs * mask)
            raw_log_ratio = (
                current_logprob_sum
                - torch.tensor(old_logprob_sum, dtype=logprobs.dtype, device=logprobs.device)
            )
            avg_log_ratio = raw_log_ratio / max(n_tokens, 1)
            log_ratio = torch.clamp(avg_log_ratio, min=-5.0, max=5.0)
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

            adv_tensor = torch.tensor(advantage, dtype=logprobs.dtype, device=logprobs.device)
            objective = torch.minimum(ratio * adv_tensor, clipped_ratio * adv_tensor)
            losses.append(-objective)
            ratios.append(float(ratio.detach().cpu().item()))
            advantages.append(advantage)

        if not losses:
            zero = torch.zeros((), dtype=torch.float32, requires_grad=True)
            return zero, {
                "grpo_loss": 0.0,
                "ratio_mean": 0.0,
                "adv_mean": 0.0,
                "n_valid_datums": 0.0,
            }

        loss = torch.stack(losses).mean()
        metrics = {
            "grpo_loss": float(loss.detach().cpu().item()),
            "ratio_mean": float(np.mean(ratios)),
            "adv_mean": float(np.mean(advantages)),
            "n_valid_datums": float(len(losses)),
        }
        return loss, metrics

    return grpo_loss


REGRESSION_TASKS = ("tulu", "code")


def build_regression_sets(
    renderer,
    *,
    max_length: int,
    n_per_task: int,
    seed: int,
    cache_dir: str | None,
) -> dict[str, list[types.Datum]]:
    """Load held-out slices from the TRAIN splits of tulu-3 and OpenCodeInstruct,
    pack them into datums, and return them keyed by task.

    These datums are used only for NLL measurement during GRPO — they never
    flow into training gradients. The seed is offset from the GRPO seed so
    these examples don't overlap the GSM8K batch order (they come from
    different datasets anyway).
    """
    if n_per_task <= 0:
        return {t: [] for t in REGRESSION_TASKS}

    print(f"Building regression check sets ({n_per_task} per task from train splits)...")
    tulu_convos = sft_data.load_tulu_conversations(n_per_task, seed, cache_dir=cache_dir)
    code_convos = sft_data.load_opencode_conversations(n_per_task, seed, cache_dir=cache_dir)

    def _pack(convos):
        out: list[types.Datum] = []
        for convo in convos:
            try:
                d = conversation_to_datum(
                    convo,
                    renderer,
                    max_length=max_length,
                    train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
                )
                out.append(d)
            except Exception:
                continue
        return out

    sets = {"tulu": _pack(tulu_convos), "code": _pack(code_convos)}
    print(f"  tulu regression datums: {len(sets['tulu'])}")
    print(f"  code regression datums: {len(sets['code'])}")
    return sets


_ALIGNMENT_DEBUG_PRINTED = False


def _datum_nll(sampler, datum: types.Datum) -> float | None:
    """Mean NLL of a single SFT datum under `sampler`.

    `sampler.compute_logprobs` returns per-token logprobs. Different tinker
    datum builders produce weights at different lengths:
      - `datum_from_model_input_weights` → weights length = tokens length - 1
        (logprob at index 0 has no prior context and is dropped)
      - `conversation_to_datum` → weights length = tokens length
        (weight[0] is masked to 0 so position 0 contributes nothing)
    Handle both: align the tail of full_logprobs with weights.
    Prints the observed alignment on the first call so any further mismatch
    is diagnosable.
    """
    global _ALIGNMENT_DEBUG_PRINTED
    weights = datum.loss_fn_inputs["weights"].data
    try:
        full_logprobs = sampler.compute_logprobs(datum.model_input).result()
    except Exception as e:
        if not _ALIGNMENT_DEBUG_PRINTED:
            print(f"  [regression] compute_logprobs raised: {type(e).__name__}: {e}")
            _ALIGNMENT_DEBUG_PRINTED = True
        return None

    L_lp = len(full_logprobs)
    L_w = len(weights)
    if L_lp == L_w:
        aligned = full_logprobs
    elif L_lp == L_w + 1:
        aligned = full_logprobs[1:]
    elif L_lp > L_w:
        aligned = full_logprobs[-L_w:]
    else:
        if not _ALIGNMENT_DEBUG_PRINTED:
            print(f"  [regression] length mismatch: logprobs={L_lp}, weights={L_w}")
            _ALIGNMENT_DEBUG_PRINTED = True
        return None

    if not _ALIGNMENT_DEBUG_PRINTED:
        print(
            f"  [regression] alignment ok: logprobs={L_lp}, weights={L_w} "
            f"(using last {len(aligned)})"
        )
        _ALIGNMENT_DEBUG_PRINTED = True

    total = 0.0
    n_active = 0
    for lp, w in zip(aligned, weights, strict=True):
        if w <= 0:
            continue
        if lp is None:
            return None
        total += float(lp) * float(w)
        n_active += 1
    if n_active == 0:
        return None
    return float(-total / n_active)


def run_regression_check(
    sampler,
    regression_sets: dict[str, list[types.Datum]],
) -> dict[str, float | None]:
    """Mean NLL per task on the held-out regression sets. Returns None for a
    task if every datum fails to score (extremely unlikely but possible if
    logprob alignment fails)."""
    out: dict[str, float | None] = {}
    for task, datums in regression_sets.items():
        if not datums:
            out[task] = None
            continue
        losses: list[float] = []
        for d in datums:
            nll = _datum_nll(sampler, d)
            if nll is not None:
                losses.append(nll)
        out[task] = float(np.mean(losses)) if losses else None
    return out


def evaluate_dev_reward(
    sampler,
    tokenizer,
    renderer,
    records: list[GSM8KRecord],
    *,
    max_examples: int,
    max_tokens: int,
) -> float | None:
    """Benchmark-aligned dev accuracy: extract answer the same way the scorer does."""
    if not records or max_examples <= 0:
        return None

    subset = records[: max_examples]
    correct = 0
    for record in subset:
        prompt = renderer.build_generation_prompt(
            [{"role": "user", "content": build_grpo_user_content(record.question)}],
            role="assistant",
        )
        response = sampler.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0),
        ).result()
        completion = decode_completion(tokenizer, response.sequences[0].tokens)
        pred = parse_math_prediction(completion)
        correct += int(pred is not None and pred == record.gold_final_answer)
    return correct / len(subset)


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K-only GRPO from an existing SFT checkpoint")
    parser.add_argument("--checkpoint_info", type=str, default=DEFAULT_CHECKPOINT_INFO)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--training_weights_path",
        type=str,
        default=None,
        help="Tinker path under .../weights/... (from save_state). Overrides checkpoint_path for GRPO.",
    )
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--renderer_name", type=str, default=None)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--prompts_per_step", type=int, default=8)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--snapshot_every", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--dev_size", type=int, default=256)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--dev_eval_size", type=int, default=64)
    parser.add_argument("--snapshot_ttl_seconds", type=int, default=86400)
    parser.add_argument(
        "--regression_n_per_task",
        type=int,
        default=32,
        help="Held-out datums per non-GSM8K task (tulu, code) used for NLL "
             "regression checks. Set to 0 to disable.",
    )
    parser.add_argument(
        "--regression_max_length",
        type=int,
        default=2048,
        help="Max token length for packing regression datums (match SFT).",
    )
    parser.add_argument(
        "--regression_warn_pct",
        type=float,
        default=0.15,
        help="Warn if NLL on tulu/code rises by more than this fraction vs. baseline.",
    )
    args = parser.parse_args()

    info: dict[str, Any] = {}
    if os.path.exists(args.checkpoint_info):
        info = load_checkpoint_info(args.checkpoint_info)
    elif not ((args.checkpoint_path or args.training_weights_path) and args.base_model):
        raise SystemExit(
            f"checkpoint_info not found at {args.checkpoint_info}; provide "
            "--checkpoint_path or --training_weights_path, and --base_model."
        )

    base_model = args.base_model or info.get("base_model")
    renderer_name = args.renderer_name or info.get("renderer_name")
    resume_path = (
        args.training_weights_path
        or info.get("training_weights_path")
        or args.checkpoint_path
        or info.get("checkpoint_path")
    )
    if not resume_path or not base_model:
        raise SystemExit(
            "Need a training weights path and base_model. Use training_weights_path in checkpoint_info.json "
            "(from SFT save_state), or pass --training_weights_path / --checkpoint_path, plus --base_model."
        )
    if "/sampler_weights/" in resume_path:
        raise SystemExit(
            "GRPO must load from a training checkpoint (tinker://.../weights/<name>), not a sampler export "
            "(.../sampler_weights/...). Re-run SFT with evaluation/train_and_publish.py (it now calls save_state) "
            "or pass --training_weights_path pointing to a save_state path from the Tinker console."
        )
    if not renderer_name:
        renderer_name = model_info.get_recommended_renderer_name(base_model)

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    cache_dir = os.environ.get("SFT_DATASETS_CACHE") or os.environ.get("HF_DATASETS_CACHE")

    print(f"Base model: {base_model}")
    print(f"Resume training weights path: {resume_path}")
    print(f"Renderer: {renderer_name}")
    print("Loading tokenizer and renderer...")
    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    print("Loading GSM8K train split only...")
    train_records, dev_records = load_gsm8k_records(
        seed=args.seed,
        train_size=args.train_size,
        dev_size=args.dev_size,
        cache_dir=cache_dir,
    )
    print(f"  Train prompts: {len(train_records)}")
    print(f"  Held-out train dev prompts: {len(dev_records)}")

    regression_sets = build_regression_sets(
        renderer,
        max_length=args.regression_max_length,
        n_per_task=args.regression_n_per_task,
        seed=args.seed + 101,
        cache_dir=cache_dir,
    )
    have_regression = any(len(v) > 0 for v in regression_sets.values())

    print("Creating training client from checkpoint path (weights only)...")
    sc = tinker.ServiceClient()
    tc = sc.create_training_client_from_state(resume_path)

    # Baseline regression NLL from the SFT checkpoint BEFORE any GRPO updates.
    regression_baseline: dict[str, float | None] = {t: None for t in REGRESSION_TASKS}
    if have_regression:
        print("Measuring baseline regression NLL on pre-GRPO (SFT) checkpoint...")
        baseline_snapshot = tc.save_weights_for_sampler(
            name=f"{args.run_id or time.strftime('%Y%m%d-%H%M%S')}-regression-baseline",
            ttl_seconds=args.snapshot_ttl_seconds,
        ).result()
        baseline_sampler = sc.create_sampling_client(model_path=baseline_snapshot.path)
        regression_baseline = run_regression_check(baseline_sampler, regression_sets)
        base_str = " ".join(
            f"{t}={regression_baseline[t]:.4f}" if regression_baseline[t] is not None else f"{t}=n/a"
            for t in REGRESSION_TASKS
        )
        print(f"  Baseline NLL: {base_str}")

    adam_params = types.AdamParams(
        learning_rate=args.lr,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )
    rng = np.random.default_rng(args.seed)

    cursor = 0
    rollout_sampler = None
    rollout_snapshot_path = None

    total_rewards: list[float] = []
    total_valid_groups = 0
    total_skipped_groups = 0
    reward_hist_bins = (0.0, 0.05, 0.15, 0.30, 0.50, 0.80, 1.0, 1.5)

    for step in range(1, args.steps + 1):
        if rollout_sampler is None or (step - 1) % args.snapshot_every == 0:
            snapshot_name = f"{run_id}-rollout-step-{step:06d}"
            snapshot_resp = tc.save_weights_for_sampler(
                name=snapshot_name,
                ttl_seconds=args.snapshot_ttl_seconds,
            ).result()
            rollout_snapshot_path = snapshot_resp.path
            rollout_sampler = sc.create_sampling_client(model_path=rollout_snapshot_path)
            print(f"[step {step}] refreshed rollout snapshot: {rollout_snapshot_path}")

        batch_records, cursor = sample_batch(train_records, cursor, args.prompts_per_step)
        grpo_datums: list[types.Datum] = []
        grpo_meta: list[tuple[float, float, int]] = []
        step_rewards: list[float] = []
        step_component_sums: dict[str, float] = {
            "r_correct": 0.0,
            "r_partial": 0.0,
            "r_close": 0.0,
            "r_structure": 0.0,
            "r_total": 0.0,
        }
        step_reward_samples = 0
        valid_groups = 0
        skipped_groups = 0

        for record in batch_records:
            prompt_model_input = renderer.build_generation_prompt(
                [{"role": "user", "content": build_grpo_user_content(record.question)}],
                role="assistant",
            )
            sampling_seed = int(rng.integers(0, 2**31 - 1))
            sample_resp = rollout_sampler.sample(
                prompt=prompt_model_input,
                num_samples=args.group_size,
                sampling_params=types.SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=1.0,
                    seed=sampling_seed,
                ),
            ).result()

            completions: list[tuple[list[int], float]] = []
            rewards: list[float] = []
            for seq in sample_resp.sequences:
                tokens = list(seq.tokens)
                text = decode_completion(tokenizer, tokens)
                reward, parts = shaped_reward(record, text)
                completions.append((tokens, reward))
                rewards.append(reward)
                step_rewards.append(reward)
                for k in step_component_sums:
                    step_component_sums[k] += parts[k]
                step_reward_samples += 1

            rewards_arr = np.asarray(rewards, dtype=np.float32)
            std = float(rewards_arr.std())
            if std < 1e-6:
                skipped_groups += 1
                continue

            valid_groups += 1
            mean = float(rewards_arr.mean())
            z_adv = (rewards_arr - mean) / max(std, 1e-6)
            order = rewards_arr.argsort()
            ranks = np.empty_like(rewards_arr)
            ranks[order] = np.arange(len(rewards_arr), dtype=np.float32)
            rank_adv = 2.0 * ranks / max(len(ranks) - 1, 1) - 1.0
            advantages = 0.5 * z_adv + 0.5 * rank_adv
            for (completion_tokens, _reward), advantage in zip(completions, advantages, strict=True):
                prompt_tokens = prompt_model_input.to_ints()
                full_tokens = prompt_tokens + completion_tokens
                if len(full_tokens) < 2:
                    continue

                full_weights = torch.zeros(len(full_tokens), dtype=torch.float32)
                full_weights[len(prompt_tokens) :] = 1.0
                datum = datum_from_model_input_weights(
                    types.ModelInput.from_ints(full_tokens),
                    full_weights,
                )
                old_result = compute_old_logprob_sum(
                    rollout_sampler,
                    types.ModelInput.from_ints(full_tokens),
                    datum.loss_fn_inputs["weights"].data,
                )
                if old_result is None:
                    continue
                old_logprob_sum, n_active_tokens = old_result
                grpo_meta.append((float(advantage), float(old_logprob_sum), n_active_tokens))
                grpo_datums.append(datum)

        if not grpo_datums:
            total_valid_groups += valid_groups
            total_skipped_groups += skipped_groups
            print(f"[step {step}] no valid GRPO datums; skipped_groups={skipped_groups}")
            continue

        loss_fn = make_grpo_loss(args.clip_eps, grpo_meta)
        fwdbwd_result = tc.forward_backward_custom(grpo_datums, loss_fn).result()
        tc.optim_step(adam_params).result()

        total_rewards.extend(step_rewards)
        total_valid_groups += valid_groups
        total_skipped_groups += skipped_groups

        metrics = dict(fwdbwd_result.metrics)
        reward_mean = float(np.mean(step_rewards)) if step_rewards else 0.0
        reward_std = float(np.std(step_rewards)) if step_rewards else 0.0
        n_batch_groups = valid_groups + skipped_groups
        skip_rate_step = float(skipped_groups / n_batch_groups) if n_batch_groups else 0.0
        skip_rate_run = float(
            total_skipped_groups / max(1, total_valid_groups + total_skipped_groups)
        )
        hist_str = format_reward_histogram(step_rewards, bins=reward_hist_bins)
        comp_means: dict[str, float] = {}
        if step_reward_samples:
            for k, s in step_component_sums.items():
                comp_means[k] = float(s / step_reward_samples)
        comp_str = (
            f"corr={comp_means.get('r_correct', 0):.3f} part={comp_means.get('r_partial', 0):.4f} "
            f"close={comp_means.get('r_close', 0):.4f} struct={comp_means.get('r_structure', 0):.4f}"
        )
        print(
            f"[step {step}/{args.steps}] datums={len(grpo_datums)} "
            f"R_mean={reward_mean:.4f} R_std={reward_std:.4f} | {comp_str} | "
            f"skip_step={skip_rate_step:.2%} skip_run={skip_rate_run:.2%} "
            f"valid_g={valid_groups} skip_g={skipped_groups} | hist={hist_str} | "
            f"loss={metrics.get('grpo_loss', 0.0):.6f} ratio={metrics.get('ratio_mean', 0.0):.4f}"
        )

        if args.save_every > 0 and step % args.save_every == 0:
            checkpoint_name = f"{run_id}-grpo-step-{step:06d}"
            saved = tc.save_weights_for_sampler(name=checkpoint_name).result()
            print(f"  Saved GRPO checkpoint: {saved.path}")

            dev_sampler = None
            dev_acc = None
            if args.dev_eval_size > 0 and dev_records:
                dev_sampler = sc.create_sampling_client(model_path=saved.path)
                dev_acc = evaluate_dev_reward(
                    dev_sampler,
                    tokenizer,
                    renderer,
                    dev_records,
                    max_examples=args.dev_eval_size,
                    max_tokens=args.max_tokens,
                )
                print(f"  Held-out train dev accuracy ({min(args.dev_eval_size, len(dev_records))} ex): {dev_acc:.4f}")

            regression_nll: dict[str, float | None] = {t: None for t in REGRESSION_TASKS}
            regression_delta_pct: dict[str, float | None] = {t: None for t in REGRESSION_TASKS}
            if have_regression:
                if dev_sampler is None:
                    dev_sampler = sc.create_sampling_client(model_path=saved.path)
                regression_nll = run_regression_check(dev_sampler, regression_sets)
                parts: list[str] = []
                regressions: list[str] = []
                for t in REGRESSION_TASKS:
                    cur = regression_nll[t]
                    base = regression_baseline.get(t)
                    if cur is None or base is None or base <= 0:
                        parts.append(f"{t}=n/a")
                        continue
                    delta_pct = (cur - base) / base
                    regression_delta_pct[t] = float(delta_pct)
                    parts.append(f"{t} NLL={cur:.4f} (Δ{delta_pct:+.1%} vs baseline)")
                    if delta_pct > args.regression_warn_pct:
                        regressions.append(t)
                print(f"  Regression check: {' | '.join(parts)}")
                if regressions:
                    print(
                        f"  WARNING: {', '.join(regressions)} NLL rose more than "
                        f"{args.regression_warn_pct:.0%} — model may be overfitting to GSM8K."
                    )

            entry = {
                "run_id": run_id,
                "step": step,
                "checkpoint_path": saved.path,
                "rollout_snapshot_path": rollout_snapshot_path,
                "base_model": base_model,
                "renderer_name": renderer_name,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "shaped_reward_means": comp_means,
                "reward_histogram": hist_str,
                "skip_rate_step": skip_rate_step,
                "skip_rate_run": skip_rate_run,
                "valid_groups": valid_groups,
                "skipped_groups": skipped_groups,
                "metrics": metrics,
                "dev_accuracy": dev_acc,
                "regression_nll": regression_nll,
                "regression_nll_baseline": dict(regression_baseline),
                "regression_delta_pct": regression_delta_pct,
                "training": {
                    "group_size": args.group_size,
                    "prompts_per_step": args.prompts_per_step,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "lr": args.lr,
                    "clip_eps": args.clip_eps,
                    "seed": args.seed,
                },
                "dataset": {
                    "source": "openai/gsm8k",
                    "config": "main",
                    "split": "train",
                    "train_prompts": len(train_records),
                    "dev_prompts": len(dev_records),
                },
            }
            _append_manifest(entry)
            print("  Evaluate with:")
            print(
                f'    python -m evaluation.eval_all --checkpoint_path "{saved.path}" '
                f"--base_model {base_model} --limit 50"
            )

    if total_rewards:
        run_skip = float(
            total_skipped_groups / max(1, total_valid_groups + total_skipped_groups)
        )
        print(
            "\nTraining finished:"
            f" R_mean={float(np.mean(total_rewards)):.4f}"
            f" R_std={float(np.std(total_rewards)):.4f}"
            f" skip_rate_run={run_skip:.2%}"
            f" valid_groups={total_valid_groups}"
            f" skipped_groups={total_skipped_groups}"
            f" hist_total={format_reward_histogram(total_rewards, bins=reward_hist_bins)}"
        )
    else:
        print("\nTraining finished with no reward-bearing updates.")


if __name__ == "__main__":
    main()
