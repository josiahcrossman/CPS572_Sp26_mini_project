"""
Load train-split SFT conversations for the multi-task baseline (IF + math + code).

Uses only training data (no IFEval / GSM8K test / HumanEval):
  - openai/gsm8k (train)
  - allenai/tulu-3-sft-mixture (train)
  - nvidia/OpenCodeInstruct (train)
  - nvidia/OpenMathInstruct-2 (train, deduplicated against GSM8K test)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

_CLEAN_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")

Conversation = list[dict[str, str]]

ROLE_ALIASES = {
    "human": "user",
    "gpt": "assistant",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
}


def _cache_dir() -> str | None:
    return os.environ.get("SFT_DATASETS_CACHE") or os.environ.get("HF_DATASETS_CACHE")


def normalize_chat_messages(messages: Any) -> Conversation | None:
    """Map Tulu-style message lists to user/assistant/system turns."""
    if not isinstance(messages, list) or not messages:
        return None
    out: Conversation = []
    for m in messages:
        if not isinstance(m, dict):
            return None
        role = m.get("role")
        if role is None:
            continue
        role = ROLE_ALIASES.get(str(role).lower(), str(role).lower())
        if role not in ("user", "assistant", "system"):
            continue
        content = m.get("content")
        if content is None:
            continue
        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
                elif isinstance(p, str):
                    parts.append(p)
            content = "\n".join(parts)
        text = str(content).strip()
        if not text:
            continue
        out.append({"role": role, "content": text})
    if not out or not any(m["role"] == "assistant" for m in out):
        return None
    return out


def _take_n_shuffled(ds: Dataset, n: int, seed: int) -> Dataset:
    n = min(int(n), len(ds))
    if n <= 0:
        return ds.select([])
    return ds.shuffle(seed=seed).select(range(n))


def load_gsm8k_conversations(n: int, seed: int, *, cache_dir: str | None = None) -> list[Conversation]:
    cache_dir = cache_dir or _cache_dir()
    ds = load_dataset("openai/gsm8k", "main", split="train", cache_dir=cache_dir)
    subset = _take_n_shuffled(ds, n, seed)
    convos: list[Conversation] = []
    for row in subset:
        q = str(row["question"]).strip()
        a = str(row["answer"]).strip()
        if q and a:
            convos.append(
                [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            )
    return convos


def load_tulu_conversations(n: int, seed: int, *, cache_dir: str | None = None) -> list[Conversation]:
    cache_dir = cache_dir or _cache_dir()
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", cache_dir=cache_dir)
    subset = _take_n_shuffled(ds, n, seed + 17)
    convos: list[Conversation] = []
    for row in subset:
        norm = normalize_chat_messages(row.get("messages"))
        if norm:
            convos.append(norm)
    return convos


def load_opencode_conversations(n: int, seed: int, *, cache_dir: str | None = None) -> list[Conversation]:
    cache_dir = cache_dir or _cache_dir()
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train", cache_dir=cache_dir)
    subset = _take_n_shuffled(ds, n, seed + 29)
    convos: list[Conversation] = []
    for row in subset:
        inp = str(row.get("input", "")).strip()
        out = str(row.get("output", "")).strip()
        if inp and out:
            convos.append(
                [
                    {"role": "user", "content": inp},
                    {"role": "assistant", "content": out},
                ]
            )
    return convos


def _word_ngrams(text: str, n: int) -> frozenset[tuple[str, ...]]:
    """Return the set of word n-grams from lowercased, punctuation-stripped text."""
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    if len(words) < n:
        return frozenset()
    return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def _build_gsm8k_test_ngrams(n: int, cache_dir: str | None) -> frozenset[tuple[str, ...]]:
    """Build the union of all word n-grams from every GSM8K test question."""
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir)
    all_ngrams: set[tuple[str, ...]] = set()
    for row in ds:
        all_ngrams |= _word_ngrams(str(row["question"]), n)
    return frozenset(all_ngrams)


def _extract_boxed_answer(text: str) -> str | None:
    """Extract content of the last \\boxed{...}, correctly handling nested braces."""
    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None  # unmatched brace


def _is_clean_number(s: str) -> bool:
    """True if s (after stripping commas) is a plain integer or simple decimal."""
    return bool(_CLEAN_NUMBER_RE.match(s.replace(",", "").strip()))


def load_openmath_conversations(
    n: int,
    seed: int,
    *,
    cache_dir: str | None = None,
    ngram_size: int = 8,
    oversample_factor: int = 15,
) -> list[Conversation]:
    """Load OpenMathInstruct-2 examples formatted to match the GSM8K answer style.

    Pipeline:
      1. Shuffle & oversample to have a large candidate pool.
      2. Extract the final answer from ``\\boxed{}`` in ``generated_solution``.
      3. Keep only examples whose answer is a clean integer or simple decimal.
      4. Discard any example whose ``problem`` shares an 8-gram (or higher) with
         any question in the GSM8K *test* split (contamination guard).
      5. Reformat assistant content as ``<solution>\\n#### <answer>``.

    Extraction failures (no ``\\boxed{}`` found) are logged at WARNING level so
    data-quality issues are visible without being fatal.
    """
    cache_dir = cache_dir or _cache_dir()

    logger.info(
        "Building GSM8K test %d-gram index for OpenMathInstruct-2 deduplication…",
        ngram_size,
    )
    test_ngrams = _build_gsm8k_test_ngrams(ngram_size, cache_dir)
    logger.info("GSM8K test index built: %d unique %d-grams.", len(test_ngrams), ngram_size)

    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", cache_dir=cache_dir)
    candidate_size = min(n * oversample_factor, len(ds))
    candidates = _take_n_shuffled(ds, candidate_size, seed + 53)

    n_extraction_failures = 0
    n_non_clean = 0
    n_contaminated = 0
    convos: list[Conversation] = []
    n_gsm8k_loaded = 0
    for row in candidates:
        if len(convos) >= n:
            break

        if "gsm" in str(row.get("source", "")).strip():
            n_gsm8k_loaded += 1
            continue
        
        problem = str(row.get("problem", "")).strip()
        solution = str(row.get("generated_solution", "")).strip()

        if not problem or not solution:
            n_extraction_failures += 1
            continue

        raw_answer = _extract_boxed_answer(solution)
        if raw_answer is None:
            n_extraction_failures += 1
            logger.debug("No \\boxed{} in solution for problem: %.80s…", problem)
            continue

        if not _is_clean_number(raw_answer):
            n_non_clean += 1
            continue

        clean_answer = raw_answer.replace(",", "").strip()

        # Contamination guard: reject if any ngram_size-gram overlaps with the test set.
        if _word_ngrams(problem, ngram_size) & test_ngrams:
            n_contaminated += 1
            continue

        convos.append(
            [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": f"{solution}\n#### {clean_answer}"},
            ]
        )

    logger.warning(
        "OpenMathInstruct-2 load summary — loaded: %d  |  \\boxed{} failures: %d  "
        "|  non-clean answers: %d  |  contaminated (test overlap): %d  |  GSM8K skipped: %d",
        len(convos),
        n_extraction_failures,
        n_non_clean,
        n_contaminated,
        n_gsm8k_loaded,
    )

    return convos


def _build_stats(n_gsm8k, n_tulu, n_code, gsm, tulu, code, seed, cache_dir):
    return {
        "n_gsm8k_requested": n_gsm8k,
        "n_tulu_requested": n_tulu,
        "n_code_requested": n_code,
        "n_gsm8k_loaded": len(gsm),
        "n_tulu_loaded": len(tulu),
        "n_code_loaded": len(code),
        "seed": seed,
        "cache_dir": cache_dir,
        "sources": [
            "openai/gsm8k (train)",
            "allenai/tulu-3-sft-mixture (train)",
            "nvidia/OpenCodeInstruct (train)",
        ],
    }


def load_mixed_sft_conversations(
    n_gsm8k: int,
    n_tulu: int,
    n_code: int,
    seed: int,
    use_gsm8k_only: bool,
    *,
    cache_dir: str | None = None,
) -> tuple[list[Conversation], dict[str, Any]]:
    """Load and concatenate three sources; returns (conversations, stats dict)."""
    cache_dir = cache_dir or _cache_dir()
    if use_gsm8k_only:
        gsm = load_gsm8k_conversations(n_gsm8k, seed, cache_dir=cache_dir)
    else:
        gsm = load_gsm8k_conversations(n_gsm8k//2, seed, cache_dir=cache_dir)
        openmath = load_openmath_conversations(n_gsm8k//2, seed, cache_dir=cache_dir)
    tulu = load_tulu_conversations(n_tulu, seed, cache_dir=cache_dir)
    code = load_opencode_conversations(n_code, seed, cache_dir=cache_dir)
    stats = _build_stats(n_gsm8k, n_tulu, n_code, gsm, tulu, code, seed, cache_dir)
    if use_gsm8k_only:
        return gsm + tulu + code, stats
    return gsm + openmath + tulu + code, stats


def load_mixed_sft_conversations_tagged(
    n_gsm8k: int,
    n_tulu: int,
    n_code: int,
    seed: int,
    use_gsm8k_only: bool,
    *,
    cache_dir: str | None = None,
) -> tuple[list[Conversation], list[str], dict[str, Any]]:
    """Load conversations with per-example source labels ('gsm8k', 'tulu', 'code')."""
    cache_dir = cache_dir or _cache_dir()
    if use_gsm8k_only:
        gsm = load_gsm8k_conversations(n_gsm8k, seed, cache_dir=cache_dir)
    else:
        gsm = load_gsm8k_conversations(n_gsm8k//2, seed, cache_dir=cache_dir)
        openmath = load_openmath_conversations(n_gsm8k//2, seed, cache_dir=cache_dir)
    tulu = load_tulu_conversations(n_tulu, seed, cache_dir=cache_dir)
    code = load_opencode_conversations(n_code, seed, cache_dir=cache_dir)
    if use_gsm8k_only:
        conversations = gsm + tulu + code
        labels = ["gsm8k"] * len(gsm) + ["tulu"] * len(tulu) + ["code"] * len(code)
    else:
        conversations = gsm + openmath + tulu + code
        labels = ["gsm8k"] * len(gsm) + ["gsm8k"] * len(openmath) + ["tulu"] * len(tulu) + ["code"] * len(code)
    stats = _build_stats(n_gsm8k, n_tulu, n_code, gsm, tulu, code, seed, cache_dir) 
    return conversations, labels, stats
