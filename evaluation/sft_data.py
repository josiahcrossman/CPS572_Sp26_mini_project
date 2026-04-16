"""
Load train-split SFT conversations for the multi-task baseline (IF + math + code).

Uses only training data (no IFEval / GSM8K test / HumanEval):
  - openai/gsm8k (train)
  - allenai/tulu-3-sft-mixture (train)
  - nvidia/OpenCodeInstruct (train)
"""

from __future__ import annotations

import os
from typing import Any

from datasets import Dataset, load_dataset

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
    *,
    cache_dir: str | None = None,
) -> tuple[list[Conversation], dict[str, Any]]:
    """Load and concatenate three sources; returns (conversations, stats dict)."""
    cache_dir = cache_dir or _cache_dir()
    gsm = load_gsm8k_conversations(n_gsm8k, seed, cache_dir=cache_dir)
    tulu = load_tulu_conversations(n_tulu, seed, cache_dir=cache_dir)
    code = load_opencode_conversations(n_code, seed, cache_dir=cache_dir)
    stats = _build_stats(n_gsm8k, n_tulu, n_code, gsm, tulu, code, seed, cache_dir)
    return gsm + tulu + code, stats


def load_mixed_sft_conversations_tagged(
    n_gsm8k: int,
    n_tulu: int,
    n_code: int,
    seed: int,
    *,
    cache_dir: str | None = None,
) -> tuple[list[Conversation], list[str], dict[str, Any]]:
    """Load conversations with per-example source labels ('gsm8k', 'tulu', 'code')."""
    cache_dir = cache_dir or _cache_dir()
    gsm = load_gsm8k_conversations(n_gsm8k, seed, cache_dir=cache_dir)
    tulu = load_tulu_conversations(n_tulu, seed, cache_dir=cache_dir)
    code = load_opencode_conversations(n_code, seed, cache_dir=cache_dir)
    conversations = gsm + tulu + code
    labels = ["gsm8k"] * len(gsm) + ["tulu"] * len(tulu) + ["code"] * len(code)
    stats = _build_stats(n_gsm8k, n_tulu, n_code, gsm, tulu, code, seed, cache_dir)
    return conversations, labels, stats
