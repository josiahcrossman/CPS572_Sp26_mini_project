"""
Produce exactly two overlay plots of OVERALL loss only:

  1. Llama-3.2-3B, batch_size=8   — 4 configs overlayed
  2. Llama-3.2-3B, batch_size=16  — 4 configs overlayed

Each plot shows, per run:
  - faint raw train loss
  - train loss EMA (dashed)
  - validation loss (solid, with markers)
  - star marker on the lowest-val-loss checkpoint
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RAW = (HERE / "raw.txt").read_text()
TEXT = re.sub(r"\s+", " ", RAW)

CMD_RE = re.compile(
    r"python\s+evaluation/train_and_publish\.py\s+(?P<args>.+?)"
    r"(?=python\s+evaluation/train_and_publish\.py|\Z)",
    re.DOTALL,
)
STEP_RE = re.compile(
    r"Step\s+(\d+)/(\d+)\s*\|\s*loss=([\d.]+)\s*\|\s*ema=([\d.]+)"
)
VAL_RE = re.compile(r"\[Val\s*@\s*step\s+(\d+)\]\s*loss=([\d.]+)")


def flag(args_block: str, name: str, default=None):
    m = re.search(rf"--{name}\s+([^\s]+)", args_block)
    return m.group(1).strip('"\'') if m else default


def parse_run(args_block: str) -> dict:
    base = flag(args_block, "base_model", "unknown")
    ckpt = flag(args_block, "checkpoint_name", "unnamed")
    batch = int(flag(args_block, "batch_size", "0") or 0)

    train = []
    for m in STEP_RE.finditer(args_block):
        train.append(
            {"step": int(m.group(1)), "loss": float(m.group(3)), "ema": float(m.group(4))}
        )
    val = [
        {"step": int(m.group(1)), "loss": float(m.group(2))}
        for m in VAL_RE.finditer(args_block)
    ]
    return {
        "base_model": base,
        "checkpoint_name": ckpt,
        "batch_size": batch,
        "train": train,
        "val": val,
    }


runs = []
for m in CMD_RE.finditer(TEXT):
    r = parse_run(m.group(0))
    if r["train"]:
        runs.append(r)

print(f"Parsed {len(runs)} runs total")

NICE_LABEL = {
    "true_baseline": "Baseline",
    "baseline_extra_data": "Baseline + extra math data",
    "curriculum_only": "Curriculum",
    "curriculum_extra_data": "Curriculum + extra math data",
}
RUN_ORDER = ["true_baseline", "baseline_extra_data", "curriculum_only", "curriculum_extra_data"]
COLORS = {
    "true_baseline": "tab:blue",
    "baseline_extra_data": "tab:orange",
    "curriculum_only": "tab:green",
    "curriculum_extra_data": "tab:red",
}


def make_overlay(subset: list[dict], title: str, fname: str) -> None:
    ordered = []
    for name in RUN_ORDER:
        for r in subset:
            if r["checkpoint_name"] == name:
                ordered.append(r)
                break

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for r in ordered:
        name = r["checkpoint_name"]
        color = COLORS.get(name, "gray")
        label = NICE_LABEL.get(name, name)

        steps_t = [d["step"] for d in r["train"]]
        loss_t = [d["loss"] for d in r["train"]]
        ema_t = [d["ema"] for d in r["train"]]
        steps_v = [d["step"] for d in r["val"]]
        loss_v = [d["loss"] for d in r["val"]]
        best = min(r["val"], key=lambda d: d["loss"]) if r["val"] else None

        ax.plot(steps_t, ema_t, color=color, alpha=0.45, linewidth=1.2, linestyle="--")
        legend_label = f"{label}  (best val = {best['loss']:.4f} @ step {best['step']})" if best else label
        ax.plot(steps_v, loss_v, color=color, linewidth=2.4, marker="o", markersize=5.0, label=legend_label)
        if best:
            ax.scatter(
                [best["step"]],
                [best["loss"]],
                color=color,
                s=220,
                marker="*",
                zorder=6,
                edgecolor="black",
                linewidth=1.0,
            )

    all_val = [d["loss"] for r in ordered for d in r["val"]]
    all_ema = [d["ema"] for r in ordered for d in r["train"]]
    lo_candidates = all_val + all_ema
    skip_early = [d["ema"] for r in ordered for d in r["train"] if d["step"] >= 100]
    hi_candidates = (all_val + skip_early) if skip_early else (all_val + all_ema)
    if lo_candidates and hi_candidates:
        y_lo = max(0.0, min(lo_candidates) - 0.03)
        y_hi = max(hi_candidates) + 0.10
        ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Cross-entropy loss", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper right",
        fontsize=10,
        title="Solid = val loss  •  Dashed = train EMA  •  ★ = best-val checkpoint",
        title_fontsize=9,
    )
    fig.tight_layout()
    out = HERE / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}  ({len(ordered)} runs overlayed)")


threeB_bs8 = [r for r in runs if r["base_model"].endswith("Llama-3.2-3B") and r["batch_size"] == 8]
threeB_bs16 = [r for r in runs if r["base_model"].endswith("Llama-3.2-3B") and r["batch_size"] == 16]

make_overlay(threeB_bs8, "Llama-3.2-3B, batch_size=8 — overall training loss (4 configs overlayed)", "overlay_3B_bs8.png")
make_overlay(threeB_bs16, "Llama-3.2-3B, batch_size=16 — overall training loss (4 configs overlayed)", "overlay_3B_bs16.png")

print("Done.")
