"""
Build a clean, presentation-quality summary graphic for the final results table.

Two panels stacked vertically:
  - Top:    Llama-3.2-3B sweeps (batch=8 and batch=16, 4 configs each)
  - Bottom: Llama-3.1-8B runs (incl. GRPO, final model highlighted)

For each run, a grouped bar shows IFEval / GSM8K / HumanEval scores.
Dashed horizontal lines mark the per-task passing thresholds.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "final_results.png"

THRESHOLDS = {"IFEval": 0.45, "GSM8K": 0.50, "HumanEval": 0.30}
TASK_COLORS = {
    "IFEval": "#4C78A8",
    "GSM8K": "#F58518",
    "HumanEval": "#54A24B",
}

THREE_B = [
    ("Baseline",                  "bs=8",  0.372, 0.378, 0.341),
    ("Baseline + extra math",     "bs=8",  0.410, 0.391, 0.353),
    ("Curriculum",                "bs=8",  0.405, 0.371, 0.341),
    ("Curriculum + extra math",   "bs=8",  0.436, 0.382, 0.390),
    ("Baseline",                  "bs=16", 0.431, 0.406, 0.390),
    ("Baseline + extra math",     "bs=16", 0.428, 0.398, 0.372),
    ("Curriculum",                "bs=16", 0.403, 0.346, 0.367),
    ("Curriculum + extra math",   "bs=16", 0.426, 0.387, 0.353),
]

EIGHT_B = [
    ("Curriculum + extra math",         "bs=8",  0.547, 0.619, 0.512, False),
    ("Baseline",                        "bs=16", 0.549, 0.616, 0.493, False),
    ("Curriculum",                      "bs=16", 0.546, 0.590, 0.518, False),
    ("Curriculum + extra math (10k ea)","bs=16", 0.533, 0.632, 0.476, False),
    ("GRPO on Curriculum",              "bs=16", 0.556, 0.636, 0.549, True),
    ("GRPO on Curriculum + extra math", "bs=16", 0.544, 0.628, 0.524, False),
]


def draw_group(ax, rows, tasks, title, model_label, bs_in_label=True, highlight_idx=None):
    n = len(rows)
    x = np.arange(n)
    width = 0.26

    for i, task in enumerate(tasks):
        vals = [r[2 + i] for r in rows]
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=TASK_COLORS[task],
            edgecolor="white",
            linewidth=0.8,
            label=task,
            zorder=3,
        )
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                v + 0.008,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#222",
                zorder=4,
            )

    for i, task in enumerate(tasks):
        thr = THRESHOLDS[task]
        ax.hlines(
            thr,
            xmin=-0.5,
            xmax=n - 0.5,
            colors=TASK_COLORS[task],
            linestyles=":",
            linewidth=1.1,
            alpha=0.55,
            zorder=2,
        )
    labels = []
    for r in rows:
        name = r[0]
        bs = r[1]
        labels.append(f"{name}\n({bs})" if bs_in_label else name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xlim(-0.6, n - 0.4)
    ax.grid(True, axis="y", alpha=0.3, zorder=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    title_full = f"{title}\n{model_label}"
    ax.set_title(title_full, fontsize=13, fontweight="bold", loc="left", pad=10)

    if highlight_idx is not None:
        y0, y1 = ax.get_ylim()
        rect = FancyBboxPatch(
            (highlight_idx - 0.48, y0 + 0.005),
            0.96,
            y1 - y0 - 0.02,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=2.2,
            edgecolor="#D62728",
            facecolor="none",
            zorder=5,
        )
        ax.add_patch(rect)
        ax.annotate(
            "FINAL MODEL",
            xy=(highlight_idx, y1 - 0.02),
            xytext=(highlight_idx, y1 - 0.02),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#D62728",
            zorder=6,
        )


fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(16, 11),
    gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.45},
)

tasks = ["IFEval", "GSM8K", "HumanEval"]

draw_group(
    ax1,
    THREE_B,
    tasks,
    "Llama-3.2-3B sweeps",
    "5000 examples per dataset • 2000 steps • batch sizes 8 and 16",
)
for i in [3, 7]:
    ax1.axvline(i + 0.5, color="#bbb", linewidth=0.9, linestyle="-", zorder=1)
ax1.text(1.5, 0.73, "batch size = 8", ha="center", fontsize=10, color="#555",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="none"))
ax1.text(5.5, 0.73, "batch size = 16", ha="center", fontsize=10, color="#555",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="none"))

highlight = next(i for i, r in enumerate(EIGHT_B) if r[5])
draw_group(
    ax2,
    EIGHT_B,
    tasks,
    "Llama-3.1-8B final runs",
    "Includes GRPO RL fine-tuning (final submitted model = GRPO on Curriculum)",
    highlight_idx=highlight,
)

fig.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color=TASK_COLORS["IFEval"]),
        plt.Rectangle((0, 0), 1, 1, color=TASK_COLORS["GSM8K"]),
        plt.Rectangle((0, 0), 1, 1, color=TASK_COLORS["HumanEval"]),
    ],
    labels=["IFEval (Instruction Following)", "GSM8K (Math Reasoning)", "HumanEval (Code Generation)"],
    loc="upper center",
    ncol=3,
    frameon=False,
    fontsize=11,
    bbox_to_anchor=(0.5, 0.985),
)

fig.suptitle(
    "CPS 572 Final Project — Multi-Task LLM Fine-Tuning: Per-Task Evaluation Scores",
    fontsize=15,
    fontweight="bold",
    y=0.998,
)

fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
