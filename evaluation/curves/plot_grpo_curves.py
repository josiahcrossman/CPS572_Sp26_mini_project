"""Learning curves for the final GRPO run on top of the 8B curriculum SFT.

Produces two figures, mirroring the SFT overlay style:
  - grpo_reward_curve.png : overall "loss-like" view, reward per step (raw + EMA)
                            with GSM8K held-out dev accuracy overlaid every 10 steps.
  - grpo_per_task.png     : three stacked panels -- GSM8K dev accuracy (target task),
                            Tulu NLL vs baseline (regression check), and Code NLL vs
                            baseline (regression check).

Data source: evaluation/curves/grpo_raw.txt
"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).parent / ".mplcache"))

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RAW = HERE / "grpo_raw.txt"


STEP_RE = re.compile(
    r"\[step\s+(\d+)/\d+\]\s+datums=(\d+)\s+"
    r"R_mean=([\-\d\.]+)\s+R_std=([\-\d\.]+)\s+\|\s+"
    r"corr=([\-\d\.]+)\s+part=([\-\d\.]+)\s+close=([\-\d\.]+)\s+struct=([\-\d\.]+)"
    r".*?loss=([\-\d\.eE]+)"
)
DEV_RE = re.compile(r"Held-out train dev accuracy \(\d+ ex\): ([\-\d\.]+)")
REG_RE = re.compile(
    r"Regression check: tulu NLL=([\-\d\.]+)\s+\(delta([+\-][\d\.]+)%\s+vs baseline\)\s+\|\s+"
    r"code NLL=([\-\d\.]+)\s+\(delta([+\-][\d\.]+)%\s+vs baseline\)"
)


def parse() -> dict:
    text = RAW.read_text()
    lines = text.splitlines()

    train = []
    dev = []
    reg = []
    current_last_step = 0

    for line in lines:
        m = STEP_RE.search(line)
        if m:
            step = int(m.group(1))
            current_last_step = step
            train.append(
                dict(
                    step=step,
                    datums=int(m.group(2)),
                    R_mean=float(m.group(3)),
                    R_std=float(m.group(4)),
                    corr=float(m.group(5)),
                    part=float(m.group(6)),
                    close=float(m.group(7)),
                    struct=float(m.group(8)),
                    loss=float(m.group(9)),
                )
            )
            continue
        m = DEV_RE.search(line)
        if m:
            dev.append(dict(step=current_last_step, acc=float(m.group(1))))
            continue
        m = REG_RE.search(line)
        if m:
            reg.append(
                dict(
                    step=current_last_step,
                    tulu_nll=float(m.group(1)),
                    tulu_delta=float(m.group(2)),
                    code_nll=float(m.group(3)),
                    code_delta=float(m.group(4)),
                )
            )
            continue

    return dict(train=train, dev=dev, reg=reg)


def ema(xs: list[float], alpha: float = 0.2) -> list[float]:
    out = []
    s = None
    for x in xs:
        s = x if s is None else alpha * x + (1 - alpha) * s
        out.append(s)
    return out


def plot_reward(data: dict) -> None:
    train = data["train"]
    dev = data["dev"]

    steps = [t["step"] for t in train]
    rewards = [t["R_mean"] for t in train]
    reward_ema = ema(rewards, alpha=0.2)

    fig, ax = plt.subplots(figsize=(11, 6.0))

    ax.plot(steps, rewards, color="#999999", alpha=0.45, linewidth=1.0, label="Mean reward (per step)")
    ax.plot(steps, reward_ema, color="#1f77b4", linewidth=2.4, label="Mean reward (EMA, alpha=0.2)")

    dev_steps = [d["step"] for d in dev]
    dev_acc = [d["acc"] for d in dev]
    ax.plot(
        dev_steps,
        dev_acc,
        color="#d62728",
        linewidth=2.2,
        marker="o",
        markersize=6.0,
        label="GSM8K held-out dev accuracy (every 10 steps)",
    )
    best_dev = max(dev, key=lambda d: d["acc"])
    ax.scatter(
        [best_dev["step"]],
        [best_dev["acc"]],
        color="#d62728",
        s=240,
        marker="*",
        zorder=6,
        edgecolor="black",
        linewidth=1.0,
        label=f"best dev acc = {best_dev['acc']:.3f} @ step {best_dev['step']}",
    )

    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("GRPO step", fontsize=12)
    ax.set_ylabel("Reward / accuracy", fontsize=12)
    ax.set_title(
        "Final GRPO run (Llama-3.1-8B, curriculum SFT init): reward + GSM8K dev accuracy",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    out = HERE / "grpo_reward_curve.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_per_task(data: dict) -> None:
    dev = data["dev"]
    reg = data["reg"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)

    # --- Panel 1: GSM8K held-out dev accuracy (target task) ---
    ax = axes[0]
    steps = [d["step"] for d in dev]
    accs = [d["acc"] for d in dev]
    ax.plot(steps, accs, color="#d62728", linewidth=2.2, marker="o", markersize=6.0, label="GSM8K held-out dev accuracy")
    best_dev = max(dev, key=lambda d: d["acc"])
    ax.scatter(
        [best_dev["step"]],
        [best_dev["acc"]],
        color="#d62728",
        s=240,
        marker="*",
        zorder=6,
        edgecolor="black",
        linewidth=1.0,
        label=f"best = {best_dev['acc']:.3f} @ step {best_dev['step']}",
    )
    ax.axhline(dev[0]["acc"], color="#d62728", linestyle=":", alpha=0.5, linewidth=1.2, label=f"step-10 baseline = {dev[0]['acc']:.3f}")
    ax.set_ylabel("GSM8K dev accuracy", fontsize=11)
    ax.set_title("Target task: GSM8K held-out train-dev accuracy (GRPO is GSM8K-only)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.0, 0.75)

    # --- Panel 2: Tulu NLL (regression check) ---
    ax = axes[1]
    steps_r = [r["step"] for r in reg]
    tulu_nll = [r["tulu_nll"] for r in reg]
    tulu_delta = [r["tulu_delta"] for r in reg]
    tulu_baseline = tulu_nll[0] / (1 + tulu_delta[0] / 100.0) if tulu_delta[0] != 0 else tulu_nll[0]
    ax.plot(steps_r, tulu_nll, color="#2ca02c", linewidth=2.2, marker="s", markersize=5.0, label="Tulu regression NLL")
    ax.axhline(tulu_baseline, color="#2ca02c", linestyle=":", alpha=0.6, linewidth=1.2, label=f"pre-GRPO baseline = {tulu_baseline:.4f}")
    ax.set_ylabel("Tulu NLL", fontsize=11)
    ax.set_title("Regression check: instruction-following (Tulu) held-out NLL", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    lo = min(min(tulu_nll), tulu_baseline) - 0.003
    hi = max(max(tulu_nll), tulu_baseline) + 0.003
    ax.set_ylim(lo, hi)

    # --- Panel 3: Code NLL (regression check) ---
    ax = axes[2]
    code_nll = [r["code_nll"] for r in reg]
    code_delta = [r["code_delta"] for r in reg]
    code_baseline = code_nll[0] / (1 + code_delta[0] / 100.0) if code_delta[0] != 0 else code_nll[0]
    ax.plot(steps_r, code_nll, color="#ff7f0e", linewidth=2.2, marker="D", markersize=5.0, label="Code regression NLL")
    ax.axhline(code_baseline, color="#ff7f0e", linestyle=":", alpha=0.6, linewidth=1.2, label=f"pre-GRPO baseline = {code_baseline:.4f}")
    ax.set_ylabel("Code NLL", fontsize=11)
    ax.set_xlabel("GRPO step", fontsize=11)
    ax.set_title("Regression check: code (OpenCodeInstruct) held-out NLL", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    lo = min(min(code_nll), code_baseline) - 0.0015
    hi = max(max(code_nll), code_baseline) + 0.0015
    ax.set_ylim(lo, hi)

    fig.suptitle(
        "Final GRPO run per-task trajectories (Llama-3.1-8B, GSM8K-only GRPO on curriculum SFT)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    out = HERE / "grpo_per_task.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> None:
    data = parse()
    print(f"parsed {len(data['train'])} train steps, {len(data['dev'])} dev evals, {len(data['reg'])} regression checks")
    plot_reward(data)
    plot_per_task(data)


if __name__ == "__main__":
    main()
