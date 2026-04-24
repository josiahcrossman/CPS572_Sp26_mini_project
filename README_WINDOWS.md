# CPS572 Final Project: Multi-Task LLM Fine-Tuning (Windows / PowerShell)

Windows PowerShell equivalent of all commands in [`README.md`](README.md). Run these in **PowerShell** (not CMD).

For project requirements, grading breakdown, deliverables, and policies, see [`PROJECT.md`](PROJECT.md).

---

## Setup

```powershell
# Create virtual environment
uv venv --python 3.11
.venv\Scripts\Activate.ps1

# Install dependencies (option A: from requirements.txt)
uv pip install -r requirements.txt

# Or install manually (option B):
# uv pip install "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git"
# uv pip install inspect_ai==0.3.170 inspect_evals==0.3.106 openai
# uv pip install "git+https://github.com/josejg/instruction_following_eval"

# Set your Tinker API key
$env:TINKER_API_KEY = "your-api-key-here"
```

> **Note:** If PowerShell blocks the activation script, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then retry.

---

## Getting Started

### Step 1: Verify Your Setup — Evaluate Baseline Models

**Do this first before any training.** Evaluate one or more base Llama models to establish baseline scores and confirm your environment works.

```powershell
# Quick test (5 samples per task, ~2 minutes):
python evaluation/eval_all.py --base_models meta-llama/Llama-3.2-3B --limit 5

# Full baseline evaluation (~30 minutes per model):
python evaluation/eval_all.py --base_models meta-llama/Llama-3.2-3B

# Multiple models:
python evaluation/eval_all.py --base_models meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B meta-llama/Llama-3.1-8B
```

**Expected baseline scores (no fine-tuning):**

| Model | IFEval (%) | GSM8K (%) | HumanEval (%) | Avg (%) |
|-------|-----------|----------|---------------|---------|
| meta-llama/Llama-3.2-1B | 22.1 | 2.7 | 0.0 | 8.3 |
| meta-llama/Llama-3.2-3B | 22.6 | 9.2 | 0.6 | 10.8 |
| meta-llama/Llama-3.1-8B | 22.2 | 9.7 | 0.0 | 10.6 |

If your numbers match within ~1%, your setup is working correctly. Results are saved to `evaluation/baseline_results.json`.

### Step 2: Run the Toy Training Example

Verify that the full train → save → publish → evaluate pipeline works:

```powershell
# Train for 10 steps on dummy data:
python evaluation/train_and_publish.py

# This saves a checkpoint path to evaluation/checkpoint_info.json and can also be found on your tinker dashboard (https://tinker-console.thinkingmachines.ai/checkpoints)

# Then evaluate it:
python evaluation/eval_all.py --checkpoint_path "tinker://your-checkpoint-path" --base_model meta-llama/Llama-3.2-3B
```

The toy example (`train_and_publish.py`) trains on fake data — the scores will be low, but it confirms the workflow is functional.

### Step 3: Implement Your Training

Replace the toy training data and logic in `train_and_publish.py` with your own implementation. This is where the real work begins:

1. Load real training datasets (see [Suggested Datasets](PROJECT.md#suggested-datasets))
2. Design your training strategy (data mixing, hyperparameters, etc.)
3. Train on a small model first (Llama-3.2-1B or 3B) to iterate quickly
4. Train on Llama-3.1-8B for your final submission

### Step 4: Evaluate and Submit

```powershell
# Full evaluation (all samples, no --limit):
python evaluation/eval_all.py --checkpoint_path "tinker://your-checkpoint-path" `
    --base_model meta-llama/Llama-3.1-8B

# With custom generation settings:
python evaluation/eval_all.py --checkpoint_path "tinker://your-checkpoint-path" `
    --base_model meta-llama/Llama-3.1-8B --temperature 0.3 --top_p 0.9
```

This produces `evaluation/submission.json`. **Upload this file to Gradescope.** The leaderboard will update with your scores in real time.

---

## Passing Baseline

These are the per-task scores your model must meet to receive full baseline credit (4/5 points per task). They were achieved with mixed SFT on Llama-3.1-8B:

| Task | Baseline Score |
|------|---------------|
| IFEval | 45.0% |
| GSM8K | 50.0% |
| HumanEval | 30.0% |

See `PROJECT.md` for full grading details.

---

## Generation Settings

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Temperature | `--temperature` | 0.0 | Controls randomness. 0 = greedy/deterministic. |
| Top-p | `--top_p` | 1.0 | Nucleus sampling. 1.0 = disabled. |

These settings are stored in `submission.json` so the instructor can reproduce your results.

---

## File Reference

| File | Purpose |
|------|---------|
| [`evaluation/eval_all.py`](evaluation/eval_all.py) | Main entry point — evaluates base models or checkpoints on all 3 tasks |
| [`evaluation/eval_ifeval.py`](evaluation/eval_ifeval.py) | IFEval evaluation (Inspect AI) |
| [`evaluation/eval_gsm8k.py`](evaluation/eval_gsm8k.py) | GSM8K evaluation (Inspect AI, zero-shot) |
| [`evaluation/eval_code.py`](evaluation/eval_code.py) | HumanEval evaluation (Inspect AI, local sandbox) |
| [`evaluation/train_and_publish.py`](evaluation/train_and_publish.py) | **Toy example** — replace with your training code |

You can also run individual tasks:

```powershell
python evaluation/eval_ifeval.py --base_model meta-llama/Llama-3.2-3B
python evaluation/eval_gsm8k.py --checkpoint_path "tinker://..." --temperature 0.3
python evaluation/eval_code.py --limit 20 --top_p 0.9
```
