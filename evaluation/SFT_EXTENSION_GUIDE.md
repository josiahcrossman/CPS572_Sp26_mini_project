# SFT Extension Guide

This note explains what kinds of experiments you can already run with the current supervised fine-tuning pipeline, using only the existing command-line interface in [evaluation/train_and_publish.py](/Users/yuanwenbo/Desktop/CPS572_Sp26_mini_project/evaluation/train_and_publish.py) and the current data loader in [evaluation/sft_data.py](/Users/yuanwenbo/Desktop/CPS572_Sp26_mini_project/evaluation/sft_data.py).

It does not require any source-code changes.

## 1. Current Data Pipeline Summary

The training script loads three train-split datasets only:

- `openai/gsm8k` for math
- `allenai/tulu-3-sft-mixture` for instruction following
- `nvidia/OpenCodeInstruct` for code

Each source is normalized into the same internal chat format:

```python
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."},
]
```

or, for multi-turn Tulu examples:

```python
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."},
  ...
]
```

After normalization, all conversations are concatenated, converted into model-ready datums, shuffled, and used for LoRA training.

Because all examples are converted into one common format, the current pipeline is already flexible enough to support several experimental extensions through command-line settings alone.

## 2. What You Can Already Experiment With

Without changing source code, you can already vary the following parts of training.

### 2.1 Data Mixing Ratios

You can control how many examples come from each source:

- `--n_gsm8k`
- `--n_tulu`
- `--n_code`

This lets you test different task balances.

Example:

```bash
python evaluation/train_and_publish.py \
  --n_gsm8k 3000 \
  --n_tulu 3000 \
  --n_code 3000
```

Math-heavy mix:

```bash
python evaluation/train_and_publish.py \
  --n_gsm8k 5000 \
  --n_tulu 1500 \
  --n_code 1500
```

Instruction-following-heavy mix:

```bash
python evaluation/train_and_publish.py \
  --n_gsm8k 1500 \
  --n_tulu 5000 \
  --n_code 1500
```

Code-heavy mix:

```bash
python evaluation/train_and_publish.py \
  --n_gsm8k 1500 \
  --n_tulu 1500 \
  --n_code 5000
```

This is the easiest existing hook for studying data selection and task tradeoffs.

### 2.2 Random Subsampling

You can vary which examples are selected using:

- `--seed`

This changes the shuffle used before taking the first `n` examples from each dataset.

Example:

```bash
python evaluation/train_and_publish.py \
  --n_gsm8k 2500 \
  --n_tulu 2500 \
  --n_code 2500 \
  --seed 42
```

and then:

```bash
python evaluation/train_and_publish.py \
  --n_gsm8k 2500 \
  --n_tulu 2500 \
  --n_code 2500 \
  --seed 123
```

This is useful for checking whether a result is robust or just due to one lucky subset.

### 2.3 Datum Order After Packing

You can separately control the shuffle order of packed training datums with:

- `--shuffle_datums_seed`

This is useful when you want the same sampled dataset but a different training order.

Example:

```bash
python evaluation/train_and_publish.py \
  --seed 42 \
  --shuffle_datums_seed 999
```

This is a lightweight way to test ordering sensitivity.

### 2.4 Context Length

You can control the packing limit with:

- `--max_length`

Example:

```bash
python evaluation/train_and_publish.py \
  --max_length 1024
```

or

```bash
python evaluation/train_and_publish.py \
  --max_length 2048
```

This can matter because longer contexts allow more multi-turn Tulu conversations to survive packing, while shorter lengths may implicitly filter out long examples.

### 2.5 Training Hyperparameters

The parser already exposes the main optimization settings:

- `--num_steps`
- `--batch_size`
- `--lr`
- `--rank`
- `--base_model`

Example:

```bash
python evaluation/train_and_publish.py \
  --base_model meta-llama/Llama-3.2-3B \
  --num_steps 500 \
  --batch_size 4 \
  --lr 1e-4 \
  --rank 32
```

You can treat these as part of the extension space, especially when studying whether a data-mixing change is actually helping.

### 2.6 Intermediate Checkpoints

You can save checkpoints during training with:

- `--save_every`
- `--checkpoint_name`
- `--run_id`

Example:

```bash
python evaluation/train_and_publish.py \
  --num_steps 800 \
  --save_every 100 \
  --run_id mix_exp_01 \
  --checkpoint_name final_mix_exp_01
```

This is especially useful for extensions like:

- early stopping
- comparing data mixes at equal training budgets
- finding whether one task degrades later in training

### 2.7 Publishing Control

You can avoid publishing while experimenting by using:

- `--no_publish`

Example:

```bash
python evaluation/train_and_publish.py \
  --num_steps 200 \
  --no_publish
```

This is useful for quick local iteration before deciding which checkpoint is worth publishing and evaluating broadly.

## 3. Extensions You Can Approximate Right Now

Even without editing code, the current CLI already lets you approximate several extension ideas.

### 3.1 Data Selection by Source Weighting

You cannot yet score examples individually, but you can still do coarse data selection by changing source counts.

Examples:

- increase `--n_tulu` if instruction following is weak
- increase `--n_gsm8k` if GSM8K accuracy is weak
- increase `--n_code` if HumanEval is weak

This is source-level selection rather than example-level selection.

### 3.2 Robustness Through Multiple Seeds

A simple experimental extension is to repeat the same configuration with different seeds.

Example pattern:

```bash
python evaluation/train_and_publish.py --seed 1 --run_id seed1 --no_publish
python evaluation/train_and_publish.py --seed 2 --run_id seed2 --no_publish
python evaluation/train_and_publish.py --seed 3 --run_id seed3 --no_publish
```

If one configuration only works for one seed, it is probably not a reliable improvement.

### 3.3 Budgeted Training

You can study data efficiency by keeping the mix fixed and varying only:

- `--num_steps`
- `--batch_size`

This helps answer questions like:

- does a better score come from better data or just more updates?
- does one mix learn faster than another?

### 3.4 Length-Based Implicit Filtering

Changing `--max_length` acts like a rough filter on long conversations.

For example:

- smaller `--max_length` may favor short QA-style data
- larger `--max_length` may better preserve multi-turn instruction-following conversations

This is not a precise filtering method, but it is still an experimental lever.

## 4. Extensions That Are Not Really Supported Yet

The current parser is useful, but some ideas are not truly possible without changing the loader or training script.

### 4.1 True Data Augmentation

You cannot currently inject synthetic extra examples from the command line alone.

Why not:

- the script always loads the same three Hugging Face datasets
- there is no parser argument for an external JSONL or custom dataset path
- there is no augmentation hook in the current loader

So true augmentation is a future code extension, not a current CLI-only extension.

### 4.2 Fine-Grained Filtering

You also cannot currently do example-level filtering such as:

- remove low-quality Tulu records
- keep only multi-turn Tulu conversations
- keep only code examples with tests
- deduplicate near-duplicates
- filter by difficulty or prompt length exactly

Why not:

- after loading, the pipeline mostly works on plain normalized conversations
- source metadata and example metadata are not exposed through the CLI

### 4.3 Curriculum Learning

You cannot currently say:

- train on Tulu first, then GSM8K
- oversample code only in later steps
- change source ratios during training

The current script loads all data once, concatenates it, shuffles it, and trains over the resulting list.

## 5. Practical Experiment Recipes

Here are some useful experiment recipes that fit the current setup.

### Recipe A: Balanced Baseline

```bash
python evaluation/train_and_publish.py \
  --base_model meta-llama/Llama-3.2-3B \
  --n_gsm8k 2500 \
  --n_tulu 2500 \
  --n_code 2500 \
  --seed 42 \
  --num_steps 500 \
  --batch_size 4 \
  --lr 1e-4 \
  --rank 32 \
  --save_every 100 \
  --run_id balanced_baseline \
  --checkpoint_name balanced_baseline_final
```

Use this as a reference point.

### Recipe B: Instruction-Following Focus

```bash
python evaluation/train_and_publish.py \
  --base_model meta-llama/Llama-3.2-3B \
  --n_gsm8k 1500 \
  --n_tulu 5000 \
  --n_code 1500 \
  --seed 42 \
  --num_steps 500 \
  --batch_size 4 \
  --lr 1e-4 \
  --rank 32 \
  --save_every 100 \
  --run_id if_focus \
  --checkpoint_name if_focus_final
```

Use this if IFEval is lagging.

### Recipe C: Math Focus

```bash
python evaluation/train_and_publish.py \
  --base_model meta-llama/Llama-3.2-3B \
  --n_gsm8k 5000 \
  --n_tulu 1500 \
  --n_code 1500 \
  --seed 42 \
  --num_steps 500 \
  --batch_size 4 \
  --lr 1e-4 \
  --rank 32 \
  --save_every 100 \
  --run_id math_focus \
  --checkpoint_name math_focus_final
```

Use this if GSM8K is weak.

### Recipe D: Code Focus

```bash
python evaluation/train_and_publish.py \
  --base_model meta-llama/Llama-3.2-3B \
  --n_gsm8k 1500 \
  --n_tulu 1500 \
  --n_code 5000 \
  --seed 42 \
  --num_steps 500 \
  --batch_size 4 \
  --lr 1e-4 \
  --rank 32 \
  --save_every 100 \
  --run_id code_focus \
  --checkpoint_name code_focus_final
```

Use this if HumanEval is weak.

### Recipe E: Same Mix, Different Seed

```bash
python evaluation/train_and_publish.py \
  --n_gsm8k 2500 \
  --n_tulu 2500 \
  --n_code 2500 \
  --seed 1 \
  --run_id seed1 \
  --checkpoint_name seed1_final \
  --no_publish
```

```bash
python evaluation/train_and_publish.py \
  --n_gsm8k 2500 \
  --n_tulu 2500 \
  --n_code 2500 \
  --seed 2 \
  --run_id seed2 \
  --checkpoint_name seed2_final \
  --no_publish
```

This checks whether the result is stable.

## 6. How To Think About Future Extensions

If the team later decides to extend the codebase, the cleanest next steps would be:

- preserve source metadata per example
- preserve optional extra metadata such as length, difficulty, or quality score
- allow loading external augmented data files
- add parser flags for filtering and weighting
- add parser flags for custom sampling strategies

A future version of the loader could return records like:

```python
{
  "source": "gsm8k",
  "conversation": [...],
  "meta": {
    "length": 312,
    "augmented": False,
    "difficulty": "medium"
  }
}
```

That would make it much easier to support:

- data augmentation
- quality filtering
- source-aware reweighting
- deduplication
- curriculum schedules
- ablation studies

## 7. Current Bottom Line

With the current parser and no source-code changes, the most realistic extensions are:

- source-ratio experiments
- seed sweeps
- training-budget sweeps
- context-length sweeps
- hyperparameter tuning
- checkpoint-based model selection

With the current code, true augmentation and fine-grained filtering are not yet command-line features.

## 8. Suggested Experimental Order

A sensible order for experiments is:

1. establish one balanced baseline
2. sweep source ratios
3. repeat the best mixes with multiple seeds
4. compare intermediate checkpoints
5. only then tune learning rate, steps, and rank

This keeps the experiments interpretable and helps separate data effects from optimization effects.

## 9. Important Constraint

Only use training splits for training.

Do not train on:

- IFEval prompts
- GSM8K test split
- HumanEval problems

That is explicitly disallowed by the project specification.
