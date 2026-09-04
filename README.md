# Qwen3.5-2B-Coder — validated Colab v1

This project implements the first gated version of the training plan in
[`plan/Qwen3.5-2B-Coder.md`](plan/Qwen3.5-2B-Coder.md). It trains
`unsloth/Qwen3.5-2B` with **16-bit BF16 LoRA**, first at 8K and then on
repository tasks at the longest measured safe length up to 32K.

The pipeline deliberately does not use QLoRA for Qwen3.5 and does not claim
that 64K or 128K training fits before measuring it.

## What is included

- Streaming preparation of 35K EER6 Refined, 15K tiered EER6 Broad, and 10K
  multilingual CommitPackFT examples.
- Exact and 64-permutation MinHash-LSH deduplication plus EvalPlus prompt
  exclusion.
- Repository-level SWE-smith splits and context reconstruction from pinned Git
  commits, including relevant files, tests, dependencies, and distractors.
- Automatic 40 GB/80 GB BF16 profiles, isolated VRAM probes, checkpoint resume,
  assistant-response-only loss, and projection audits for Qwen3.5's hybrid
  attention architecture.
- Generation-only HumanEval+/MBPP+ evaluation, repository structural scoring,
  stage gates, merged-BF16 verification, and optional GGUF export.

## Colab setup

Use [`notebooks/Qwen3_5_2B_Coder_Colab.ipynb`](notebooks/Qwen3_5_2B_Coder_Colab.ipynb).
Place or clone this `coder_SFT` directory under `/content/coder_SFT`, then run
the notebook cells in order. Outputs default to:

```text
/content/drive/MyDrive/qwen35-2b-coder
```

Set `HF_TOKEN` and `WANDB_API_KEY` through Colab Secrets only. Hub upload is
disabled until `training.hub_model_id` is set in a local config override.

For a shell-based setup on a compatible Colab runtime:

```bash
python -m pip install --upgrade uv
uv pip install --system -r requirements-colab.txt
uv pip install --system --no-build-isolation -r requirements-kernels.txt
uv pip install --system --no-deps -r requirements-accelerator.txt
uv pip install --system -e .
export CODER_SFT_WORKDIR=/content/drive/MyDrive/qwen35-2b-coder
```

Qwen3.5 requires the linear-attention kernel build in
`requirements-kernels.txt`; the separate no-build-isolation step and the
A100/H100 accelerator additions match the current official notebook family
and can take several minutes.

## End-to-end commands

All commands accept multiple `--config` flags, merged from left to right.

```bash
prepare-data --config configs/base.yaml --config configs/data_v1.yaml

build-repo-context \
  --config configs/base.yaml \
  --config configs/data_v1.yaml

profile-memory \
  --config configs/base.yaml \
  --config configs/hardware.yaml \
  --config configs/stage2_repo_32k.yaml

train-sft \
  --config configs/base.yaml \
  --config configs/hardware.yaml \
  --config configs/stage1_8k.yaml \
  --stage stage1 \
  --resume auto

train-sft \
  --config configs/base.yaml \
  --config configs/hardware.yaml \
  --config configs/stage2_repo_32k.yaml \
  --stage stage2 \
  --adapter "$CODER_SFT_WORKDIR/outputs/q35-2b-coder-s1-8k/final_adapter" \
  --resume auto
```

Repository construction performs many filtered Git clones and should be run
before the GPU session when possible. Failures and rejected rows are recorded
in the adjacent manifest instead of being silently converted into context-free
examples.

## Evaluation without running untrusted code in Colab

Generate the same samples for the base model and candidate:

```bash
generate-eval \
  --config configs/base.yaml \
  --config configs/data_v1.yaml \
  --model unsloth/Qwen3.5-2B \
  --dataset humaneval \
  --output "$CODER_SFT_WORKDIR/reports/base_humaneval.jsonl"

generate-eval \
  --config configs/base.yaml \
  --config configs/data_v1.yaml \
  --model "$CODER_SFT_WORKDIR/outputs/q35-2b-coder-s1-8k/final_adapter" \
  --dataset mbpp \
  --output "$CODER_SFT_WORKDIR/reports/stage1_mbpp.jsonl"
```

Copy the generated JSONL files to a Docker-capable machine. EvalPlus executes
model-generated Python, so keep it isolated:

```bash
docker run --rm -v "$PWD:/app" ganler/evalplus:latest \
  --dataset humaneval --samples /app/base_humaneval.jsonl
docker run --rm -v "$PWD:/app" ganler/evalplus:latest \
  --dataset mbpp --samples /app/base_mbpp.jsonl
```

Create normalized score files using fractions, not percentages:

```json
{
  "benchmarks": {
    "humaneval_plus_pass_at_1": 0.22,
    "mbpp_plus_pass_at_1": 0.31
  },
  "smoke": {
    "finite_validation_loss": true,
    "repetition_regression": false,
    "termination_regression": false
  }
}
```

Apply the Stage-1 gate:

```bash
compare-runs \
  --baseline reports/base.json \
  --candidate reports/stage1.json \
  --stage stage1 \
  --output "$CODER_SFT_WORKDIR/reports/stage1_gate.json"
```

Stage-2 training validates that exact report and refuses to start if it is
missing, is not a Stage-1 report, or has `passed: false`.

Repository evaluation never runs project code. It measures localization,
valid diff structure, changed-file precision, and `git apply --check`:

```bash
generate-eval \
  --config configs/base.yaml \
  --config configs/data_v1.yaml \
  --model "$CODER_SFT_WORKDIR/outputs/q35-2b-coder-s2-repo/final_adapter" \
  --dataset repo \
  --output "$CODER_SFT_WORKDIR/reports/stage2_repo.jsonl"
```

## Export

```bash
export-model \
  --adapter "$CODER_SFT_WORKDIR/outputs/q35-2b-coder-s2-repo/final_adapter" \
  --output "$CODER_SFT_WORKDIR/exports/q35-coder-merged" \
  --format merged

export-model \
  --adapter "$CODER_SFT_WORKDIR/outputs/q35-2b-coder-s2-repo/final_adapter" \
  --output "$CODER_SFT_WORKDIR/exports/q35-coder-gguf" \
  --format gguf \
  --quantization q5_k_m \
  --quantization q4_k_m
```

GGUF export is refused unless fixed greedy prompts produce exactly the same
outputs from the adapter and merged BF16 model.

## Tests

CPU-only tests do not download models or datasets:

```bash
python -m unittest discover -s tests -v
```

Create a small data and one-step GPU smoke run before the full job:

```bash
prepare-data --config configs/base.yaml --config configs/data_v1.yaml --limit 8
train-sft \
  --config configs/base.yaml --config configs/hardware.yaml \
  --config configs/stage1_8k.yaml --stage stage1 --resume none \
  --limit 8 --max-steps 1 \
  --output-dir "$CODER_SFT_WORKDIR/outputs/smoke"
```

Re-run `prepare-data` without `--limit` before full training. The smoke run must
produce finite loss, a valid loss mask, and a resumable adapter.
