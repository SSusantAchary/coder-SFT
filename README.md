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

## Credit-efficient split Colab workflow

Use the two notebooks in order:

1. [`01_Prepare_Data_CPU.ipynb`](notebooks/01_Prepare_Data_CPU.ipynb) runs in a
   CPU-only runtime. It prepares Stage 1 and repository-context data once,
   validates both manifests, and uploads only the reusable training bundle to
   a private Hugging Face **dataset** repository.
2. Release the CPU runtime, select an A100 40 GB or A100/H100 80 GB runtime,
   and open
   [`02_Train_and_Publish_GPU.ipynb`](notebooks/02_Train_and_Publish_GPU.ipynb).
   It downloads and validates that bundle, runs the GPU stages, verifies the
   merged result, and pushes the final adapter and model to a private
   Hugging Face **model** repository.

Changing a Colab runtime can replace the VM, so `/content` must not be used to
hand data from CPU to GPU. The private Dataset Hub repository is the durable
handoff. Repository clone caches are intentionally excluded: training needs
only `stage1_v1.jsonl`, `repo_v1.jsonl`, and their provenance manifests.
Once that repository exists, later training attempts can start directly with
the GPU notebook without paying the data-preparation cost again.

Both notebooks automatically clone
[`SSusantAchary/coder-SFT`](https://github.com/SSusantAchary/coder-SFT) into
`/content/coder_SFT`. Add one write-enabled `HF_TOKEN` in Colab Secrets and
grant each notebook access. The namespace is derived from that token; no token
is written to a file. W&B is not used and no W&B key is required.

The default Hub targets are:

```text
YOUR_HF_USERNAME/Qwen3.5-2B-Coder-Data  # private dataset handoff
YOUR_HF_USERNAME/Qwen3.5-2B-Coder-SFT   # private final model
```

The original
[`Qwen3_5_2B_Coder_Colab.ipynb`](notebooks/Qwen3_5_2B_Coder_Colab.ipynb) is
kept as an all-in-one convenience notebook, but it spends GPU runtime on data
preparation. Prefer the split notebooks when conserving Colab credits.

For a shell-based GPU setup on a compatible Colab runtime:

```bash
python -m pip install --upgrade uv
uv pip install --system -r requirements-colab.txt
uv pip install --system --no-build-isolation -r requirements-kernels.txt
uv pip install --system --no-deps -r requirements-accelerator.txt
uv pip install --system -e .
export CODER_SFT_WORKDIR=/content/qwen35-2b-coder
```

Qwen3.5 requires the linear-attention kernel build in
`requirements-kernels.txt`; the separate no-build-isolation step and the
A100/H100 accelerator additions match the current official notebook family
and can take several minutes.

## End-to-end commands

All commands accept multiple `--config` flags, merged from left to right.
The notebooks run these commands for you. For a manual split, prepare and
upload from the CPU runtime:

```bash
prepare-data --config configs/base.yaml --config configs/data_v1.yaml

build-repo-context \
  --config configs/base.yaml \
  --config configs/data_v1.yaml

sync-data upload \
  --config configs/base.yaml \
  --repo-id YOUR_HF_USERNAME/Qwen3.5-2B-Coder-Data
```

Then set `CODER_SFT_WORKDIR` in the new GPU runtime and download before
profiling or training:

```bash
sync-data download \
  --config configs/base.yaml \
  --repo-id YOUR_HF_USERNAME/Qwen3.5-2B-Coder-Data

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

Repository construction performs many filtered Git clones and therefore runs
in the CPU phase. Failures and rejected rows are recorded in the adjacent
manifest instead of being silently converted into context-free examples. The
GPU phase refuses an incomplete upload or download.

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

Because the GPU working directory is on Colab's ephemeral disk, use Colab's
Files panel to download the generated JSONL files and upload the resulting
normalized `base.json` and `stage1.json` score reports back into
`/content/qwen35-2b-coder/reports/` before running the Stage-2 cell.

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

Final Hub publication is accepted only for an adapter whose lineage identifies
it as a completed Stage-2 output. The Hub repository is private by default.

```bash
export-model \
  --adapter "$CODER_SFT_WORKDIR/outputs/q35-2b-coder-s2-repo/final_adapter" \
  --output "$CODER_SFT_WORKDIR/exports/q35-coder-merged" \
  --format merged \
  --hub-model-id "YOUR_HF_USERNAME/Qwen3.5-2B-Coder-SFT"
```

Add `--public` only if the model should be published publicly. The uploader
creates the repository when necessary, uploads the merged BF16 model at its
root, and stores the original Stage-2 LoRA adapter under `adapter/`. Re-running
the command is safe when an upload is interrupted.

GGUF remains a separate opt-in export:

```bash
export-model \
  --adapter "$CODER_SFT_WORKDIR/outputs/q35-2b-coder-s2-repo/final_adapter" \
  --output "$CODER_SFT_WORKDIR/exports/q35-coder-gguf" \
  --format gguf \
  --quantization q5_k_m \
  --quantization q4_k_m
```

Every merged or GGUF export is refused unless fixed greedy prompts produce
exactly the same outputs from the adapter and merged BF16 model.

## Tests

CPU-only tests do not download models or datasets:

```bash
python -m unittest discover -s tests -v
```

Run a one-step GPU smoke test against the already downloaded full dataset
before the full job:

```bash
train-sft \
  --config configs/base.yaml --config configs/hardware.yaml \
  --config configs/stage1_8k.yaml --stage stage1 --resume none \
  --limit 8 --max-steps 1 \
  --output-dir "$CODER_SFT_WORKDIR/outputs/smoke"
```

The smoke run must produce finite loss, a valid loss mask, and a resumable
adapter. It does not overwrite or truncate the prepared dataset.
