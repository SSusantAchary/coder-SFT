# Qwen3.5-2B-Coder implementation guide

The executable v1 lives one directory above this guide. Start with
[`../README.md`](../README.md), then run the CPU preparation notebook followed
by the GPU training notebook in `../notebooks/`.

The most important change from the original research plan is that v1 uses
BF16 LoRA. Current Qwen3.5 guidance does not recommend 4-bit QLoRA because the
quantization difference is unusually large. The code rejects a 4-bit training
configuration rather than silently accepting it.

The workflow is deliberately gated:

1. On a CPU runtime, prepare and audit the 60K Stage-1 mixture.
2. Build pinned-commit SWE-smith repository contexts on that CPU runtime.
3. Validate and upload the prepared JSONL and manifests to a private HF dataset
   repository.
4. On a new GPU runtime, download and validate the immutable data handoff.
5. Run base-model generation, the one-step smoke test, and 8K Stage-1 training.
6. Score HumanEval+/MBPP+ externally in the EvalPlus Docker image.
7. Profile real backward steps through 32K and start Stage 2 only after the
   Stage-1 quality gate passes.
8. Verify adapter/merged equivalence before final model publication or optional
   GGUF export.

The CPU and GPU notebooks use different ephemeral `/content` work directories.
`sync-data upload` and `sync-data download` bridge them through a private
Hugging Face dataset repository and reject missing or incomplete manifests.
The HF token stays in process memory. After the gated Stage-2 run, the final
export command verifies the merged BF16 model and pushes it together with the
adapter and lineage to a private Hugging Face model repository by default.
