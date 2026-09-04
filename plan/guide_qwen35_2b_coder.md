# Qwen3.5-2B-Coder implementation guide

The executable v1 lives one directory above this guide. Start with
[`../README.md`](../README.md) and the Colab notebook in `../notebooks/`.

The most important change from the original research plan is that v1 uses
BF16 LoRA. Current Qwen3.5 guidance does not recommend 4-bit QLoRA because the
quantization difference is unusually large. The code rejects a 4-bit training
configuration rather than silently accepting it.

The workflow is deliberately gated:

1. Prepare and audit the 60K Stage-1 mixture.
2. Run the base-model benchmark generation.
3. Perform the one-step GPU smoke test and 8K Stage-1 training.
4. Score HumanEval+/MBPP+ externally in the EvalPlus Docker image.
5. Build pinned-commit SWE-smith repository contexts.
6. Profile real backward steps through 32K.
7. Start Stage 2 only after the Stage-1 quality gate passes.
8. Verify adapter/merged equivalence before optional GGUF export.

The profiler, manifests, resolved configuration, environment versions,
training summaries, and evaluation generations are stored under
`$CODER_SFT_WORKDIR`, which the notebook places on Google Drive.

