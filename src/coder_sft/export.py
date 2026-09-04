"""Save adapters, merge BF16 weights, and optionally export verified GGUFs."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from .config import safe_output_path
from .constants import SYSTEM_PROMPT
from .modeling import load_inference_model
from .utils import write_json

PROBE_PROMPTS = [
    "Write a Python function that returns the first non-repeated character in a string.",
    "Explain why a null pointer check must happen before dereferencing an object.",
    "Return a unified diff that changes `return a-b` to `return a+b` in calc.py.",
]


def greedy_outputs(model: Any, tokenizer: Any, max_new_tokens: int = 128) -> list[str]:
    import torch

    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    outputs = []
    for prompt in PROBE_PROMPTS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            tokens = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs.append(
            tokenizer.decode(
                tokens[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            ).strip()
        )
    return outputs


def release_cuda_memory() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def export(adapter: str, output: Path, format_name: str, quantizations: list[str]) -> dict[str, Any]:
    output = safe_output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_inference_model(adapter, 8192)
    if format_name == "adapter":
        model.save_pretrained(str(output))
        tokenizer.save_pretrained(str(output))
        result = {"format": "adapter", "output": str(output)}
        write_json(result, output / "export_summary.json")
        return result

    adapter_outputs = greedy_outputs(model, tokenizer)
    merged = output if format_name == "merged" else output / "merged_bf16"
    model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
    del model
    del tokenizer
    release_cuda_memory()
    merged_model, merged_tokenizer = load_inference_model(str(merged), 8192)
    merged_outputs = greedy_outputs(merged_model, merged_tokenizer)
    equivalent = adapter_outputs == merged_outputs
    if not equivalent:
        raise RuntimeError("Merged BF16 model failed greedy adapter-equivalence probes")
    del merged_model
    del merged_tokenizer
    release_cuda_memory()
    result: dict[str, Any] = {
        "format": format_name,
        "merged_output": str(merged),
        "greedy_equivalent": True,
        "probe_outputs": adapter_outputs,
    }
    if format_name == "gguf":
        gguf_model, gguf_tokenizer = load_inference_model(adapter, 8192)
        gguf_dir = output / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        for quantization in quantizations:
            gguf_model.save_pretrained_gguf(
                str(gguf_dir), gguf_tokenizer, quantization_method=quantization
            )
        result["gguf_output"] = str(gguf_dir)
        result["quantizations"] = quantizations
    write_json(result, output / "export_summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=("adapter", "merged", "gguf"), required=True)
    parser.add_argument(
        "--quantization",
        action="append",
        choices=("q5_k_m", "q4_k_m"),
        default=None,
    )
    args = parser.parse_args()
    result = export(
        args.adapter,
        Path(args.output),
        args.format,
        args.quantization or ["q5_k_m", "q4_k_m"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
