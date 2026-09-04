"""Isolated real-step VRAM profiling for Qwen3.5 BF16 LoRA context lengths."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import load_config, safe_output_path, validate_training_config
from .hardware import discover_hardware, resolve_hardware
from .modeling import load_model_and_tokenizer
from .utils import environment_manifest, write_json

RESULT_MARKER = "CODER_SFT_MEMORY_RESULT="


def worker(config: dict[str, Any], length: int) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    resolved = resolve_hardware(config, "stage2", discover_hardware())
    resolved["training"]["max_length"] = length
    validate_training_config(resolved)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, audit = load_model_and_tokenizer(resolved, length)
    batch_size = int(resolved["training"]["per_device_train_batch_size"])
    vocab_size = int(getattr(tokenizer, "vocab_size", 248_320))
    generator = torch.Generator(device="cuda").manual_seed(int(config.get("seed", 3407)))
    input_ids = torch.randint(
        low=1000,
        high=max(1001, vocab_size - 256),
        size=(batch_size, length),
        generator=generator,
        device="cuda",
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    started = time.perf_counter()
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    loss = output.loss
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    total = torch.cuda.get_device_properties(0).total_memory
    peak_reserved = torch.cuda.max_memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    return {
        "length": length,
        "batch_size": batch_size,
        "success": True,
        "loss": float(loss.detach().float().cpu()),
        "elapsed_seconds": elapsed,
        "tokens_per_second": batch_size * length / elapsed,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_bytes": peak_allocated,
        "total_vram_bytes": total,
        "reserved_fraction": peak_reserved / total,
        "headroom_fraction": 1.0 - peak_reserved / total,
        "model_audit": audit,
    }


def run_probe(config_paths: list[str], length: int) -> dict[str, Any]:
    command = [sys.executable, "-m", "coder_sft.memory_profile"]
    for path in config_paths:
        command.extend(["--config", path])
    command.extend(["--worker", "--length", str(length)])
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    parsed: dict[str, Any] | None = None
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            parsed = json.loads(line[len(RESULT_MARKER) :])
            break
    if result.returncode == 0 and parsed is not None:
        return parsed
    error = (result.stderr or result.stdout)[-4000:]
    return {"length": length, "success": False, "error": error}


def profile(config: dict[str, Any], config_paths: list[str], extended: bool) -> dict[str, Any]:
    info = discover_hardware()
    resolved = resolve_hardware(config, "stage2", info)
    profile_config = resolved.get("memory_profile", {})
    lengths = list(profile_config.get("lengths", [8192, 16384, 24576, 32768]))
    if extended:
        lengths.extend(profile_config.get("extended_lengths", [49152, 65536, 98304, 131072]))
    lengths = sorted(set(int(length) for length in lengths))
    results = [run_probe(config_paths, length) for length in lengths]
    required_headroom = float(profile_config.get("required_headroom", 0.10))
    passing = [
        int(result["length"])
        for result in results
        if result.get("success") and float(result.get("headroom_fraction", 0)) >= required_headroom
    ]
    selected = max(passing, default=0)
    report = {
        "hardware": resolved["resolved_hardware"],
        "environment": environment_manifest(),
        "required_headroom": required_headroom,
        "selected_max_length": selected,
        "results": results,
    }
    output = safe_output_path(os.path.expandvars(profile_config["output"]))
    write_json(report, output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(*args.config)
    if args.worker:
        if not args.length:
            raise SystemExit("--worker requires --length")
        try:
            result = worker(config, args.length)
            print(RESULT_MARKER + json.dumps(result, sort_keys=True))
        except Exception as exc:
            print(
                RESULT_MARKER
                + json.dumps(
                    {"length": args.length, "success": False, "error": repr(exc)},
                    sort_keys=True,
                )
            )
            raise
    else:
        print(json.dumps(profile(config, args.config, args.extended), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
