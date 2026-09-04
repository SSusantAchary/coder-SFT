"""Small deterministic and environment utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

T = TypeVar("T")


def stable_int(value: str, seed: int = 3407) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def deterministic_split(key: str, seed: int = 3407) -> str:
    bucket = stable_int(key, seed) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def priority_sample(rows: Iterable[T], count: int, key, seed: int = 3407) -> list[T]:
    """Keep the rows with the smallest deterministic hashes without loading all rows."""
    import heapq

    heap: list[tuple[int, int, T]] = []
    for index, row in enumerate(rows):
        score = stable_int(str(key(row)), seed)
        item = (-score, -index, row)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda x: (-x[0], -x[1]))]


def batched(values: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_checked(args: list[str], cwd: str | Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def package_versions() -> dict[str, str]:
    from importlib import metadata

    packages = [
        "torch",
        "transformers",
        "trl",
        "peft",
        "unsloth",
        "unsloth-zoo",
        "datasets",
        "bitsandbytes",
        "accelerate",
        "xformers",
        "triton",
        "flash-linear-attention",
        "causal-conv1d",
        "apache-tvm-ffi",
        "tilelang",
        "torchao",
        "evalplus",
        "wandb",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def environment_manifest() -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    try:
        import torch

        value["cuda_available"] = torch.cuda.is_available()
        value["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            value["gpu"] = torch.cuda.get_device_name(0)
            value["gpu_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
            value["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        value["cuda_available"] = False
    return value


def hub_revision(repo_id: str, repo_type: str) -> str | None:
    """Resolve an immutable Hub revision for provenance without making it mandatory."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = (
            api.dataset_info(repo_id=repo_id)
            if repo_type == "dataset"
            else api.model_info(repo_id=repo_id)
        )
        return str(info.sha) if info.sha else None
    except Exception:  # A cached/offline run remains usable and records the absence.
        return None


def write_json(value: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def extract_fenced_code(text: str) -> str:
    matches = re.findall(r"```(?:[A-Za-z0-9_+.#-]+)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return max(matches, key=len).strip() if matches else text.strip()


def finalize_chat_render(text: str, eos_token: str | None) -> str:
    """Remove template trailing whitespace and require its native EOS marker."""
    rendered = text.rstrip()
    if not eos_token or not rendered.endswith(eos_token):
        raise ValueError("Native chat template did not terminate the assistant turn with EOS")
    return rendered


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
