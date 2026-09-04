"""YAML configuration loading, merging, and validation."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from .constants import LORA_TARGET_MODULES


class ConfigError(ValueError):
    """Raised when a run configuration is unsafe or inconsistent."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_environment(child) for key, child in value.items()}
    if isinstance(value, list):
        return [expand_environment(child) for child in value]
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    return value


def load_config(*paths: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency message
        raise RuntimeError("Install PyYAML to read training configuration files") from exc

    merged: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
        if not isinstance(value, dict):
            raise ConfigError(f"Configuration root must be a mapping: {path}")
        merged = deep_merge(merged, value)
    return merged


def write_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        text = yaml.safe_dump(config, sort_keys=False)
    except ImportError:
        text = json.dumps(config, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")


def validate_training_config(config: dict[str, Any]) -> None:
    model = config.get("model", {})
    training = config.get("training", {})
    lora = config.get("lora", {})

    if model.get("load_in_4bit", False):
        raise ConfigError("Qwen3.5 v1 forbids QLoRA; use BF16 LoRA")
    if not model.get("load_in_16bit", True):
        raise ConfigError("Qwen3.5 v1 requires load_in_16bit=true")
    if training.get("bf16") is not True:
        raise ConfigError("Qwen3.5 v1 requires bf16=true")
    if int(lora.get("rank", 32)) != 32 or int(lora.get("alpha", 32)) != 32:
        raise ConfigError("Validated v1 requires LoRA rank=32 and alpha=32")
    if float(lora.get("dropout", 0.0)) != 0.0:
        raise ConfigError("Validated v1 requires LoRA dropout=0")
    targets = set(lora.get("target_modules", LORA_TARGET_MODULES))
    missing_targets = sorted(set(LORA_TARGET_MODULES) - targets)
    if missing_targets:
        raise ConfigError(f"LoRA target_modules omit required Qwen3.5 projections: {missing_targets}")
    max_length = int(training.get("max_length", 0))
    if max_length < 512 or max_length > 262_144:
        raise ConfigError("training.max_length must be between 512 and 262144")


def safe_output_path(path: str | Path) -> Path:
    if "$" in str(path):
        raise ConfigError(f"Refusing output path with unresolved environment variable: {path}")
    resolved = Path(path).expanduser().resolve()
    forbidden = {Path("/"), Path.home().resolve()}
    if resolved in forbidden or len(resolved.parts) < 3:
        raise ConfigError(f"Refusing unsafe output path: {resolved}")
    return resolved
