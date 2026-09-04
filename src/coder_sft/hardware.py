"""GPU discovery and deterministic 40/80 GB profile selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import ConfigError, deep_merge


@dataclass(frozen=True)
class HardwareInfo:
    name: str
    total_vram_gb: float
    bf16_supported: bool
    cuda_version: str | None
    capability: tuple[int, int]


def discover_hardware() -> HardwareInfo:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - GPU environment only
        raise ConfigError("PyTorch is not installed") from exc
    if not torch.cuda.is_available():
        raise ConfigError("A CUDA GPU is required")
    props = torch.cuda.get_device_properties(0)
    return HardwareInfo(
        name=torch.cuda.get_device_name(0),
        total_vram_gb=props.total_memory / 1024**3,
        bf16_supported=bool(torch.cuda.is_bf16_supported()),
        cuda_version=torch.version.cuda,
        capability=tuple(torch.cuda.get_device_capability(0)),
    )


def select_profile(profiles: dict[str, dict[str, Any]], info: HardwareInfo) -> tuple[str, dict[str, Any]]:
    if not info.bf16_supported:
        raise ConfigError(f"{info.name} does not support BF16")
    eligible = [
        (float(profile.get("min_vram_gb", 0)), name, profile)
        for name, profile in profiles.items()
        if info.total_vram_gb >= float(profile.get("min_vram_gb", 0))
    ]
    if not eligible:
        required = min(float(profile.get("min_vram_gb", 0)) for profile in profiles.values())
        raise ConfigError(
            f"{info.total_vram_gb:.1f} GB VRAM is below the supported minimum ({required:.0f} GB)"
        )
    _, name, profile = max(eligible, key=lambda item: (item[0], item[1]))
    return name, profile


def resolve_hardware(config: dict[str, Any], stage: str, info: HardwareInfo) -> dict[str, Any]:
    name, profile = select_profile(config["hardware_profiles"], info)
    stage_overrides = profile.get("stages", {}).get(stage)
    if not isinstance(stage_overrides, dict):
        raise ConfigError(f"Hardware profile {name!r} has no settings for stage {stage!r}")
    resolved = deep_merge(config, {"training": stage_overrides})
    resolved["resolved_hardware"] = {"profile": name, **asdict(info)}
    return resolved

