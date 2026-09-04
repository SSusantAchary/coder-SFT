"""Unsloth model loading and LoRA target audits."""

from __future__ import annotations

from typing import Any

from .config import ConfigError
from .constants import LORA_TARGET_MODULES, PROJECTION_FAMILIES

NATIVE_MAX_POSITION_EMBEDDINGS = 262_144


def audit_native_context(model: Any) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    value = getattr(text_config, "max_position_embeddings", None)
    if int(value or 0) != NATIVE_MAX_POSITION_EMBEDDINGS:
        raise ConfigError(
            "Expected Qwen3.5 native max_position_embeddings="
            f"{NATIVE_MAX_POSITION_EMBEDDINGS}, found {value!r}"
        )
    return int(value)


def audit_projection_modules(model: Any, targets: list[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {name: [] for name in PROJECTION_FAMILIES}
    target_set = set(targets)
    for module_name, _ in model.named_modules():
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf not in target_set:
            continue
        for family, names in PROJECTION_FAMILIES.items():
            if leaf in names:
                found[family].append(module_name)
    missing = [family for family, names in found.items() if not names]
    if missing:
        raise ConfigError(f"Model is missing LoRA projection families: {missing}")
    return found


def audit_trainable_parameters(model: Any) -> dict[str, int]:
    trainable = 0
    total = 0
    forbidden: list[str] = []
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
            lowered = name.lower()
            if ".visual." in lowered or ".vision_" in lowered or "vision_model" in lowered:
                forbidden.append(name)
    if forbidden:
        raise ConfigError(f"Vision parameters unexpectedly trainable: {forbidden[:5]}")
    if trainable == 0:
        raise ConfigError("No trainable LoRA parameters were found")
    return {"trainable": trainable, "total": total}


def load_model_and_tokenizer(
    config: dict[str, Any], max_length: int, adapter: str | None = None
) -> tuple[Any, Any, dict[str, Any]]:
    # Unsloth must be imported before Transformers/TRL so its patches are active.
    from unsloth import FastLanguageModel

    model_config = config.get("model", {})
    source = adapter or model_config.get("name") or config.get("model_name")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source,
        max_seq_length=max_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    native_context = audit_native_context(model)
    targets = list(config.get("lora", {}).get("target_modules", LORA_TARGET_MODULES))
    projection_audit = audit_projection_modules(model, targets)
    has_adapter = bool(getattr(model, "peft_config", None))
    if not has_adapter:
        lora = config.get("lora", {})
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora.get("rank", 32)),
            target_modules=targets,
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.0)),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=int(config.get("seed", 3407)),
            use_rslora=bool(lora.get("use_rslora", False)),
            loftq_config=None,
            max_seq_length=max_length,
        )
    FastLanguageModel.for_training(model)
    trainable = audit_trainable_parameters(model)
    return model, tokenizer, {
        "source": str(source),
        "continued_adapter": has_adapter,
        "projection_matches": {key: len(value) for key, value in projection_audit.items()},
        "native_max_position_embeddings": native_context,
        **trainable,
    }


def enable_inference(model: Any) -> None:
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)


def load_inference_model(source: str, max_length: int) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source,
        max_seq_length=max_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    audit_native_context(model)
    FastLanguageModel.for_inference(model)
    return model, tokenizer
