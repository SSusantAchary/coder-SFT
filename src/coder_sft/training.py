"""Train Stage 1 or Stage 2 with BF16 LoRA and resumable checkpoints."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from .config import (
    deep_merge,
    expand_environment,
    load_config,
    safe_output_path,
    validate_training_config,
    write_resolved_config,
)
from .hardware import discover_hardware, resolve_hardware
from .modeling import load_model_and_tokenizer
from .utils import environment_manifest, finalize_chat_render, stable_int, write_json

LOGGER = logging.getLogger(__name__)


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def validate_stage1_gate(training: dict[str, Any]) -> str | None:
    gate_path_value = training.get("stage1_gate_report")
    gate_required = training.get("require_stage1_gate", True)
    if not gate_path_value:
        if gate_required:
            raise ValueError("Stage 2 requires training.stage1_gate_report")
        return None
    gate_path = Path(_expand(str(gate_path_value)))
    if not gate_path.exists():
        if gate_required:
            raise FileNotFoundError(
                f"Run compare-runs for Stage 1 first; missing gate report {gate_path}"
            )
        return None
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("stage") != "stage1" or gate.get("passed") is not True:
        raise RuntimeError(
            f"Stage-1 gate did not pass in {gate_path}; refusing Stage-2 training"
        )
    return str(gate_path)


def validate_stage1_adapter(adapter: str | None) -> dict[str, Any]:
    if not adapter:
        raise ValueError("Stage 2 requires a Stage-1 adapter")
    adapter_path = Path(_expand(adapter))
    lineage_path = adapter_path / "training_lineage.json"
    if not adapter_path.is_dir() or not lineage_path.is_file():
        raise FileNotFoundError(
            f"Stage-2 adapter must be a local Stage-1 output with {lineage_path.name}: "
            f"{adapter_path}"
        )
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    if lineage.get("stage") != "stage1":
        raise RuntimeError(f"Invalid Stage-2 adapter lineage in {lineage_path}")
    return lineage


def validate_data_manifest(data_path: str) -> dict[str, Any]:
    path = Path(_expand(data_path))
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset or manifest is missing: {path}, {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise RuntimeError(f"Dataset manifest is incomplete: {manifest_path}")
    return manifest


def resolve_profiled_config(config: dict[str, Any], stage: str) -> dict[str, Any]:
    gate_verified = validate_stage1_gate(config.get("training", {})) if stage == "stage2" else None
    resolved = resolve_hardware(config, stage, discover_hardware())
    resolved = expand_environment(resolved)
    if gate_verified:
        resolved["training"]["stage1_gate_verified"] = gate_verified
    memory_report_path = resolved.get("training", {}).get("memory_report")
    if stage == "stage2" and memory_report_path:
        path = Path(_expand(memory_report_path))
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            requested = int(resolved["training"]["max_length"])
            minimum = int(resolved["training"].get("minimum_length", 16_384))
            selected = min(requested, int(report.get("selected_max_length", 0)))
            if selected < minimum:
                raise RuntimeError(
                    f"Memory profile selected {selected}, below Stage-2 minimum {minimum}"
                )
            resolved["training"]["max_length"] = selected
            resolved["training"]["memory_downgrade"] = selected != requested
        elif resolved["training"].get("require_memory_report", True):
            raise FileNotFoundError(f"Run profile-memory first; missing {path}")
    validate_training_config(resolved)
    return resolved


def _load_training_data(config: dict[str, Any], stage: str):
    from datasets import concatenate_datasets, load_dataset

    data_config = config["training_data"]
    primary_path = _expand(data_config["stage1_path"] if stage == "stage1" else data_config["repo_path"])
    validate_data_manifest(primary_path)
    primary = load_dataset("json", data_files=primary_path, split="train")
    if stage == "stage1":
        return primary
    rehearsal_path = _expand(data_config["stage1_path"])
    validate_data_manifest(rehearsal_path)
    rehearsal = load_dataset("json", data_files=rehearsal_path, split="train")
    rehearsal = rehearsal.filter(lambda row: row["split"] == "train")
    count = int(data_config.get("stage2_rehearsal_count", 556))
    order = sorted(range(len(rehearsal)), key=lambda index: stable_int(rehearsal[index]["id"]))[:count]
    rehearsal = rehearsal.select(order)
    return concatenate_datasets([primary, rehearsal]).shuffle(seed=int(config.get("seed", 3407)))


def _render_dataset(dataset, tokenizer: Any, num_proc: int):
    def render(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        return {
            "text": [
                finalize_chat_render(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    ),
                    tokenizer.eos_token,
                )
                for messages in batch["messages"]
            ]
        }

    return dataset.map(render, batched=True, num_proc=num_proc, desc="Applying Qwen chat template")


def _split_validation(dataset):
    train = dataset.filter(lambda row: row["split"] == "train") if "split" in dataset.column_names else dataset
    validation = (
        dataset.filter(lambda row: row["split"] == "validation")
        if "split" in dataset.column_names
        else None
    )
    return train, validation if validation is not None and len(validation) else None


def _last_checkpoint(output: Path) -> str | None:
    checkpoints = []
    for path in output.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[-1]), path))
        except ValueError:
            continue
    return str(max(checkpoints)[1]) if checkpoints else None


def validate_supervised_text(decoded: str) -> None:
    if "<|im_start|>user" in decoded or "<|im_start|>system" in decoded:
        raise RuntimeError("User or system turns leaked into assistant-only labels")


def _subsequence_positions(values: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    return [
        index
        for index in range(len(values) - len(needle) + 1)
        if values[index : index + len(needle)] == needle
    ]


def validate_assistant_mask(
    input_ids: list[int],
    labels: list[int],
    system_marker: list[int],
    user_marker: list[int],
    response_marker: list[int],
) -> int:
    responses = _subsequence_positions(input_ids, response_marker)
    users = _subsequence_positions(input_ids, user_marker)
    systems = _subsequence_positions(input_ids, system_marker)
    if not responses or not users or not systems:
        raise RuntimeError("Could not find Qwen system/user/assistant markers in mask audit")
    if any(label != -100 for label in labels[: responses[0] + len(response_marker)]):
        raise RuntimeError("System or initial user tokens are supervised")
    checked = 0
    for user_start in users:
        response_start = next((value for value in responses if value > user_start), None)
        if response_start is None:
            raise RuntimeError("User turn has no following assistant response")
        masked_end = response_start + len(response_marker)
        if any(label != -100 for label in labels[user_start:masked_end]):
            raise RuntimeError("User tokens are supervised")
        checked += 1
    for system_start in systems:
        response_start = next((value for value in responses if value > system_start), None)
        if response_start is None:
            raise RuntimeError("System turn has no following assistant response")
        if any(
            label != -100
            for label in labels[system_start : response_start + len(response_marker)]
        ):
            raise RuntimeError("System tokens are supervised")
    return checked


def _validate_response_labels(trainer: Any, tokenizer: Any) -> dict[str, int]:
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"]
    supervised = int((labels != -100).sum().item())
    masked = int((labels == -100).sum().item())
    if supervised <= 0 or masked <= 0:
        raise RuntimeError(
            f"Assistant-only loss mask is invalid: supervised={supervised}, masked={masked}"
        )
    supervised_ids = labels[labels != -100].detach().cpu().tolist()
    decoded = tokenizer.decode(supervised_ids, skip_special_tokens=False)
    validate_supervised_text(decoded)
    system_marker = tokenizer.encode("<|im_start|>system\n", add_special_tokens=False)
    user_marker = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    response_marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    checked_turns = 0
    for row_ids, row_labels in zip(batch["input_ids"], labels):
        checked_turns += validate_assistant_mask(
            row_ids.detach().cpu().tolist(),
            row_labels.detach().cpu().tolist(),
            system_marker,
            user_marker,
            response_marker,
        )
    return {
        "supervised_tokens_in_probe": supervised,
        "masked_tokens_in_probe": masked,
        "fully_masked_user_turns_in_probe": checked_turns,
    }


def train(
    config: dict[str, Any],
    stage: str,
    adapter: str | None,
    resume: str,
    max_steps: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    # Importing Unsloth through the loader occurs before TRL/Transformers imports.
    resolved = resolve_profiled_config(config, stage)
    parent_lineage = validate_stage1_adapter(adapter) if stage == "stage2" else None
    training = resolved["training"]
    output = safe_output_path(_expand(training["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    write_resolved_config(resolved, output / "resolved_config.yaml")
    write_json(environment_manifest(), output / "environment.json")

    max_length = int(training["max_length"])
    model, tokenizer, model_audit = load_model_and_tokenizer(resolved, max_length, adapter)
    raw_dataset = _load_training_data(resolved, stage)
    unfiltered_examples = len(raw_dataset)
    if "token_count" in raw_dataset.column_names:
        raw_dataset = raw_dataset.filter(
            lambda row: int(row["token_count"]) <= max_length,
            desc=f"Keeping examples that fit {max_length} tokens",
        )
    length_filtered_examples = unfiltered_examples - len(raw_dataset)
    train_raw, validation_raw = _split_validation(raw_dataset)
    if not len(train_raw):
        raise RuntimeError(f"No training examples fit the resolved max_length={max_length}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        train_raw = train_raw.select(range(min(limit, len(train_raw))))
        if validation_raw is not None:
            validation_raw = validation_raw.select(range(min(max(1, limit // 4), len(validation_raw))))
    num_proc = int(training.get("dataset_num_proc", 2))
    train_data = _render_dataset(train_raw, tokenizer, num_proc)
    validation_data = (
        _render_dataset(validation_raw, tokenizer, num_proc) if validation_raw is not None else None
    )

    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    report_to = list(training.get("report_to", ["tensorboard"]))
    if training.get("enable_wandb", True) and os.environ.get("WANDB_API_KEY"):
        if "wandb" not in report_to:
            report_to.append("wandb")
    args = SFTConfig(
        output_dir=str(output),
        dataset_text_field="text",
        max_length=max_length,
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training.get("learning_rate", 1e-4)),
        num_train_epochs=float(training.get("num_train_epochs", 1)),
        max_steps=max_steps if max_steps is not None else -1,
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=float(training.get("weight_decay", 0.01)),
        logging_steps=int(training.get("logging_steps", 10)),
        save_strategy="steps",
        save_steps=int(training.get("save_steps", 250)),
        save_total_limit=int(training.get("save_total_limit", 3)),
        eval_strategy="steps" if validation_data is not None else "no",
        eval_steps=int(training.get("eval_steps", 250)) if validation_data is not None else None,
        bf16=True,
        fp16=False,
        tf32=True,
        packing=bool(training.get("packing", stage == "stage1")),
        eval_packing=False,
        seed=int(resolved.get("seed", 3407)),
        data_seed=int(resolved.get("seed", 3407)),
        dataset_num_proc=num_proc,
        report_to=report_to,
        run_name=training.get("run_name", f"q35-coder-{stage}"),
        remove_unused_columns=True,
    )
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_data,
        "eval_dataset": validation_data,
    }
    if "processing_class" in inspect.signature(SFTTrainer).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:  # TRL 0.22.2 / Unsloth compatibility path
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    mask_audit = _validate_response_labels(trainer, tokenizer)
    checkpoint = _last_checkpoint(output) if resume == "auto" else resume if resume != "none" else None
    result = trainer.train(resume_from_checkpoint=checkpoint)
    if not math.isfinite(float(result.metrics.get("train_loss", math.nan))):
        raise RuntimeError(f"Non-finite training loss: {result.metrics.get('train_loss')}")
    final_adapter = output / "final_adapter"
    model.save_pretrained(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))
    lineage = {
        "stage": stage,
        "model_name": resolved.get("model_name") or resolved.get("model", {}).get("name"),
        "max_length": max_length,
        "length_filtered_examples": length_filtered_examples,
        "parent_adapter": str(Path(_expand(adapter)).resolve()) if adapter else None,
        "parent_lineage": parent_lineage,
    }
    write_json(lineage, final_adapter / "training_lineage.json")
    trainer.save_state()
    metrics = {
        "stage": stage,
        "checkpoint_resumed": checkpoint,
        "train_examples": len(train_data),
        "validation_examples": len(validation_data) if validation_data is not None else 0,
        "max_length": max_length,
        "model_audit": model_audit,
        "mask_audit": mask_audit,
        "train_metrics": result.metrics,
        "adapter": str(final_adapter),
        "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
        "lineage": lineage,
    }
    if validation_data is not None:
        metrics["eval_metrics"] = trainer.evaluate()
    write_json(metrics, output / "training_summary.json")

    hub_id = training.get("hub_model_id")
    if hub_id:
        if not os.environ.get("HF_TOKEN"):
            LOGGER.warning("hub_model_id is configured but HF_TOKEN is absent; skipping upload")
        else:
            model.push_to_hub(hub_id, token=os.environ["HF_TOKEN"])
            tokenizer.push_to_hub(hub_id, token=os.environ["HF_TOKEN"])
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--adapter", help="Stage-1 adapter path when continuing Stage 2")
    parser.add_argument("--resume", default="auto", help="auto, none, or checkpoint path")
    parser.add_argument("--max-steps", type=int, help="Override duration for a smoke run")
    parser.add_argument("--limit", type=int, help="Limit training examples for a smoke run")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    if args.stage == "stage2" and not args.adapter:
        raise SystemExit("--adapter is required for Stage 2")
    config = load_config(*args.config)
    if args.output_dir:
        config = deep_merge(config, {"training": {"output_dir": args.output_dir}})
    metrics = train(
        config,
        args.stage,
        args.adapter,
        args.resume,
        max_steps=args.max_steps,
        limit=args.limit,
    )
    print(json.dumps(metrics, indent=2, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
