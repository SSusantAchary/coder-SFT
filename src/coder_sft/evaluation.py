"""Generate benchmark answers without executing model-produced code and compare gates."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import load_config, safe_output_path
from .constants import SYSTEM_PROMPT
from .modeling import load_inference_model
from .quality import changed_files_from_patch
from .repo_context import GitObjectStore, RepoRevision
from .schema import read_jsonl
from .utils import extract_fenced_code, write_json


def generate_text(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    import torch

    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    continuation = generated[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()


def _evalplus_rows(dataset: str) -> Iterable[tuple[str, dict[str, Any]]]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    values = get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()
    return values.items()


def generate_evalplus(
    model: Any,
    tokenizer: Any,
    dataset: str,
    output: Path,
    max_new_tokens: int,
    limit: int | None,
) -> dict[str, Any]:
    rows = list(_evalplus_rows(dataset))
    if limit is not None:
        rows = rows[:limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task_id, problem in rows:
            prompt = str(problem["prompt"])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Complete the following Python programming task. Return the complete "
                        f"implementation in one code block.\n\n{prompt}"
                    ),
                },
            ]
            response = generate_text(model, tokenizer, messages, max_new_tokens)
            solution = extract_fenced_code(response)
            handle.write(
                json.dumps(
                    {"task_id": task_id, "solution": solution, "raw_response": response},
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {
        "dataset": dataset,
        "samples": len(rows),
        "output": str(output),
        "code_executed": False,
        "generation": {"do_sample": False, "seed": 3407, "max_new_tokens": max_new_tokens},
    }
    write_json(summary, output.with_suffix(".summary.json"))
    return summary


def _ranked_values(text: str) -> list[str]:
    values = []
    for line in text.splitlines():
        value = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip("`'")
        if value:
            values.append(value)
    return values


def _recall_at(expected: list[str], predicted: list[str], k: int) -> float:
    expected_set = set(expected)
    return len(expected_set.intersection(predicted[:k])) / len(expected_set) if expected_set else 0.0


def git_apply_check(
    store: GitObjectStore,
    repository: str,
    commit: str,
    patch: str,
    temp_root: str | Path,
) -> bool:
    revision = RepoRevision(repository, commit)
    source = store.ensure(revision)
    root = safe_output_path(temp_root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="apply-check-", dir=root) as temporary:
        checkout = Path(temporary) / "repo"
        subprocess.run(
            ["git", "clone", "--shared", "--no-checkout", str(source), str(checkout)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", commit],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        patch_path = Path(temporary) / "candidate.diff"
        patch_path.write_text(patch, encoding="utf-8")
        return (
            subprocess.run(
                ["git", "-C", str(checkout), "apply", "--check", str(patch_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )


def generate_repo(
    model: Any,
    tokenizer: Any,
    data_path: str,
    output: Path,
    config: dict[str, Any],
    max_new_tokens: int,
    limit: int | None,
) -> dict[str, Any]:
    rows = [row for row in read_jsonl(data_path) if row.split == "test"]
    if limit is not None:
        rows = rows[:limit]
    cache = os.path.expandvars(config["repo_data"]["cache_dir"])
    temp_root = os.path.expandvars(config["evaluation"]["temp_dir"])
    store = GitObjectStore(cache)
    metric_values: dict[str, list[float]] = defaultdict(list)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt_messages = row.messages[:-1]
            response = generate_text(model, tokenizer, prompt_messages, max_new_tokens)
            expected = row.messages[-1]["content"]
            record: dict[str, Any] = {
                "id": row.id,
                "task_type": row.task_type,
                "expected": expected,
                "response": response,
            }
            if row.task_type == "file_localization":
                expected_values = list(row.metadata.get("changed_files") or [])
                predicted = _ranked_values(response)
                for k in (1, 3, 5):
                    value = _recall_at(expected_values, predicted, k)
                    record[f"recall_at_{k}"] = value
                    metric_values[f"file_recall_at_{k}"].append(value)
            elif row.task_type == "function_localization":
                expected_values = list(row.metadata.get("functions") or [])
                predicted = _ranked_values(response)
                for k in (1, 3, 5):
                    value = _recall_at(expected_values, predicted, k)
                    record[f"recall_at_{k}"] = value
                    metric_values[f"function_recall_at_{k}"].append(value)
            elif row.task_type == "patch_generation":
                patch = extract_fenced_code(response)
                expected_files = set(row.metadata.get("changed_files") or [])
                predicted_files = set(changed_files_from_patch(patch))
                valid = bool(re.search(r"^--- ", patch, re.MULTILINE) and re.search(r"^\+\+\+ ", patch, re.MULTILINE))
                precision = (
                    len(expected_files.intersection(predicted_files)) / len(predicted_files)
                    if predicted_files
                    else 0.0
                )
                applies = False
                if valid:
                    try:
                        applies = git_apply_check(
                            store,
                            str(row.repo_id),
                            str(row.metadata["commit"]),
                            patch,
                            temp_root,
                        )
                    except (OSError, subprocess.SubprocessError):
                        applies = False
                record.update(valid_diff=valid, changed_file_precision=precision, applies=applies)
                metric_values["valid_diff_rate"].append(float(valid))
                metric_values["changed_file_precision"].append(precision)
                metric_values["patch_apply_rate"].append(float(applies))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics = {
        key: sum(values) / len(values) if values else 0.0 for key, values in metric_values.items()
    }
    summary = {
        "dataset": "repo",
        "samples": len(rows),
        "task_counts": dict(Counter(row.task_type for row in rows)),
        "metrics": metrics,
        "output": str(output),
        "code_executed": False,
        "generation": {"do_sample": False, "seed": 3407, "max_new_tokens": max_new_tokens},
    }
    write_json(summary, output.with_suffix(".summary.json"))
    return summary


def _metric(report: dict[str, Any], key: str) -> float:
    if key in report:
        return float(report[key])
    for section in ("benchmarks", "metrics", "repo", "smoke"):
        value = report.get(section)
        if isinstance(value, dict) and key in value:
            return float(value[key])
    raise KeyError(f"Metric {key!r} is absent")


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any], stage: str) -> dict[str, Any]:
    core_keys = ("humaneval_plus_pass_at_1", "mbpp_plus_pass_at_1")
    base_core = [_metric(baseline, key) for key in core_keys]
    candidate_core = [_metric(candidate, key) for key in core_keys]
    core_gain = sum(candidate_core) / 2 - sum(base_core) / 2
    core_regression = max(base - current for base, current in zip(base_core, candidate_core))
    checks = {
        "core_average_gain_at_least_1pp": core_gain >= 0.01,
        "no_core_regression_over_2pp": core_regression <= 0.02,
    }
    if stage == "stage1":
        checks.update(
            finite_validation_loss=bool(candidate.get("smoke", {}).get("finite_validation_loss", False)),
            no_repetition_regression=not bool(candidate.get("smoke", {}).get("repetition_regression", True)),
            no_termination_regression=not bool(candidate.get("smoke", {}).get("termination_regression", True)),
        )
    else:
        repo_gain = max(
            _metric(candidate, "file_recall_at_3") - _metric(baseline, "file_recall_at_3"),
            _metric(candidate, "patch_apply_rate") - _metric(baseline, "patch_apply_rate"),
        )
        checks = {
            "repo_gain_at_least_5pp": repo_gain >= 0.05,
            "no_core_regression_over_2pp": core_regression <= 0.02,
        }
    return {
        "stage": stage,
        "passed": all(checks.values()),
        "checks": checks,
        "core_average_delta": core_gain,
    }


def generate_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=("humaneval", "mbpp", "repo"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(*args.config)
    output = safe_output_path(args.output)
    max_length = int(config.get("evaluation", {}).get("max_length", 8192))
    model, tokenizer = load_inference_model(args.model, max_length)
    if args.dataset == "repo":
        summary = generate_repo(
            model,
            tokenizer,
            os.path.expandvars(config["training_data"]["repo_path"]),
            output,
            config,
            args.max_new_tokens,
            args.limit,
        )
    else:
        summary = generate_evalplus(
            model, tokenizer, args.dataset, output, args.max_new_tokens, args.limit
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


def compare_main() -> None:
    parser = argparse.ArgumentParser(description="Compare normalized evaluation reports and apply a stage gate")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    result = compare_reports(baseline, candidate, args.stage)
    if args.output:
        write_json(result, safe_output_path(args.output))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    generate_main()
