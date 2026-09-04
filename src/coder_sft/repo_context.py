"""Build objective repository-context SFT tasks from language-specific SWE-smith rows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .config import load_config, safe_output_path
from .constants import SYSTEM_PROMPT
from .quality import (
    changed_files_from_patch,
    function_names_from_patch,
    largest_remainder_counts,
    language_for_path,
)
from .schema import NormalizedExample, write_jsonl
from .utils import (
    deterministic_split,
    finalize_chat_render,
    hub_revision,
    priority_sample,
    stable_int,
    write_json,
)

LOGGER = logging.getLogger(__name__)
SAFE_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SWE_SMITH_SUFFIX = {
    "python": "py",
    "javascript": "js",
    "typescript": "ts",
    "cpp": "cpp",
    "rust": "rs",
}


@dataclass(frozen=True)
class RepoRevision:
    github_repo: str
    commit: str


def parse_swesmith_revision(row: dict[str, Any]) -> RepoRevision:
    value = str(row.get("repo") or "")
    if value.startswith("swesmith/"):
        value = value[len("swesmith/") :]
    try:
        repository, commit = value.rsplit(".", 1)
        owner, name = repository.split("__", 1)
    except ValueError as exc:
        raise ValueError(f"Cannot parse SWE-smith repository identifier: {value!r}") from exc
    github_repo = f"{owner}/{name}"
    if not SAFE_REPO.fullmatch(github_repo) or not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
        raise ValueError(f"Unsafe SWE-smith repository identifier: {value!r}")
    return RepoRevision(github_repo=github_repo, commit=commit.lower())


class GitObjectStore:
    """Read pinned Git objects without creating mutable worktrees."""

    def __init__(self, cache_root: str | Path):
        self.cache_root = safe_output_path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _path(self, repository: str) -> Path:
        return self.cache_root / repository.replace("/", "__")

    def ensure(self, revision: RepoRevision) -> Path:
        path = self._path(revision.github_repo)
        if not path.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    f"https://github.com/{revision.github_repo}.git",
                    str(path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        exists = subprocess.run(
            ["git", "-C", str(path), "cat-file", "-e", f"{revision.commit}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
        if not exists:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(path),
                    "fetch",
                    "--depth=1",
                    "origin",
                    revision.commit,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        return path

    def list_files(self, revision: RepoRevision) -> list[str]:
        path = self.ensure(revision)
        result = subprocess.run(
            ["git", "-C", str(path), "ls-tree", "-r", "--name-only", revision.commit],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return [line for line in result.stdout.splitlines() if line]

    def read_file(self, revision: RepoRevision, file_path: str, max_bytes: int) -> str | None:
        if file_path.startswith("/") or ".." in PurePosixPath(file_path).parts:
            return None
        path = self.ensure(revision)
        result = subprocess.run(
            ["git", "-C", str(path), "show", f"{revision.commit}:{file_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode:
            return None
        raw = result.stdout[: max_bytes + 1]
        if len(raw) > max_bytes or b"\x00" in raw:
            return None
        return raw.decode("utf-8", errors="replace")


def test_paths(row: dict[str, Any]) -> list[str]:
    values = list(row.get("FAIL_TO_PASS") or []) + list(row.get("PASS_TO_PASS") or [])
    return list(dict.fromkeys(str(value).split("::", 1)[0] for value in values if value))


def dependency_candidates(relevant: dict[str, str], all_files: list[str]) -> list[str]:
    referenced: set[str] = set()
    patterns = [
        r"(?:from|import)\s+([A-Za-z_][\w.]*)",
        r"(?:require\(|from\s+)[\"']([^\"']+)[\"']",
        r"#include\s*[<\"]([^>\"]+)[>\"]",
        r"(?:use|mod)\s+([A-Za-z_][\w:]*)",
    ]
    for content in relevant.values():
        for pattern in patterns:
            for match in re.findall(pattern, content):
                referenced.update(part for part in re.split(r"[./:]", match) if part)
    result = []
    for path in all_files:
        stem = PurePosixPath(path).stem
        if stem in referenced and path not in relevant:
            result.append(path)
    return result


def _block(path: str, content: str) -> str:
    language = language_for_path(path)
    return f"### FILE: {path}\n```{language}\n{content}\n```"


def _truncate_to_tokens(tokenizer: Any, text: str, budget: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return text
    if budget < 32:
        return tokenizer.decode(ids[:budget])
    left = budget * 2 // 3
    right = budget - left
    return tokenizer.decode(ids[:left]) + "\n...<TRUNCATED>...\n" + tokenizer.decode(ids[-right:])


def arrange_file_blocks(
    relevant: list[tuple[str, str]],
    distractors: list[tuple[str, str]],
    position_bucket: int,
) -> list[tuple[str, str]]:
    if not 0 <= position_bucket <= 4:
        raise ValueError("position_bucket must be in [0, 4]")
    ratio = (0.1, 0.3, 0.5, 0.7, 0.9)[position_bucket]
    before = round(len(distractors) * ratio)
    return distractors[:before] + relevant + distractors[before:]


def choose_task_type(source_id: str, functions: list[str], tests: list[str], seed: int) -> str:
    bucket = stable_int(source_id, seed) % 100
    desired = (
        "file_localization"
        if bucket < 40
        else "function_localization"
        if bucket < 60
        else "patch_generation"
        if bucket < 90
        else "test_localization"
    )
    if desired == "function_localization" and not functions:
        return "patch_generation"
    if desired == "test_localization" and not tests:
        return "patch_generation"
    return desired


def task_response(
    task_type: str,
    patch: str,
    changed_files: list[str],
    functions: list[str],
    tests: list[str],
) -> str:
    if task_type == "file_localization":
        return "\n".join(f"{index + 1}. {path}" for index, path in enumerate(changed_files))
    if task_type == "function_localization":
        return "\n".join(f"{index + 1}. {name}" for index, name in enumerate(functions))
    if task_type == "test_localization":
        return "\n".join(f"{index + 1}. {path}" for index, path in enumerate(tests))
    return patch.strip()


def task_instruction(task_type: str) -> str:
    return {
        "file_localization": "Rank the files that must change. Return only a numbered file list.",
        "function_localization": "Rank the functions or classes that must change. Return only a numbered list.",
        "test_localization": "List the tests that validate this issue. Return only a numbered test list.",
        "patch_generation": "Produce the smallest correct unified diff and no unrelated changes.",
    }[task_type]


def build_repo_example(
    row: dict[str, Any],
    language: str,
    tokenizer: Any,
    store: GitObjectStore,
    seed: int,
    max_file_bytes: int,
) -> NormalizedExample:
    revision = parse_swesmith_revision(row)
    patch = str(row.get("patch") or "")
    changed = changed_files_from_patch(patch)
    if not changed:
        raise ValueError("gold patch has no changed files")
    all_files = store.list_files(revision)
    relevant = {
        path: content
        for path in changed
        if (content := store.read_file(revision, path, max_file_bytes)) is not None
    }
    if not relevant:
        raise ValueError("no changed pre-change files were readable")
    tests = test_paths(row)
    dependencies = dependency_candidates(relevant, all_files)
    source_id = str(row.get("instance_id") or f"{revision.github_repo}:{revision.commit}")
    split = deterministic_split(revision.github_repo, seed)
    functions = function_names_from_patch(patch)
    task_type = choose_task_type(source_id, functions, tests, seed)
    position_bucket = stable_int(f"position:{source_id}", seed) % 5
    context_bucket = stable_int(f"length:{source_id}", seed) % 10
    target_tokens = 16_384 if context_bucket < 6 else 24_576 if context_bucket < 8 else 32_768

    priority_paths = list(dict.fromkeys(tests + dependencies))
    same_language = [
        path
        for path in all_files
        if path not in relevant
        and path not in priority_paths
        and language_for_path(path) == language
    ]
    same_language.sort(key=lambda path: stable_int(f"{source_id}:{path}", seed))
    candidate_paths = priority_paths + same_language
    distractors: list[tuple[str, str]] = []
    for path in candidate_paths:
        content = store.read_file(revision, path, max_file_bytes)
        if content is not None:
            distractors.append((path, content))
        if len(distractors) >= 80:
            break

    relevant_blocks = [(path, content) for path, content in relevant.items()]
    ordered = arrange_file_blocks(relevant_blocks, distractors, position_bucket)
    raw_tree = "\n".join(all_files[:4000])
    tree = _truncate_to_tokens(tokenizer, raw_tree, max(512, target_tokens // 8))
    issue = str(row.get("problem_statement") or "").strip()
    fixed = (
        f"Repository: {revision.github_repo}@{revision.commit}\n\n"
        f"Repository tree:\n{tree}\n\nIssue:\n{issue}\n\n"
        f"Task:\n{task_instruction(task_type)}\n\nRepository context:\n"
    )
    response = task_response(task_type, patch, changed, functions, tests)
    response_tokens = len(tokenizer.encode(response, add_special_tokens=False))
    fixed_tokens = len(tokenizer.encode(fixed, add_special_tokens=False))
    context_budget = target_tokens - response_tokens - fixed_tokens - 256
    if context_budget < 512:
        raise ValueError("issue or response leaves no repository-context budget")

    # A low floor ensures all late-position relevant files remain reachable even
    # when many distractors are present. Every changed file is verified below.
    per_block = max(64, context_budget // max(1, len(ordered)))
    rendered_blocks: list[str] = []
    rendered_paths: set[str] = set()
    consumed = 0
    for path, content in ordered:
        rendered = _block(path, content)
        rendered = _truncate_to_tokens(tokenizer, rendered, min(per_block, context_budget - consumed))
        tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
        if consumed + tokens > context_budget or tokens <= 0:
            continue
        rendered_blocks.append(rendered)
        rendered_paths.add(path)
        consumed += tokens
        if context_budget - consumed < 64:
            break
    missing_relevant = set(relevant) - rendered_paths
    if missing_relevant:
        raise ValueError(f"token budget omitted changed files: {sorted(missing_relevant)}")
    prompt = fixed + "\n\n".join(rendered_blocks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    rendered = finalize_chat_render(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        ),
        tokenizer.eos_token,
    )
    token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
    if token_count > target_tokens:
        raise ValueError(f"rendered task exceeds target: {token_count}>{target_tokens}")
    return NormalizedExample(
        id=f"swesmith:{source_id}:{task_type}",
        source="swe_smith",
        source_id=source_id,
        repo_id=revision.github_repo,
        language=language,
        task_type=task_type,
        quality_tier="executable_source",
        split=split,
        messages=messages,
        token_count=token_count,
        metadata={
            "commit": revision.commit,
            "changed_files": changed,
            "functions": functions,
            "tests": tests,
            "position_bucket": position_bucket,
            "target_context_tokens": target_tokens,
            "repository_tree_truncated": tree != raw_tree,
            "image_name": row.get("image_name"),
        },
    )


def _load_swesmith(language: str) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    suffix = SWE_SMITH_SUFFIX.get(language, language)
    return load_dataset(f"SWE-bench/SWE-smith-{suffix}", split="train", streaming=True)


def build(config: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    repo_config = config["repo_data"]
    seed = int(config.get("seed", 3407))
    output = safe_output_path(os.path.expandvars(repo_config["output"]))
    store = GitObjectStore(os.path.expandvars(repo_config["cache_dir"]))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    desired = {
        "train": int(repo_config.get("train_count", 5000)),
        "validation": int(repo_config.get("validation_count", 100)),
        "test": int(repo_config.get("test_count", 100)),
    }
    weights = repo_config["language_weights"]
    targets = {split: largest_remainder_counts(count, weights) for split, count in desired.items()}
    accepted: list[NormalizedExample] = []
    accepted_counts: dict[str, Counter[str]] = defaultdict(Counter)
    overflow: dict[str, list[NormalizedExample]] = defaultdict(list)
    skips: Counter[str] = Counter()
    skip_details: list[dict[str, str]] = []
    multiplier = int(repo_config.get("candidate_multiplier", 3))
    max_file_bytes = int(repo_config.get("max_file_bytes", 100_000))

    for language in weights:
        need = max(
            targets["train"][language] * multiplier,
            targets["validation"][language] * 25,
            targets["test"][language] * 25,
            100,
        )
        rows = priority_sample(
            _load_swesmith(language),
            need,
            key=lambda row: row.get("instance_id"),
            seed=seed,
        )
        for row in rows:
            try:
                split = deterministic_split(parse_swesmith_revision(row).github_repo, seed)
                if accepted_counts[split][language] >= targets[split][language] and len(
                    overflow[split]
                ) >= desired[split]:
                    continue
                example = build_repo_example(
                    row, language, tokenizer, store, seed, max_file_bytes
                )
                if accepted_counts[split][language] < targets[split][language]:
                    accepted.append(example)
                    accepted_counts[split][language] += 1
                else:
                    overflow[split].append(example)
            except (ValueError, subprocess.SubprocessError, OSError) as exc:
                reason = type(exc).__name__
                skips[reason] += 1
                if len(skip_details) < int(repo_config.get("max_skip_details", 1000)):
                    skip_details.append(
                        {"instance_id": str(row.get("instance_id")), "reason": str(exc)}
                    )

    actual = Counter(example.split for example in accepted)
    for split, requested in desired.items():
        missing = requested - actual[split]
        if missing > 0:
            extras = sorted(overflow[split], key=lambda row: stable_int(row.id, seed))
            accepted.extend(extras[:missing])
            actual[split] += min(missing, len(extras))
    shortages = {
        split: desired[split] - actual[split]
        for split in desired
        if actual[split] < desired[split]
    }
    accepted.sort(key=lambda row: stable_int(row.id, seed))
    if not shortages:
        write_jsonl(accepted, output)
    dataset_revisions = {
        language: hub_revision(
            f"SWE-bench/SWE-smith-{SWE_SMITH_SUFFIX.get(language, language)}",
            "dataset",
        )
        for language in weights
    }
    manifest = {
        "name": "qwen35_coder_repo_v1",
        "version": 1,
        "seed": seed,
        "examples": len(accepted),
        "splits": dict(actual),
        "languages": dict(Counter(row.language for row in accepted)),
        "task_types": dict(Counter(row.task_type for row in accepted)),
        "position_buckets": dict(
            Counter(str(row.metadata["position_bucket"]) for row in accepted)
        ),
        "requested_language_quotas": targets,
        "resolved_language_quotas": {
            split: dict(Counter(row.language for row in accepted if row.split == split))
            for split in desired
        },
        "complete": not shortages,
        "shortages": shortages,
        "skips": dict(skips),
        "skip_details": skip_details,
        "source_revisions": dataset_revisions,
        "output": str(output),
    }
    write_json(manifest, output.with_suffix(".manifest.json"))
    if shortages:
        raise RuntimeError(
            f"Unable to fill repository split targets: {shortages}. "
            f"Inspect {output.with_suffix('.manifest.json')} and increase candidate_multiplier."
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    print(json.dumps(build(load_config(*args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
