"""Prepare the Stage-1 EER6 and CommitPackFT mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import quote

from .config import deep_merge, load_config, safe_output_path
from .constants import MODEL_NAME, SYSTEM_PROMPT
from .dedup import MinHashDeduper
from .quality import (
    broad_quality_tier,
    infer_language,
    largest_remainder_counts,
    unified_diff,
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


def make_messages(instruction: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction.strip()},
        {"role": "assistant", "content": response.strip()},
    ]


def render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return finalize_chat_render(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        ),
        tokenizer.eos_token,
    )


def count_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    rendered = render_messages(tokenizer, messages)
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def _source_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("id") or row.get("commit") or index)


def _sample_rows(
    rows: Iterable[dict[str, Any]],
    count: int,
    seed: int,
    multiplier: float,
    scan_limit: int | None = None,
) -> list[dict[str, Any]]:
    if scan_limit is not None:
        from itertools import islice

        rows = islice(rows, scan_limit)
    candidate_count = max(count, math.ceil(count * multiplier))
    if scan_limit is not None:
        candidate_count = max(candidate_count, count + 16)
    return priority_sample(
        rows,
        candidate_count,
        key=lambda row: row.get("id") or row.get("commit") or json.dumps(row, sort_keys=True),
        seed=seed,
    )


def select_broad_rows(
    rows: Iterable[dict[str, Any]],
    total: int,
    tier_weights: dict[str, float],
    seed: int,
    multiplier: float,
    scan_limit: int | None = None,
) -> list[tuple[dict[str, Any], str]]:
    targets = largest_remainder_counts(total, tier_weights)
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    limits = {
        tier: max(value, math.ceil(value * multiplier), value + 8 if scan_limit else value)
        for tier, value in targets.items()
    }

    # Use one bounded deterministic heap per tier while scanning the stream.
    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    import heapq

    for index, row in enumerate(rows):
        if scan_limit is not None and index >= scan_limit:
            break
        tier = broad_quality_tier(row)
        if tier not in limits or limits[tier] <= 0:
            continue
        score = stable_int(_source_id(row, index), seed)
        item = (-score, -index, row)
        heap = heaps[tier]
        if len(heap) < limits[tier]:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    for tier, heap in heaps.items():
        candidates[tier] = [item[2] for item in sorted(heap, key=lambda x: (-x[0], -x[1]))]
    return [(row, tier) for tier in targets for row in candidates[tier]]


def commitpack_url(language: str) -> str:
    source_name = {"cpp": "c++", "shell": "shell"}.get(language, language)
    return (
        "https://huggingface.co/datasets/bigcode/commitpackft/resolve/main/"
        f"data/{quote(source_name, safe='')}/data.jsonl"
    )


def build_eer6_example(
    row: dict[str, Any],
    source: str,
    tier: str,
    tokenizer: Any,
    seed: int,
) -> NormalizedExample:
    source_id = str(row["id"])
    messages = make_messages(str(row["input"]), str(row["output"]))
    language = infer_language(str(row["output"]))
    return NormalizedExample(
        id=f"{source}:{source_id}",
        source=source,
        source_id=source_id,
        repo_id=None,
        language=language,
        task_type="code_generation",
        quality_tier=tier,
        split=deterministic_split(f"{source}:{source_id}", seed),
        messages=messages,
        token_count=count_tokens(tokenizer, messages),
        metadata={
            "domain": row.get("domain"),
            "generation_algorithm": row.get("generation_algorithm"),
            "average_test_score": row.get("average_test_score"),
        },
    )


def build_commit_example(
    row: dict[str, Any], tokenizer: Any, seed: int
) -> NormalizedExample | None:
    old_file = str(row.get("old_file") or "file")
    new_file = str(row.get("new_file") or old_file)
    old = str(row.get("old_contents") or "")
    new = str(row.get("new_contents") or "")
    patch = unified_diff(old, new, old_file, new_file)
    if not patch.strip():
        return None
    instruction = (
        f"Repository file: {old_file}\n\n"
        f"Task: {row.get('subject') or row.get('message') or 'Apply the requested change.'}\n\n"
        f"Current content:\n```{row.get('lang') or ''}\n{old}\n```\n\n"
        "Return the smallest correct unified diff."
    )
    messages = make_messages(instruction, patch)
    source_id = str(row.get("commit") or hashlib.sha256(patch.encode()).hexdigest())
    repo_id = str(row.get("repos") or "") or None
    source_language = infer_language("", str(row.get("lang") or "unknown"))
    language = infer_language(new, source_language)
    return NormalizedExample(
        id=f"commitpackft:{source_id}:{new_file}",
        source="commitpackft",
        source_id=source_id,
        repo_id=repo_id,
        language=language,
        task_type="code_edit",
        quality_tier="filtered",
        split=deterministic_split(repo_id or source_id, seed),
        messages=messages,
        token_count=count_tokens(tokenizer, messages),
        metadata={
            "old_file": old_file,
            "new_file": new_file,
            "license": row.get("license"),
            "source_language": source_language,
        },
    )


def load_benchmark_prompts() -> list[str]:
    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
    except ImportError as exc:
        raise RuntimeError(
            "EvalPlus is required for benchmark-contamination exclusion; install the eval extra"
        ) from exc
    prompts = [row["prompt"] for row in get_human_eval_plus().values()]
    prompts.extend(row["prompt"] for row in get_mbpp_plus().values())
    return prompts


def _accept_examples(
    candidates: Iterable[NormalizedExample],
    requested: int,
    max_tokens: int,
    deduper: MinHashDeduper,
    benchmark_deduper: MinHashDeduper,
    counters: Counter[str],
) -> list[NormalizedExample]:
    if requested <= 0:
        return []
    accepted: list[NormalizedExample] = []
    for example in candidates:
        if example.token_count > max_tokens:
            counters["too_long"] += 1
            continue
        user_text = "\n".join(
            message["content"] for message in example.messages if message["role"] == "user"
        )
        if benchmark_deduper.is_duplicate(user_text):
            counters["benchmark_overlap"] += 1
            continue
        dedup_text = "\n".join(
            message["content"]
            for message in example.messages
            if message["role"] in {"user", "assistant"}
        )
        if not deduper.add_if_unique(dedup_text):
            counters["duplicate"] += 1
            continue
        accepted.append(example)
        if len(accepted) == requested:
            break
    return accepted


def _load_stream(dataset_name: str, split: str = "train") -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    return load_dataset(dataset_name, split=split, streaming=True)


def prepare(config: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    from transformers import AutoTokenizer

    data = config["data"]
    smoke_scan_limit: int | None = None
    if limit is not None:
        if limit < 3:
            raise ValueError("--limit must be at least 3 so every source is represented")
        scaled = largest_remainder_counts(limit, data["counts"])
        config = deep_merge(config, {"data": {"counts": scaled}})
        data = config["data"]
        smoke_scan_limit = int(data.get("smoke_scan_limit", 5000))
    seed = int(config.get("seed", 3407))
    output = safe_output_path(os.path.expandvars(data["stage1_output"]))
    tokenizer = AutoTokenizer.from_pretrained(config.get("model_name", MODEL_NAME))
    max_tokens = int(data.get("max_tokens", 8192))
    multiplier = float(data.get("candidate_multiplier", 1.3))
    counters: Counter[str] = Counter()
    deduper = MinHashDeduper(threshold=float(data.get("dedup_threshold", 0.85)))
    benchmark_deduper = MinHashDeduper(threshold=float(data.get("dedup_threshold", 0.85)))
    for prompt in load_benchmark_prompts():
        benchmark_deduper.add_if_unique(prompt)

    refined_count = int(data["counts"]["refined"])
    refined_rows = _sample_rows(
        _load_stream(data["datasets"]["refined"]),
        refined_count,
        seed,
        multiplier,
        smoke_scan_limit,
    )
    refined_candidates = (
        build_eer6_example(row, "eer6_refined", "refined", tokenizer, seed)
        for row in refined_rows
    )
    refined = _accept_examples(
        refined_candidates, refined_count, max_tokens, deduper, benchmark_deduper, counters
    )

    broad_count = int(data["counts"]["broad"])
    broad_rows = select_broad_rows(
        _load_stream(data["datasets"]["broad"]),
        broad_count,
        data["broad_tier_weights"],
        seed,
        multiplier,
        smoke_scan_limit,
    )
    broad_targets = largest_remainder_counts(broad_count, data["broad_tier_weights"])
    broad: list[NormalizedExample] = []
    for tier, requested in broad_targets.items():
        broad_candidates = (
            build_eer6_example(row, "eer6_broad", row_tier, tokenizer, seed)
            for row, row_tier in broad_rows
            if row_tier == tier
        )
        broad.extend(
            _accept_examples(
                broad_candidates,
                requested,
                max_tokens,
                deduper,
                benchmark_deduper,
                counters,
            )
        )

    commit_total = int(data["counts"]["commitpack"])
    commit_counts = largest_remainder_counts(commit_total, data["language_weights"])
    commits: list[NormalizedExample] = []
    commit_extras: list[NormalizedExample] = []
    from datasets import load_dataset

    for language, requested in commit_counts.items():
        if requested <= 0:
            continue
        stream = load_dataset(
            "json", data_files=commitpack_url(language), split="train", streaming=True
        )
        rows = _sample_rows(stream, requested, seed, multiplier, smoke_scan_limit)
        candidates = [
            value
            for value in (build_commit_example(row, tokenizer, seed) for row in rows)
            if value is not None
        ]
        accepted_language = _accept_examples(
            candidates,
            len(candidates),
            max_tokens,
            deduper,
            benchmark_deduper,
            counters,
        )
        commits.extend(accepted_language[:requested])
        commit_extras.extend(accepted_language[requested:])
    commit_shortfall = commit_total - len(commits)
    if commit_shortfall > 0:
        commit_extras.sort(key=lambda row: stable_int(row.id, seed))
        commits.extend(commit_extras[:commit_shortfall])

    examples = refined + broad + commits
    expected = refined_count + broad_count + commit_total
    examples.sort(key=lambda row: stable_int(row.id, seed))
    complete = len(examples) == expected
    if complete:
        write_jsonl(examples, output)
    model_name = config.get("model_name", MODEL_NAME)
    dataset_names = {
        "refined": data["datasets"]["refined"],
        "broad": data["datasets"]["broad"],
        "commitpack": "bigcode/commitpackft",
    }
    resolved_commit_languages = dict(
        Counter(str(row.metadata.get("source_language", row.language)) for row in commits)
    )
    manifest = {
        "name": "qwen35_coder_stage1_v1",
        "version": 1,
        "model": model_name,
        "model_revision": hub_revision(model_name, "model"),
        "source_revisions": {
            name: hub_revision(repo_id, "dataset")
            for name, repo_id in dataset_names.items()
        },
        "seed": seed,
        "examples": len(examples),
        "expected_examples": expected,
        "complete": complete,
        "sources": dict(Counter(row.source for row in examples)),
        "splits": dict(Counter(row.split for row in examples)),
        "languages": dict(Counter(row.language for row in examples)),
        "quality_tiers": dict(Counter(row.quality_tier for row in examples)),
        "token_stats": token_stats(row.token_count for row in examples),
        "dedup": {"method": "64-permutation MinHash-LSH", "benchmark_exclusion": True},
        "commitpack_language_quotas": commit_counts,
        "commitpack_resolved_languages": resolved_commit_languages,
        "commitpack_quota_redistributed": resolved_commit_languages != commit_counts,
        "rejections": dict(counters),
        "output": str(output),
    }
    write_json(manifest, output.with_suffix(".manifest.json"))
    if not complete:
        raise RuntimeError(
            f"Prepared {len(examples):,}/{expected:,} examples. "
            f"Inspect {output.with_suffix('.manifest.json')} and increase "
            f"data.candidate_multiplier; rejection counts: {dict(counters)}"
        )
    return manifest


def token_stats(values: Iterable[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}

    def percentile(p: float) -> int:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--limit", type=int, help="Scale source counts to a small smoke dataset")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    manifest = prepare(load_config(*args.config), args.limit)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
