"""Dataset quality, language, split, and diff helpers."""

from __future__ import annotations

import difflib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import LANGUAGE_ALIASES


def _walk_scores(value: Any) -> list[float]:
    scores: list[float] = []
    if isinstance(value, dict):
        if isinstance(value.get("score"), (int, float)):
            scores.append(float(value["score"]))
        for child in value.values():
            scores.extend(_walk_scores(child))
    elif isinstance(value, list):
        for child in value:
            scores.extend(_walk_scores(child))
    return scores


def parse_judge_scores(raw: Any) -> list[float]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return [float(x) for x in re.findall(r'"score"\s*:\s*([0-9.]+)', raw)]
    return _walk_scores(raw)


def parse_test_score(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        match = re.search(r"[0-9]+(?:\.[0-9]+)?", raw)
        if match:
            value = float(match.group())
            return value / 100.0 if value > 1.0 else value
    return 0.0


def broad_quality_tier(row: dict[str, Any]) -> str | None:
    test_score = parse_test_score(row.get("average_test_score"))
    scores = parse_judge_scores(row.get("llm_judgement"))
    minimum_judge = min(scores) if scores else 0.0
    if test_score >= 1.0 and minimum_judge >= 5.0:
        return "A"
    if test_score >= 1.0 and minimum_judge >= 4.0:
        return "B"
    if test_score >= 0.9 and minimum_judge >= 4.0:
        return "C"
    if test_score >= 0.8 and minimum_judge >= 4.0:
        return "D"
    return None


def infer_language(text: str, fallback: str = "unknown") -> str:
    labels = re.findall(r"```\s*([A-Za-z0-9_+.#-]+)", text)
    if labels:
        normalized = LANGUAGE_ALIASES.get(labels[0].lower(), labels[0].lower())
        return normalized
    checks = [
        ("python", r"\bdef\s+\w+\s*\(|\bimport\s+\w+|\bfrom\s+\w+\s+import\b"),
        ("typescript", r"\binterface\s+\w+|:\s*(?:string|number|boolean)\b"),
        ("javascript", r"\b(?:const|let|var)\s+\w+|=>"),
        ("java", r"\bpublic\s+(?:static\s+)?(?:class|void|int|String)\b"),
        ("cpp", r"#include\s*[<\"]|\bstd::"),
        ("go", r"\bpackage\s+main\b|\bfunc\s+\w+\s*\("),
        ("rust", r"\bfn\s+\w+\s*\(|\blet\s+mut\b"),
        ("shell", r"^#!.*\b(?:bash|sh)\b"),
        ("sql", r"\bSELECT\b.+\bFROM\b"),
    ]
    for language, pattern in checks:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL):
            return language
    normalized_fallback = fallback.strip().lower()
    return LANGUAGE_ALIASES.get(normalized_fallback, normalized_fallback)


def largest_remainder_counts(total: int, quotas: dict[str, float]) -> dict[str, int]:
    if total < 0 or not quotas:
        raise ValueError("total must be nonnegative and quotas cannot be empty")
    quota_sum = sum(quotas.values())
    if quota_sum <= 0:
        raise ValueError("quota weights must sum to a positive number")
    raw = {key: total * weight / quota_sum for key, weight in quotas.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def redistribute_shortfall(
    requested: dict[str, int], available: dict[str, int]
) -> dict[str, int]:
    selected = {key: min(value, available.get(key, 0)) for key, value in requested.items()}
    missing = sum(requested.values()) - sum(selected.values())
    while missing:
        candidates = sorted(
            (key for key in available if selected.get(key, 0) < available[key]),
            key=lambda key: (-(available[key] - selected.get(key, 0)), key),
        )
        if not candidates:
            break
        for key in candidates:
            if missing == 0:
                break
            selected[key] = selected.get(key, 0) + 1
            missing -= 1
    return selected


def unified_diff(old: str, new: str, old_file: str, new_file: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{old_file}",
            tofile=f"b/{new_file}",
        )
    )


def changed_files_from_patch(patch: str) -> list[str]:
    values = re.findall(r"^diff --git a/(.*?) b/(.*?)$", patch, flags=re.MULTILINE)
    if values:
        return list(dict.fromkeys(new for _, new in values))
    return list(
        dict.fromkeys(
            value
            for value in re.findall(r"^\+\+\+ b/(.*?)$", patch, flags=re.MULTILINE)
            if value != "/dev/null"
        )
    )


def function_names_from_patch(patch: str) -> list[str]:
    names: list[str] = []
    for header in re.findall(r"^@@.*?@@\s*(.*)$", patch, flags=re.MULTILINE):
        match = re.search(
            r"(?:def|class|function|func|fn)\s+([A-Za-z_$][\w$]*)|"
            r"(?:[\w:<>,*&]+\s+)+([A-Za-z_$][\w$]*)\s*\(",
            header,
        )
        if match:
            names.append(next(value for value in match.groups() if value))
    return list(dict.fromkeys(names))


def language_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    for language, suffixes in {
        "python": {".py"},
        "javascript": {".js", ".jsx", ".mjs", ".cjs"},
        "typescript": {".ts", ".tsx"},
        "java": {".java"},
        "cpp": {".cc", ".cpp", ".cxx", ".h", ".hpp"},
        "go": {".go"},
        "rust": {".rs"},
        "shell": {".sh"},
        "sql": {".sql"},
    }.items():
        if suffix in suffixes:
            return language
    return "unknown"
