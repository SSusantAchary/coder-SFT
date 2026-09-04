"""Normalized JSONL schema shared by all source datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


VALID_ROLES = {"system", "user", "assistant", "tool"}
VALID_SPLITS = {"train", "validation", "test"}


def canonical_messages(messages: list[dict[str, str]]) -> str:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(canonical_messages(messages).encode("utf-8")).hexdigest()


@dataclass(slots=True)
class NormalizedExample:
    id: str
    source: str
    source_id: str
    repo_id: str | None
    language: str
    task_type: str
    quality_tier: str
    split: str
    messages: list[dict[str, str]]
    token_count: int
    content_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_sha256:
            self.content_sha256 = content_hash(self.messages)
        self.validate()

    def validate(self) -> None:
        if not self.id or not self.source or not self.source_id:
            raise ValueError("id, source, and source_id are required")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Unsupported split: {self.split}")
        if self.token_count < 0:
            raise ValueError("token_count cannot be negative")
        if not self.messages or self.messages[-1].get("role") != "assistant":
            raise ValueError("messages must end with an assistant response")
        for message in self.messages:
            if message.get("role") not in VALID_ROLES:
                raise ValueError(f"Invalid message role: {message.get('role')}")
            if not isinstance(message.get("content"), str):
                raise ValueError("Every message needs string content")
        if self.content_sha256 != content_hash(self.messages):
            raise ValueError("content_sha256 does not match messages")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NormalizedExample":
        return cls(**value)


def write_jsonl(examples: Iterable[NormalizedExample], path: str | Path) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterable[NormalizedExample]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield NormalizedExample.from_dict(json.loads(line))

