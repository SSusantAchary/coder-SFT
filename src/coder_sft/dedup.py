"""Deterministic exact and MinHash-LSH near-duplicate filtering."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def shingles(text: str, width: int = 5) -> set[str]:
    tokens = normalize_text(text).split()
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def minhash_signature(text: str, permutations: int = 64) -> tuple[int, ...]:
    values = shingles(text)
    if not values:
        return tuple(0 for _ in range(permutations))
    prime = (1 << 61) - 1
    hashed = [
        int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")
        % prime
        for value in values
    ]
    signature = [prime] * permutations
    for value in hashed:
        second = ((value >> 29) | 1) % prime
        for seed in range(permutations):
            candidate = (value + (seed + 1) * second + seed * seed) % prime
            if candidate < signature[seed]:
                signature[seed] = candidate
    return tuple(signature)


def signature_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("MinHash signatures must have the same length")
    return sum(a == b for a, b in zip(left, right)) / len(left)


@dataclass
class MinHashDeduper:
    threshold: float = 0.85
    bands: int = 8
    rows_per_band: int = 8
    _exact: set[str] = field(default_factory=set)
    _signatures: list[tuple[int, ...]] = field(default_factory=list)
    _buckets: dict[tuple[int, tuple[int, ...]], list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def __post_init__(self) -> None:
        if self.bands * self.rows_per_band != 64:
            raise ValueError("This implementation requires 64 total MinHash rows")

    def is_duplicate(self, text: str) -> bool:
        normalized = normalize_text(text)
        exact = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if exact in self._exact:
            return True
        signature = minhash_signature(normalized)
        return self._signature_is_duplicate(signature)

    def _signature_is_duplicate(self, signature: tuple[int, ...]) -> bool:
        candidates: set[int] = set()
        for band in range(self.bands):
            start = band * self.rows_per_band
            key = (band, signature[start : start + self.rows_per_band])
            candidates.update(self._buckets.get(key, []))
        if any(
            signature_similarity(signature, self._signatures[index]) >= self.threshold
            for index in candidates
        ):
            return True
        return False

    def add_if_unique(self, text: str) -> bool:
        normalized = normalize_text(text)
        exact = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if exact in self._exact:
            return False
        signature = minhash_signature(normalized)
        if self._signature_is_duplicate(signature):
            return False
        index = len(self._signatures)
        self._signatures.append(signature)
        self._exact.add(exact)
        for band in range(self.bands):
            start = band * self.rows_per_band
            key = (band, signature[start : start + self.rows_per_band])
            self._buckets[key].append(index)
        return True

    def seed(self, texts: Iterable[str]) -> None:
        for text in texts:
            self.add_if_unique(text)
