import tempfile
import unittest
from pathlib import Path

from coder_sft.dedup import MinHashDeduper, minhash_signature, signature_similarity
from coder_sft.schema import NormalizedExample, read_jsonl, write_jsonl
from coder_sft.utils import deterministic_split
from coder_sft.utils import finalize_chat_render


class SchemaAndDedupTests(unittest.TestCase):
    def example(self):
        return NormalizedExample(
            id="example-1",
            source="fixture",
            source_id="1",
            repo_id=None,
            language="python",
            task_type="code_generation",
            quality_tier="A",
            split="train",
            messages=[
                {"role": "user", "content": "Add two numbers"},
                {"role": "assistant", "content": "def add(a,b): return a+b"},
            ],
            token_count=12,
        )

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            write_jsonl([self.example()], path)
            loaded = list(read_jsonl(path))
        self.assertEqual(loaded[0].content_sha256, self.example().content_sha256)

    def test_invalid_last_role(self):
        value = self.example().to_dict()
        value["messages"][-1]["role"] = "user"
        value["content_sha256"] = ""
        with self.assertRaises(ValueError):
            NormalizedExample.from_dict(value)

    def test_exact_and_normalized_duplicate(self):
        deduper = MinHashDeduper()
        self.assertTrue(deduper.add_if_unique("Implement add(a, b) correctly."))
        self.assertFalse(deduper.add_if_unique(" implement ADD a b correctly "))

    def test_minhash_similarity(self):
        left = minhash_signature("one two three four five six seven eight nine")
        right = minhash_signature("one two three four five six seven eight ten")
        self.assertGreater(signature_similarity(left, right), 0.2)

    def test_split_is_deterministic(self):
        first = [deterministic_split(f"repo-{index}", 3407) for index in range(100)]
        second = [deterministic_split(f"repo-{index}", 3407) for index in range(100)]
        self.assertEqual(first, second)

    def test_chat_render_requires_eos(self):
        self.assertEqual(finalize_chat_render("answer<eos>\n", "<eos>"), "answer<eos>")
        with self.assertRaises(ValueError):
            finalize_chat_render("answer", "<eos>")


if __name__ == "__main__":
    unittest.main()
