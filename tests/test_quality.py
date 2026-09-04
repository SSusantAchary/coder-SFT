import unittest

from coder_sft.quality import (
    broad_quality_tier,
    changed_files_from_patch,
    function_names_from_patch,
    infer_language,
    largest_remainder_counts,
    parse_judge_scores,
    redistribute_shortfall,
    unified_diff,
)


class QualityTests(unittest.TestCase):
    def test_quality_tiers(self):
        row = {
            "average_test_score": "1",
            "llm_judgement": '{"correctness":{"score":5},"style":{"score":5}}',
        }
        self.assertEqual(broad_quality_tier(row), "A")
        row["llm_judgement"] = '{"correctness":{"score":4},"style":{"score":4}}'
        self.assertEqual(broad_quality_tier(row), "B")
        row["average_test_score"] = "90%"
        self.assertEqual(broad_quality_tier(row), "C")
        row["average_test_score"] = "0.8"
        self.assertEqual(broad_quality_tier(row), "D")
        row["average_test_score"] = "0.7"
        self.assertIsNone(broad_quality_tier(row))

    def test_nested_judge_scores(self):
        self.assertEqual(
            parse_judge_scores({"a": {"score": 5}, "b": [{"score": 4}] }),
            [5.0, 4.0],
        )

    def test_quota_rounding_and_redistribution(self):
        counts = largest_remainder_counts(10, {"python": 0.35, "cpp": 0.10, "rust": 0.05})
        self.assertEqual(sum(counts.values()), 10)
        redistributed = redistribute_shortfall(
            {"python": 5, "cpp": 5}, {"python": 10, "cpp": 2}
        )
        self.assertEqual(redistributed, {"python": 8, "cpp": 2})

    def test_language_inference(self):
        self.assertEqual(infer_language("```py\ndef f(): pass\n```"), "python")
        self.assertEqual(infer_language("#include <vector>\nstd::vector<int> x;"), "cpp")
        self.assertEqual(infer_language("plain text", "c++"), "cpp")

    def test_diff_and_targets(self):
        patch = unified_diff(
            "def add(a, b):\n    return a - b\n",
            "def add(a, b):\n    return a + b\n",
            "calc.py",
            "calc.py",
        )
        self.assertIn("--- a/calc.py", patch)
        git_patch = "diff --git a/calc.py b/calc.py\n" + patch
        self.assertEqual(changed_files_from_patch(git_patch), ["calc.py"])
        header_patch = "@@ -1,2 +1,2 @@ def add(a, b):\n-return a-b\n+return a+b\n"
        self.assertEqual(function_names_from_patch(header_patch), ["add"])


if __name__ == "__main__":
    unittest.main()
