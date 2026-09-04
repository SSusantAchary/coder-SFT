import unittest

from coder_sft.repo_context import (
    SWE_SMITH_SUFFIX,
    arrange_file_blocks,
    build_repo_example,
    choose_task_type,
    dependency_candidates,
    parse_swesmith_revision,
    test_paths as extract_test_paths,
)


class RepoContextTests(unittest.TestCase):
    def test_language_dataset_suffixes(self):
        self.assertEqual(SWE_SMITH_SUFFIX["rust"], "rs")
        self.assertEqual(SWE_SMITH_SUFFIX["typescript"], "ts")

    def test_parse_revision(self):
        revision = parse_swesmith_revision(
            {"repo": "swesmith/oauthlib__oauthlib.1fd52536"}
        )
        self.assertEqual(revision.github_repo, "oauthlib/oauthlib")
        self.assertEqual(revision.commit, "1fd52536")

    def test_rejects_unsafe_revision(self):
        with self.assertRaises(ValueError):
            parse_swesmith_revision({"repo": "swesmith/a__../bad.1fd52536"})

    def test_position_buckets_move_relevant_later(self):
        relevant = [("target.py", "target")]
        distractors = [(f"d{i}.py", str(i)) for i in range(10)]
        positions = [
            arrange_file_blocks(relevant, distractors, bucket).index(relevant[0])
            for bucket in range(5)
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertGreater(positions[-1], positions[0])

    def test_dependency_and_tests(self):
        relevant = {"src/main.py": "from helpers import parse\n"}
        files = ["src/main.py", "src/helpers.py", "tests/test_main.py"]
        self.assertEqual(dependency_candidates(relevant, files), ["src/helpers.py"])
        row = {"FAIL_TO_PASS": ["tests/test_main.py::test_x"], "PASS_TO_PASS": []}
        self.assertEqual(extract_test_paths(row), ["tests/test_main.py"])

    def test_task_type_falls_back_when_gold_missing(self):
        for index in range(1000):
            value = choose_task_type(str(index), [], [], 3407)
            self.assertNotIn(value, {"function_localization", "test_localization"})

    def test_repository_context_preserves_token_budget(self):
        class Tokenizer:
            eos_token = "<eos>"

            def encode(self, text, add_special_tokens=False):
                return text.split()

            def decode(self, ids):
                return " ".join(ids)

            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(message["content"] for message in messages) + self.eos_token

        class Store:
            files = {
                "src/calc.py": "def add(a, b):\n    return a - b\n",
                "src/helpers.py": "def normalize(x):\n    return x\n",
                "tests/test_calc.py": "def test_add():\n    assert add(1, 2) == 3\n",
            }

            def list_files(self, revision):
                return list(self.files)

            def read_file(self, revision, path, max_bytes):
                return self.files.get(path)

        row = {
            "instance_id": "owner__repo.abcdef1.task",
            "repo": "swesmith/owner__repo.abcdef1",
            "patch": (
                "diff --git a/src/calc.py b/src/calc.py\n"
                "--- a/src/calc.py\n+++ b/src/calc.py\n"
                "@@ -1,2 +1,2 @@ def add(a, b):\n-    return a - b\n+    return a + b\n"
            ),
            "FAIL_TO_PASS": ["tests/test_calc.py::test_add"],
            "PASS_TO_PASS": [],
            "problem_statement": "Addition returns the wrong result.",
        }
        example = build_repo_example(row, "python", Tokenizer(), Store(), 3407, 100_000)
        self.assertLessEqual(example.token_count, example.metadata["target_context_tokens"])
        self.assertIn("### FILE: src/calc.py", example.messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
