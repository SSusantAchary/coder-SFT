import tempfile
import unittest
from pathlib import Path

from coder_sft.evaluation import compare_reports
from coder_sft.modeling import (
    audit_native_context,
    audit_projection_modules,
    audit_trainable_parameters,
)
from coder_sft.dedup import MinHashDeduper
from coder_sft.prepare import _accept_examples, build_commit_example, build_eer6_example
from coder_sft.schema import read_jsonl, write_jsonl
from coder_sft.training import validate_assistant_mask, validate_supervised_text
from coder_sft.config import ConfigError


class FakeTokenizer:
    bos_token = "<bos>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"<{m['role']}>\n{m['content']}" for m in messages) + self.eos_token

    def encode(self, text, add_special_tokens=False):
        return text.split()


class PipelineTests(unittest.TestCase):
    def test_cpu_fixture_pipeline(self):
        tokenizer = FakeTokenizer()
        eer = build_eer6_example(
            {"id": "e1", "input": "Add", "output": "```py\ndef add(a,b): return a+b\n```"},
            "eer6_refined",
            "refined",
            tokenizer,
            3407,
        )
        commit = build_commit_example(
            {
                "commit": "abc",
                "old_file": "calc.py",
                "new_file": "calc.py",
                "old_contents": "return a-b\n",
                "new_contents": "return a+b\n",
                "subject": "Fix addition",
                "lang": "python",
                "repos": "owner/repo",
                "license": "mit",
            },
            tokenizer,
            3407,
        )
        self.assertIsNotNone(commit)
        self.assertEqual(commit.metadata["source_language"], "python")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            write_jsonl([eer, commit], path)
            rows = list(read_jsonl(path))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row.task_type for row in rows}, {"code_generation", "code_edit"})

    def test_stage1_gate(self):
        baseline = {
            "benchmarks": {
                "humaneval_plus_pass_at_1": 0.20,
                "mbpp_plus_pass_at_1": 0.30,
            }
        }
        candidate = {
            "benchmarks": {
                "humaneval_plus_pass_at_1": 0.22,
                "mbpp_plus_pass_at_1": 0.31,
            },
            "smoke": {
                "finite_validation_loss": True,
                "repetition_regression": False,
                "termination_regression": False,
            },
        }
        self.assertTrue(compare_reports(baseline, candidate, "stage1")["passed"])

    def test_projection_family_audit(self):
        class Model:
            def named_modules(self):
                names = [
                    "model.layers.0.self_attn.q_proj",
                    "model.layers.0.self_attn.in_proj_qkv",
                    "model.layers.0.mlp.gate_proj",
                ]
                return [(name, object()) for name in names]

        targets = ["q_proj", "in_proj_qkv", "gate_proj"]
        audit = audit_projection_modules(Model(), targets)
        self.assertTrue(all(audit.values()))
        with self.assertRaises(ConfigError):
            audit_projection_modules(Model(), ["q_proj", "gate_proj"])

    def test_native_context_audit(self):
        class Config:
            max_position_embeddings = 262_144

        class Model:
            config = Config()

        self.assertEqual(audit_native_context(Model()), 262_144)
        Model.config.max_position_embeddings = 32_768
        with self.assertRaises(ConfigError):
            audit_native_context(Model())

    def test_trainable_vision_parameters_are_rejected(self):
        class Parameter:
            requires_grad = True

            def numel(self):
                return 4

        class Model:
            def named_parameters(self):
                return [("model.visual.encoder.lora_A", Parameter())]

        with self.assertRaises(ConfigError):
            audit_trainable_parameters(Model())

    def test_assistant_supervision_marker_audit(self):
        validate_supervised_text("def add(a, b): return a + b<|im_end|>")
        with self.assertRaises(RuntimeError):
            validate_supervised_text("<|im_start|>user\nshould be masked")
        input_ids = [10, 11, 12, 20, 21, 22, 30, 31, 40, 41]
        labels = [-100, -100, -100, -100, -100, -100, -100, -100, 40, 41]
        self.assertEqual(
            validate_assistant_mask(input_ids, labels, [10, 11], [20, 21], [30, 31]),
            1,
        )
        labels[5] = 22
        with self.assertRaises(RuntimeError):
            validate_assistant_mask(input_ids, labels, [10, 11], [20, 21], [30, 31])

    def test_benchmark_prompt_exclusion(self):
        tokenizer = FakeTokenizer()
        candidate = build_eer6_example(
            {"id": "benchmark-copy", "input": "Implement exact benchmark prompt", "output": "pass"},
            "eer6_refined",
            "refined",
            tokenizer,
            3407,
        )
        benchmark = MinHashDeduper()
        benchmark.seed(["Implement exact benchmark prompt"])
        counters = Counter()
        accepted = _accept_examples(
            [candidate], 1, 8192, MinHashDeduper(), benchmark, counters
        )
        self.assertEqual(accepted, [])
        self.assertEqual(counters["benchmark_overlap"], 1)


if __name__ == "__main__":
    unittest.main()
from collections import Counter
