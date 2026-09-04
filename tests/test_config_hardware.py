import json
import os
import tempfile
import unittest
from pathlib import Path

from coder_sft.config import (
    ConfigError,
    deep_merge,
    expand_environment,
    safe_output_path,
    validate_training_config,
)
from coder_sft.hardware import HardwareInfo, select_profile
from coder_sft.training import (
    validate_data_manifest,
    validate_stage1_adapter,
    validate_stage1_gate,
)


class ConfigHardwareTests(unittest.TestCase):
    def valid_config(self):
        return {
            "model": {"load_in_4bit": False, "load_in_16bit": True},
            "training": {"bf16": True, "max_length": 8192},
            "lora": {"rank": 32, "alpha": 32},
        }

    def test_rejects_qlora(self):
        config = self.valid_config()
        config["model"]["load_in_4bit"] = True
        with self.assertRaises(ConfigError):
            validate_training_config(config)

    def test_rejects_missing_projection_target(self):
        config = self.valid_config()
        config["lora"]["target_modules"] = ["q_proj", "in_proj_qkv", "gate_proj"]
        with self.assertRaises(ConfigError):
            validate_training_config(config)

    def test_profile_selection(self):
        profiles = {
            "40gb": {"min_vram_gb": 35},
            "80gb": {"min_vram_gb": 70},
        }
        info = HardwareInfo("A100", 79.0, True, "12.8", (8, 0))
        self.assertEqual(select_profile(profiles, info)[0], "80gb")

    def test_rejects_non_bf16(self):
        info = HardwareInfo("T4", 16.0, False, "12.8", (7, 5))
        with self.assertRaises(ConfigError):
            select_profile({"any": {"min_vram_gb": 1}}, info)

    def test_deep_merge(self):
        self.assertEqual(
            deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}),
            {"a": {"b": 3, "c": 2}},
        )

    def test_environment_expansion(self):
        os.environ["CODER_SFT_TEST_ROOT"] = "/tmp/coder-sft-test"
        self.assertEqual(
            expand_environment({"path": "${CODER_SFT_TEST_ROOT}/output"})["path"],
            "/tmp/coder-sft-test/output",
        )

    def test_rejects_unsafe_output_path(self):
        with self.assertRaises(ConfigError):
            safe_output_path("/")
        with self.assertRaises(ConfigError):
            safe_output_path("${MISSING_CODER_SFT_ROOT}/output")

    def test_stage2_requires_passing_stage1_lineage(self):
        with self.assertRaises(ValueError):
            validate_stage1_gate({"require_stage1_gate": True})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps({"stage": "stage1", "passed": False}))
            with self.assertRaises(RuntimeError):
                validate_stage1_gate({"stage1_gate_report": str(path)})
            path.write_text(json.dumps({"stage": "stage1", "passed": True}))
            self.assertEqual(
                validate_stage1_gate({"stage1_gate_report": str(path)}), str(path)
            )

    def test_stage2_rejects_wrong_adapter_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "training_lineage.json").write_text(
                json.dumps({"stage": "stage2"}), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                validate_stage1_adapter(str(path))
            (path / "training_lineage.json").write_text(
                json.dumps({"stage": "stage1"}), encoding="utf-8"
            )
            self.assertEqual(validate_stage1_adapter(str(path))["stage"], "stage1")

    def test_incomplete_data_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            path.with_suffix(".manifest.json").write_text(
                json.dumps({"complete": False}), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                validate_data_manifest(str(path))


if __name__ == "__main__":
    unittest.main()
