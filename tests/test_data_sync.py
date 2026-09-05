import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coder_sft.data_sync import _api, bundle_paths, validate_bundle, verify_bundle_metadata


class DataSyncTests(unittest.TestCase):
    def fixture(self, root: Path, complete: bool = True):
        data = root / "data"
        data.mkdir()
        stage1 = data / "stage1_v1.jsonl"
        repo = data / "repo_v1.jsonl"
        stage1.write_text('{"id":"stage1"}\n', encoding="utf-8")
        repo.write_text('{"id":"repo"}\n', encoding="utf-8")
        stage1.with_suffix(".manifest.json").write_text(
            json.dumps(
                {
                    "complete": complete,
                    "examples": 1,
                    "source_revisions": {"source": "stage1-revision"},
                }
            ),
            encoding="utf-8",
        )
        repo.with_suffix(".manifest.json").write_text(
            json.dumps(
                {
                    "complete": complete,
                    "examples": 1,
                    "source_revisions": {"source": "repo-revision"},
                }
            ),
            encoding="utf-8",
        )
        return {
            "training_data": {
                "stage1_path": str(stage1),
                "repo_path": str(repo),
            }
        }

    def test_bundle_validation_records_files_counts_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(Path(directory))
            root, paths = bundle_paths(config)
            bundle = validate_bundle(config)
            self.assertEqual(root, (Path(directory) / "data").resolve())
            self.assertEqual(bundle["files"], [path.name for path in paths])
            self.assertEqual(bundle["examples"], {"stage1": 1, "repo": 1})
            self.assertEqual(set(bundle["file_sha256"]), set(bundle["files"]))
            self.assertTrue(all(len(value) == 64 for value in bundle["file_sha256"].values()))

    def test_incomplete_or_missing_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(Path(directory), complete=False)
            with self.assertRaises(RuntimeError):
                validate_bundle(config)
            Path(config["training_data"]["repo_path"]).unlink()
            with self.assertRaises(FileNotFoundError):
                validate_bundle(config)

    def test_paths_must_share_directory_and_resolve_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            config["training_data"]["repo_path"] = str(root / "other" / "repo.jsonl")
            with self.assertRaises(ValueError):
                bundle_paths(config)
        with patch.dict(os.environ, {}, clear=True):
            config = {
                "training_data": {
                    "stage1_path": "${MISSING_SYNC_ROOT}/data/stage1.jsonl",
                    "repo_path": "${MISSING_SYNC_ROOT}/data/repo.jsonl",
                }
            }
            with self.assertRaises(ValueError):
                bundle_paths(config)

    def test_hub_repo_id_is_validated_before_token(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "namespace/dataset-name"):
                _api("missing-namespace")
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                _api("owner/dataset")

    def test_download_metadata_must_match_content_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(Path(directory))
            local = validate_bundle(config)
            remote = dict(local)
            remote["root"] = "/content/a-different-runtime/data"
            verify_bundle_metadata(remote, local)
            remote["file_sha256"] = dict(local["file_sha256"])
            remote["file_sha256"]["stage1_v1.jsonl"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "file_sha256"):
                verify_bundle_metadata(remote, local)


if __name__ == "__main__":
    unittest.main()
