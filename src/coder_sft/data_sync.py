"""Upload or download a validated prepared-data bundle through the HF Dataset Hub."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .config import load_config, safe_output_path
from .utils import write_json

REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _expand(value: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in data path: {value}")
    return Path(expanded).resolve()


def bundle_paths(config: dict[str, Any]) -> tuple[Path, list[Path]]:
    data = config["training_data"]
    stage1 = _expand(str(data["stage1_path"]))
    repo = _expand(str(data["repo_path"]))
    if stage1.parent != repo.parent:
        raise ValueError("Stage-1 and repository datasets must share one data directory")
    root = safe_output_path(stage1.parent)
    files = [
        stage1,
        stage1.with_suffix(".manifest.json"),
        repo,
        repo.with_suffix(".manifest.json"),
    ]
    return root, files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundle(config: dict[str, Any]) -> dict[str, Any]:
    root, files = bundle_paths(config)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared-data bundle is incomplete; missing: {missing}")
    manifests = []
    for manifest_path in files[1::2]:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            raise RuntimeError(f"Prepared-data manifest is incomplete: {manifest_path}")
        manifests.append(manifest)
    return {
        "version": 1,
        "root": str(root),
        "files": [path.name for path in files],
        "file_sha256": {path.name: _sha256(path) for path in files},
        "examples": {
            "stage1": int(manifests[0].get("examples", 0)),
            "repo": int(manifests[1].get("examples", 0)),
        },
        "source_revisions": {
            "stage1": manifests[0].get("source_revisions", {}),
            "repo": manifests[1].get("source_revisions", {}),
        },
    }


def _api(repo_id: str):
    if not REPO_ID_PATTERN.fullmatch(repo_id):
        raise ValueError("--repo-id must use the namespace/dataset-name format")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for prepared-data synchronization")
    from huggingface_hub import HfApi

    return HfApi(token=token), token


def verify_bundle_metadata(remote: dict[str, Any], local: dict[str, Any]) -> None:
    checked_fields = ("version", "files", "file_sha256", "examples", "source_revisions")
    mismatches = [field for field in checked_fields if remote.get(field) != local[field]]
    if mismatches:
        raise RuntimeError(
            "Downloaded data does not match dataset_bundle.json; "
            f"mismatched fields: {mismatches}"
        )


def upload(config: dict[str, Any], repo_id: str, private: bool) -> dict[str, Any]:
    root, _ = bundle_paths(config)
    bundle = validate_bundle(config)
    write_json(bundle, root / "dataset_bundle.json")
    readme = root / "README.md"
    readme.write_text(
        "---\n"
        "license: other\n"
        "task_categories:\n- text-generation\n"
        "tags:\n- code\n- sft\n- qwen3.5\n"
        "---\n\n"
        "# Qwen3.5-2B-Coder prepared SFT data\n\n"
        "Private prepared-data handoff between the CPU and GPU Colab phases. "
        "See `dataset_bundle.json` and the adjacent manifests for provenance.\n",
        encoding="utf-8",
    )
    api, token = _api(repo_id)
    repo = api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(root),
        token=token,
        commit_message="Upload validated Qwen3.5 coder training data",
        allow_patterns=bundle["files"] + ["dataset_bundle.json", "README.md"],
    )
    result = {
        "action": "upload",
        "repo_id": repo_id,
        "repo_url": str(repo),
        "private": private,
        "commit": str(commit),
        "bundle": bundle,
    }
    write_json(result, root / "upload_summary.json")
    return result


def download(config: dict[str, Any], repo_id: str) -> dict[str, Any]:
    root, files = bundle_paths(config)
    root.mkdir(parents=True, exist_ok=True)
    _, token = _api(repo_id)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(root),
        token=token,
        allow_patterns=[path.name for path in files] + ["dataset_bundle.json", "README.md"],
    )
    bundle = validate_bundle(config)
    remote_bundle_path = root / "dataset_bundle.json"
    if not remote_bundle_path.is_file():
        raise FileNotFoundError("Downloaded repository has no dataset_bundle.json")
    remote_bundle = json.loads(remote_bundle_path.read_text(encoding="utf-8"))
    verify_bundle_metadata(remote_bundle, bundle)
    result = {"action": "download", "repo_id": repo_id, "bundle": bundle}
    write_json(result, root / "download_summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "download"))
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public dataset repository (private is the default)",
    )
    args = parser.parse_args()
    config = load_config(*args.config)
    result = (
        upload(config, args.repo_id, private=not args.public)
        if args.action == "upload"
        else download(config, args.repo_id)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
