"""A local immutable model registry with atomic promotion and explicit rollback."""

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from fraud_platform.contracts import FEATURE_VERSION
from fraud_platform.train import train


def release_id(manifest: dict) -> str:
    identity = {k: v for k, v in manifest.items() if k != "trained_at_utc"}
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def checked_release(root: Path, version: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{24}", version):
        raise ValueError("Invalid release identifier")
    folder = root / "releases" / version
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["feature_version"] != FEATURE_VERSION or release_id(manifest) != version:
        raise ValueError("Release identity mismatch")
    actual = hashlib.sha256((folder / "model.joblib").read_bytes()).hexdigest()
    if actual != manifest["model_sha256"]:
        raise ValueError("Artifact checksum mismatch")
    return folder


def register(root: Path, rows: int, seed: int) -> str:
    releases = root / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=releases, prefix=".staging-"))
    try:
        manifest = train(staging, rows=rows, seed=seed)
        version = release_id(manifest)
        try:
            staging.rename(releases / version)
        except OSError:
            if not (releases / version).is_dir():
                raise
        checked_release(root, version)
        return version
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def promote(root: Path, version: str) -> None:
    checked_release(root, version)
    with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as file:
        file.write(version + "\n")
        file.flush()
        os.fsync(file.fileno())
        temp = Path(file.name)
    try:
        os.replace(temp, root / "active")
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/registry"))
    commands = parser.add_subparsers(dest="command", required=True)
    register_command = commands.add_parser("register")
    register_command.add_argument("--rows", type=int, default=4000)
    register_command.add_argument("--seed", type=int, default=42)
    commands.add_parser("promote").add_argument("version")
    commands.add_parser("active")
    args = parser.parse_args()
    if args.command == "register":
        print(register(args.root, args.rows, args.seed))
    elif args.command == "promote":
        promote(args.root, args.version)
        print(args.version)
    else:
        version = (args.root / "active").read_text().strip()
        checked_release(args.root, version)
        print(version)
