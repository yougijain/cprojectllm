"""What produced a result: recorded in every predictions and results file.

A number in the README is only worth something if it can be traced to the
model, adapter, prompt and code that made it.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from jpe.fields import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parent.parent

# The files that define a LoRA adapter. Checkpoint optimizer state and the
# README peft writes do not change the model, so they are not hashed.
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def adapter_sha256(adapter_dir: Path) -> str:
    """One hash for an adapter directory, over the files that define it."""
    h = hashlib.sha256()
    found = False
    for name in _ADAPTER_FILES:
        p = Path(adapter_dir) / name
        if p.exists():
            found = True
            h.update(name.encode())
            h.update(file_sha256(p).encode())
    if not found:
        raise FileNotFoundError(f"no adapter files in {adapter_dir}")
    return h.hexdigest()


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def run_meta(name: str, identity: dict, prompt_version: str | None, **extra) -> dict:
    return {
        "run": name,
        "model": identity,
        "prompt_version": prompt_version,
        "schema_version": SCHEMA_VERSION,
        "git": git_state(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
