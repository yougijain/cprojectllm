"""Silver labels: the teacher's records for the training pool.

data/silver/pool.json   the eligible posts, in a fixed random order
data/silver/labels.jsonl one teacher record per post (fields only, no text)
data/silver/meta.json    teacher, prompt version, counts

The pool's order is drawn once. Every training-set size in the learning
curve is a prefix of it, so the 500-example set contains the 250-example
set, and a bigger set only adds data rather than swapping it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from jpe.corpus import DATA

SILVER_DIR = DATA / "silver"
POOL_FILE = SILVER_DIR / "pool.json"
LABELS_FILE = SILVER_DIR / "labels.jsonl"
META_FILE = SILVER_DIR / "meta.json"
SPOT_FILE = SILVER_DIR / "spot_check.jsonl"

POOL_SEED = 20261106


def ordered_pool(eligible: list[int], seed: int = POOL_SEED) -> list[int]:
    order = sorted(eligible)
    random.Random(seed).shuffle(order)
    return order


def save_pool(order: list[int], dropped: dict, n_gold: int, path: Path = POOL_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "seed": POOL_SEED,
        "excluded_gold_candidates": n_gold,
        "excluded_near_duplicates": {str(k): v for k, v in sorted(dropped.items())},
        "order": order,
    }
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")


def load_pool(path: Path = POOL_FILE) -> list[int]:
    return [int(i) for i in json.loads(path.read_text(encoding="utf-8"))["order"]]


def read_labels(path: Path = LABELS_FILE) -> dict[int, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {int(r["post_id"]): r for r in rows}


def write_labels(rows: list[dict], path: Path = LABELS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in sorted(rows, key=lambda r: r["post_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def training_examples(n: int | None = None) -> list[tuple[int, dict]]:
    """The first `n` usable silver labels in pool order. Posts the teacher
    failed on are skipped, not replaced by a guess."""
    labels = read_labels()
    out = []
    for pid in load_pool():
        row = labels.get(pid)
        if row is None or row.get("record") is None:
            continue
        out.append((pid, row["record"]))
        if n is not None and len(out) == n:
            break
    return out
