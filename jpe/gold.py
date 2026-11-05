"""The hand-labeled gold set: sampling, the dev/test split, and freezing.

Files under data/gold/:
    split.json         which posts to label, in order, for each split
    labels.jsonl       one hand label per line (append-only)
    labels_second.jsonl a second annotator's labels, for agreement
    FROZEN             sha256 of split.json + labels.jsonl once labeling ends

The split is a queue, not a list. Each split gets more candidates than it
needs, in a fixed random order. Labeling walks the queue; a post that turns
out not to be a job post is skipped and the next candidate takes its place,
so the dev split ends at exactly 50 labeled posts and test at 250 without
re-sampling anything.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from jpe.corpus import DATA

GOLD_DIR = DATA / "gold"
SPLIT_FILE = GOLD_DIR / "split.json"
LABELS_FILE = GOLD_DIR / "labels.jsonl"
SECOND_FILE = GOLD_DIR / "labels_second.jsonl"
FROZEN_FILE = GOLD_DIR / "FROZEN"

TARGETS = {"dev": 50, "test": 250}
# Worked examples for the few-shot baseline (B3). Drawn from outside the gold
# set, so B3 never sees a test post in its prompt.
FEWSHOT_TARGET = 5
# Extra candidates per split, for posts skipped as not-a-job-post.
RESERVE = {"dev": 15, "test": 60}
FEWSHOT_CANDIDATES = 15


@dataclass
class Split:
    seed: int
    months: list[str]
    queues: dict[str, list[int]]  # "dev", "test", "fewshot"
    targets: dict[str, int]

    def to_json(self) -> dict:
        return {"seed": self.seed, "months": self.months, "targets": self.targets, "queues": self.queues}

    @classmethod
    def from_json(cls, d: dict) -> "Split":
        return cls(d["seed"], d["months"], {k: [int(i) for i in v] for k, v in d["queues"].items()}, d["targets"])

    def all_gold_candidates(self) -> set[int]:
        return set(self.queues["dev"]) | set(self.queues["test"])


def sample_split(
    posts_by_month: dict[str, list[int]],
    seed: int,
    targets: dict[str, int] = TARGETS,
    reserve: dict[str, int] = RESERVE,
    fewshot: int = FEWSHOT_CANDIDATES,
) -> Split:
    """Draw the gold queues, stratified by month.

    Each month contributes in proportion to its size, so a busy month is not
    under-represented and the set covers the whole two years. The draw is a
    pure function of `posts_by_month` and `seed`.
    """
    rng = random.Random(seed)
    months = sorted(posts_by_month)
    total = sum(len(v) for v in posts_by_month.values())
    need = sum(targets.values()) + sum(reserve.values())

    # Largest-remainder allocation of `need` across months.
    shares = {m: need * len(posts_by_month[m]) / total for m in months}
    alloc = {m: int(shares[m]) for m in months}
    for m in sorted(months, key=lambda m: shares[m] - alloc[m], reverse=True)[: need - sum(alloc.values())]:
        alloc[m] += 1

    drawn: list[int] = []
    leftover: list[int] = []
    for m in months:
        ids = sorted(posts_by_month[m])
        rng.shuffle(ids)
        drawn += ids[: alloc[m]]
        leftover += ids[alloc[m]:]
    rng.shuffle(drawn)

    queues: dict[str, list[int]] = {}
    start = 0
    for name in ("dev", "test"):
        n = targets[name] + reserve[name]
        queues[name] = drawn[start:start + n]
        start += n
    rng.shuffle(leftover)
    queues["fewshot"] = leftover[:fewshot]
    return Split(seed, months, queues, {**targets, "fewshot": FEWSHOT_TARGET})


def load_split(path: Path = SPLIT_FILE) -> Split:
    return Split.from_json(json.loads(path.read_text(encoding="utf-8")))


def save_split(split: Split, path: Path = SPLIT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_json(), indent=1) + "\n", encoding="utf-8", newline="\n")


def read_labels(path: Path = LABELS_FILE) -> list[dict]:
    """Every label line, oldest first. A later line for the same post
    replaces an earlier one (that is how a label is corrected)."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_label(entry: dict, path: Path = LABELS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def latest(entries: Iterable[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for e in entries:
        out[int(e["post_id"])] = e
    return out


def gold_ids(split: Split, entries: Iterable[dict], name: str) -> list[int]:
    """The posts in split `name`: walk its queue, skip posts marked not a job
    post, and stop at the target."""
    by_post = latest(entries)
    out = []
    for pid in split.queues[name]:
        if len(out) == split.targets.get(name, len(split.queues[name])):
            break
        e = by_post.get(pid)
        if e is None:
            continue
        if e.get("skip"):
            continue
        out.append(pid)
    return out


def next_to_label(split: Split, entries: Iterable[dict], names=("dev", "test")) -> list[tuple[str, int]]:
    """Posts still to label, interleaved across splits so that conventions
    that drift during labeling do not land on one split."""
    by_post = latest(entries)
    pending: dict[str, list[int]] = defaultdict(list)
    for name in names:
        done = len(gold_ids(split, by_post.values(), name))
        want = split.targets.get(name, len(split.queues[name])) - done
        for pid in split.queues[name]:
            if want <= 0:
                break
            if pid not in by_post:
                pending[name].append(pid)
                want -= 1
    out: list[tuple[str, int]] = []
    queues = [(n, pending[n]) for n in names]
    while any(q for _, q in queues):
        for name, q in queues:
            # Test is five times dev; take five test posts per dev post.
            for _ in range(max(1, split.targets.get(name, 1) // min(split.targets.get(n, 1) for n in names))):
                if q:
                    out.append((name, q.pop(0)))
    return out


def load_gold(name: str, split: Split | None = None, path: Path = LABELS_FILE, validate: bool = True) -> dict[int, dict]:
    """post id -> record for one split.

    Records go through jpe.validate.validate_record, so a malformed label
    fails here rather than scoring as a model error. Pass validate=False
    only before the validator is written.
    """
    split = split or load_split()
    entries = read_labels(path)
    by_post = latest(entries)
    ids = gold_ids(split, entries, name)
    out = {pid: by_post[pid]["record"] for pid in ids}
    if validate:
        from jpe.validate import validate_record

        out = {pid: validate_record(r) for pid, r in out.items()}
    return out


# --- Freezing ---------------------------------------------------------------

def gold_digest(split_path: Path = SPLIT_FILE, labels_path: Path = LABELS_FILE) -> str:
    h = hashlib.sha256()
    for p in (split_path, labels_path):
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n") if p.exists() else b"")
    return h.hexdigest()


def freeze(path: Path = FROZEN_FILE, **kw) -> str:
    digest = gold_digest(**kw)
    path.write_text(digest + "\n", newline="\n")
    return digest


def check_frozen(path: Path = FROZEN_FILE, **kw) -> bool | None:
    """None if never frozen, True if unchanged since, False if edited."""
    if not path.exists():
        return None
    return path.read_text().strip() == gold_digest(**kw)
