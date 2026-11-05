"""Where posts live on disk.

Committed (the reproducible part):
    data/threads.json            month -> thread id, title, time
    data/post_ids/YYYY-MM.txt    one post id per line

Local only (rebuilt from the IDs by scripts/fetch_posts.py --from-ids):
    data/raw/posts/YYYY-MM.jsonl the posts themselves, as the API returned them

Re-hosting a dump of other people's posts is what the ID-only split avoids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from jpe.hn import Post, Thread

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
THREADS_FILE = DATA / "threads.json"
IDS_DIR = DATA / "post_ids"
RAW_DIR = DATA / "raw" / "posts"


def load_threads(path: Path = THREADS_FILE) -> dict[str, Thread]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {m: Thread(id=v["id"], month=m, title=v["title"], time=v["time"]) for m, v in data.items()}


def save_threads(threads: dict[str, Thread], path: Path = THREADS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {m: {"id": t.id, "title": t.title, "time": t.time} for m, t in sorted(threads.items())}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_ids(month: str, ids_dir: Path = IDS_DIR) -> list[int]:
    path = ids_dir / f"{month}.txt"
    if not path.exists():
        return []
    return [int(line) for line in path.read_text().split() if line.strip()]


def write_ids(month: str, ids: Iterable[int], ids_dir: Path = IDS_DIR) -> None:
    ids_dir.mkdir(parents=True, exist_ok=True)
    (ids_dir / f"{month}.txt").write_text("".join(f"{i}\n" for i in sorted(set(ids))), newline="\n")


def read_posts(month: str, raw_dir: Path = RAW_DIR) -> list[Post]:
    path = raw_dir / f"{month}.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [Post.from_json(json.loads(line)) for line in f if line.strip()]


def write_posts(month: str, posts: Iterable[Post], raw_dir: Path = RAW_DIR) -> None:
    """Merge `posts` into the month's file, keyed by id."""
    merged = {p.id: p for p in read_posts(month, raw_dir)}
    merged.update({p.id: p for p in posts})
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / f"{month}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for pid in sorted(merged):
            f.write(json.dumps(merged[pid].to_json(), ensure_ascii=False) + "\n")


def load_corpus(months: Iterable[str] | None = None, raw_dir: Path = RAW_DIR) -> dict[int, Post]:
    """Every post on disk, keyed by id."""
    if months is None:
        months = sorted(p.stem for p in raw_dir.glob("*.jsonl"))
    corpus: dict[int, Post] = {}
    for month in months:
        for post in read_posts(month, raw_dir):
            corpus[post.id] = post
    return corpus
