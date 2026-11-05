"""Fetch "Who is hiring?" posts from the Hacker News API.

    # First build: two years of threads.
    python scripts/fetch_posts.py --since 2024-11 --until 2026-10

    # Pick up posts added to the newest threads since the last run.
    python scripts/fetch_posts.py --since 2026-10

    # Rebuild data/raw/ from the committed IDs (a fresh clone, or CI).
    python scripts/fetch_posts.py --from-ids
    python scripts/fetch_posts.py --from-ids --only data/gold/split.json:dev

Writes data/threads.json and data/post_ids/ (committed) and data/raw/posts/
(not committed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jpe import corpus  # noqa: E402
from jpe.hn import HNClient, fetch_by_ids, fetch_posts, list_hiring_threads  # noqa: E402


def _only_ids(spec: str) -> set[int]:
    """`path.json:key` -> the ids listed under that key (e.g. a gold split)."""
    path, _, key = spec.partition(":")
    data = json.loads(Path(path).read_text())
    return {int(i) for i in (data[key] if key else data)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", help="first month, YYYY-MM")
    ap.add_argument("--until", help="last month, YYYY-MM (default: newest)")
    ap.add_argument("--from-ids", action="store_true", help="rehydrate from data/post_ids/")
    ap.add_argument("--only", help="with --from-ids: restrict to ids in FILE.json:key")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    client = HNClient()

    if args.from_ids:
        threads = corpus.load_threads()
        by_id = {t.id: t for t in threads.values()}
        wanted = _only_ids(args.only) if args.only else None
        for month in sorted(threads):
            ids = [i for i in corpus.read_ids(month) if wanted is None or i in wanted]
            have = {p.id for p in corpus.read_posts(month)}
            todo = [i for i in ids if i not in have]
            if not todo:
                continue
            posts = fetch_by_ids(client, todo, by_id, workers=args.workers)
            corpus.write_posts(month, posts)
            gone = len(todo) - len(posts)
            print(f"{month}: rehydrated {len(posts)}" + (f", {gone} deleted since" if gone else ""))
        return 0

    if not args.since:
        ap.error("--since is required unless --from-ids is given")

    threads = corpus.load_threads()
    found = list_hiring_threads(client, args.since, args.until)
    if not found:
        print("No hiring threads in that range.", file=sys.stderr)
        return 1

    for thread in found:
        threads[thread.month] = thread
        known = set(corpus.read_ids(thread.month))
        new = fetch_posts(client, thread, skip_ids=known, workers=args.workers)
        corpus.write_posts(thread.month, new)
        corpus.write_ids(thread.month, known | {p.id for p in new})
        print(f"{thread.month} (thread {thread.id}): {len(new)} new, {len(known) + len(new)} total")

    corpus.save_threads(threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
