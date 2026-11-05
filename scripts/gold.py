"""Manage the gold set.

    python scripts/gold.py sample --until 2026-10     # draw data/gold/split.json (once)
    python scripts/gold.py status                     # how far labeling has got
    python scripts/gold.py freeze                     # write data/gold/FROZEN
    python scripts/gold.py check                      # fail if gold changed since freezing

Labeling itself is scripts/label.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jpe import corpus, gold  # noqa: E402

DEFAULT_SEED = 20261105


def cmd_sample(args) -> int:
    if gold.SPLIT_FILE.exists() and not args.force:
        print(f"{gold.SPLIT_FILE} exists. The sample is drawn once; pass --force to redraw.", file=sys.stderr)
        return 1
    if gold.read_labels() and args.force:
        print("Labels exist for the current split; redrawing would orphan them.", file=sys.stderr)
        return 1
    months = [m for m in sorted(corpus.load_threads()) if (not args.since or m >= args.since) and m <= args.until]
    by_month = {m: corpus.read_ids(m) for m in months}
    split = gold.sample_split(by_month, seed=args.seed)
    gold.save_split(split)
    n = sum(len(v) for v in by_month.values())
    print(f"Drew from {n} posts over {len(months)} months ({months[0]} to {months[-1]}), seed {args.seed}.")
    for name, q in split.queues.items():
        print(f"  {name}: {len(q)} candidates for {split.targets[name]} labels")
    return 0


def cmd_status(args) -> int:
    split = gold.load_split()
    entries = gold.read_labels()
    skipped = sum(1 for e in gold.latest(entries).values() if e.get("skip"))
    for name in ("dev", "test", "fewshot"):
        done = len(gold.gold_ids(split, entries, name))
        print(f"{name:8s} {done:4d} / {split.targets[name]}")
    print(f"skipped as not a job post: {skipped}")
    second = gold.read_labels(gold.SECOND_FILE)
    if second:
        print(f"second annotator: {len(gold.latest(second))} posts")
    frozen = gold.check_frozen()
    print({None: "not frozen", True: "frozen, unchanged", False: "FROZEN BUT CHANGED SINCE"}[frozen])
    return 0


def cmd_freeze(args) -> int:
    split = gold.load_split()
    entries = gold.read_labels()
    short = [n for n in ("dev", "test") if len(gold.gold_ids(split, entries, n)) < split.targets[n]]
    if short and not args.force:
        print(f"Not fully labeled yet: {', '.join(short)}. Pass --force to freeze anyway.", file=sys.stderr)
        return 1
    print(f"Frozen: {gold.freeze()}")
    return 0


def cmd_check(args) -> int:
    state = gold.check_frozen()
    if state is False:
        print("data/gold changed after it was frozen.", file=sys.stderr)
        return 1
    print("not frozen yet" if state is None else "gold set unchanged since freezing")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--since")
    s.add_argument("--until", required=True, help="last complete month, YYYY-MM")
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--force", action="store_true")
    sub.add_parser("status")
    f = sub.add_parser("freeze")
    f.add_argument("--force", action="store_true")
    sub.add_parser("check")
    args = ap.parse_args(argv)
    return {"sample": cmd_sample, "status": cmd_status, "freeze": cmd_freeze, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
