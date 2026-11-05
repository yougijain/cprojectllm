"""Label posts by hand, in the terminal.

    python scripts/label.py                       # the gold set, dev and test interleaved
    python scripts/label.py --set fewshot         # the five few-shot examples for B3
    python scripts/label.py --second --annotator sam   # 40 posts for inter-annotator agreement
    python scripts/label.py --redo 45812345       # relabel one post

Nothing is pre-filled. Showing a model's or the rules baseline's guess would
anchor the label to it, and the gold set exists to measure those guesses.

At any prompt: ":skip" marks the post as not a job post (the next candidate
takes its place), ":back" restarts the post, ":quit" stops. Labels append to
data/gold/; a relabel adds a newer line rather than editing an old one.
Conventions are in data/gold/LABELING.md.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jpe import corpus, gold  # noqa: E402
from jpe.fields import EMPTY_RECORD, NEW_GRAD, REMOTE, SCHEMA_VERSION, SENIORITY, VISA  # noqa: E402
from jpe.normalize import canonical_stack  # noqa: E402

AGREEMENT_N = 40
AGREEMENT_SEED = 7


class Skip(Exception):
    pass


class Back(Exception):
    pass


class Quit(Exception):
    pass


def ask(prompt: str) -> str:
    raw = input(prompt).strip()
    if raw == ":skip":
        raise Skip
    if raw == ":back":
        raise Back
    if raw == ":quit":
        raise Quit
    return raw


def parse_choice(raw: str, choices: tuple[str, ...]) -> str | None:
    """'y' / 'yes' / 'h' -> the choice it uniquely starts; '' -> unspecified."""
    raw = raw.strip().lower()
    if not raw:
        return "unspecified"
    hits = [c for c in choices if c.startswith(raw)]
    return hits[0] if len(hits) == 1 else None


_AMOUNT = re.compile(r"^(\d+(?:\.\d+)?)(k)?$", re.I)


def parse_amount(tok: str) -> float | None:
    """'120000', '120k', '120,000' -> 120000. '-' -> None."""
    tok = tok.replace(",", "")
    if tok in ("-", ""):
        return None
    m = _AMOUNT.match(tok)
    if not m:
        raise ValueError(f"not an amount: {tok!r}")
    return float(m.group(1)) * (1000 if m.group(2) else 1)


def parse_salary_input(raw: str) -> tuple[float | None, float | None, str | None]:
    """'' -> no salary. '120k 150k USD', '150k - USD' (open max), '- 200k USD',
    '140k USD' (point salary: min = max), '120k 150k' (currency not stated)."""
    toks = raw.split()
    if not toks:
        return None, None, None
    cur = None
    if re.fullmatch(r"[A-Za-z]{3}", toks[-1]):
        cur = toks.pop().upper()
    if len(toks) == 1:
        v = parse_amount(toks[0])
        lo = hi = v
    elif len(toks) == 2:
        lo, hi = parse_amount(toks[0]), parse_amount(toks[1])
    else:
        raise ValueError("expected: MIN MAX [CUR]")
    if lo is None and hi is None:
        raise ValueError("give at least one amount, or leave it blank for no salary")
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("min is above max")
    return lo, hi, cur


def _ask_until(prompt: str, parse):
    while True:
        raw = ask(prompt)
        try:
            out = parse(raw)
        except ValueError as e:
            print(f"  {e}")
            continue
        if out is None:
            print("  not one of the choices")
            continue
        return out


def label_record() -> tuple[dict, str]:
    r = dict(EMPTY_RECORD)
    r["company"] = ask("company: ") or None
    r["role_title"] = ask("role title (the first listed role): ") or None
    r["seniority"] = _ask_until("seniority [i]ntern [n]ew_grad [m]id [s]enior, Enter = unspecified: ", lambda s: parse_choice(s, SENIORITY))
    r["location"] = ask("location (where the hire must be; Enter = none): ") or None
    r["remote"] = _ask_until("remote [y]es [n]o [h]ybrid, Enter = unspecified: ", lambda s: parse_choice(s, REMOTE))
    r["visa_sponsorship"] = _ask_until("visa sponsorship [y]es [n]o, Enter = unspecified: ", lambda s: parse_choice(s, VISA))
    lo, hi, cur = _ask_until("salary, annual base (e.g. '120k 150k USD', '150k - USD'; Enter = none): ", parse_salary_input)
    r["salary_min"], r["salary_max"], r["salary_currency"] = lo, hi, cur
    stack = ask("tech stack (comma-separated): ")
    r["tech_stack"] = canonical_stack(s for s in stack.split(","))
    r["new_grad_eligible"] = _ask_until("new grads eligible [y]es [n]o, Enter = unspecified: ", lambda s: parse_choice(s, NEW_GRAD))
    note = ask("note (start with 'ambiguous: <field>' for judgment calls; Enter = none): ")
    return r, note


def check(record: dict) -> str | None:
    """The validator's complaint, if it is written yet."""
    try:
        from jpe.validate import InvalidRecord, validate_record

        validate_record(record)
    except NotImplementedError:
        return None
    except InvalidRecord as e:
        return str(e)
    return None


def show(post, header: str) -> None:
    width = min(100, shutil.get_terminal_size((100, 40)).columns)
    print("\n" + "=" * width)
    print(header)
    print(f"{post.url}   ({post.month})")
    print("-" * width)
    for para in post.text.split("\n"):
        print(textwrap.fill(para, width) if para else "")
    print("-" * width)


def run(queue, annotator: str, out: Path, totals) -> int:
    posts = corpus.load_corpus()
    n = 0
    for split_name, pid in queue:
        post = posts.get(pid)
        if post is None:
            print(f"post {pid} is not on disk; run scripts/fetch_posts.py --from-ids", file=sys.stderr)
            continue
        while True:
            show(post, f"[{split_name}] {totals(split_name)}   post {pid}")
            started = time.monotonic()
            entry = {"post_id": pid, "split": split_name, "annotator": annotator, "schema_version": SCHEMA_VERSION}
            try:
                record, note = label_record()
            except Back:
                continue
            except Skip:
                reason = input("why is this not a job post? ").strip()
                entry.update(skip=True, note=reason)
                break
            except (Quit, EOFError, KeyboardInterrupt):
                print(f"\nStopped. {n} labeled this session.")
                return n
            problem = check(record)
            if problem:
                print(f"  rejected by the validator: {problem}")
                continue
            entry.update(record=record, note=note)
            break
        entry["seconds"] = round(time.monotonic() - started, 1)
        entry["labeled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        gold.append_label(entry, out)
        n += 1
    print(f"Nothing left to label. {n} labeled this session.")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["gold", "fewshot"], default="gold")
    ap.add_argument("--second", action="store_true", help="label the agreement subset as a second annotator")
    ap.add_argument("--annotator", default="author")
    ap.add_argument("--redo", type=int, help="relabel one post")
    args = ap.parse_args(argv)

    split = gold.load_split()

    if args.second:
        entries = gold.read_labels()
        pool = sorted(gold.gold_ids(split, entries, "dev") + gold.gold_ids(split, entries, "test"))
        if len(pool) < AGREEMENT_N:
            print("Label the gold set first; the agreement subset is drawn from it.", file=sys.stderr)
            return 1
        subset = random.Random(AGREEMENT_SEED).sample(pool, AGREEMENT_N)
        done = gold.latest(gold.read_labels(gold.SECOND_FILE))
        queue = [("agreement", pid) for pid in subset if pid not in done]
        return 0 if run(queue, args.annotator, gold.SECOND_FILE, lambda _: f"{AGREEMENT_N - len(queue)} done of {AGREEMENT_N}") >= 0 else 1

    if args.redo:
        name = next((n for n, q in split.queues.items() if args.redo in q), None)
        if name is None:
            print(f"{args.redo} is not in any gold queue", file=sys.stderr)
            return 1
        run([(name, args.redo)], args.annotator, gold.LABELS_FILE, lambda n: "relabel")
        return 0

    names = ("fewshot",) if args.set == "fewshot" else ("dev", "test")

    def totals(name):
        done = len(gold.gold_ids(split, gold.read_labels(), name))
        return f"{done}/{split.targets[name]} labeled"

    queue = gold.next_to_label(split, gold.read_labels(), names)
    run(queue, args.annotator, gold.LABELS_FILE, totals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
