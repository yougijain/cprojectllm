"""Label the training pool with the open-weight teacher.

    python scripts/silver_label.py pool                 # build data/silver/pool.json (once)
    python scripts/silver_label.py label --n 4000 --workers 16

The pool is every post from the complete months that is neither a gold
candidate (dev, test or few-shot, reserves included) nor a near-duplicate
of one. The teacher is configured under `teacher:` in configs/models.yaml
and must have a license that allows training on its outputs. Calls are
cached, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jpe import corpus, dedup, extract, gold, silver  # noqa: E402
from jpe.cache import CachedClient  # noqa: E402
from jpe.clients import make_client  # noqa: E402
from jpe.config import model_config  # noqa: E402
from jpe.prompts import load_prompt  # noqa: E402
from jpe.provenance import run_meta  # noqa: E402


def cmd_pool(args) -> int:
    if silver.POOL_FILE.exists() and not args.force:
        print(f"{silver.POOL_FILE} exists; the pool order is drawn once. --force to rebuild.", file=sys.stderr)
        return 1
    split = gold.load_split()
    months = [m for m in split.months]
    texts = {pid: p.text for pid, p in corpus.load_corpus(months).items()}
    gold_candidates = set().union(*(set(q) for q in split.queues.values()))
    eligible, dropped = dedup.training_pool(texts, gold_candidates)
    order = silver.ordered_pool(eligible)
    silver.save_pool(order, dropped, len(gold_candidates))
    print(f"{len(texts)} posts in {months[0]}..{months[-1]}: {len(gold_candidates)} gold candidates excluded, "
          f"{len(dropped)} near-duplicates of them excluded, {len(order)} eligible for training.")
    return 0


def cmd_label(args) -> int:
    cfg = model_config("teacher")
    if not cfg.get("license"):
        print("Set teacher.license in configs/models.yaml: training on outputs must be allowed.", file=sys.stderr)
        return 1
    prompt = load_prompt()
    client = CachedClient(make_client(cfg))
    texts = corpus.load_corpus()
    todo = [(pid, texts[pid].text) for pid in silver.load_pool()[: args.n] if pid in texts]

    def progress(done, total):
        if done % 100 == 0 or done == total:
            print(f"  {done}/{total}", file=sys.stderr, flush=True)

    preds = extract.run(client, prompt, todo, workers=args.workers, progress=progress)
    rows = [{"post_id": p.post_id, "status": p.status, "record": p.record, "error": p.error,
             "input_tokens": p.input_tokens, "output_tokens": p.output_tokens} for p in preds]
    silver.write_labels(rows)
    counts = {s: sum(r["status"] == s for r in rows) for s in (extract.OK, extract.OK_AFTER_RETRY, extract.FAILED)}
    meta = run_meta("silver", client.identity(), prompt.version, license=cfg["license"], n=len(rows), statuses=counts,
                    cache={"hits": client.hits, "misses": client.misses})
    silver.META_FILE.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"silver labels: {counts}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--force", action="store_true")
    q = sub.add_parser("label")
    q.add_argument("--n", type=int, default=4000)
    q.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    return {"pool": cmd_pool, "label": cmd_label}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
