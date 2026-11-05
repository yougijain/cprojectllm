"""Run one model over a gold split and save every prediction.

    python scripts/run_model.py B1 --split test
    python scripts/run_model.py B4 --split test --workers 8
    python scripts/run_model.py M1 --split dev
    python scripts/run_model.py M1-q4 --split test --name M1-q4

Predictions go to results/predictions/<split>/<name>.json with the run's
provenance (model, adapter hash, prompt version, git commit). They cover
every candidate in the split's queue, so a run never has to wait for
labeling to finish; scoring keeps only the posts that ended up in the gold
set.

Model calls go through the disk cache, so rerunning a finished run is free.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jpe import corpus, extract, gold, rules  # noqa: E402
from jpe.cache import CachedClient  # noqa: E402
from jpe.clients import make_client  # noqa: E402
from jpe.config import model_config  # noqa: E402
from jpe.prompts import load_prompt  # noqa: E402
from jpe.provenance import run_meta  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "predictions"


def split_posts(split_name: str) -> list[tuple[int, str]]:
    split = gold.load_split()
    posts = corpus.load_corpus()
    ids = split.queues[split_name]
    missing = [pid for pid in ids if pid not in posts]
    if missing:
        print(f"{len(missing)} posts are not on disk; run scripts/fetch_posts.py --from-ids", file=sys.stderr)
    return [(pid, posts[pid].text) for pid in ids if pid in posts]


def fewshot_examples() -> list[tuple[str, dict]]:
    posts = corpus.load_corpus()
    labels = gold.load_gold("fewshot")
    if len(labels) < gold.FEWSHOT_TARGET:
        raise SystemExit(f"{len(labels)} few-shot examples labeled; label {gold.FEWSHOT_TARGET} with scripts/label.py --set fewshot")
    return [(posts[pid].text, record) for pid, record in sorted(labels.items())]


def progress(done: int, total: int) -> None:
    if done == total or done % 10 == 0:
        print(f"  {done}/{total}", file=sys.stderr, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="a key under models: in configs/models.yaml")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--name", help="output name (default: the model key)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent calls, for API and server models")
    ap.add_argument("--batch-size", type=int, default=8, help="for in-process models")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    cfg = model_config(args.model)
    name = args.name or args.model
    posts = split_posts(args.split)
    started = time.time()

    if cfg["kind"] == "rules":
        preds = [extract.Prediction(pid, extract.OK, rules.extract(text)) for pid, text in posts]
        meta = run_meta(name, {"kind": "rules", "version": rules.VERSION}, None, split=args.split, n_posts=len(preds), json_applies=False)
    else:
        prompt = load_prompt(fewshot=fewshot_examples() if cfg.get("fewshot") else None)
        client = make_client(cfg)
        if not args.no_cache:
            client = CachedClient(client)
        if cfg["kind"] == "hf_local":
            preds = extract.run_batched(client, prompt, posts, batch_size=args.batch_size)
        else:
            preds = extract.run(client, prompt, posts, workers=args.workers, progress=progress)
        meta = run_meta(
            name,
            client.identity(),
            prompt.version,
            split=args.split,
            n_posts=len(preds),
            json_applies=True,
            description=cfg.get("description"),
            cache={"hits": getattr(client, "hits", None), "misses": getattr(client, "misses", None)},
        )
    meta["wall_seconds"] = round(time.time() - started, 1)

    out = RESULTS / args.split / f"{name}.json"
    extract.write_predictions(out, meta, preds)
    statuses = {s: sum(p.status == s for p in preds) for s in (extract.OK, extract.OK_AFTER_RETRY, extract.FAILED)}
    print(f"{name} on {args.split}: {statuses} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
