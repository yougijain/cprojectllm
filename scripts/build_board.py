"""Build the job board's data from the newest threads.

    python scripts/build_board.py --extractor rules --months 3
    python scripts/build_board.py --extractor M1-q4 --months 3   # llama-server running

Extracts every post from the last --months threads that the record store
does not already hold for this extractor, then writes site/data/jobs.json
for the static page in site/. The store (data/board/records.jsonl) is what
makes the weekly run incremental; the workflow keeps it as a Release asset.

The extractor is named in the page's footer. Until a model has been
published the board runs on B1, the rules baseline, and says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jpe import corpus, extract, rules  # noqa: E402
from jpe.clients import make_client  # noqa: E402
from jpe.config import model_config  # noqa: E402
from jpe.prompts import load_prompt  # noqa: E402

STORE = ROOT / "data" / "board" / "records.jsonl"
SITE_DATA = ROOT / "site" / "data" / "jobs.json"


def read_store(path: Path = STORE) -> dict[int, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {int(r["post_id"]): r for r in rows}


def write_store(rows: dict[int, dict], path: Path = STORE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for pid in sorted(rows):
            f.write(json.dumps(rows[pid], ensure_ascii=False) + "\n")


def board_row(post, record: dict) -> dict:
    return {
        "id": post.id,
        "url": post.url,
        "month": post.month,
        "posted": post.time,
        "company": record["company"],
        "role": record["role_title"],
        "location": record["location"],
        "remote": record["remote"],
        "visa": record["visa_sponsorship"],
        "seniority": record["seniority"],
        "new_grad": record["new_grad_eligible"],
        "salary_min": record["salary_min"],
        "salary_max": record["salary_max"],
        "currency": record["salary_currency"],
        "stack": record["tech_stack"],
    }


def build(posts: list, store: dict[int, dict], extractor: str) -> tuple[list[dict], dict[int, dict], dict]:
    """Extract what is new, return (board rows, updated store, extractor info)."""
    if extractor == "rules":
        ident = {"kind": "rules", "version": rules.VERSION}
        label = "B1, the rules baseline"
    else:
        client = make_client(model_config(extractor))
        ident = client.identity()
        label = f"{extractor}, the fine-tuned model"
    key = json.dumps(ident, sort_keys=True)

    todo = [p for p in posts if store.get(p.id, {}).get("extractor") != key]
    if todo:
        if extractor == "rules":
            results = [(p.id, extract.OK, rules.extract(p.text)) for p in todo]
        else:
            preds = extract.run(client, load_prompt(), [(p.id, p.text) for p in todo])
            results = [(pr.post_id, pr.status, pr.record) for pr in preds]
        for pid, status, record in results:
            store[pid] = {"post_id": pid, "extractor": key, "status": status, "record": record}

    by_id = {p.id: p for p in posts}
    rows = [board_row(by_id[pid], store[pid]["record"]) for pid in sorted(by_id, reverse=True)
            if store.get(pid, {}).get("record") is not None and store[pid].get("extractor") == key]
    info = {"label": label, "identity": ident, "extracted_now": len(todo)}
    return rows, store, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor", default="rules", help="'rules', or a served model in configs/models.yaml (e.g. M1-q4)")
    ap.add_argument("--months", type=int, default=3)
    args = ap.parse_args(argv)

    months = sorted(corpus.load_threads())[-args.months:]
    posts = sorted(corpus.load_corpus(months).values(), key=lambda p: p.id)
    rows, store, info = build(posts, read_store(), args.extractor)
    write_store(store)

    SITE_DATA.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "months": months,
        "extractor": info["label"],
        "count": len(rows),
        "jobs": rows,
    }
    SITE_DATA.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
    failed = sum(1 for p in posts if store.get(p.id, {}).get("status") == extract.FAILED)
    print(f"{len(rows)} jobs from {months[0]}..{months[-1]} by {info['label']}; "
          f"{info['extracted_now']} extracted this run, {failed} failed and left off the board")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
