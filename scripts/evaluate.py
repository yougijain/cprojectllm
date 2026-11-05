"""Score every saved run on a split and write the results table.

    python scripts/evaluate.py --split test
    python scripts/evaluate.py --split dev --reference M1

Reads results/predictions/<split>/*.json and results/serving/*.json (for
throughput-based cost), and writes results/<split>_results.json and
results/<split>_results.md. With --errors it also writes one error sheet
per run to results/errors/<split>/<run>.csv for hand-tagging causes, and
the cause counts of any sheets already tagged to the markdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jpe import errors, gold, report  # noqa: E402
from jpe.config import load_models  # noqa: E402
from jpe.extract import read_predictions  # noqa: E402
from jpe.score import score_record  # noqa: E402

RESULTS = ROOT / "results"


def load_runs(split: str) -> dict:
    runs = {}
    for path in sorted((RESULTS / "predictions" / split).glob("*.json")):
        meta, preds = read_predictions(path)
        runs[path.stem] = (meta, preds)
    return runs


def load_serving() -> dict:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (RESULTS / "serving").glob("*.json")}


def error_section(split: str, runs: dict, gold_records: dict) -> str:
    entries = gold.latest(gold.read_labels())
    notes = {pid: entries[pid].get("note", "") for pid in gold_records if pid in entries}
    sheets = RESULTS / "errors" / split
    counts = {}
    for name, (meta, preds) in runs.items():
        if report._CURVE.match(name):
            continue
        scores = {pid: score_record(preds[pid]["record"] if pid in preds and preds[pid]["status"] != "failed" else None, g) for pid, g in gold_records.items()}
        path = sheets / f"{name}.csv"
        errors.write_sheet(path, errors.error_rows(preds, gold_records, scores, notes))
        counts[name] = errors.count_causes(errors.read_sheet(path))
    if not counts:
        return ""
    causes = list(errors.CAUSES) + ["untagged"]
    lines = ["", "### Why fields were wrong", "",
             "Every wrong field, by cause: hand-tagged in results/errors/, with mechanical suggestions where untagged.", "",
             "| Model | " + " | ".join(causes) + " | total |", "|---|" + "---|" * (len(causes) + 1)]
    for name, c in counts.items():
        lines.append(f"| {name} | " + " | ".join(str(c.get(k, 0)) for k in causes) + f" | {sum(c.values())} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--reference", default="M1")
    ap.add_argument("--errors", action="store_true", help="write error sheets and count causes")
    args = ap.parse_args(argv)

    if gold.check_frozen() is False:
        print("data/gold changed after it was frozen; refusing to score against it.", file=sys.stderr)
        return 1
    gold_records = gold.load_gold(args.split)
    runs = load_runs(args.split)
    if not runs:
        print(f"no predictions under results/predictions/{args.split}/", file=sys.stderr)
        return 1

    prices = load_models().get("prices", {})
    res = report.evaluate(runs, gold_records, score_record, prices=prices, serving=load_serving(), reference=args.reference)
    md = report.to_markdown(res, args.split)
    if args.errors:
        md += error_section(args.split, runs, gold_records)

    (RESULTS / f"{args.split}_results.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    (RESULTS / f"{args.split}_results.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
