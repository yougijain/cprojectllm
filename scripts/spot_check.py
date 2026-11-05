"""Hand-check 100 random silver labels and report the teacher's error rate.

    python scripts/spot_check.py           # judge, resuming where you stopped
    python scripts/spot_check.py --report  # write data/silver/spot_check.md

Unlike gold labeling, the teacher's record is shown: the question here is
whether it is right, field by field. Press Enter if a field is right, "n"
if it is wrong. The per-field error rate is the noise in the training
labels, and roughly the ceiling on how well M1 can learn each field from
them.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jpe import corpus, silver  # noqa: E402
from jpe.fields import SCORED_FIELDS  # noqa: E402
from jpe.stats import bootstrap_mean_ci  # noqa: E402
from scripts.label import show  # noqa: E402

N = 100
SEED = 11
REPORT = silver.SILVER_DIR / "spot_check.md"


def sample_ids() -> list[int]:
    usable = sorted(pid for pid, r in silver.read_labels().items() if r.get("record"))
    return random.Random(SEED).sample(usable, min(N, len(usable)))


def shown(record: dict, field: str) -> str:
    if field == "salary":
        return f"{record['salary_min']} - {record['salary_max']} {record['salary_currency']}"
    return json.dumps(record[field], ensure_ascii=False)


def judge() -> int:
    labels = silver.read_labels()
    posts = corpus.load_corpus()
    done = {}
    if silver.SPOT_FILE.exists():
        done = {int(json.loads(line)["post_id"]) for line in silver.SPOT_FILE.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [pid for pid in sample_ids() if pid not in done]
    for i, pid in enumerate(todo, start=len(done) + 1):
        show(posts[pid], f"[spot check] {i}/{N}   post {pid}")
        record = labels[pid]["record"]
        verdicts = {}
        try:
            for field in SCORED_FIELDS:
                raw = input(f"{field:18s} {shown(record, field):50s} right? [Enter=yes, n=no] ").strip().lower()
                verdicts[field] = raw not in ("n", "no")
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            return 0
        with silver.SPOT_FILE.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"post_id": pid, "correct": verdicts, "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
    print("Spot check complete. Run with --report.")
    return 0


def report() -> int:
    rows = [json.loads(line) for line in silver.SPOT_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [
        "# Silver label spot check",
        "",
        f"{len(rows)} silver labels drawn at random (seed {SEED}) and judged by hand, field by field.",
        "",
        "| Field | Error rate | 95% CI |",
        "|---|---|---|",
    ]
    total = []
    for field in SCORED_FIELDS:
        errs = [0.0 if r["correct"][field] else 1.0 for r in rows]
        total += errs
        m, lo, hi = bootstrap_mean_ci(errs)
        lines.append(f"| {field} | {m:.0%} | {lo:.0%} to {hi:.0%} |")
    m, lo, hi = bootstrap_mean_ci(total)
    lines.append(f"| **all fields** | **{m:.1%}** | {lo:.1%} to {hi:.1%} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    return report() if args.report else judge()


if __name__ == "__main__":
    raise SystemExit(main())
