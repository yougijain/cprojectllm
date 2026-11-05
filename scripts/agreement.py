"""Inter-annotator agreement on the 40-post subset.

    python scripts/agreement.py

Compares data/gold/labels_second.jsonl with the author's labels, field by
field, using the same scorer the models are measured with. Writes
data/gold/agreement.md. Low agreement on a field is a finding, not a
failure: it is the ceiling a model can reasonably be held to there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jpe import gold  # noqa: E402
from jpe.fields import ENUMS, SCORED_FIELDS  # noqa: E402
from jpe.score import score_record  # noqa: E402
from jpe.stats import bootstrap_mean_ci, cohen_kappa  # noqa: E402

OUT_MD = gold.GOLD_DIR / "agreement.md"
OUT_JSON = gold.GOLD_DIR / "agreement.json"


def main() -> int:
    first = {pid: e for pid, e in gold.latest(gold.read_labels()).items() if not e.get("skip")}
    second = {pid: e for pid, e in gold.latest(gold.read_labels(gold.SECOND_FILE)).items() if not e.get("skip")}
    both = sorted(set(first) & set(second))
    if not both:
        print("No posts labeled by both annotators yet.", file=sys.stderr)
        return 1

    try:
        scores = [score_record(second[p]["record"], first[p]["record"]) for p in both]
    except NotImplementedError:
        print("Write jpe/score.py first: agreement uses the same scorer as the models.", file=sys.stderr)
        return 1

    rows = {}
    for field in SCORED_FIELDS:
        mean, lo, hi = bootstrap_mean_ci([s[field] for s in scores])
        row = {"agreement": mean, "ci": [lo, hi]}
        if field in ENUMS:
            row["kappa"] = cohen_kappa([first[p]["record"][field] for p in both], [second[p]["record"][field] for p in both])
        rows[field] = row

    OUT_JSON.write_text(json.dumps({"n": len(both), "fields": rows}, indent=2) + "\n", newline="\n")
    lines = [
        "# Inter-annotator agreement",
        "",
        f"{len(both)} posts labeled independently by two people, scored with jpe/score.py.",
        "",
        "| Field | Agreement | 95% CI | Cohen's kappa |",
        "|---|---|---|---|",
    ]
    for field, r in rows.items():
        kappa = f"{r['kappa']:.2f}" if "kappa" in r else ""
        lines.append(f"| {field} | {r['agreement']:.0%} | {r['ci'][0]:.0%} to {r['ci'][1]:.0%} | {kappa} |")
    OUT_MD.write_text("\n".join(lines) + "\n", newline="\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
