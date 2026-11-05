"""The CI evaluation gate: fail a PR that costs more than 2 points on dev.

    python scripts/eval_gate.py --check                    # is the gate able to run yet?
    python scripts/eval_gate.py --run results/predictions/dev/M1-q4-gate.json
    python scripts/eval_gate.py --run ... --update-baseline

The workflow runs the published 4-bit M1 over the 50 dev posts with the
PR's code (prompt, validation, extraction loop), scores it, and compares
field accuracy with results/dev_gate_baseline.json. A drop of more than
TOLERANCE fails the check. Decoding is greedy and the model file is pinned
by hash, so on unchanged code the score is exactly the baseline: any move
is the PR's doing, not noise.

The baseline is updated on purpose, by a person, after a change that is
meant to move it (--update-baseline, then commit the file in that PR).

Until there are dev labels, a written scorer and validator, and a
published model, the gate reports itself inactive and passes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jpe import gold, report  # noqa: E402
from jpe.extract import read_predictions  # noqa: E402
from jpe.fields import SCORED_FIELDS  # noqa: E402

BASELINE = ROOT / "results" / "dev_gate_baseline.json"
TOLERANCE = 0.02


def summary(text: str) -> None:
    print(text)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def inactive_reason() -> str | None:
    split = gold.load_split()
    if len(gold.gold_ids(split, gold.read_labels(), "dev")) < split.targets["dev"]:
        return "the dev split is not fully labeled yet"
    try:
        from jpe.score import score_record
        from jpe.validate import parse_record

        score_record(None, dict.fromkeys(SCORED_FIELDS))
        parse_record("{}")
    except NotImplementedError as e:
        return f"{e} is not written yet"
    except Exception:
        pass  # written, and rejecting the probe input as it should
    return None


def decide(baseline: dict, current: dict, tolerance: float = TOLERANCE) -> tuple[bool, str]:
    b, c = baseline["field_accuracy"], current["field_accuracy"]
    drop = b - c
    ok = drop <= tolerance + 1e-12
    verdict = "passes" if ok else "FAILS"
    msg = f"Dev field accuracy {100 * c:.1f} vs baseline {100 * b:.1f} ({-100 * drop:+.1f} points): {verdict} (limit -{100 * tolerance:.0f})."
    return ok, msg


def score(path: Path) -> dict:
    from jpe.score import score_record

    meta, preds = read_predictions(path)
    run = report.score_run("gate", meta, preds, gold.load_gold("dev"), score_record)
    per_post = run.per_post()
    return {
        "field_accuracy": sum(per_post) / len(per_post),
        "fields": {f: sum(run.fields[f]) / len(run.ids) for f in SCORED_FIELDS},
        "model": meta.get("model"),
        "prompt_version": meta.get("prompt_version"),
        "git": meta.get("git"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="print active=true/false for the workflow")
    ap.add_argument("--run", type=Path, help="predictions to score")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args(argv)

    reason = inactive_reason()
    if args.check:
        line = f"active={'false' if reason else 'true'}"
        print(line)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(line + "\n")
        if reason:
            summary(f"Evaluation gate inactive: {reason}.")
        return 0

    if reason:
        summary(f"Evaluation gate inactive: {reason}.")
        return 0
    current = score(args.run)

    if args.update_baseline:
        BASELINE.write_text(json.dumps(current, indent=1) + "\n", encoding="utf-8", newline="\n")
        summary(f"Baseline set: dev field accuracy {100 * current['field_accuracy']:.1f}.")
        return 0
    if not BASELINE.exists():
        summary(f"No baseline yet; dev field accuracy is {100 * current['field_accuracy']:.1f}. Set one with --update-baseline.")
        return 0

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    ok, msg = decide(baseline, current)
    lines = ["### Evaluation gate", "", msg, "", "| Field | Baseline | This PR |", "|---|---|---|"]
    for f in SCORED_FIELDS:
        lines.append(f"| {f} | {100 * baseline['fields'][f]:.1f} | {100 * current['fields'][f]:.1f} |")
    summary("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
