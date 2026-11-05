"""Error analysis: every wrong field, sorted into one cause.

    ambiguous       the post supports more than one reading (the gold note
                    says "ambiguous: <field>")
    missed          the post states it; the model gave null / unspecified / []
    hallucinated    the post does not state it; the model gave a value
    normalization   the right fact in the wrong form ("120" for 120,000,
                    "San Francisco" for "SF", monthly pay not annualized)
    invalid_json    the post failed: no record after the retry
    label_error     the gold label is wrong (fix it, and note it here)

suggest_cause fills in what can be read off mechanically. The rest needs a
person: the sheet has a `cause` column for the hand-assigned cause, which
always wins, and regenerating the sheet keeps what was entered.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from jpe.fields import EMPTY_RECORD, SCORED_FIELDS

CAUSES = ("ambiguous", "missed", "hallucinated", "normalization", "invalid_json", "label_error")
COLUMNS = ("post_id", "field", "gold", "pred", "score", "suggested", "cause", "note")


def _value(record: dict | None, field: str):
    if record is None:
        return None
    if field == "salary":
        return [record["salary_min"], record["salary_max"], record["salary_currency"]]
    return record[field]


def _empty(field: str, value) -> bool:
    if field == "salary":
        return value is None or all(v is None for v in value)
    return value in (None, [], "", EMPTY_RECORD.get(field))


def _loose(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _normalization(field: str, pred, gold) -> bool:
    if field == "salary":
        if pred[2] != gold[2] and (pred[0], pred[1]) == (gold[0], gold[1]):
            return False  # a different currency is a different fact
        for p, g in zip(pred[:2], gold[:2]):
            if p is None or g is None or not g:
                continue
            ratio = p / g
            if any(abs(ratio - f) / f < 0.05 for f in (1 / 1000, 1000, 1 / 12, 12, 1 / 2080, 2080)):
                return True
        return False
    if isinstance(gold, str) and isinstance(pred, str):
        return _loose(pred) == _loose(gold) or _loose(pred) in _loose(gold) or _loose(gold) in _loose(pred)
    return False


def suggest_cause(field: str, pred_record: dict | None, gold_record: dict, status: str, gold_note: str = "") -> str:
    if status == "failed" or pred_record is None:
        return "invalid_json"
    if re.search(rf"ambiguous:\s*[^\n]*\b{re.escape(field)}\b", gold_note or "", re.I):
        return "ambiguous"
    pred, gold = _value(pred_record, field), _value(gold_record, field)
    if _empty(field, pred) and not _empty(field, gold):
        return "missed"
    if _empty(field, gold) and not _empty(field, pred):
        return "hallucinated"
    if _normalization(field, pred, gold):
        return "normalization"
    return ""


def error_rows(predictions: dict[int, dict], gold: dict[int, dict], scores: dict[int, dict], notes: dict[int, str]) -> list[dict]:
    rows = []
    for pid in sorted(gold):
        p = predictions.get(pid)
        record = p["record"] if p else None
        status = p["status"] if p else "failed"
        for field in SCORED_FIELDS:
            s = scores[pid][field]
            if s >= 1.0:
                continue
            rows.append({
                "post_id": pid,
                "field": field,
                "gold": json.dumps(_value(gold[pid], field), ensure_ascii=False),
                "pred": json.dumps(_value(record, field), ensure_ascii=False),
                "score": round(s, 3),
                "suggested": suggest_cause(field, record, gold[pid], status, notes.get(pid, "")),
                "cause": "",
                "note": "",
            })
    return rows


def write_sheet(path: Path, rows: list[dict]) -> None:
    """Write the sheet, keeping any cause and note already entered by hand."""
    kept = {}
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                kept[(r["post_id"], r["field"])] = (r.get("cause", ""), r.get("note", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["cause"], r["note"] = kept.get((str(r["post_id"]), r["field"]), ("", ""))
            w.writerow(r)


def read_sheet(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def count_causes(rows: list[dict]) -> Counter:
    """The hand-assigned cause where there is one, else the suggestion, else
    'untagged'."""
    c = Counter()
    for r in rows:
        cause = (r.get("cause") or r.get("suggested") or "untagged").strip()
        if cause not in CAUSES:
            cause = "untagged"
        c[cause] += 1
    return c
