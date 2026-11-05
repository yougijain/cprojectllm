"""Aggregation, comparisons, cost and error causes, with a stand-in scorer.

The real scorer is hand-written (jpe/score.py); these tests use an
exact-match stand-in so they check the aggregation, not the scoring rule.
"""

import pytest

from jpe import errors, report
from jpe.cost import api_cost_per_1k, gpu_cost_per_1k
from jpe.fields import EMPTY_RECORD, SCORED_FIELDS


def exact(pred, gold):
    if pred is None:
        return {f: 0.0 for f in SCORED_FIELDS}
    out = {}
    for f in SCORED_FIELDS:
        if f == "salary":
            out[f] = float(all(pred[k] == gold[k] for k in ("salary_min", "salary_max", "salary_currency")))
        else:
            out[f] = float(pred[f] == gold[f])
    return out


def rec(**kw):
    r = dict(EMPTY_RECORD, company="Acme")
    r.update(kw)
    return r


GOLD = {i: rec(remote="yes" if i % 2 else "no") for i in range(1, 21)}


def preds(fn):
    return {pid: {"post_id": pid, "status": "ok", "record": fn(pid, g), "input_tokens": 1000, "output_tokens": 200} for pid, g in GOLD.items()}


def test_perfect_and_failed_runs():
    perfect = preds(lambda pid, g: dict(g))
    failed = {pid: {"post_id": pid, "status": "failed", "record": None, "input_tokens": 0, "output_tokens": 0} for pid in GOLD}
    res = report.evaluate({"M1": ({}, perfect), "B2": ({}, failed)}, GOLD, exact, n_boot=200)
    assert res["models"]["M1"]["field_accuracy"]["mean"] == 1.0
    assert res["models"]["B2"]["field_accuracy"]["mean"] == 0.0
    assert res["models"]["B2"]["json_valid_final"] == 0.0
    c = res["comparisons"][0]
    assert c["b"] == "B2" and c["field_accuracy_diff"]["mean"] == 1.0
    assert all(t["p_holm"] < 0.05 for t in c["fields"].values())


def test_missing_prediction_counts_as_wrong():
    partial = {pid: p for pid, p in preds(lambda pid, g: dict(g)).items() if pid <= 10}
    res = report.evaluate({"M1": ({}, partial)}, GOLD, exact, n_boot=200)
    assert res["models"]["M1"]["missing"] == 10
    assert res["models"]["M1"]["field_accuracy"]["mean"] == pytest.approx(0.5)


def test_one_wrong_field_moves_field_accuracy_by_a_ninth():
    run = preds(lambda pid, g: dict(g, remote="hybrid"))
    res = report.evaluate({"M1": ({}, run)}, GOLD, exact, n_boot=200)
    assert res["models"]["M1"]["fields"]["remote"]["mean"] == 0.0
    assert res["models"]["M1"]["field_accuracy"]["mean"] == pytest.approx(8 / 9)


def test_learning_curve_seeds_and_quantization():
    good = preds(lambda pid, g: dict(g))
    worse = preds(lambda pid, g: dict(g, remote="hybrid") if pid <= 6 else dict(g))
    runs = {
        "M1": ({}, good),
        "M1-n250-s0": ({}, worse),
        "M1-nall-s0": ({}, good),
        "M1-nall-s1": ({}, worse),
        "M1-q4": ({}, worse),
    }
    res = report.evaluate(runs, GOLD, exact, n_boot=200)
    assert [r["n_train"] for r in res["learning_curve"]] == ["250", "all"]
    assert res["seeds"]["seeds"] == [0, 1] and res["seeds"]["max"] == 1.0
    q = res["quantization"]
    assert q["fields"]["remote"]["a_only"] == 6 and q["fields"]["remote"]["b_only"] == 0
    md = report.to_markdown(res, "test")
    assert "Learning curve" in md and "Quantization" in md
    assert all(c["b"] != "M1-n250-s0" for c in res["comparisons"])


def test_cost():
    run = list(preds(lambda pid, g: dict(g)).values())
    # 1,000 input and 200 output tokens per post at $4 / $20 per million: $0.008 per post.
    assert api_cost_per_1k(run, 4.0, 20.0) == pytest.approx(8.0)
    # 10 posts/s is 100 s per 1,000 posts; at $0.36/hour that is $0.01.
    assert gpu_cost_per_1k(10.0, 0.36) == pytest.approx(0.01)
    res = report.evaluate({"B4": ({"model": {"kind": "anthropic", "model": "claude-opus-5-5"}}, {p["post_id"]: p for p in run})}, GOLD, exact,
                          prices={"claude-opus-5-5": {"input_per_mtok": 4.0, "output_per_mtok": 20.0, "as_of": "x"}}, n_boot=200)
    assert res["models"]["B4"]["cost_per_1k"]["usd"] == pytest.approx(8.0)


@pytest.mark.parametrize(
    "field,pred,gold,status,note,expected",
    [
        ("remote", rec(), rec(remote="yes"), "ok", "", "missed"),
        ("remote", rec(remote="yes"), rec(), "ok", "", "hallucinated"),
        ("salary", rec(salary_min=120, salary_max=150, salary_currency="USD"), rec(salary_min=120_000, salary_max=150_000, salary_currency="USD"), "ok", "", "normalization"),
        ("company", rec(company="Acme Inc"), rec(company="acme, inc."), "ok", "", "normalization"),
        ("remote", rec(remote="no"), rec(remote="yes"), "ok", "ambiguous: remote, says both", "ambiguous"),
        ("remote", None, rec(remote="yes"), "failed", "", "invalid_json"),
        ("remote", rec(remote="no"), rec(remote="yes"), "ok", "", ""),
    ],
)
def test_suggest_cause(field, pred, gold, status, note, expected):
    assert errors.suggest_cause(field, pred, gold, status, note) == expected


def test_error_sheet_keeps_hand_tags(tmp_path):
    p = preds(lambda pid, g: dict(g, remote="hybrid"))
    scores = {pid: exact(p[pid]["record"], g) for pid, g in GOLD.items()}
    rows = errors.error_rows(p, GOLD, scores, {})
    assert len(rows) == 20 and {r["field"] for r in rows} == {"remote"}
    path = tmp_path / "M1.csv"
    errors.write_sheet(path, rows)
    sheet = errors.read_sheet(path)
    sheet[0]["cause"] = "label_error"
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=errors.COLUMNS)
        w.writeheader()
        w.writerows(sheet)
    errors.write_sheet(path, rows)  # regenerate
    again = errors.read_sheet(path)
    assert again[0]["cause"] == "label_error"
    assert errors.count_causes(again)["label_error"] == 1
