"""Turn saved predictions into the results table.

Every model is scored on the same gold posts with jpe.score.score_record.
A post a model has no prediction for, or failed on, scores zero on every
field: it stays in the denominator.

  field accuracy   mean over posts of the mean over the nine scored fields
                   (stack F1 counts as its F1), with a bootstrap 95% CI
  per field        the same, one field at a time
  JSON valid       share of posts valid on the first reply, and after the
                   one retry
  vs reference     per field, exact McNemar on right/wrong (Holm-adjusted
                   across the nine fields), plus a paired bootstrap of the
                   difference in field accuracy
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from jpe import stats
from jpe.cost import api_cost_per_1k, gpu_cost_per_1k
from jpe.fields import SCORED_FIELDS, USER_FIELDS


@dataclass
class RunScores:
    name: str
    meta: dict
    ids: list[int]
    fields: dict[str, list[float]]
    status: list[str]
    preds: dict[int, dict]

    def per_post(self) -> list[float]:
        return [sum(self.fields[f][i] for f in SCORED_FIELDS) / len(SCORED_FIELDS) for i in range(len(self.ids))]

    def correct(self, field: str) -> list[bool]:
        return [s >= 1.0 for s in self.fields[field]]


def score_run(name: str, meta: dict, preds: dict[int, dict], gold: dict[int, dict], score: Callable) -> RunScores:
    ids = sorted(gold)
    fields: dict[str, list[float]] = {f: [] for f in SCORED_FIELDS}
    status = []
    for pid in ids:
        p = preds.get(pid)
        record = p["record"] if p and p.get("status") != "failed" else None
        status.append(p["status"] if p else "missing")
        s = score(record, gold[pid])
        for f in SCORED_FIELDS:
            fields[f].append(float(s[f]))
    return RunScores(name, meta, ids, fields, status, preds)


def summarize(run: RunScores, n_boot: int = stats.N_BOOT) -> dict:
    m, lo, hi = stats.bootstrap_mean_ci(run.per_post(), n_boot=n_boot)
    out = {
        "n": len(run.ids),
        "field_accuracy": {"mean": m, "ci": [lo, hi]},
        "fields": {},
        "missing": run.status.count("missing"),
    }
    for f in SCORED_FIELDS:
        fm, flo, fhi = stats.bootstrap_mean_ci(run.fields[f], n_boot=n_boot)
        out["fields"][f] = {"mean": fm, "ci": [flo, fhi]}
    if run.meta.get("json_applies", True):
        n = len(run.status)
        out["json_valid_first"] = run.status.count("ok") / n if n else float("nan")
        out["json_valid_final"] = (run.status.count("ok") + run.status.count("ok_after_retry")) / n if n else float("nan")
    return out


def compare(a: RunScores, b: RunScores, n_boot: int = stats.N_BOOT) -> dict:
    if a.ids != b.ids:
        raise ValueError("runs were scored on different posts")
    tests = {f: stats.mcnemar_exact(a.correct(f), b.correct(f)) for f in SCORED_FIELDS}
    adjusted = stats.holm({f: t["p"] for f, t in tests.items()})
    d, lo, hi = stats.paired_bootstrap_diff(a.per_post(), b.per_post(), n_boot=n_boot)
    return {
        "a": a.name,
        "b": b.name,
        "field_accuracy_diff": {"mean": d, "ci": [lo, hi]},
        "fields": {f: {**tests[f], "p_holm": adjusted[f], "diff": sum(a.fields[f]) / len(a.ids) - sum(b.fields[f]) / len(b.ids)} for f in SCORED_FIELDS},
    }


def cost_per_1k(run: RunScores, prices: dict, serving: dict[str, dict]) -> dict | None:
    model = run.meta.get("model", {})
    if model.get("kind") == "rules":
        return {"usd": 0.0, "basis": "CPU, negligible"}
    if model.get("kind") == "anthropic":
        price = prices.get(model.get("model"))
        if not price:
            return None
        usd = api_cost_per_1k(run.preds.values(), price["input_per_mtok"], price["output_per_mtok"])
        return {"usd": usd, "basis": f"list price, {price['as_of']}"}
    bench = serving.get(run.name) or (serving.get("M1-vllm") if run.name == "M1" else None)
    if bench and bench.get("hardware") == "T4" and "t4_gpu_hour" in prices:
        pps = bench["throughput_posts_per_s"]
        return {"usd": gpu_cost_per_1k(pps, prices["t4_gpu_hour"]["usd"]), "basis": f"T4 at {pps:.2f} posts/s (vLLM)"}
    return None


_CURVE = re.compile(r"^M1-n(\w+)-s(\d+)$")


def evaluate(runs: dict[str, tuple[dict, dict]], gold: dict[int, dict], score: Callable, prices: dict | None = None,
             serving: dict | None = None, reference: str = "M1", n_boot: int = stats.N_BOOT) -> dict:
    prices, serving = prices or {}, serving or {}
    scored = {name: score_run(name, meta, preds, gold, score) for name, (meta, preds) in runs.items()}
    out = {"n_gold": len(gold), "reference": reference, "models": {}, "comparisons": [], "learning_curve": [], "seeds": None, "quantization": None}

    for name, run in scored.items():
        s = summarize(run, n_boot)
        s["meta"] = {k: run.meta.get(k) for k in ("model", "prompt_version", "git", "created_at", "description")}
        s["cost_per_1k"] = cost_per_1k(run, prices, serving)
        out["models"][name] = s

    ref = scored.get(reference)
    if ref:
        for name, run in scored.items():
            if name != reference and not _CURVE.match(name):
                out["comparisons"].append(compare(ref, run, n_boot))

    curve = {}
    for name, run in scored.items():
        m = _CURVE.match(name)
        if m:
            curve.setdefault(m.group(1), {})[int(m.group(2))] = out["models"][name]["field_accuracy"]
    for size in sorted(curve, key=lambda s: float("inf") if s == "all" else int(s)):
        if 0 in curve[size]:
            out["learning_curve"].append({"n_train": size, **curve[size][0]})
    if "all" in curve and len(curve["all"]) > 1:
        vals = [v["mean"] for v in curve["all"].values()]
        out["seeds"] = {"seeds": sorted(curve["all"]), "mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}

    full = scored.get("M1-vllm") or scored.get(reference)
    quant = scored.get("M1-q4")
    if full and quant:
        out["quantization"] = compare(full, quant, n_boot)
    return out


# --- Markdown -----------------------------------------------------------------

def _pct(x) -> str:
    return "—" if x is None or x != x else f"{100 * x:.1f}"


def _ci(d) -> str:
    return f"{_pct(d['mean'])} ({_pct(d['ci'][0])}–{_pct(d['ci'][1])})"


def _usd(c) -> str:
    if not c:
        return "—"
    return "≈ $0" if c["usd"] == 0 else f"${c['usd']:.2f}"


def to_markdown(res: dict, split: str) -> str:
    order = [m for m in ("B1", "B2", "B3", "B4", "M1", "M1-vllm", "M1-q4") if m in res["models"]]
    order += [m for m in res["models"] if m not in order and not _CURVE.match(m)]
    lines = [
        f"Scored on {res['n_gold']} hand-labeled {split} posts. Percent, with bootstrap 95% intervals.",
        "",
        "| Model | Field accuracy | " + " | ".join(f.replace("_", " ") for f in USER_FIELDS) + " | JSON valid (1st / retry) | Cost per 1k posts |",
        "|---|---|" + "---|" * len(USER_FIELDS) + "---|---|",
    ]
    for name in order:
        m = res["models"][name]
        json_cell = "n/a" if "json_valid_first" not in m else f"{_pct(m['json_valid_first'])} / {_pct(m['json_valid_final'])}"
        user = " | ".join(_ci(m["fields"][f]) for f in USER_FIELDS)
        lines.append(f"| {name} | {_ci(m['field_accuracy'])} | {user} | {json_cell} | {_usd(m['cost_per_1k'])} |")

    lines += ["", "### Every field", "", "| Field | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for f in SCORED_FIELDS:
        lines.append(f"| {f} | " + " | ".join(_pct(res["models"][n]["fields"][f]["mean"]) for n in order) + " |")

    if res["comparisons"]:
        lines += ["", f"### {res['reference']} against each model", "",
                  "Difference in field accuracy (points, paired bootstrap 95% CI), and the fields where McNemar's test (Holm-adjusted) finds a difference at p < 0.05.", "",
                  "| vs | Difference | Fields that differ |", "|---|---|---|"]
        for c in res["comparisons"]:
            d = c["field_accuracy_diff"]
            sig = [f"{f} ({'+' if t['diff'] > 0 else ''}{100 * t['diff']:.1f})" for f, t in c["fields"].items() if t["p_holm"] < 0.05]
            lines.append(f"| {c['b']} | {100 * d['mean']:+.1f} ({100 * d['ci'][0]:+.1f} to {100 * d['ci'][1]:+.1f}) | {', '.join(sig) or 'none'} |")

    if res["learning_curve"]:
        lines += ["", "### Learning curve (seed 0)", "", "| Training examples | Field accuracy |", "|---|---|"]
        for row in res["learning_curve"]:
            lines.append(f"| {row['n_train']} | {_ci(row)} |")
    if res["seeds"]:
        s = res["seeds"]
        lines += ["", f"Final configuration over seeds {s['seeds']}: mean {_pct(s['mean'])}, range {_pct(s['min'])}–{_pct(s['max'])}."]

    q = res.get("quantization")
    if q:
        d = q["field_accuracy_diff"]
        lines += ["", f"### Quantization: {q['a']} vs {q['b']}", "",
                  f"Field accuracy lost to 4-bit: {100 * d['mean']:.1f} points ({100 * d['ci'][0]:.1f} to {100 * d['ci'][1]:.1f}).", "",
                  "| Field | Points lost | Posts only full precision got right | Posts only 4-bit got right | p (Holm) |", "|---|---|---|---|---|"]
        for f, t in q["fields"].items():
            lines.append(f"| {f} | {100 * t['diff']:.1f} | {t['a_only']} | {t['b_only']} | {t['p_holm']:.3f} |")
    return "\n".join(lines) + "\n"
