"""Cost per 1,000 posts.

API model: what the tokens it actually used cost at list price, retries
and thinking included. Own model: GPU-hours at the measured throughput,
times what renting that GPU costs. Both are per post actually processed,
so a model that needs retries pays for them.
"""

from __future__ import annotations

from typing import Iterable


def api_cost_per_1k(predictions: Iterable[dict], input_per_mtok: float, output_per_mtok: float) -> float:
    preds = list(predictions)
    if not preds:
        return float("nan")
    usd = sum(p["input_tokens"] * input_per_mtok + p["output_tokens"] * output_per_mtok for p in preds) / 1e6
    return usd / len(preds) * 1000


def gpu_cost_per_1k(posts_per_second: float, usd_per_hour: float) -> float:
    if posts_per_second <= 0:
        return float("nan")
    hours = 1000 / posts_per_second / 3600
    return hours * usd_per_hour


def tokens_per_post(predictions: Iterable[dict]) -> tuple[float, float]:
    preds = list(predictions)
    if not preds:
        return float("nan"), float("nan")
    return (sum(p["input_tokens"] for p in preds) / len(preds), sum(p["output_tokens"] for p in preds) / len(preds))
