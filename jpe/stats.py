"""Intervals and paired tests.

Every model is scored on the same 250 posts, so comparisons are paired:
resample posts, not models, and test the disagreements, not the totals.
"""

from __future__ import annotations

from collections import Counter
from math import comb
from typing import Sequence

import numpy as np

N_BOOT = 10_000


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap (1 - alpha) interval, resampling items."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """mean(a) - mean(b) with an interval, resampling the same posts for both."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must be the same length")
    return bootstrap_mean_ci(a - b, n_boot=n_boot, alpha=alpha, seed=seed)


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Exact (binomial) McNemar test on paired right/wrong outcomes.

    Only the posts where exactly one model is right carry information:
    `a_only` and `b_only`. Under the null each is equally likely, so the
    two-sided p-value is the binomial tail at 1/2. Exact rather than
    chi-squared because the disagreement counts here are small.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("paired samples must be the same length")
    a_only = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    b_only = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return {"a_only": a_only, "b_only": b_only, "p": p}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, for testing many fields at once."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[key] = running
    return out


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa for two annotators' categorical labels."""
    if len(a) != len(b):
        raise ValueError("label lists must be the same length")
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)
