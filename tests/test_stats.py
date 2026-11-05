"""Intervals and paired tests, against hand-computed values."""

import pytest

from jpe.stats import bootstrap_mean_ci, cohen_kappa, holm, mcnemar_exact, paired_bootstrap_diff


def test_bootstrap_constant_has_no_width():
    assert bootstrap_mean_ci([1.0] * 50) == (1.0, 1.0, 1.0)


def test_bootstrap_interval_covers_mean_and_is_reproducible():
    vals = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 25
    mean, lo, hi = bootstrap_mean_ci(vals, n_boot=2000, seed=1)
    assert mean == pytest.approx(0.7)
    assert lo < 0.7 < hi
    # Normal approximation: sqrt(.7 * .3 / 250) = 0.029, so the 95% interval is about +-0.057.
    assert hi - lo == pytest.approx(2 * 1.96 * 0.029, abs=0.015)
    assert bootstrap_mean_ci(vals, n_boot=2000, seed=1) == (mean, lo, hi)


def test_bootstrap_empty():
    mean, lo, hi = bootstrap_mean_ci([])
    assert mean != mean and lo != lo  # NaN


def test_paired_diff():
    a = [1, 1, 0, 1] * 50
    assert paired_bootstrap_diff(a, a) == (0.0, 0.0, 0.0)
    d, lo, hi = paired_bootstrap_diff([1] * 100, [0] * 20 + [1] * 80, n_boot=2000)
    assert d == pytest.approx(0.2) and lo > 0
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 0], [1])


def test_mcnemar_exact_by_hand():
    # 9 posts only A gets right, 1 only B: p = 2 * (C(10,0) + C(10,1)) / 2^10.
    a = [True] * 9 + [False] * 1 + [True] * 5
    b = [False] * 9 + [True] * 1 + [True] * 5
    r = mcnemar_exact(a, b)
    assert (r["a_only"], r["b_only"]) == (9, 1)
    assert r["p"] == pytest.approx(22 / 1024)


def test_mcnemar_no_disagreements():
    assert mcnemar_exact([True, False], [True, False])["p"] == 1.0


def test_mcnemar_balanced_is_one():
    assert mcnemar_exact([True, False] * 5, [False, True] * 5)["p"] == pytest.approx(1.0)


def test_holm():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)  # monotone: never below the previous step


def test_cohen_kappa_by_hand():
    # Agreement 3/4; chance agreement .5*.25 + .5*.75 = .5; kappa = (.75 - .5) / .5.
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert cohen_kappa(["yes", "no"], ["yes", "no"]) == 1.0
    assert cohen_kappa(["yes"] * 4, ["yes"] * 4) == 1.0
