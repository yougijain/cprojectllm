"""The scorer's contract (jpe/score.py). Skipped until it is written."""

import pytest

from jpe.fields import EMPTY_RECORD, SCORED_FIELDS
from jpe.score import salary_matches, score_record, stack_f1

pytestmark = pytest.mark.core


def rec(**kw):
    r = dict(EMPTY_RECORD, company="Acme", role_title="Backend Engineer", location="New York, NY")
    r.update(kw)
    return r


def salary(lo, hi, cur="USD"):
    return rec(salary_min=lo, salary_max=hi, salary_currency=cur)


def test_identical_records_score_one_everywhere():
    g = rec(remote="yes", seniority="senior", tech_stack=["python"], salary_min=150000, salary_max=180000, salary_currency="USD")
    s = score_record(dict(g), g)
    assert tuple(s) == SCORED_FIELDS
    assert all(v == 1.0 for v in s.values())


def test_failed_post_scores_zero_everywhere():
    s = score_record(None, rec())
    assert tuple(s) == SCORED_FIELDS
    assert all(v == 0.0 for v in s.values())


def test_scores_are_in_unit_interval():
    s = score_record(rec(remote="no", tech_stack=["go", "rust"]), rec(remote="yes", tech_stack=["go"]))
    assert all(0.0 <= v <= 1.0 for v in s.values())


@pytest.mark.parametrize("field,a,b", [("remote", "yes", "hybrid"), ("seniority", "mid", "senior"), ("visa_sponsorship", "no", "unspecified"), ("new_grad_eligible", "yes", "unspecified")])
def test_enums_are_exact_match(field, a, b):
    assert score_record(rec(**{field: a}), rec(**{field: a}))[field] == 1.0
    assert score_record(rec(**{field: a}), rec(**{field: b}))[field] == 0.0


def test_salary_within_five_percent():
    gold = salary(150_000, 200_000)
    assert salary_matches(salary(157_000, 200_000), gold)  # 4.7% off
    assert not salary_matches(salary(158_000, 200_000), gold)  # 5.3% off
    assert salary_matches(salary(150_000, 209_000), gold)
    assert not salary_matches(salary(150_000, 211_000), gold)


def test_salary_null_handling():
    none = salary(None, None, None)
    assert salary_matches(none, none)  # correct null
    assert not salary_matches(salary(150_000, 200_000), none)  # invented
    assert not salary_matches(none, salary(150_000, 200_000))  # missed
    assert salary_matches(salary(150_000, None), salary(150_000, None))
    assert not salary_matches(salary(150_000, 150_000), salary(150_000, None))  # open max vs point


def test_salary_currency_must_match():
    assert not salary_matches(salary(150_000, 200_000, "CAD"), salary(150_000, 200_000, "USD"))
    assert salary_matches(salary(150_000, 200_000, None), salary(150_000, 200_000, None))


def test_salary_field_uses_salary_matches():
    assert score_record(salary(150_000, 200_000), salary(150_000, 200_000))["salary"] == 1.0
    assert score_record(salary(100_000, 200_000), salary(150_000, 200_000))["salary"] == 0.0


def test_stack_f1():
    assert stack_f1(["python", "go"], ["python", "rust", "postgresql"]) == pytest.approx(0.4)
    assert stack_f1(["Golang", "Postgres"], ["go", "postgresql"]) == 1.0
    assert stack_f1(["python", "python"], ["python"]) == 1.0
    assert stack_f1([], []) == 1.0
    assert stack_f1([], ["python"]) == 0.0
    assert stack_f1(["python"], []) == 0.0
