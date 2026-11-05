"""The validation contract (jpe/validate.py). Skipped until it is written."""

import json

import pytest

from jpe.fields import EMPTY_RECORD, RECORD_KEYS
from jpe.validate import InvalidRecord, parse_record, validate_record

pytestmark = pytest.mark.core


def record(**changes):
    r = dict(EMPTY_RECORD, company="Acme", role_title="Backend Engineer")
    r.update(changes)
    return r


def test_valid_record_round_trips():
    r = record(remote="hybrid", salary_min=120000, salary_max=150000, salary_currency="USD", tech_stack=["Python", "Postgres"])
    out = validate_record(r)
    assert list(out) == list(RECORD_KEYS)
    assert out["remote"] == "hybrid"
    assert out["salary_min"] == 120000 and out["salary_max"] == 150000
    assert out is not r


def test_empty_record_is_valid():
    assert validate_record(dict(EMPTY_RECORD))["tech_stack"] == []


@pytest.mark.parametrize("key", ["company", "tech_stack", "salary_currency"])
def test_missing_key_is_invalid(key):
    r = record()
    del r[key]
    with pytest.raises(InvalidRecord):
        validate_record(r)


def test_extra_key_is_invalid():
    with pytest.raises(InvalidRecord):
        validate_record(record(equity="0.5%"))


@pytest.mark.parametrize(
    "key,value",
    [("seniority", "staff"), ("remote", "sometimes"), ("visa_sponsorship", "maybe"), ("new_grad_eligible", "junior")],
)
def test_value_outside_enum_is_invalid(key, value):
    with pytest.raises(InvalidRecord):
        validate_record(record(**{key: value}))


@pytest.mark.parametrize("value", ["120k", "120000", True])
def test_salary_must_be_a_number(value):
    with pytest.raises(InvalidRecord):
        validate_record(record(salary_min=value, salary_currency="USD"))


def test_salary_min_above_max_is_invalid():
    with pytest.raises(InvalidRecord):
        validate_record(record(salary_min=200000, salary_max=150000, salary_currency="USD"))


def test_currency_without_salary_is_invalid():
    with pytest.raises(InvalidRecord):
        validate_record(record(salary_currency="USD"))


@pytest.mark.parametrize("value", ["usd", "US$", "dollars"])
def test_currency_must_be_a_code(value):
    with pytest.raises(InvalidRecord):
        validate_record(record(salary_min=100000, salary_currency=value))


def test_open_ended_salary_is_valid():
    out = validate_record(record(salary_min=150000, salary_currency="USD"))
    assert out["salary_max"] is None


def test_tech_stack_must_be_a_list_of_strings():
    with pytest.raises(InvalidRecord):
        validate_record(record(tech_stack="Python, Go"))
    with pytest.raises(InvalidRecord):
        validate_record(record(tech_stack=["Python", ""]))


def test_not_a_dict_is_invalid():
    with pytest.raises(InvalidRecord):
        validate_record(["Acme"])


def test_parse_plain_and_fenced_json():
    r = record()
    assert parse_record(json.dumps(r))["company"] == "Acme"
    assert parse_record("```json\n" + json.dumps(r) + "\n```")["company"] == "Acme"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "Here is the JSON: {}",
        '{"company": "Acme"',  # truncated
        "not json at all",
    ],
)
def test_parse_rejects_non_json(raw):
    with pytest.raises(InvalidRecord):
        parse_record(raw)


def test_parse_rejects_prose_around_the_object():
    with pytest.raises(InvalidRecord):
        parse_record("Sure! " + json.dumps(record()) + " Hope that helps.")


def test_error_message_says_what_is_wrong():
    with pytest.raises(InvalidRecord) as exc:
        validate_record(record(remote="sometimes"))
    assert "remote" in str(exc.value)
