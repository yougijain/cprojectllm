"""Gold sampling, the label queue and freezing."""

from collections import Counter

import pytest

from jpe import gold
from scripts.label import parse_amount, parse_choice, parse_salary_input


def by_month():
    # Uneven months, as in the real corpus.
    return {f"2025-{m:02d}": list(range(m * 1000, m * 1000 + n)) for m, n in [(1, 200), (2, 400), (3, 300), (4, 100)]}


def test_sample_is_deterministic_and_disjoint():
    a = gold.sample_split(by_month(), seed=1)
    b = gold.sample_split(by_month(), seed=1)
    assert a.queues == b.queues
    dev, test, few = (set(a.queues[k]) for k in ("dev", "test", "fewshot"))
    assert not (dev & test) and not (few & (dev | test))
    assert len(a.queues["dev"]) == gold.TARGETS["dev"] + gold.RESERVE["dev"]
    assert len(a.queues["test"]) == gold.TARGETS["test"] + gold.RESERVE["test"]
    assert a.targets == {"dev": 50, "test": 250, "fewshot": gold.FEWSHOT_TARGET}


def test_sample_is_proportional_to_month_size():
    s = gold.sample_split(by_month(), seed=3)
    drawn = s.queues["dev"] + s.queues["test"]
    per_month = Counter(pid // 1000 for pid in drawn)
    need = len(drawn)
    for m, n in [(1, 200), (2, 400), (3, 300), (4, 100)]:
        assert abs(per_month[m] - need * n / 1000) <= 1


def test_different_seed_different_sample():
    assert gold.sample_split(by_month(), seed=1).queues != gold.sample_split(by_month(), seed=2).queues


def entry(pid, skip=False, **record):
    e = {"post_id": pid, "record": record or {"company": "x"}}
    if skip:
        e = {"post_id": pid, "skip": True}
    return e


def small_split():
    return gold.Split(0, ["2025-01"], {"dev": [1, 2, 3, 4], "test": [10, 11, 12, 13, 14, 15], "fewshot": [20, 21]}, {"dev": 2, "test": 4, "fewshot": 1})


def test_gold_ids_skips_non_posts_and_stops_at_target():
    s = small_split()
    entries = [entry(1), entry(2, skip=True), entry(3), entry(4)]
    assert gold.gold_ids(s, entries, "dev") == [1, 3]


def test_later_label_replaces_earlier():
    s = small_split()
    entries = [entry(1), entry(1, skip=True), entry(2), entry(3)]
    assert gold.gold_ids(s, entries, "dev") == [2, 3]
    assert gold.latest(entries)[1].get("skip")


def test_next_to_label_interleaves_and_respects_targets():
    s = small_split()
    queue = gold.next_to_label(s, [entry(1)])
    assert queue[0] == ("dev", 2)
    assert [pid for name, pid in queue if name == "dev"] == [2]  # dev needs one more
    assert [pid for name, pid in queue if name == "test"] == [10, 11, 12, 13]
    assert [n for n, _ in queue[:3]] == ["dev", "test", "test"]


def test_freeze_detects_any_edit(tmp_path):
    split_path, labels_path, frozen = tmp_path / "split.json", tmp_path / "labels.jsonl", tmp_path / "FROZEN"
    gold.save_split(small_split(), split_path)
    gold.append_label(entry(1), labels_path)
    kw = dict(split_path=split_path, labels_path=labels_path)
    assert gold.check_frozen(frozen, **kw) is None
    gold.freeze(frozen, **kw)
    assert gold.check_frozen(frozen, **kw) is True
    gold.append_label(entry(1, skip=True), labels_path)
    assert gold.check_frozen(frozen, **kw) is False


def test_committed_gold_has_not_changed_since_freezing():
    # Fails the build if anyone edits data/gold after `scripts/gold.py freeze`.
    assert gold.check_frozen() is not False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", (None, None, None)),
        ("120k 150k USD", (120_000, 150_000, "USD")),
        ("150k - usd", (150_000, None, "USD")),
        ("- 200,000 EUR", (None, 200_000, "EUR")),
        ("140000 USD", (140_000, 140_000, "USD")),
        ("120k 150k", (120_000, 150_000, None)),
    ],
)
def test_salary_input(raw, expected):
    assert parse_salary_input(raw) == expected


@pytest.mark.parametrize("raw", ["150k 120k USD", "- - USD", "lots", "1 2 3 USD"])
def test_salary_input_rejects(raw):
    with pytest.raises(ValueError):
        parse_salary_input(raw)


def test_choice_prefixes():
    assert parse_choice("h", ("yes", "no", "hybrid", "unspecified")) == "hybrid"
    assert parse_choice("", ("yes", "no", "unspecified")) == "unspecified"
    assert parse_choice("n", ("intern", "new_grad", "mid", "senior", "unspecified")) == "new_grad"
    assert parse_choice("x", ("yes", "no", "unspecified")) is None
    assert parse_amount("1.5k") == 1500
