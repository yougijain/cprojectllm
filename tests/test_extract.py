"""The extraction loop and the cache, with a scripted fake model."""

import json

import pytest

from jpe import extract
from jpe.cache import CachedClient, cache_key
from jpe.clients.base import Completion
from jpe.fields import EMPTY_RECORD
from jpe.prompts import load_prompt
from jpe.validate import InvalidRecord


class Scripted:
    """Replies from a list, in order, and records what it was sent."""

    def __init__(self, replies, model="fake"):
        self.replies = list(replies)
        self.sent = []
        self.model = model

    def identity(self):
        return {"kind": "fake", "model": self.model}

    def complete(self, messages, max_tokens):
        self.sent.append(messages)
        return Completion(text=self.replies.pop(0), input_tokens=100, output_tokens=40, latency_s=0.5, model=self.model)


def parse(raw):
    """A stand-in validator: JSON object with a company key."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        raise InvalidRecord("the reply is not JSON.")
    if not isinstance(obj, dict) or "company" not in obj:
        raise InvalidRecord("the key 'company' is missing.")
    return obj


GOOD = json.dumps(dict(EMPTY_RECORD, company="Acme"))
PROMPT = load_prompt()


def test_valid_first_reply():
    model = Scripted([GOOD])
    p = extract.extract_post(model, PROMPT, 1, "Acme | Engineer", parse=parse)
    assert p.status == extract.OK and p.record["company"] == "Acme"
    assert p.calls == 1 and p.outputs == [GOOD]


def test_retry_once_with_the_error_shown():
    model = Scripted(["Sure! Here it is", GOOD])
    p = extract.extract_post(model, PROMPT, 1, "Acme | Engineer", parse=parse)
    assert p.status == extract.OK_AFTER_RETRY
    retry = model.sent[1]
    assert retry[:-2] == model.sent[0]  # same conversation, extended
    assert retry[-2] == {"role": "assistant", "content": "Sure! Here it is"}
    assert "not JSON" in retry[-1]["content"]
    assert (p.calls, p.input_tokens, p.output_tokens) == (2, 200, 80)


def test_two_invalid_replies_fail_without_a_record():
    model = Scripted(["nope", '{"role": "x"}'])
    p = extract.extract_post(model, PROMPT, 1, "Acme | Engineer", parse=parse)
    assert p.status == extract.FAILED and p.record is None
    assert "company" in p.error
    assert len(p.outputs) == 2


def test_never_more_than_one_retry():
    model = Scripted(["a", "b", GOOD])
    extract.extract_post(model, PROMPT, 1, "x", parse=parse)
    assert len(model.replies) == 1  # the third reply was never asked for


def test_run_keeps_order_with_workers():
    model = Scripted([GOOD] * 6)
    preds = extract.run(model, PROMPT, [(i, f"post {i}") for i in range(6)], workers=3, parse=parse)
    assert sorted(p.post_id for p in preds) == list(range(6))


def test_cache_hits_on_repeat_and_misses_on_change(tmp_path):
    inner = Scripted([GOOD, GOOD])
    cached = CachedClient(inner, tmp_path)
    msgs = PROMPT.messages("Acme | Engineer")
    a = cached.complete(msgs, 512)
    b = cached.complete(msgs, 512)
    assert not a.cached and b.cached and b.text == a.text
    assert (cached.hits, cached.misses) == (1, 1)
    cached.complete(PROMPT.messages("Beta | Designer"), 512)
    assert cached.misses == 2 and not inner.replies


def test_cache_key_depends_on_model_and_prompt():
    msgs = PROMPT.messages("x")
    assert cache_key({"model": "a"}, msgs, 512) != cache_key({"model": "b"}, msgs, 512)
    assert cache_key({"model": "a"}, msgs, 512) != cache_key({"model": "a"}, msgs, 256)
    assert cache_key({"model": "a"}, msgs, 512) == cache_key({"model": "a"}, PROMPT.messages("x"), 512)


def test_prompt_version_tracks_content():
    base = load_prompt()
    shot = load_prompt(fewshot=[("Acme | Engineer", dict(EMPTY_RECORD, company="Acme"))])
    assert base.version.startswith("extract_v1@")
    assert shot.version.startswith("extract_v1+fewshot@") and shot.version != base.version
    msgs = shot.messages("Beta | Designer")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]


def test_predictions_round_trip(tmp_path):
    preds = [extract.Prediction(2, extract.OK, {"company": "B"}), extract.Prediction(1, extract.FAILED, None, error="x")]
    path = tmp_path / "p.json"
    extract.write_predictions(path, {"run": "t"}, preds)
    meta, by_id = extract.read_predictions(path)
    assert meta == {"run": "t"} and list(by_id) == [1, 2]
    assert by_id[1]["record"] is None


@pytest.mark.parametrize("n", [1, 3])
def test_run_batched_retries_one_by_one(n):
    class Batched(Scripted):
        def complete_batch(self, batch, max_tokens):
            return [self.complete(m, max_tokens) for m in batch]

    model = Batched(["bad"] + [GOOD] * n)
    preds = extract.run_batched(model, PROMPT, [(i, "x") for i in range(n)], batch_size=n, parse=parse)
    assert preds[0].status == extract.OK_AFTER_RETRY
    assert all(p.status != extract.FAILED for p in preds)
