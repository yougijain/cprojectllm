"""The extraction loop: one post in, one validated record (or a failure) out.

Every reply goes through jpe.validate.parse_record. If it is invalid the
model gets exactly one retry, with the validator's complaint appended to the
conversation. Retrying the same messages would be pointless: decoding is
greedy, so the model would say the same thing again (and the cache would
hand back the same reply without asking). If the retry fails too, the post
is marked failed with no record. Nothing here ever fills in a field.

Every prediction keeps the raw replies, token counts and latency, so error
analysis and cost can be worked out later without rerunning anything.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from jpe.clients.base import ChatClient
from jpe.prompts import Prompt
from jpe.validate import InvalidRecord, parse_record

# A record is about 150 tokens of JSON. 512 leaves room for a long stack
# without letting a model that never stops run for thousands of tokens.
MAX_TOKENS = 512

RETRY_MESSAGE = (
    "Your reply could not be used: {error} "
    "Reply again with only the corrected JSON object, with exactly the keys listed."
)

OK, OK_AFTER_RETRY, FAILED = "ok", "ok_after_retry", "failed"


@dataclass
class Prediction:
    post_id: int
    status: str
    record: dict | None
    error: str | None = None
    outputs: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    calls: int = 0
    cached: bool = False

    def to_json(self) -> dict:
        return asdict(self)


def extract_post(
    client: ChatClient,
    prompt: Prompt,
    post_id: int,
    text: str,
    parse: Callable[[str], dict] = parse_record,
    max_tokens: int = MAX_TOKENS,
) -> Prediction:
    messages = prompt.messages(text)
    pred = Prediction(post_id=post_id, status=FAILED, record=None)

    def call(msgs):
        c = client.complete(msgs, max_tokens)
        pred.outputs.append(c.text)
        pred.input_tokens += c.input_tokens
        pred.output_tokens += c.output_tokens
        pred.latency_s += c.latency_s
        pred.calls += 1
        pred.cached = pred.cached or c.cached
        return c.text

    first = call(messages)
    try:
        pred.record, pred.status = parse(first), OK
        return pred
    except InvalidRecord as e:
        error = str(e) or "the reply is not a valid record."

    retry = messages + [
        {"role": "assistant", "content": first},
        {"role": "user", "content": RETRY_MESSAGE.format(error=error)},
    ]
    second = call(retry)
    try:
        pred.record, pred.status = parse(second), OK_AFTER_RETRY
    except InvalidRecord as e:
        pred.error = str(e) or error
    return pred


def run(
    client: ChatClient,
    prompt: Prompt,
    posts: Iterable[tuple[int, str]],
    workers: int = 1,
    parse: Callable[[str], dict] = parse_record,
    progress: Callable[[int, int], None] | None = None,
) -> list[Prediction]:
    """Extract every post. `workers` > 1 runs calls concurrently, for API
    and server clients; in-process models should use run_batched."""
    posts = list(posts)
    done = 0

    def one(item):
        nonlocal done
        pid, text = item
        p = extract_post(client, prompt, pid, text, parse=parse)
        done += 1
        if progress:
            progress(done, len(posts))
        return p

    if workers <= 1:
        return [one(item) for item in posts]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, posts))


def run_batched(client, prompt: Prompt, posts: Iterable[tuple[int, str]], batch_size: int = 8, parse=parse_record) -> list[Prediction]:
    """For in-process models: first attempts in batches, retries one by one.

    Batching only changes speed; the messages, and so the outputs, are the
    same as in `run`.
    """
    posts = list(posts)
    preds: list[Prediction] = []
    for start in range(0, len(posts), batch_size):
        chunk = posts[start:start + batch_size]
        replies = client.complete_batch([prompt.messages(t) for _, t in chunk], MAX_TOKENS)
        for (pid, text), c in zip(chunk, replies):
            p = Prediction(pid, FAILED, None, outputs=[c.text], input_tokens=c.input_tokens,
                           output_tokens=c.output_tokens, latency_s=c.latency_s, calls=1)
            try:
                p.record, p.status = parse(c.text), OK
            except InvalidRecord as e:
                retry = prompt.messages(text) + [
                    {"role": "assistant", "content": c.text},
                    {"role": "user", "content": RETRY_MESSAGE.format(error=str(e))},
                ]
                c2 = client.complete(retry, MAX_TOKENS)
                p.outputs.append(c2.text)
                p.input_tokens += c2.input_tokens
                p.output_tokens += c2.output_tokens
                p.latency_s += c2.latency_s
                p.calls += 1
                try:
                    p.record, p.status = parse(c2.text), OK_AFTER_RETRY
                except InvalidRecord as e2:
                    p.error = str(e2)
            preds.append(p)
    return preds


def write_predictions(path: Path, meta: dict, preds: list[Prediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"meta": meta, "predictions": [p.to_json() for p in sorted(preds, key=lambda p: p.post_id)]}
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_predictions(path: Path) -> tuple[dict, dict[int, dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["meta"], {int(p["post_id"]): p for p in data["predictions"]}
