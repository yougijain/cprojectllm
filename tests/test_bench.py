"""The benchmark's arithmetic, with a fake server."""

import pytest

from jpe.clients.base import Completion
from scripts.bench_serving import bench, summarize


def test_summarize():
    lat = [1.0] * 19 + [3.0]
    r = summarize(lat, wall_s=10.0, in_tok=[800] * 20, out_tok=[150] * 20, concurrency=4)
    assert r["throughput_posts_per_s"] == 2.0
    assert r["latency_p50_s"] == 1.0
    assert r["latency_p95_s"] == pytest.approx(1.1)  # linear interpolation between the 19th and 20th
    assert r["output_tokens_per_s"] == 300.0


class Fake:
    def complete(self, messages, max_tokens):
        return Completion(text="{}", input_tokens=10, output_tokens=5, latency_s=0.25)


def test_bench_counts_every_post():
    r = bench(Fake(), [[{"role": "user", "content": "x"}]] * 12, concurrency=3)
    assert r["n"] == 12 and r["latency_p50_s"] == 0.25 and r["concurrency"] == 3
