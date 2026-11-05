"""Throughput and latency of a served model.

    # vLLM on a Colab T4 (see docs/serving.md to start the server)
    python scripts/bench_serving.py M1-vllm --hardware T4 --concurrency 1 8 32

    # the 4-bit llama.cpp build on a laptop
    python scripts/bench_serving.py M1-q4 --hardware "laptop CPU: <model>, <cores>" --concurrency 1

Sends every post in the split's queue to the server at each concurrency
level, uncached, after a short warm-up. Latency is per request (one
extraction, first attempt only); throughput is posts finished per second of
wall time. Writes results/serving/<model>.json; the cost per 1,000 posts in
the results table uses the best T4 throughput from it.

Accuracy is measured separately (scripts/run_model.py), so the numbers here
are speed only.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jpe.clients import make_client  # noqa: E402
from jpe.config import model_config  # noqa: E402
from jpe.extract import MAX_TOKENS  # noqa: E402
from jpe.prompts import load_prompt  # noqa: E402
from jpe.provenance import run_meta  # noqa: E402
from scripts.run_model import split_posts  # noqa: E402

OUT = ROOT / "results" / "serving"
WARMUP = 3


def summarize(latencies: list[float], wall_s: float, in_tok: list[int], out_tok: list[int], concurrency: int) -> dict:
    lat = np.asarray(latencies)
    return {
        "concurrency": concurrency,
        "n": int(lat.size),
        "wall_s": round(wall_s, 2),
        "throughput_posts_per_s": round(lat.size / wall_s, 3) if wall_s > 0 else None,
        "latency_p50_s": round(float(np.percentile(lat, 50)), 3),
        "latency_p95_s": round(float(np.percentile(lat, 95)), 3),
        "mean_input_tokens": round(float(np.mean(in_tok)), 1),
        "mean_output_tokens": round(float(np.mean(out_tok)), 1),
        "output_tokens_per_s": round(float(np.sum(out_tok)) / wall_s, 1) if wall_s > 0 else None,
    }


def bench(client, messages: list[list[dict]], concurrency: int) -> dict:
    def one(m):
        c = client.complete(m, MAX_TOKENS)
        return c.latency_s, c.input_tokens, c.output_tokens

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, messages))
    wall = time.perf_counter() - started
    lat, inp, out = zip(*results)
    return summarize(list(lat), wall, list(inp), list(out), concurrency)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="an openai_compat entry in configs/models.yaml")
    ap.add_argument("--hardware", required=True, help='"T4", or a description of the machine')
    ap.add_argument("--split", default="test")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1])
    ap.add_argument("--limit", type=int, help="only the first N posts (quick checks)")
    args = ap.parse_args(argv)

    cfg = model_config(args.model)
    if cfg["kind"] != "openai_compat":
        print("benchmark a served model (kind: openai_compat)", file=sys.stderr)
        return 1
    client = make_client(cfg)  # deliberately uncached
    prompt = load_prompt()
    posts = split_posts(args.split)[: args.limit]
    messages = [prompt.messages(text) for _, text in posts]

    for m in messages[:WARMUP]:
        client.complete(m, MAX_TOKENS)

    runs = []
    for c in args.concurrency:
        r = bench(client, messages, c)
        runs.append(r)
        print(f"concurrency {c:3d}: {r['throughput_posts_per_s']} posts/s, p50 {r['latency_p50_s']} s, p95 {r['latency_p95_s']} s")

    result = {
        "model": args.model,
        "hardware": args.hardware,
        "host": {"platform": platform.platform(), "processor": platform.processor()},
        "runs": runs,
        "throughput_posts_per_s": max(r["throughput_posts_per_s"] or 0 for r in runs),
        "latency_p50_s_single": next((r["latency_p50_s"] for r in runs if r["concurrency"] == 1), None),
        "latency_p95_s_single": next((r["latency_p95_s"] for r in runs if r["concurrency"] == 1), None),
        "meta": run_meta(args.model, client.identity(), prompt.version, split=args.split),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.model}.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
