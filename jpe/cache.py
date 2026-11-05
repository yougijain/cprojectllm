"""A disk cache in front of every model call, so reruns cost nothing.

The key is everything that changes the output: the client's identity (model,
adapter hash, decoding settings), the exact messages and the token budget.
Changing the prompt or the adapter therefore misses, and repeating a run
hits. Entries are plain JSON under .cache/llm/, which is not committed: it
holds post text and API responses.

Benchmarks bypass the cache, since a cache hit has no latency to measure.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from jpe.clients.base import ChatClient, Completion

CACHE_DIR = Path(os.environ.get("JPE_CACHE", Path(__file__).resolve().parent.parent / ".cache" / "llm"))


def cache_key(identity: dict, messages: list[dict], max_tokens: int) -> str:
    payload = json.dumps({"id": identity, "messages": messages, "max_tokens": max_tokens}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class CachedClient:
    def __init__(self, inner: ChatClient, cache_dir: Path = CACHE_DIR):
        self.inner = inner
        self.dir = Path(cache_dir)
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def identity(self) -> dict:
        return self.inner.identity()

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def complete(self, messages: list[dict], max_tokens: int) -> Completion:
        key = cache_key(self.identity(), messages, max_tokens)
        path = self._path(key)
        if path.exists():
            with self._lock:
                self.hits += 1
            c = Completion.from_json(json.loads(path.read_text(encoding="utf-8")))
            c.cached = True
            return c
        c = self.inner.complete(messages, max_tokens)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{threading.get_ident()}")
        tmp.write_text(json.dumps(c.to_json(), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        with self._lock:
            self.misses += 1
        return c

    def __getattr__(self, name):
        # complete_batch and anything else client-specific passes through.
        return getattr(self.inner, name)
