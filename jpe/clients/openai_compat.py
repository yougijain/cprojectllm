"""Any server that speaks the /v1/chat/completions protocol.

Used for the open-weight models only: vLLM on the T4, llama.cpp's
llama-server on a laptop or a CI runner, and a hosted open-weight teacher.
Decoding is greedy (temperature 0) so a rerun reproduces its outputs.
"""

from __future__ import annotations

import os
import time

import requests

from jpe.clients.base import Completion


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        seed: int | None = 0,
        json_schema: dict | None = None,
        timeout: float = 300.0,
        retries: int = 3,
        label: str | None = None,
        artifact_sha256: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.json_schema = json_schema
        self.timeout = timeout
        self.retries = retries
        self.label = label or model
        # Hash of the weights behind the server (adapter dir or GGUF file),
        # so the cache cannot serve answers from a previous build.
        self.artifact_sha256 = artifact_sha256
        self.session = requests.Session()

    @classmethod
    def from_config(cls, cfg: dict) -> "OpenAICompatClient":
        if not cfg.get("base_url"):
            raise ValueError(f"{cfg.get('model')}: set base_url in configs/models.yaml")
        key = os.environ.get(cfg["api_key_env"]) if cfg.get("api_key_env") else None
        schema = None
        if cfg.get("constrained"):
            from jpe.fields import JSON_SCHEMA

            schema = JSON_SCHEMA
        artifact = None
        if cfg.get("artifact"):
            from pathlib import Path

            from jpe.provenance import adapter_sha256, file_sha256

            path = Path(cfg["artifact"])
            artifact = adapter_sha256(path) if path.is_dir() else file_sha256(path)
        return cls(
            base_url=cfg["base_url"],
            model=cfg["model"],
            api_key=key,
            temperature=cfg.get("temperature", 0.0),
            seed=cfg.get("seed", 0),
            json_schema=schema,
            label=cfg.get("label"),
            artifact_sha256=artifact,
        )

    def identity(self) -> dict:
        return {
            "kind": "openai_compat",
            "model": self.label,
            "temperature": self.temperature,
            "seed": self.seed,
            "constrained": self.json_schema is not None,
            "artifact_sha256": self.artifact_sha256,
        }

    def complete(self, messages: list[dict], max_tokens: int) -> Completion:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if self.json_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "job_post", "schema": self.json_schema}}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        delay = 2.0
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                r = self.session.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    raise requests.HTTPError(f"{r.status_code}", response=r)
                r.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
                if attempt == self.retries:
                    raise
                time.sleep(delay)
                delay *= 2
        latency = time.perf_counter() - started
        data = r.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return Completion(
            text=choice["message"].get("content") or "",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_s=latency,
            model=data.get("model", self.model),
            meta={"finish_reason": choice.get("finish_reason")},
        )
