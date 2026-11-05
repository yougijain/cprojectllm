"""Model clients. Each one turns chat messages into a Completion.

    openai_compat  vLLM, llama.cpp's llama-server, and hosted open-weight
                   models (the teacher): anything speaking /v1/chat/completions
    anthropic      B4, the frontier baseline, through the Anthropic SDK
    hf_local       transformers on a Colab GPU, with or without a LoRA adapter
"""

from __future__ import annotations

from jpe.clients.base import ChatClient, Completion


def make_client(cfg: dict) -> ChatClient:
    """Build a client from one entry of configs/models.yaml."""
    kind = cfg["kind"]
    if kind == "openai_compat":
        from jpe.clients.openai_compat import OpenAICompatClient

        return OpenAICompatClient.from_config(cfg)
    if kind == "anthropic":
        from jpe.clients.anthropic_client import AnthropicClient

        return AnthropicClient.from_config(cfg)
    if kind == "hf_local":
        from jpe.clients.hf_local import HFLocalClient

        return HFLocalClient.from_config(cfg)
    raise ValueError(f"unknown client kind {kind!r}")


__all__ = ["ChatClient", "Completion", "make_client"]
