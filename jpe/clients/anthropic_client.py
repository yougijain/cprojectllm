"""B4: a frontier model through the Anthropic API.

B4 is the quality ceiling and the cost comparison. It gets the same prompt
as the open models, zero-shot, with unconstrained output, so its JSON
validity is measured on the same terms as theirs. Its outputs are used for
comparison only: Anthropic's terms do not allow training models on them,
and the silver labels come from an open-weight teacher instead.

Claude Opus 5.5 thinks before answering on every request (it cannot be
turned off); `effort` sets how much. Thinking tokens are billed as output
and are counted in output_tokens, so the cost per 1,000 posts includes
them. No sampling parameters are sent: the current models reject them.

No refusal fallback is configured. A fallback would quietly answer some
posts with a different model, and B4 has to be one model. A refusal (very
unlikely on job posts) comes back as an empty reply with
stop_reason "refusal", fails validation, and is counted as a failure.
"""

from __future__ import annotations

import time

import anthropic

from jpe.clients.base import Completion


class AnthropicClient:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 16000):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        # Credentials from ANTHROPIC_API_KEY or an `ant auth login` profile.
        self.client = anthropic.Anthropic()

    @classmethod
    def from_config(cls, cfg: dict) -> "AnthropicClient":
        return cls(model=cfg["model"], effort=cfg.get("effort", "medium"), max_tokens=cfg.get("max_tokens", 16000))

    def identity(self) -> dict:
        return {"kind": "anthropic", "model": self.model, "effort": self.effort}

    def complete(self, messages: list[dict], max_tokens: int) -> Completion:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        started = time.perf_counter()
        # max_tokens covers thinking as well as the answer, so the caller's
        # budget (sized for a JSON object) is raised to the configured one.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max(max_tokens, self.max_tokens),
            system=system,
            messages=turns,
            output_config={"effort": self.effort},
        )
        latency = time.perf_counter() - started
        text = "".join(block.text for block in response.content if block.type == "text")
        meta = {"stop_reason": response.stop_reason, "request_id": response._request_id}
        if response.stop_reason == "refusal" and response.stop_details:
            meta["refusal_category"] = response.stop_details.category
        return Completion(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=latency,
            model=response.model,
            meta=meta,
        )
