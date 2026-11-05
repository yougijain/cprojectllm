from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""
    # Anything else worth keeping about the call: stop reason, request id.
    meta: dict = field(default_factory=dict)
    cached: bool = False

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Completion":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class ChatClient(Protocol):
    # Everything that changes the output, as a stable dict: model, adapter
    # hash, decoding settings. The cache key and the results file use it.
    def identity(self) -> dict: ...

    def complete(self, messages: list[dict], max_tokens: int) -> Completion: ...
