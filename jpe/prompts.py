"""Prompts, versioned by content.

A prompt's version is its file name plus a hash of its text
("extract_v1@3f9a2c1d"), so editing a prompt without renaming it still
changes the version recorded in every results file, and the LLM cache
misses instead of returning answers to the old wording.

Every model sees the same prompt: B2 (zero-shot), B3 (few-shot) and M1
(fine-tuned) differ only in examples and weights, and B4 only in the model.

B3's five examples are hand-labeled posts from outside the gold set (the
"fewshot" queue in data/gold/split.json). They are built from the labels
and the fetched posts at run time, so no post text is committed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from jpe.corpus import ROOT

PROMPTS_DIR = ROOT / "prompts"

# A guard, not a cut that happens: the longest of the 7,791 posts is 4,764
# characters (the median is about 1,000). It keeps a future outlier from
# blowing a small model's context; 6,000 characters is about 1,500 tokens.
MAX_POST_CHARS = 6000


@dataclass(frozen=True)
class Prompt:
    name: str
    system: str
    fewshot: tuple[tuple[str, dict], ...] = field(default=())

    @property
    def version(self) -> str:
        h = hashlib.sha256(self.system.encode())
        for text, record in self.fewshot:
            h.update(text.encode())
            h.update(json.dumps(record, sort_keys=True).encode())
        suffix = "+fewshot" if self.fewshot else ""
        return f"{self.name}{suffix}@{h.hexdigest()[:8]}"

    def user_message(self, post_text: str) -> str:
        text = post_text if len(post_text) <= MAX_POST_CHARS else post_text[:MAX_POST_CHARS] + "\n[post truncated]"
        return f"Job post:\n\n{text}"

    def messages(self, post_text: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.system}]
        for text, record in self.fewshot:
            msgs.append({"role": "user", "content": self.user_message(text)})
            msgs.append({"role": "assistant", "content": json.dumps(record, ensure_ascii=False)})
        msgs.append({"role": "user", "content": self.user_message(post_text)})
        return msgs

    def target(self, record: dict) -> str:
        """The assistant reply a fine-tuned model is trained to produce."""
        return json.dumps(record, ensure_ascii=False)


def load_prompt(name: str = "extract_v1", fewshot: list[tuple[str, dict]] | None = None) -> Prompt:
    system = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
    return Prompt(name=name, system=system, fewshot=tuple(fewshot or ()))
