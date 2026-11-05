"""Open models run in-process with transformers, on a Colab T4.

For B2 and B3 (the base model zero- and few-shot) and M1 (base + LoRA
adapter) when no vLLM server is running. Greedy decoding. The base model
loads in fp16, the dtype the T4 supports (it has no bf16), so these numbers
are the unquantized reference that the 4-bit llama.cpp build is compared
against.

torch, transformers and peft are imported lazily: nothing else in the
project needs a GPU stack installed.
"""

from __future__ import annotations

import time
from pathlib import Path

from jpe.clients.base import Completion
from jpe.provenance import adapter_sha256


class HFLocalClient:
    def __init__(self, model: str, adapter: str | None = None, dtype: str = "float16", label: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model
        self.adapter = adapter
        self.adapter_hash = adapter_sha256(Path(adapter)) if adapter else None
        self.label = label or (f"{model}+{Path(adapter).name}" if adapter else model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype), device_map="auto")
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self._torch = torch

    @classmethod
    def from_config(cls, cfg: dict) -> "HFLocalClient":
        return cls(model=cfg["model"], adapter=cfg.get("adapter"), dtype=cfg.get("dtype", "float16"), label=cfg.get("label"))

    def identity(self) -> dict:
        return {"kind": "hf_local", "model": self.model_id, "adapter_sha256": self.adapter_hash, "decoding": "greedy"}

    def complete(self, messages: list[dict], max_tokens: int) -> Completion:
        return self.complete_batch([messages], max_tokens)[0]

    def complete_batch(self, batch: list[list[dict]], max_tokens: int) -> list[Completion]:
        torch = self._torch
        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        latency = (time.perf_counter() - started) / len(batch)
        width = enc["input_ids"].shape[1]
        results = []
        for i in range(len(batch)):
            new = out[i, width:]
            n_out = int((new != self.tokenizer.pad_token_id).sum())
            results.append(
                Completion(
                    text=self.tokenizer.decode(new, skip_special_tokens=True),
                    input_tokens=int(enc["attention_mask"][i].sum()),
                    output_tokens=n_out,
                    latency_s=latency,
                    model=self.label,
                    meta={"batched": len(batch)},
                )
            )
        return results
