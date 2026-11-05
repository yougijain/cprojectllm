"""The QLoRA training configuration, built from configs/train.yaml.

The three build_* functions are written by hand (see the README's build
rules); tests/test_train_config.py pins what the hardware and the method
fix. scripts/train_qlora.py is the plumbing around them: it loads the
model with build_bnb_config, hands build_lora_config and build_sft_config
to TRL's SFTTrainer, and saves the adapter and the loss history.

Fixed by the setup, and so by the tests:
  * A Colab T4 has no bf16 support. Compute in fp16, and do not enable bf16.
  * QLoRA means the frozen base weights are 4-bit NF4.
  * Evaluate on the dev posts after every epoch, save a checkpoint each
    epoch, and keep the one with the lowest dev loss (load_best_model_at_end
    on eval_loss, lower is better). Text vs. Timing's BERT was best after
    one epoch and worse after; this is what stops that being shipped.
  * The seed passed in is the seed used.
  * Loss only on the answer (the JSON), not on the prompt.
Everything else (rank, alpha, dropout, target modules, learning rate,
schedule, batch size) is a choice, made on the dev set.

transformers, peft and trl are imported inside the functions, so the
rest of the project does not need them installed.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "configs" / "train.yaml"


def _nulls(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if v is None:
            yield key
        elif isinstance(v, dict):
            yield from _nulls(v, key + ".")


def load_train_config(path: Path = TRAIN_FILE) -> dict:
    """The YAML, refusing to return while any value is still undecided."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = list(_nulls(cfg))
    if missing:
        raise ValueError(f"{path} has values still to choose: {', '.join(missing)}")
    return cfg


def build_bnb_config(cfg: dict):
    """A transformers.BitsAndBytesConfig for the 4-bit base model.

    Must: load in 4 bit, NF4 quantization, fp16 compute dtype.
    """
    raise NotImplementedError("jpe.train_config.build_bnb_config")


def build_lora_config(cfg: dict):
    """A peft.LoraConfig from cfg["lora"].

    Must: task type causal LM; r, lora_alpha, lora_dropout and
    target_modules taken from cfg["lora"].
    """
    raise NotImplementedError("jpe.train_config.build_lora_config")


def build_sft_config(cfg: dict, output_dir: str, seed: int):
    """A trl.SFTConfig from cfg["training"] and cfg["max_seq_length"].

    Must: output_dir and seed as given; fp16 on and bf16 off; evaluation
    and checkpointing every epoch; load_best_model_at_end on eval_loss with
    greater_is_better False; loss on the completion only; the learning
    rate and epoch count from cfg. The dataset uses TRL's conversational
    prompt-completion format (see scripts/train_qlora.py), so the chat
    template is applied by TRL.
    """
    raise NotImplementedError("jpe.train_config.build_sft_config")
