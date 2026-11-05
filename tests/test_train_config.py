"""The QLoRA configuration's contract (jpe/train_config.py).

Needs transformers, peft and trl, so it runs on Colab (the notebook runs it
before training) and is skipped where they are not installed. It is also
skipped until the builders are written.
"""

import pytest

from jpe.train_config import TRAIN_FILE, build_bnb_config, build_lora_config, build_sft_config, load_train_config

pytestmark = pytest.mark.core

CFG = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "max_seq_length": 1536,
    "lora": {"r": 8, "alpha": 16, "dropout": 0.1, "target_modules": ["q_proj", "v_proj"]},
    "training": {
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.0,
    },
    "seeds": [0, 1, 2],
    "learning_curve": [250, "all"],
}


def test_undecided_values_are_refused(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text(TRAIN_FILE.read_text(encoding="utf-8").replace("max_seq_length: null", "max_seq_length: 1536"))
    with pytest.raises(ValueError, match="lora.r"):
        load_train_config(path)


def test_bnb_is_4bit_nf4_fp16():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    bnb = build_bnb_config(CFG)
    assert bnb.load_in_4bit
    assert bnb.bnb_4bit_quant_type == "nf4"
    assert bnb.bnb_4bit_compute_dtype == torch.float16  # the T4 has no bf16


def test_lora_takes_its_values_from_the_config():
    peft = pytest.importorskip("peft")
    lora = build_lora_config(CFG)
    assert lora.task_type == peft.TaskType.CAUSAL_LM
    assert (lora.r, lora.lora_alpha, lora.lora_dropout) == (8, 16, 0.1)
    assert set(lora.target_modules) == {"q_proj", "v_proj"}


def test_sft_keeps_the_best_epoch_by_dev_loss(tmp_path):
    pytest.importorskip("trl")
    args = build_sft_config(CFG, str(tmp_path), seed=2)
    assert args.output_dir == str(tmp_path)
    assert args.seed == 2
    assert args.fp16 and not args.bf16
    assert str(args.eval_strategy).endswith("epoch") and str(args.save_strategy).endswith("epoch")
    assert args.load_best_model_at_end
    assert args.metric_for_best_model in ("eval_loss", "loss")
    assert args.greater_is_better is False
    assert args.learning_rate == 1e-4 and args.num_train_epochs == 3
    assert args.completion_only_loss is not False
