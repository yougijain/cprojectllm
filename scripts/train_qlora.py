"""Fine-tune M1 with QLoRA on silver labels (on a Colab T4).

    python scripts/train_qlora.py --n-train all --seed 0 --out runs/m1/nall-s0

Train on the first N silver labels in pool order, evaluate on the 50 dev
posts (gold labels) after every epoch, and keep the epoch with the lowest
dev loss. The held-out 250 are never touched here.

Writes to --out:
    adapter/          the kept adapter
    train_log.json    loss history, the epoch kept, adapter hash, provenance
    loss.png          train and dev loss by epoch (if matplotlib is present)

The configuration itself is jpe/train_config.py and configs/train.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jpe import corpus, gold, silver  # noqa: E402
from jpe.prompts import load_prompt  # noqa: E402
from jpe.provenance import adapter_sha256, run_meta  # noqa: E402
from jpe.train_config import build_bnb_config, build_lora_config, build_sft_config, load_train_config  # noqa: E402


def to_dataset(prompt, pairs):
    """TRL's conversational prompt-completion format: the loss falls on the
    completion (the JSON record) and the prompt is context only."""
    from datasets import Dataset

    rows = [
        {"prompt": prompt.messages(text), "completion": [{"role": "assistant", "content": prompt.target(record)}]}
        for text, record in pairs
    ]
    return Dataset.from_list(rows)


def kept_epoch(log_history: list[dict]) -> dict | None:
    evals = [e for e in log_history if "eval_loss" in e]
    return min(evals, key=lambda e: e["eval_loss"]) if evals else None


def plot(log_history: list[dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    train = [(e["epoch"], e["loss"]) for e in log_history if "loss" in e]
    dev = [(e["epoch"], e["eval_loss"]) for e in log_history if "eval_loss" in e]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if train:
        ax.plot(*zip(*train), label="train", alpha=0.6)
    if dev:
        ax.plot(*zip(*dev), marker="o", label="dev (50 gold posts)")
        best = min(dev, key=lambda p: p[1])
        ax.axvline(best[0], ls="--", c="grey", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", default="all", help="number of silver examples, or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cfg = load_train_config()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTTrainer

    n = None if args.n_train == "all" else int(args.n_train)
    posts = corpus.load_corpus()
    train_pairs = [(posts[pid].text, rec) for pid, rec in silver.training_examples(n) if pid in posts]
    if n is not None and len(train_pairs) < n:
        print(f"only {len(train_pairs)} silver examples available for --n-train {n}", file=sys.stderr)
        return 1
    dev_pairs = [(posts[pid].text, rec) for pid, rec in sorted(gold.load_gold("dev").items())]

    prompt = load_prompt()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=build_bnb_config(cfg),
        dtype=torch.float16,
        device_map="auto",
    )
    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(cfg, str(args.out / "checkpoints"), args.seed),
        train_dataset=to_dataset(prompt, train_pairs),
        eval_dataset=to_dataset(prompt, dev_pairs),
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg),
    )

    started = time.time()
    trainer.train()
    seconds = time.time() - started

    adapter_dir = args.out / "adapter"
    trainer.save_model(str(adapter_dir))
    history = trainer.state.log_history
    best = kept_epoch(history)
    log = run_meta(
        args.out.name,
        {"kind": "qlora", "base_model": cfg["base_model"], "adapter_sha256": adapter_sha256(adapter_dir)},
        prompt.version,
        n_train=len(train_pairs),
        n_dev=len(dev_pairs),
        seed=args.seed,
        train_seconds=round(seconds),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        kept={"epoch": best and best["epoch"], "eval_loss": best and best["eval_loss"], "checkpoint": trainer.state.best_model_checkpoint},
        config=cfg,
        log_history=history,
    )
    (args.out / "train_log.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8", newline="\n")
    plot(history, args.out / "loss.png")
    print(f"kept epoch {log['kept']['epoch']} (dev loss {log['kept']['eval_loss']}) after {seconds / 60:.0f} min -> {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
