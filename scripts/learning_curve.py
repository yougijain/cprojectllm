"""Train M1 at each training-set size and seed, then predict dev and test.

    # The learning curve: every size, one seed.
    python scripts/learning_curve.py --sizes 250 500 1000 2000 all --seeds 0 \
        --runs-dir /content/drive/MyDrive/cprojectllm/runs

    # The final configuration, three seeds.
    python scripts/learning_curve.py --sizes all --seeds 0 1 2 --runs-dir ...

Resumable: a run whose adapter already exists is not retrained, and its
predictions are not redone if they exist. That matters on Colab, where a
session can end mid-way; point --runs-dir at Google Drive.

Predictions land in results/predictions/<split>/M1-n<size>-s<seed>.json.
The all-data, seed-0 adapter is copied to runs/m1/final/adapter, the path
configs/models.yaml calls M1.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jpe import extract  # noqa: E402
from jpe.clients.hf_local import HFLocalClient  # noqa: E402
from jpe.prompts import load_prompt  # noqa: E402
from jpe.provenance import run_meta  # noqa: E402
from jpe.train_config import load_train_config  # noqa: E402
from scripts.run_model import RESULTS, split_posts  # noqa: E402


def run_name(size: str, seed: int) -> str:
    return f"M1-n{size}-s{seed}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", default=None, help="default: learning_curve in configs/train.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs" / "m1")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args(argv)

    cfg = load_train_config()
    sizes = [str(s) for s in (args.sizes or cfg["learning_curve"])]
    prompt = load_prompt()

    for size in sizes:
        for seed in args.seeds:
            name = run_name(size, seed)
            out = args.runs_dir / f"n{size}-s{seed}"
            adapter = out / "adapter"
            if not (adapter / "adapter_config.json").exists():
                print(f"== training {name}", flush=True)
                subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / "train_qlora.py"), "--n-train", size, "--seed", str(seed), "--out", str(out)],
                    check=True,
                )
            todo = [s for s in ("dev", "test") if not (RESULTS / s / f"{name}.json").exists()]
            if todo:
                client = HFLocalClient(cfg["base_model"], adapter=str(adapter), label=name)
                for split in todo:
                    print(f"== predicting {name} on {split}", flush=True)
                    preds = extract.run_batched(client, prompt, split_posts(split), batch_size=args.batch_size)
                    meta = run_meta(name, client.identity(), prompt.version, split=split, n_posts=len(preds),
                                    json_applies=True, n_train=size, seed=seed)
                    extract.write_predictions(RESULTS / split / f"{name}.json", meta, preds)
                del client
            if size == "all" and seed == 0:
                final = ROOT / "runs" / "m1" / "final" / "adapter"
                if final.resolve() != adapter.resolve():
                    shutil.copytree(adapter, final, dirs_exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
