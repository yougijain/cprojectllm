"""Merge M1's adapter into the base model and build a 4-bit GGUF for llama.cpp.

    python scripts/export_gguf.py --adapter runs/m1/final/adapter --llama-cpp ~/src/llama.cpp

Steps, each skipped if its output already exists:
  1. merge the LoRA adapter into the fp16 base weights (peft merge_and_unload)
  2. convert to an fp16 GGUF with llama.cpp's convert_hf_to_gguf.py
  3. quantize to Q4_K_M with llama-quantize
Writes models/m1-merged/, models/m1-f16.gguf, models/m1-q4_k_m.gguf and
models/m1-q4_k_m.json (hashes, sizes and versions, so a GGUF can always be
traced to its adapter).

Needs torch, transformers and peft (requirements-train.txt) and a built
llama.cpp checkout. The merge fits on a CPU: 1.5B parameters in fp16 is
about 3 GB.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jpe.provenance import adapter_sha256, file_sha256  # noqa: E402

QUANT = "Q4_K_M"


def merge(base: str, adapter: Path, out: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def find_tool(llama_cpp: Path, name: str) -> Path:
    for p in (llama_cpp / "build" / "bin" / name, llama_cpp / "build" / "bin" / "Release" / f"{name}.exe", llama_cpp / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found under {llama_cpp}; build llama.cpp first (cmake -B build && cmake --build build)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, default=ROOT / "runs" / "m1" / "final" / "adapter")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "models")
    args = ap.parse_args(argv)

    merged = args.out / "m1-merged"
    f16 = args.out / "m1-f16.gguf"
    q4 = args.out / f"m1-{QUANT.lower()}.gguf"
    args.out.mkdir(parents=True, exist_ok=True)

    if not (merged / "config.json").exists():
        print("merging adapter into base", flush=True)
        merge(args.base, args.adapter, merged)
    if not f16.exists():
        print("converting to GGUF (f16)", flush=True)
        subprocess.run([sys.executable, str(args.llama_cpp / "convert_hf_to_gguf.py"), str(merged), "--outtype", "f16", "--outfile", str(f16)], check=True)
    if not q4.exists():
        print(f"quantizing to {QUANT}", flush=True)
        subprocess.run([str(find_tool(args.llama_cpp, "llama-quantize")), str(f16), str(q4), QUANT], check=True)

    llama_rev = subprocess.run(["git", "-C", str(args.llama_cpp), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {
        "base_model": args.base,
        "adapter_sha256": adapter_sha256(args.adapter),
        "quantization": QUANT,
        "llama_cpp_commit": llama_rev or None,
        "gguf": {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size} for p in (f16, q4)},
    }
    (args.out / f"m1-{QUANT.lower()}.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
