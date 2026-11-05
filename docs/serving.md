# Serving M1

Two setups, measured the same way: the fine-tuned model under vLLM on a
GPU, and the same model merged and quantized to 4 bits for llama.cpp on a
CPU. Both speak the OpenAI chat-completions protocol, so one client
(`jpe/clients/openai_compat.py`), one accuracy run (`scripts/run_model.py`)
and one benchmark (`scripts/bench_serving.py`) cover both.

## GPU: vLLM on a T4

In a fresh Colab runtime (vLLM pins its own torch):

```bash
pip install -r requirements-serve.txt
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype half \
  --max-model-len 4096 \
  --enable-lora --lora-modules m1=runs/m1/final/adapter --max-lora-rank 64 &
```

`--dtype half` because the T4 has no bf16. The adapter is served under the
name `m1`, which is what `M1-vllm` in `configs/models.yaml` asks for; set
`--max-lora-rank` to at least the rank you trained with. vLLM's support for
older GPUs changes between releases: if the installed version refuses a
Turing card, pin the last release that accepts one.

```bash
python scripts/run_model.py M1-vllm --split test --workers 8
python scripts/bench_serving.py M1-vllm --hardware T4 --concurrency 1 8 32
```

Concurrency 1 gives single-request latency. Higher concurrency is where
vLLM's continuous batching shows: requests join the running batch as others
finish, so throughput climbs while each request's latency grows only
modestly. The best throughput sets M1's cost per 1,000 posts.

## CPU: 4-bit llama.cpp

Build llama.cpp once (CPU only):

```bash
git clone https://github.com/ggml-org/llama.cpp ~/src/llama.cpp
cmake -S ~/src/llama.cpp -B ~/src/llama.cpp/build -DGGML_NATIVE=ON
cmake --build ~/src/llama.cpp/build -j --target llama-server llama-quantize
pip install -r ~/src/llama.cpp/requirements.txt   # for convert_hf_to_gguf.py
```

Merge, convert and quantize:

```bash
python scripts/export_gguf.py --adapter runs/m1/final/adapter --llama-cpp ~/src/llama.cpp
```

Serve and measure:

```bash
~/src/llama.cpp/build/bin/llama-server -m models/m1-q4_k_m.gguf --alias m1-q4_k_m -c 4096 --port 8080 &
python scripts/run_model.py M1-q4 --split test
python scripts/bench_serving.py M1-q4 --hardware "laptop CPU: <model>, <cores> cores" --concurrency 1
```

`scripts/evaluate.py` then reports the accuracy lost to quantization field
by field, with McNemar's test on the posts where only one of the two builds
is right. One pattern worth checking for, not assuming: loss concentrated
where a single token carries the answer (the digits of a salary, an enum
value the model was nearly undecided on), while fields copied verbatim
from the header (company, role) survive.

## Publishing the GGUF

The weekly board refresh and the CI evaluation gate both run M1 on a
GitHub runner's CPU, so the 4-bit file is published as a Release asset:

```bash
gh release create model-latest --title "M1, 4-bit GGUF" --notes "Built by scripts/export_gguf.py; see m1-q4_k_m.json for hashes." || true
gh release upload model-latest models/m1-q4_k_m.gguf models/m1-q4_k_m.json --clobber
```

Qwen2.5-1.5B-Instruct is Apache-2.0, which allows publishing derived
weights.
