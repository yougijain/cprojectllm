# Runbook

Every step, in order, with the command that does it. Steps marked
**(Colab)** need a T4; everything else runs on a laptop.

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python scripts/fetch_posts.py --from-ids        # rebuild post text from the committed IDs
pytest -q
```

## M0: gold labels (about 7 hours)

The sample is already drawn (`data/gold/split.json`, seed 20261105).

1. Settle the open choices at the bottom of `data/gold/LABELING.md`.
2. Write `jpe/validate.py` until `pytest tests/test_validate.py` passes. The
   labeling tool uses it to reject malformed labels.
3. Label:
   ```bash
   python scripts/label.py                  # 50 dev + 250 test, interleaved
   python scripts/label.py --set fewshot    # 5 examples for B3
   python scripts/gold.py status
   ```
4. Optional: a second annotator labels 40 posts, then
   `python scripts/label.py --second --annotator <name>` and
   `python scripts/agreement.py`.
5. Freeze, and commit `data/gold/`:
   ```bash
   python scripts/gold.py freeze
   ```

## M1: scorer and first numbers (about 4 hours)

1. Write `jpe/score.py` until `pytest tests/test_score.py` passes.
2. Run the baselines that need no GPU:
   ```bash
   python scripts/run_model.py B1 --split dev
   python scripts/run_model.py B1 --split test
   export ANTHROPIC_API_KEY=...             # or `ant auth login`
   python scripts/run_model.py B4 --split dev --workers 8
   python scripts/run_model.py B4 --split test --workers 8
   python scripts/evaluate.py --split dev --reference B4
   ```
   B2 runs on Colab in M3; it can also run on any machine with a GPU.

## M2: silver labels (about 3 hours)

1. Point `teacher.base_url` in `configs/models.yaml` at an endpoint serving
   the teacher, and export its key as `TEACHER_API_KEY`.
2. ```bash
   python scripts/silver_label.py label --n 4000 --workers 16
   python scripts/spot_check.py             # judge 100 by hand
   python scripts/spot_check.py --report
   ```
3. Commit `data/silver/` (fields only; no post text).

## M3: fine-tuning (about 6 hours, Colab)

1. Write `jpe/train_config.py` and fill every value in `configs/train.yaml`.
   Choose them on dev: dev loss by epoch, and dev field accuracy from
   `scripts/evaluate.py --split dev`.
2. Open `notebooks/train_colab.ipynb` in Colab with a T4, and run it top to
   bottom: core tests, B2 and B3, the learning curve, three seeds.
3. Copy `results/predictions/` and the training logs back from Drive and
   commit them.

## M4: serving (about 5 hours)

Follow `docs/serving.md`:

```bash
python scripts/run_model.py M1-vllm --split test --workers 8          # Colab
python scripts/bench_serving.py M1-vllm --hardware T4 --concurrency 1 8 32
python scripts/export_gguf.py --llama-cpp ~/src/llama.cpp             # laptop
python scripts/run_model.py M1-q4 --split test
python scripts/bench_serving.py M1-q4 --hardware "laptop CPU: ..." --concurrency 1
gh release upload model-latest models/m1-q4_k_m.gguf models/m1-q4_k_m.json --clobber
```

## M5: the board and the gate (about 5 hours)

1. Settings → Pages → Source: **GitHub Actions**.
2. Run the refresh once by hand: Actions → refresh board → Run workflow.
3. Set the gate's baseline from main and commit it, with llama-server
   serving the published GGUF (`docs/serving.md`):
   ```bash
   python scripts/run_model.py M1-q4 --split dev --name M1-q4-gate
   python scripts/eval_gate.py --run results/predictions/dev/M1-q4-gate.json --update-baseline
   ```

## M6: the write-up (about 3 hours)

```bash
python scripts/evaluate.py --split test --errors   # then tag causes in results/errors/test/*.csv
python scripts/evaluate.py --split test --errors   # recount with the tags
python scripts/sync_readme.py
```

Then write "What didn't work" in the README, and report the result whatever
it is.
