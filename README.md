# Job Post Extractor

[![tests](https://github.com/yougijain/cprojectllm/actions/workflows/tests.yml/badge.svg)](https://github.com/yougijain/cprojectllm/actions/workflows/tests.yml)
[![eval gate](https://github.com/yougijain/cprojectllm/actions/workflows/eval-gate.yml/badge.svg)](https://github.com/yougijain/cprojectllm/actions/workflows/eval-gate.yml)
[![refresh board](https://github.com/yougijain/cprojectllm/actions/workflows/refresh.yml/badge.svg)](https://github.com/yougijain/cprojectllm/actions/workflows/refresh.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Can a 1.5B-parameter open model, fine-tuned on a few thousand
machine-labeled examples, pull structured fields out of job posts as well
as a frontier API model, for a small fraction of the cost?

This project measures that on Hacker News's monthly "Who is hiring?"
threads. A small model (Qwen2.5-1.5B-Instruct) is fine-tuned with QLoRA to
turn each post into eleven fields: company, role, seniority, location,
remote, visa sponsorship, salary range and currency, tech stack, and
whether new grads can apply. It is scored against a rules baseline, the
same small model zero- and few-shot, and a frontier model, on 300 posts
labeled by hand before anything was trained. It is then served two ways
(vLLM on a T4, and 4-bit llama.cpp on a CPU), and its output runs a
searchable job board that refreshes every week.

**[Open the job board →](https://yougijain.github.io/cprojectllm/)**

## Results

<!-- results:table -->
_Pending._ The gold set is drawn ([`data/gold/split.json`](data/gold/split.json))
and is being labeled. Every model then runs on the same 250 held-out posts,
and this block is filled from `results/test_results.md` by
`scripts/sync_readme.py`, never typed by hand.

The table will report, for each model: field accuracy with a bootstrap 95%
interval; the three fields a job seeker filters on (new-grad eligibility,
remote, salary); JSON validity on the first reply and after one retry; and
cost per 1,000 posts. Below it: every field separately, M1 against each
baseline (paired bootstrap, and McNemar's test per field with Holm
correction), the learning curve, the spread over three seeds, and what
4-bit quantization costs, field by field.
<!-- /results:table -->

### Why fields were wrong

<!-- results:errors -->
_Pending._ Every wrong field gets one cause: an ambiguous post, a missed
field, a hallucinated value, wrong normalization ("120" for 120,000), invalid
JSON, or a label error. Counts per model go here.
<!-- /results:errors -->

### What didn't work

Kept as the project runs: what was tried, what happened, and the number.

- **Text similarity was not enough to keep test posts out of training.**
  Companies repost almost the same text every month, so the first leakage
  filter dropped training posts that were near-duplicates of a gold post
  (Jaccard ≥ 0.5 on word 5-grams). Reading pairs below that threshold showed
  the same company's post a few months apart at 0.3, 0.2 and even 0.15
  similarity. The filter now also drops every post from a gold candidate's
  company: 1,429 posts excluded instead of 666, leaving 5,785 for training.

## Models compared

All on the same posts, with the same prompt ([`prompts/extract_v1.md`](prompts/extract_v1.md)).

| | Model | What it is |
|---|---|---|
| B1 | Rules | Split the header line on pipes, keyword lists, a salary parser. No model. |
| B2 | Qwen2.5-1.5B-Instruct | Zero-shot. |
| B3 | Qwen2.5-1.5B-Instruct | Five hand-labeled examples in the prompt, from outside the gold set. |
| B4 | Claude Opus 5.5 | Zero-shot through the API. The quality ceiling and the cost comparison. Never used as training data. |
| M1 | Qwen2.5-1.5B-Instruct + LoRA | QLoRA on silver labels from an open-weight (Apache-2.0) teacher. |
| M1-vllm | M1 | Served by vLLM on a T4, fp16. |
| M1-q4 | M1, merged | 4-bit (Q4_K_M) llama.cpp build on a CPU. |

The student is 1.5B rather than 3B partly for licensing (the 1.5B model is
Apache-2.0, the 3B one is not) and partly so the 4-bit build is fast enough
to run on a CI runner's CPU every week.

## How it is measured

**Data.** 7,791 top-level posts from the 25 threads between November 2024
and November 2026, read from the public HN API. Only post IDs are committed
([`data/post_ids/`](data/post_ids)); `scripts/fetch_posts.py --from-ids`
rebuilds the text.

**Gold labels.** 300 posts from the 24 complete months, stratified by
month, drawn once with a fixed seed and labeled by hand in the terminal with
nothing pre-filled: 50 for development, 250 held out. Conventions are in
[`data/gold/LABELING.md`](data/gold/LABELING.md). A second person labels 40
of them independently for per-field agreement. The labels are then frozen by
hash, and CI fails if they change.

**Silver labels.** Up to 4,000 training posts labeled by the open-weight
teacher, with a random 100 checked by hand to measure their error rate.
Training posts exclude every gold candidate's company and anything
textually close to a gold post (above).

**Scoring.** Exact match for the enum fields; set F1 after canonicalizing
names for the stack; salary right only when min, max and currency all are,
each within 5%, or correctly absent. A post a model fails on scores zero on
every field rather than dropping out. Every prediction is saved, so every
comparison is paired.

**Training.** QLoRA (4-bit NF4 base, fp16 compute, since a T4 has no bf16)
on a free Colab T4. Hyperparameters are chosen on the 50 dev posts only;
the dev loss is checked every epoch and the lowest-loss epoch is kept. The
final configuration is trained with three seeds.

**Engineering.** Pydantic validation of every reply, one retry with the
validator's complaint shown to the model, then the post is marked failed,
never guessed. Every model call is cached on disk by model, adapter hash,
prompt and parameters, so reruns are free. Every results file records the
model, adapter hash, prompt version and git commit.

## Serving and the board

[`docs/serving.md`](docs/serving.md) has the commands. Throughput and p50/p95
latency come from [`scripts/bench_serving.py`](scripts/bench_serving.py),
against the same OpenAI-compatible endpoint shape for vLLM and llama.cpp.

The [board](https://yougijain.github.io/cprojectllm/) is a static page over
the last three threads, with filters for remote, salary, new-grad
eligibility, visa sponsorship, seniority and stack. A weekly GitHub Actions
job ([`refresh.yml`](.github/workflows/refresh.yml)) fetches new posts,
extracts them on the runner's CPU with the 4-bit model, and redeploys. No
GPU is involved. Until a model is published it runs on the rules baseline,
and its footer says which extractor produced the fields.

On every pull request, [`eval-gate.yml`](.github/workflows/eval-gate.yml)
runs the published model over the 50 dev posts with the PR's code and fails
if field accuracy drops by more than 2 points.

## Where to look

| Question | Where it is answered |
|---|---|
| What counts as correct? | [`jpe/score.py`](jpe/score.py), [`tests/test_score.py`](tests/test_score.py) |
| What makes a reply valid? | [`jpe/validate.py`](jpe/validate.py), [`tests/test_validate.py`](tests/test_validate.py) |
| How is the gold set drawn and protected? | [`jpe/gold.py`](jpe/gold.py), [`scripts/label.py`](scripts/label.py) |
| How are test posts kept out of training? | [`jpe/dedup.py`](jpe/dedup.py) |
| How is M1 trained? | [`jpe/train_config.py`](jpe/train_config.py), [`configs/train.yaml`](configs/train.yaml), [`scripts/train_qlora.py`](scripts/train_qlora.py) |
| How are models compared? | [`jpe/report.py`](jpe/report.py), [`jpe/stats.py`](jpe/stats.py) |
| What is the rules baseline? | [`jpe/rules.py`](jpe/rules.py), [`jpe/normalize.py`](jpe/normalize.py) |
| How do I reproduce it? | [`docs/RUNBOOK.md`](docs/RUNBOOK.md) |

## Status

- [x] Fetch script, schema, post IDs for 25 threads
- [x] Gold sample drawn (50 dev / 250 test), labeling tool, freeze check
- [x] Rules baseline, salary parser, model clients, cache, extraction loop
- [x] Training pool with leakage control, teacher and spot-check scripts
- [x] QLoRA, learning-curve, serving, benchmark and board tooling; CI
- [ ] Hand-written core: scorer, schema validation, QLoRA configuration
- [ ] Label and freeze the gold set; second-annotator agreement
- [ ] Silver labels and their spot check
- [ ] Baselines, M1, learning curve, three seeds
- [ ] vLLM and llama.cpp benchmarks, quantization loss
- [ ] Error analysis, then the results above

## Credits

Posts are written by Hacker News users and belong to them; the board links
every card to its original post. Qwen2.5 models are Apache-2.0. Code is MIT.
