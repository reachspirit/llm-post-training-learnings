# Session A kit — PostTrainBench GSM8K cell (baseline + harness + decontamination)

Code for Session A of `labs/posttrainbench-gsm8k-weekend-runsheet.md` (~2 h of the
~10 h weekend run). The run sheet is the plan; this folder is the executable
version of its three Session A items. Prepared 2026-10-10.

## Run order

1. **Verify the kit itself** (any machine, 10 seconds):
   `python selftest.py` — checks answer extraction/normalization,
   pass@1 / pass@K / never-solved aggregation, and that the decontamination
   check catches a planted test item. Verified passing on synthetic fixtures
   2026-10-10; re-run before trusting a score anyway.
2. **Decontaminate the training candidates** (any machine):
   `python decontaminate.py --train <candidates.jsonl> --out <clean.jsonl> --report decontamination-report.json`
   Defaults: 10-word n-gram overlap vs the GSM8K test split (HF `gsm8k`,
   config `main`), or `--test-file` for a local copy. Paste the report summary
   into `dataset-card.md` — that is the run sheet's provenance/decontamination
   terminal-state check, done from file contents, not split names.
3. **Baseline eval** (A100 box):
   `python eval_gsm8k.py --model Qwen/Qwen3-1.7B-Base --limit 200 --fewshot 4 --out results/base-fewshot-200.jsonl`
   Also run `--fewshot 0` for the zero-shot number. Both go in the run sheet's
   scoring block before any training starts. `--samples 8` adds pass@K +
   never-solved fraction when you want the Sharpening-Tax-style view; default
   `--samples 1` is pass@1 only, cheap and sufficient for Session A.

## Patch — 2026-10-11 (pre-first-run audit)

Two bugs found by auditing the kit before its first A100 run, both reproduced
with the kit's own code, both fixed, selftest re-run green:

1. **Continuation corruption (score validity).** `generate()` sets no stop
   sequence, so a base model can continue past its own `#### N` with an
   invented `Question: ... #### M` block — and `extract_answer` took the
   *last* `####`, i.e. the fabricated problem's answer. Reproduced: a
   generation ending `#### 72` + invented problem scored **35**, not 72.
   Fix: `extract_answer` now truncates at the first `\nQuestion:` block
   (`truncate_continuation`) before extracting. Raw generations are still
   stored verbatim in the results file.
2. **Results lost after a full run (fail-fast).** `main()` opened `--out`
   only *after* generation finished, and the documented path
   `results/base-fewshot-200.jsonl` needs a `results/` dir that doesn't
   exist in a fresh kit — first run would have crashed *after* spending
   the eval compute. Fix: output parent dir is created and checked
   before any generation; same mkdir guard added to `decontaminate.py`.
3. **Provenance.** The summary now also logs `limit`, `samples_requested`,
   `temperature` (null for greedy), `max_new_tokens` and `seed`
   (new `--seed`, default 1234, applied to sampled runs) — previously a
   score could not be reproduced from its own summary.

Still unverified: the GPU generation path itself (see below) — the seed
wiring in `generate()` is code-reviewed only, like the rest of that path.

## Patch — 2026-10-14 (decontamination short-item blind spot)

The 10-word n-gram check in `decontaminate.py` was blind to any text
shorter than 10 normalized words: such a text produces zero n-grams, so
an exact duplicate of a short test question could never be flagged.
Reproduced with the kit's own code: `check()` on an 8-word exact
duplicate returned zero flags, while the run sheet's terminal-state
check claims "no GSM8K test items in any train file — check contents."
Fix: `check()` now also tests verbatim containment both ways — a test
text shorter than n inside a training item, or a training item shorter
than n inside a test text — and the Session A selftest covers a short
exact duplicate (flagged) and a short clean item (passed). How many
real GSM8K test questions normalize to under 10 words is unknown (no
`datasets` in the prep environment, same limit as before); if zero,
nothing changes for this run — the fix closes the guarantee gap for
short items and for reuse of the checker on other datasets.

## What's verified / what isn't

- ✅ Scoring logic (extraction, aggregation) and the decontamination CLI were
  run here on synthetic fixtures — planted duplicate flagged, clean item kept.
  Extraction now also covered against invented-continuation generations
  (patch above, 2026-10-11).
- ⚠️ Generation (`generate()` in `eval_gsm8k.py`, transformers + bf16) has
  **not** been run — this prep environment has no GPU/torch. First A100 step is
  still the run sheet's own check: checkpoint loads and generates non-garbled
  output with the intended template. If `datasets`/`transformers` versions on
  the A100 box differ, that's the step that will show it.
- The 4-shot prompt in `eval_gsm8k.py` uses canonical GSM8K *train*-split
  exemplars, so few-shot prompting does not contaminate the test slice.

## Files

- `eval_gsm8k.py` — baseline harness (exact-match, results JSONL + summary)
- `decontaminate.py` — train-vs-test n-gram decontamination + report
- `selftest.py` — GPU-free logic tests
- `dataset-card.md` — provenance card skeleton for the Session A data step
