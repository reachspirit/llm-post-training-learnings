# Session B kit — PostTrainBench GSM8K cell (SFT v1 → v2 → v3)

Code for Session B of `labs/posttrainbench-gsm8k-weekend-runsheet.md` (~5 h).
Prepared 2026-10-11, ahead of the week of Mon 2026-10-12, so Session B —
like Session A after its kit landed — is running, not writing code.
Assumes Session A is done: a logged base score (zero-shot + 4-shot on the
200-slice) exists to beat, from the *patched* Session A harness.

## Run order

1. **Verify the prep logic** (any machine, seconds):
   `python selftest.py` — annotation stripping, eval-format construction,
   dedupe, gold filtering. Verified passing here 2026-10-11.
2. **Prepare training candidates** (any machine):
   `python prepare_sft_data.py --out data/sft_candidates.jsonl`
   GSM8K *train* split → `prompt`/`completion` rows in exact eval format
   (`Question: …\nAnswer:` + ` #### N` target), calculator annotations
   (`<<…>>`) stripped, unparseable-gold rows dropped, questions deduped.
   Prints counts + sha256 of the output — paste both into
   `../session-a/dataset-card.md` (that completes the card Session A left open).
3. **Decontaminate** (any machine) — do not skip, do not reorder:
   `python ../session-a/decontaminate.py --train data/sft_candidates.jsonl --out data/sft_clean.jsonl --report data/decontamination-report.json`
   Verified compatible: the prep output's `question` field is what the
   decontaminator checks. Paste the report summary into the dataset card.
4. **Pre-flight, smoke, then v1** (A100 box):
   `python preflight.py --data data/sft_clean.jsonl`
   `python train_sft.py --data data/sft_clean.jsonl --out ckpt/smoke --max-steps 2`
   `python train_sft.py --data data/sft_clean.jsonl --out ckpt/sft-v1`
   Smoke gets its own `--out` on purpose: `train_sft.py` writes a merged
   checkpoint to whatever `--out` says, so sharing `ckpt/sft-v1` between
   smoke and the full run lets a later smoke silently replace the scored
   v1 checkpoint while its results file still points at that path.
   v1 defaults: LoRA r16 / α32, lr 1e-4 cosine, 2 epochs, batch 8 ×
   grad-accum 2 (effective 16), max_length 512, completion-only loss,
   bf16. Output is a **merged** checkpoint + `train_config.json`.
5. **Score v1 with the Session A harness, unchanged:**
   `python ../session-a/eval_gsm8k.py --model ckpt/sft-v1 --limit 200 --fewshot 4 --out results/sft-v1-fewshot-200.jsonl`
   (also `--fewshot 0`). Log in the run sheet's scoring block; keep the
   best checkpoint, not the last (run sheet rule).

## v1 → v2 → v3: one change per version

Change exactly one flag (or the data) per version, so the delta is
attributable. Levers, in the order worth trying at 1.7B scale:

| Version | Change one of | Flag |
|---|---|---|
| v2 | epochs 2 → 3, *or* lr 1e-4 → 2e-4, *or* data limit (full train vs subset) | `--epochs` / `--lr` / re-run prep with `--limit` |
| v3 | LoRA rank 16 → 32, *or* whichever v2 lever won, pushed one step | `--lora-r` |

## What's verified / what isn't

- ✅ Data-prep logic ran here on synthetic fixtures, including an
  end-to-end prep → decontaminate pass on a fixture file (kept=1,
  flagged=0 against an unrelated test question).
- ⚠️ `train_sft.py` has **not** been run — no GPU/torch/trl in the prep
  environment (same status as Session A's generation path). Its API
  shape follows the current TRL pattern checked 2026-10-11
  (`SFTTrainer` + `SFTConfig` carrying `max_length` /
  `completion_only_loss`, `peft_config=LoraConfig`). If the A100 box's
  trl version differs, the `--max-steps 2` smoke is where that shows —
  run it before the real v1.
- Defaults are a starting point fitted to 1× A100 80GB and the ~5 h
  Session B budget, not tuned values. The run sheet's hour-7 stop rule
  and best-checkpoint retention still govern the session.

## Audit — 2026-10-12 (pre-flight, same pass style as Session A's)

- **Truncation can silently eat the answer.** `train_sft.py` sets
  `max_length=512`; current TRL docs give `truncation_mode="keep_start"`
  as the default, so an over-length prompt+completion is cut from the
  end — losing the completion's trailing `#### {gold}` and the EOS token
  TRL appends at tokenization. Those rows would train answers that never
  terminate. How many GSM8K rows are affected is unknown until counted
  with the real tokenizer, so `preflight.py` counts them (tokenizer only,
  no GPU) and fails the run order above if any row is over length or any
  completion's `#### {gold}` ending is broken. Its counting logic is
  covered by `python preflight.py --selftest` (green 2026-10-12).
- **No-EOS-in-JSONL is not a bug.** TRL appends the EOS token during SFT
  tokenization; the risk is only truncation slicing it off (above), which
  is why the pre-flight budgets +1 token per row for it.
- **Smoke/full `--out` collision fixed** in the run order above (smoke
  now writes `ckpt/smoke`, not `ckpt/sft-v1`).

## Audit — 2026-10-13 (ending-invariant format mismatch)

- **Preflight's ending check disagreed with prep and with the eval.**
  `gold_answer()` in prep normalises the gold field (`#### 1,000` →
  gold `1000`, `#### 72.00` → `72`), and the Session A eval normalises
  the same way — its selftest explicitly covers the comma case. But
  `preflight.py` compared the completion with an exact string,
  `endswith("#### {gold}")`, so a row prep kept and eval would score
  correct was flagged as a bad ending and preflight exited
  "NOT SAFE TO TRAIN". Reproduced 2026-10-13 with the kit's own code:
  3 fixture rows kept by prep, 2 flagged by preflight, all 3 scored
  correct by the eval extractor. How many real GSM8K train rows use
  comma/decimal final-answer formatting is unknown — the prep
  environment has no `datasets` module, so the real file can't be
  counted here; the mismatch is proven on formats the kit's own
  parsers handle.
- **Fixes (both selftests re-run green 2026-10-13):** `preflight.py`
  now extracts the trailing `#### N` and compares *normalised values*
  (same normalisation as the eval) against the gold — comma/decimal
  endings pass, a trailing `####` with the wrong value still fails,
  and a missing trailing marker still fails. `prepare_sft_data.py`
  now canonicalises the trailing `#### N` to the normalised gold, so
  training targets carry one format. Net effect at run time: if the
  real count is zero, nothing changes; if it is non-zero, Session B
  no longer halts (or drops valid rows) at the pre-flight step.

## Files

- `prepare_sft_data.py` — GSM8K train → eval-format SFT candidates (tested)
- `train_sft.py` — TRL LoRA SFT, merged-checkpoint output (untested, see above)
- `preflight.py` — token-length / truncation / ending-invariant check (selftest green)
- `selftest.py` — GPU-free logic tests for the prep step
