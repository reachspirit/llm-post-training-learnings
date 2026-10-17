# Weekend Run Sheet — PostTrainBench Single Cell (Qwen3-1.7B-Base × GSM8K)

**Prepared:** 2026-10-09 (Fri evening, America/Los_Angeles) for the weekend of Sat 2026-10-10 – Sun 2026-10-11 / week of Mon 2026-10-12.
**Source brief:** `labs/posttrainbench-single-cell-a100.md` (filed 2026-10-07). This sheet operationalizes that brief; it does not change its recommendations.
**Budget fit:** ~10 h of run time inside Ran's 10–15 h/week budget, on the guaranteed 1× A100 80GB. A100 ≠ H100 throughput: keep the benchmark's 10 h wall-clock discipline, shrink tokens/steps, start with LoRA/QLoRA at 1.7B — do not assume token parity with the H100 reference.

## Why this cell first
- Verifiable exact-match reward on GSM8K — directly reusable in a Phase 2 GRPO stage.
- Cheap, fast eval; leaves room for 2–3 SFT iterations inside the window (where the paper's agents actually gained).
- 1.7B fits comfortably on one A100, so the lab tests data + training judgment, not VRAM survival.
- Paper calibration (arXiv:2603.08640): best controlled agent 23.2% weighted avg vs 7.5% zero-shot base and 51.1% official instruct. Success here = beat base with a clean, auditable run — not match human post-training.

## Session plan (split to taste — e.g. 2×5 h or 3 sessions)

### Session A — Baseline + harness (~2 h)
- [ ] Zero-shot and few-shot base score on a limited GSM8K eval slice; log it before any training.
- [ ] Confirm the checkpoint loads cleanly and generates non-garbled output with the intended chat template.
- [ ] Start a wall-clock timer; check it at every iteration boundary (paper: time discipline correlated with scores; many agents stopped at 2–3 h of 10 h).
- [ ] Data curation only: non-benchmark training sources, format matched to the eval prompt, decontamination check against the GSM8K test set, provenance log (source, split, license, why it is not test data). Done when the dataset card is written.

### Session B — SFT iterations (~5 h)
- [ ] v1 → v2 → v3, one change per version (data mix, formatting, LR/epochs, LoRA rank).
- [ ] Eval on the limited slice after each version; keep the best checkpoint, not the last.
- [ ] Per version log: score, config, data hash, wall time used/remaining.

### Session C — Optional GRPO + final (~3 h)
- [ ] GRPO stage 2 **only if** the SFT checkpoint is stable: exact-match correctness reward, small step budget. This mirrors the only RL pattern in successful agent runs (SFT first, RL second, no learned RM). Skip if SFT is unstable — a skipped stage 2 beats a hacked one.
- [ ] Hour-7 rule from the brief: if the run is failing at hour 7, stop adding methods. Reward-hacking cases in the paper clustered after long failed stretches.
- [ ] Final: largest feasible eval on the selected checkpoint, package the run summary + provenance locally.

## Terminal-state checks (from `labs/phase-2-eval-checklist.md` — grade the checkpoint, not the run story)
- [ ] Checkpoint loads and generates non-garbled output.
- [ ] Provably derived from Qwen3-1.7B-Base — no substituted instruct checkpoint.
- [ ] Provenance log exists; no GSM8K test items in any train file (split names are not proof — check contents).
- [ ] Eval harness unmodified; scores from the eval script, not a narrative.

## Scoring block (copy per checkpoint — extends the Phase 2 template)
- **Checkpoint / script version:**
- **Method:** SFT (LoRA/QLoRA/full) → optional GRPO
- **Wall time used / remaining:**
- **pass@1 (GSM8K):**
- **pass@K + never-solved fraction:** _(optional — fields from `labs/phase-2-grpo-logging-addendum-draft.md`, which is still a DRAFT pending Ran's approval from the 2026-10-10 message; logging them here is cheap, merging them into the protocol is his call)_
- **Data sources + decontamination check:**
- **GPU-hours:**
- **Decision:** keep / iterate data / iterate hyperparams / stage-2 GRPO / stop

## Open decisions carried from the source brief (not made here)
1. Cell: GSM8K (recommended above) vs BFCL alternate — BFCL only if tool-use is the point, and only with a full data-source audit first (the paper documents a BFCL "train" split containing eval data).
2. Whether PostTrainBench earns a `reading-list.md` slot or stays a finding only.
3. Local-only run vs investigating the repo's Harbor adapter on Modal for a closer-to-official setup.
