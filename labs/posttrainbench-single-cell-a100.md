# Lab Brief — PostTrainBench, One Cell, One A100

**Purpose:** Reproduce the *shape* of a PostTrainBench run at Ran's scale — not the 28-cell benchmark. One base × one target, ~10 hours on 1× A100 80GB, checkpoint + provenance at the end.

**Constraint translation (read first)**
- Benchmark budget: 10 h on 1× H100 80GB. Ran's guarantee: 1× A100 80GB, 10–15 h/week.
- Same VRAM class, materially lower training throughput on A100. Planning rule: keep the benchmark's 10 h wall-clock discipline, but shrink tokens/steps to fit — start with the 1.7B base and LoRA/QLoRA before attempting a 3–4B full run. Do not assume "10 H100-hours = 10 A100-hours" of training tokens.
- Official full benchmark is cluster/Modal-shaped (HTCondor reference; Harbor adapter on Modal per repo README). This brief is a local single-cell adaptation using the same prompt rules and eval logic, not a claim of an official leaderboard submission.

## Recommended cell

**Primary: Qwen3-1.7B-Base × GSM8K**
- Verifiable exact-match reward — directly reusable in Phase 2 GRPO.
- Cheap, fast eval; leaves time for 2–3 SFT iterations inside the window (which is where agents actually gained).
- Fits comfortably on one A100, so the lab tests data + training judgment, not VRAM survival.
- Connects to existing path: Phase 1 SFT → Phase 2 GRPO-on-math (see `learning-path.md`, `labs/phase-2-eval-checklist.md`).

**Alternate (pick one only if tool-use is the point): SmolLM3-3B-Base or Gemma-3-4B-PT × BFCL**
- This is where targeted agents beat official instruct models — and where the contamination trap lives (a dataset split labelled "train" containing eval data). Only take this cell if you explicitly want the Phase 2 state/tool-checking angle, and audit every data source before training.

Avoid for a first cell: AIME 2025, GPQA, ArenaHard-Writing — paper shows minimal movement in 10 h; low signal per Ran-hour.

## 10-hour run plan

| Window | Work | Exit criterion |
|---|---|---|
| 0:00–1:00 | Baseline + harness. Zero-shot and few-shot base score on a limited eval slice; confirm `final_model` loading + chat template works before any training. Start a wall-clock timer and check it at every iteration boundary. | Baseline number logged; eval runs end-to-end on A100 |
| 1:00–2:00 | Data curation only. Non-benchmark training sources, format matching the eval prompt, decontamination check against the target test set, provenance log (source, split, license, why it is not test data). | Dataset card written; no benchmark test items in train |
| 2:00–7:00 | SFT iterations v1 → v2 → v3. Change one thing per version (data mix, formatting, LR/epochs, LoRA rank). Eval on the limited slice after each; keep the best checkpoint, not the last. | Each version: score, config, data hash logged |
| 7:00–9:00 | *Optional* stage 2, only if SFT checkpoint is stable: GRPO with exact-match correctness reward on verifiable items, small step budget. This mirrors the only RL pattern that appeared in successful agent runs: SFT first, RL second, simple correctness reward, no learned RM. Skip if SFT is still unstable — a skipped stage 2 beats a hacked one. | GRPO checkpoint beats SFT on limited slice *and* does not collapse coverage (below) |
| 9:00–10:00 | Final eval + packaging. Full (or largest feasible) eval on the selected checkpoint, package as `final_model`, write run summary. | Checkpoint + summary + provenance submitted locally |

If a run is failing at hour 7: stop adding methods. The paper's reward-hacking cases clustered after long failed stretches — that is exactly when test-data shortcuts start looking rational. Cut scope, keep the audit clean.

## Terminal-state checks (from Phase 2 checklist)

Grade the checkpoint, not the story of the run:
- [ ] `final_model` loads in the starting environment and generates non-garbled output
- [ ] Checkpoint is provably derived from the specified base — no downloaded instruct checkpoint, no other model fine-tuned
- [ ] Training data provenance log exists; no benchmark test questions/answers in any train file (check split names — "train" labels are not proof)
- [ ] Eval harness unmodified; scores come from the provided `evaluate.py`-style script
- [ ] No external API synthetic data unless explicitly allowed in your own run rules and logged

## Scoring template

Copy per checkpoint (extends `labs/phase-2-eval-checklist.md`):

- **Cell:** Qwen3-1.7B-Base × GSM8K
- **Checkpoint / script version:**
- **Method:** SFT (LoRA / QLoRA / full) → optional GRPO stage 2
- **Wall time used / remaining:**
- **pass@1 (target metric):**
- **pass@K + never-solved fraction:** (Sharpening-Tax guard — did pass@1 rise by collapsing coverage?)
- **Data sources + decontamination check:**
- **Cost:** GPU-hours (+ API cost if any)
- **Decision:** keep / iterate data / iterate hyperparams / stage-2 GRPO / stop

## What success looks like for Ran
Not a leaderboard number. A clean, auditable single-cell run where: baseline → SFT v_n improves the target, coverage does not silently collapse, the checkpoint passes the terminal-state checks above, and the whole thing fits in one 10–15 h week on the guaranteed A100. That is a Phase 1→2 bridge lab with an external reference point (PostTrainBench paper: best controlled agent 23.2% weighted avg vs 51.1% official instruct — expect to beat base, not to match human post-training).

## Open decision for Ran
1. Cell choice: GSM8K primary (recommended) vs BFCL alternate.
2. Whether to file PostTrainBench in `reading-list.md` (Core/Additions) or leave as a finding only.
3. Local-only run vs investigating the repo's Harbor/Modal adapter for a closer-to-official setup.
