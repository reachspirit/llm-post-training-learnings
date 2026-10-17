# Finding — Sharpening Tax in Post-Training (Oct 2026)

**Source:** AI Weekly alert summarizing a new arXiv paper (original report linked from the alert).
**Link:** https://aiweekly.co/alerts/sharpening-tax-paper-rl-post-training-cuts-passk-coverage
**Filed:** 2026-10-06

## What it reports (short summary)
- RL post-training appears to trade solution coverage for single-shot accuracy: models get better at pass@1 but lose pass@K coverage versus their base models.
- The headline says RL post-training hurt multi-sample coverage in 36 of 42 tested model-benchmark pairs.
- The authors tested 14 base/post-trained model pairs across four model families and three agentic benchmarks (42 cases total).
- Their explanation, in their words, is that post-training pushes tasks toward "two extremes, always solved or never solved."
- They propose posterior-tempered group sampling (PTGS), a Bayesian sampler that adapts temperature per prompt by estimated difficulty, and report that it reduces the tax versus a fixed-temperature baseline.

## Why this matters here
- Directly relevant to the Phase 2 GRPO/verifier work: GRPO optimization can polarize task solvability, improving efficiency and consistency while shrinking the set of tasks the model can ever solve under sampling.
- Connects straight to the ThinkingBox three-view framing already in the learning path (single-attempt success, at-least-once breadth, all-20 consistency). Sharpening Tax is the pass@K version of the same measurement instinct.
- For the Phase 2 labs here, this suggests tracking pass@K alongside pass@1 and watching whether training is expanding or collapsing the solvable set over time.

## Practical next step
Add pass@K and task-polarization tracking to the Phase 2 GRPO lab protocol:
- log per-task solve counts across checkpoints
- report pass@1, pass@K, and the fraction of tasks never solved
- watch for training runs that improve pass@1 while shrinking solvable coverage

## Uncertainty
- This summary is based on the AI Weekly alert, not a full paper read. Per-benchmark numbers are not yet checked here.
- PTGS is presented as a sampling fix; unclear from the alert how expensive it is to adopt in an existing TRL/verifiers pipeline.

## Verified 2026-10-09 (paper + repo read)
- Paper: arXiv:2610.01509, "Sharpening Tax in Post-Training", Changdae Oh et al. (Meta Superintelligence Labs / UW–Madison / NYU / Stanford). https://arxiv.org/abs/2610.01509
- Setup confirmed: 14 open base/post-trained checkpoint pairs, 4 families (Gemma-4, Ministral-3, Qwen2.5, Qwen3.5), 3 agentic benchmarks (BFCL v4 multi-turn, WebShop, ACEBench), 128 rollouts per task, 42 model-benchmark combinations.
- Headline confirmed: TaxS(128) > 0 in **36 of the 42** combinations (paper §4 / Fig. 6 discussion). Mechanism confirmed: post-training bimodalizes per-task success rates toward always-solved / never-solved, trading coverage (pass@K) for sampling efficiency and consistency (pass^K).
- PTGS adoption cost: low. Paper Algorithm 1 — per-prompt Beta posterior over success rate with forgetting factor γ, Thompson-sampled difficulty estimate mapped to a per-group sampling temperature Tx ∈ [1/τ, τ] (τ ∈ [1.2, 1.5], γ = 0.95, target rate ramp 0.25→0.5 in their runs). The paper states PTGS requires no change to the RL update itself and brings no computational overhead. Evaluated with PPO and GRPO on Qwen2.5-7B-Instruct (Sokoban, FrozenLake, 200 steps).
- Repo: https://github.com/changdaeoh/sharpening-tax (public, CC BY-NC 4.0). Metrics and PTGS run on CPU with numpy; the repo is explicitly "minimal working examples… not a reproduction package" — benchmark checkouts and RL infrastructure are left to the user. Contains `data/outcomes_42cells.csv.gz` with per-task (c, n) counts for all 42 cells.
- Diagnostic cheapness confirmed: TaxS estimated from 8 rollouts predicts TaxS at 32 rollouts with Spearman ρ = 0.85 (paper §4, Fig. 7), so a lightweight per-checkpoint logging addition captures most of the signal.
