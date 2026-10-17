# DRAFT — Phase 2 GRPO logging addendum (Sharpening Tax)
Status: drafted 2026-10-09, pending approval. Source: arXiv:2610.01509 (verified read).

## Why
ThinkingBox's checklist already asks for pass@1 / pass@20 breadth / 20-20 dependability. Sharpening Tax adds the base-anchored version: is training *shrinking the set of tasks the model can ever solve*, even while pass@1 rises? Cheap to log now, expensive to reconstruct later.

## Per-checkpoint logging (add to the GRPO lab protocol)
For every checkpoint (base, SFT, DPO, GRPO step N), same task set, same harness:

| Field | Notes |
|---|---|
| `task_id` | stable ID |
| `n_rollouts` | use 32 if budget allows; 8 is enough for an early TaxS estimate (paper: TaxS(8) predicts TaxS(32), Spearman rho 0.85) |
| `c_success` | successes out of n |
| `pass@1` | c / n per task, averaged |
| `pass@K` | 1 - C(n-c, K)/C(n, K), K = 8 and 32 |
| `never_solved_frac` | share of tasks with c = 0 |
| `always_solved_frac` | share of tasks with c = n |

Watch: pass@1 up + never_solved_frac up = polarization. That is the failure mode this log exists to catch.

## Optional: Sharpening Tax vs base
Compute TaxS(K) = S_base(K) - S_post(K) from the (c, n) table (definitions in paper S4, Eqs. 1-3). Reference implementation: `sharpening_tax.metrics.Pair(...).tax([K])` in https://github.com/changdaeoh/sharpening-tax (CPU/numpy). Positive = post-training reduced test-time scalability vs base.

## PTGS note (if polarization appears)
PTGS (paper S5, Algorithm 1) is a sampling-side mitigation: per-prompt Beta posterior on success rate, temperature Tx in [1/tau, tau], tau 1.2-1.5, forgetting gamma 0.95. No change to the RL update rule. Repo marks itself as minimal examples, not a reproduction package — budget integration time accordingly.
