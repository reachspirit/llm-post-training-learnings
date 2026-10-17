# NB-LoRA — Reasoning-Preserving Fine-Tuning of Post-RL LLMs with Null-Basis LoRA

- **Paper:** arXiv:2609.25618, submitted Sep 22, 2026
- **Link:** https://arxiv.org/abs/2609.25618
- **Status:** v1 preprint, no public code yet (as of 2026-10-01 note). Treat results as promising, not settled.
- **Reading-list:** requested 2026-10-06; tracked as item 10 in `reading-list.md`. Not yet added to the interactive learning-path page (which currently ends at ThinkingBox, item 09).

## Problem
Once RL buys a model reasoning ability, fine-tuning onto a new domain can overwrite that reasoning.

## Method (as reported)
Estimate an approximate null basis from the model's reasoning activations — finding: reasoning concentrates in low-dimensional subspaces, leaving null-space capacity to adapt in — then reparameterize LoRA updates through that fixed basis under a layer-wise hidden-state preservation constraint.

## Reported result
Matches standard LoRA on adaptation while holding reasoning near pre-fine-tuning levels, generalizing to held-out reasoning benchmarks.

## Why it fits this path
- LoRA-scale → actually runnable on 1× A100 80GB (unlike cluster-scale benchmarks).
- Null-basis estimation from activations is linear-algebra-meets-systems — direct transfer from ML infra background.

## What to verify / reproduce
- [ ] Null-basis estimation procedure: which activations, which layers, rank chosen?
- [ ] Preservation constraint: exact form, cost during training?
- [ ] Held-out reasoning evals used; any length / verbosity confounds?
- [ ] Code release?

## Open questions
- Does the preservation hold after GRPO (not just SFT) on top?
- Interaction with DAPO / Dr. GRPO length-bias instrumentation?
