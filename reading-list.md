# Reading List

Prioritized. Read in rank order. "Current" = 2025–26 reasoning-RL wave; "canonical" = older but still required.

Interactive page count as of 2026-10-06: **9 resources** (items 01–09 below). NB-LoRA (item 10) is tracked here — it is **not yet in the interactive page**; ThinkingBox (09) is.

## Core

| # | Item | Date | Why |
|---|---|---|---|
| 01 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948v2) | Jan 2025 | The 2025 watershed. Pure RL (GRPO, rule-based rewards) elicited long reasoning; full recipe adds cold-start SFT, synthetic data, further SFT/RL, distillation. |
| 02 | [DAPO](https://arxiv.org/abs/2503.14476v2) | Mar 2025 | Open, instrumented R1-style recipe. Four GRPO fixes: Clip-Higher, dynamic sampling, token-level loss, overlong reward shaping. Paper + code. |
| 03 | [Understanding R1-Zero-Like Training / Dr. GRPO](https://arxiv.org/abs/2503.20783v1) | Mar 2025 | Antidote to hype. GRPO normalization biases toward longer answers; pretraining explains much of "emergent" reasoning. Paper + code. |
| 04 | [Direct Preference Optimization](https://huggingface.co/papers/2305.18290) | 2023 | Pragmatic offline default. KL-regularized RLHF optimum as a supervised loss — no RM, no RL loop. Canonical. |
| 05 | [The RLHF Book — Nathan Lambert](https://rlhfbook.com) | 2025– | Structured reference. Living textbook by the Tülu / Open Instruct lead. |
| 06 | [Fine-tuning & RL for LLMs — DeepLearning.AI](https://www.deeplearning.ai/courses/fine-tuning-and-reinforcement-learning-for-llms-intro-to-post-training) | Oct 2025 | Best course-shaped path. SFT, RM, PPO, GRPO, LoRA, synthetic data, evals, production go/no-go. |
| 07 | [Hugging Face TRL](https://huggingface.co/docs/trl) | Living | Fastest route from paper to run. SFTTrainer / DPOTrainer / GRPOTrainer / RewardTrainer + CLI. |
| 08 | [Reward Hacking in RL — Lilian Weng](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) | Nov 2024 | Failure-mode bible. Oracle vs human vs proxy rewards; sycophancy, U-Sophistry. Canonical. |
| 09 | [ThinkingBox — Microsoft / Hugging Face](https://huggingface.co/blog/microsoft/thinkingbox) | 2026 | Grade the state, not the story. 507 stateful workflows, 20 clean runs each, scored on terminal backend state. Distilled into `labs/phase-2-eval-checklist.md`. |

## Additions (post-2026-09-23 path build)

| # | Item | Date | Status |
|---|---|---|---|
| 10 | NB-LoRA — "Reasoning-Preserving Fine-Tuning of Post-RL LLMs with Null-Basis LoRA" ([arXiv:2609.25618](https://arxiv.org/abs/2609.25618)) | Sep 22, 2026 | **Requested 2026-10-06; not yet in interactive page.** LoRA-scale, runnable on 1× A100. Estimates a null basis from reasoning activations; reparameterizes LoRA through it under a layer-wise preservation constraint. v1 preprint, no public code yet — promising, not settled. Note: `papers/nb-lora.md` |
| — | MLPerf Training v6.1 LLM post-training benchmark ([MLCommons announcement](https://mlcommons.org/2026/09/mlperf-training-llm-post-training/)) | Sep 2026 | Read-only reference (not runnable on 1× A100). Production-grade RLVR design doc: GRPO on Qwen 3.5 397B MoE, OpenHands, NeMo-RL. See `findings/2026-09-30-mlperf-post-training.md` |

## To file
- Kolibri 78B MoE — hardware-fit note (queued for delivery 2026-10-06; file when delivered).
- BaiZe single-A100 codebase — Phase 0.5/1 slot-in brief (prepared 2026-10-05; file when confirmed).
- Nebius async-RL talk — Phase 2 watch brief (prepared 2026-10-06; file when confirmed).

## New findings filed 2026-10-06
- Sharpening Tax in Post-Training — RL post-training can raise pass@1 while collapsing pass@K coverage; relevant to Phase 2 GRPO eval design. See `findings/2026-10-06-sharpening-tax.md`.
- CorrGRPO — Pearson-correlation fix to GRPO's multi-reward normalization bias; 0.5B–8B code models, code released. Natural follow-on to Dr. GRPO (item 03). See `findings/2026-10-06-corrgrpo.md`.
