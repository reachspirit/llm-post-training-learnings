# Notes — key papers (read 2026-09-23, index)

## DeepSeek-R1 — arXiv:2501.12948 (Jan 2025; Nature 645, 2025)
URL: https://arxiv.org/abs/2501.12948v2
- Pure RL (GRPO) on DeepSeek-V3-Base, no SFT for R1-Zero; rule-based accuracy + format rewards; no learned RM (gets hacked).
- Emergent: self-verification, reflection, "aha moment"; AIME 2024 pass@1 15.6% → 71.0%.
- Full R1: cold-start SFT → GRPO → synthetic data → SFT → GRPO; distillation to small dense models. PRM and MCTS didn't help.
- Sources: arXiv abs; zylcode deepseek-research.md; whisperre ai-agent-roadmap; trishambp implementation notes.

## DAPO — arXiv:2503.14476 (Mar 2025, ByteDance Seed + Tsinghua AIR + HKU)
URL: https://arxiv.org/abs/2503.14476v2
- Four GRPO fixes: Clip-Higher (decoupled clip bounds), dynamic sampling, token-level policy gradient loss, overlong reward shaping.
- 50 on AIME 2024 with Qwen2.5-32B; code + dataset open, built on verl. Project page: https://dapo-sia.github.io/

## Understanding R1-Zero-Like Training / Dr. GRPO — arXiv:2503.20783 (Mar 2025, Sea AI Lab/NUS/SMU)
URL: https://arxiv.org/abs/2503.20783v1
- GRPO's length + std normalization biases toward longer (often wrong) answers; Dr. GRPO removes both.
- DeepSeek-V3-Base already showed "aha moments" pre-RL → pretraining explains much of emergence.
- Minimalist 7B recipe: 43.3% AIME 2024. Code: https://github.com/sail-sg/understand-r1-zero

## GRPO origin — DeepSeekMath, arXiv:2402.03300 (2024)
- Confirmed via GRPO-family locator table (zjunlp/sciatlas). Group-relative advantage, no critic.

## DPO — Rafailov et al., NeurIPS 2023
- Paper page confirmed: https://huggingface.co/papers/2305.18290 (via TRL docs). Closed-form RLHF optimum → supervised loss.

## InstructGPT — Ouyang et al., 2022 — arXiv:2203.02155 (confirmed via sage wiki).
## Tülu 3 — Lambert et al., AllenAI, Dec 2024 — SFT→DPO→RLVR open recipe; repo verified, paper arXiv page not verified this session.
