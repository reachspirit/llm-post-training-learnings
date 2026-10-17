# Phased Learning Path

Paced for a few study sessions per week. Hardware tracks: **1× A100 80GB** for quick iteration, plus an **8× H200 (1024 GB)** cluster track for full-scale labs — see `labs/cluster-labs-h200.md`.

Source: research report 2026-09-23 (`report.md` in research notes). Interactive version: artifact `llm-post-training-learning-path`.

## Phase 0 — Foundations (3–4 weeks)
*Goal: speak the language.*
Transformer mechanics at "can read a training script" level; PyTorch distributed basics (DDP/FSDP/ZeRO); RL fundamentals (policy gradients, value functions, PPO's clipped objective).

**Build:** reimplement a tiny policy-gradient loop (bandit / toy env) to internalize advantage estimation before touching LLMs.

## Phase 1 — Core post-training methods (5–7 weeks)
*Goal: run the full offline stack.*
SFT with correct loss masking, DPO + one variant (SimPO or KTO), reward modeling with Bradley-Terry loss. Read InstructGPT, DPO, LIMA, Constitutional AI; Lambert's RLHF Book in parallel.

**Build:** SFT a 0.5–7B model with TRL or axolotl → train a reward model on UltraFeedback → DPO it → evaluate with an LLM judge and try to detect reward hacking (length, sycophancy). Write up what broke.

*Cluster variant (8× H200):* full-parameter SFT of an 8B model head-to-head against the LoRA run, and an 8B reward model — labs C1/C2 in `labs/cluster-labs-h200.md`.

*Slot-in candidate:* BaiZe (single-A100 tokenizer→DPO codebase) as a Phase 0.5 / early Phase 1 whole-pipeline read — see `findings/` when filed.

## Phase 2 — Online RL & reasoning (5–7 weeks)
*Goal: understand the 2025 frontier mechanistically.*
GRPO (DeepSeekMath), DeepSeek-R1, DAPO, Dr. GRPO; RLVR with verifiable rewards; reward-hacking science (Weng).

**Build:** GRPO run on math (TRL `GRPOTrainer` or `verifiers`) — reproduce the Dr. GRPO length-bias observation yourself; then try one DAPO fix (e.g., decoupled clip). Use `labs/phase-2-eval-checklist.md` for state-based, repeated-trial eval before trusting any checkpoint.

*Cluster variant (8× H200):* GRPO at 7–14B with verl + vLLM rollouts and a PostTrainBench cell at the benchmark's native 10 h single-GPU fidelity — labs C3/C4 in `labs/cluster-labs-h200.md`.

*Watch brief candidate:* Nebius async-RL talk (staleness, in-flight weight updates, CISPO vs DAPO) as a pre-lab instrumentation checklist — see `findings/` when filed.

## Phase 3 — Systems & frontier (ongoing)
*Goal: operate at team scale.*
verl or OpenRLHF on multi-GPU; eval-harness ownership (lm-eval / oe-eval); HybridFlow paper; agentic RL (RLEF/SWiRL) and the GRPO-family wave.

**Build:** a reproducible eval + training pipeline others can run; a substantive open-source contribution.

*Cluster capstone (8× H200):* 32B GRPO with separated rollout/actor and the eval harness wired in — lab C5 in `labs/cluster-labs-h200.md`.

## Career note
Staff-level differentiators in post-training: reward/eval design taste, training-stability debugging, RL systems work (rollout generation is the bottleneck). A recommender-systems + ML-infrastructure background transfers unusually well (Bradley-Terry = learning-to-rank; bandits → RL exploration; distributed training / serving / data flywheels). Background reading: `notes/career.md`.
