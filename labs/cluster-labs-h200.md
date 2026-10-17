# Cluster Labs — 8× H200 Track

Companion to `learning-path.md`. The A100 track (PostTrainBench GSM8K Sessions A/B) stays the quick-iteration path; this track uses an 8× H200 cluster (1024 GB total HBM) for the labs that need real scale. Added 2026-10-17.

**Working rule:** cluster time is the scarce resource. Rehearse every config, data pipeline, and eval on the A100 track first; the cluster runs the rehearsed version at scale. The Phase 2 eval checklist gates (`labs/phase-2-eval-checklist.md`) apply unchanged — bigger runs fail the same ways, just faster.

## Capability envelope (rules of thumb, BF16 + AdamW)

- Full fine-tuning state ≈ 16 bytes/param (weights + grads + Adam m/v + master weights), sharded by ZeRO-3/FSDP.
  - 8B ≈ 128 GB total → ~16 GB/GPU sharded. Comfortable.
  - 32B ≈ 512 GB total → fits in 1024 GB with activation checkpointing. The practical full-FT ceiling here.
  - 70B ≈ 1.1 TB → only with optimizer CPU offload; stretch, expect slow steps.
- RL: rollout generation dominates wall-clock. verl-style separation (rollout GPUs on vLLM, actor GPUs training) is the layout these labs teach.

## The labs

### C1 — Full-parameter SFT, head-to-head with LoRA (Phase 1)
Full FT of Qwen3-8B (ZeRO-3, 8× H200) on the same SFT data and eval slice as the A100 LoRA run from Session B. Deliverable: a comparison note — score delta, stability, wall-clock, and what full FT actually bought over LoRA at this scale. This is the experiment that turns "LoRA vs full FT" from folklore into a measured answer.

### C2 — Reward model at 8B (Phase 1 → 2 bridge)
Bradley–Terry reward model at 8B on a preference set (UltraFeedback-scale), with the reward-hacking probes from the path (length, sycophancy) run as first-class evals. The recsys transfer point: this is pairwise learning-to-rank with a transformer scorer.

### C3 — GRPO at 7–14B with real rollout throughput (Phase 2)
GRPO on Qwen3-7B (then 14B if the pipeline is stable) using verl + vLLM, rollout/actor split across the 8 GPUs. Reproduce the Dr. GRPO length-bias observation at this scale, then apply one DAPO fix (decoupled clip) and measure the delta with the repeated-trial eval protocol. Success = a run where the bottleneck analysis (rollout vs training time) is measured, not guessed.

### C4 — PostTrainBench at native fidelity (Phase 2)
The benchmark's reference cell is 10 h on one H100-class GPU — now runnable as specified instead of translated down to A100. Two variants: (a) one native cell for direct comparability with the paper's numbers; (b) a 32B-base cell to see where the 10 h discipline breaks at scale. The GSM8K A100 run sheet remains the rehearsal.

### C5 — Capstone: 32B GRPO with separated rollout (Phase 3)
End-to-end pipeline at 32B: ZeRO-3 training, tensor-parallel vLLM rollout, eval harness wired to the Phase 2 checklist, checkpoint + provenance discipline throughout. This is the portfolio artifact: a reproducible eval + training pipeline others can run.

## Sequencing

C1 after Session B v1 is stable on the A100. C2 any time after C1. C3 after the A100 GRPO taster (Session C) or directly if Session B is solid. C4 once C3's pipeline exists. C5 is the Phase 3 build target from the learning path, now actually schedulable.
