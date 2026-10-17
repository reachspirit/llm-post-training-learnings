# MLPerf Training v6.1 — first LLM post-training benchmark

- **Date found:** 2026-09-30 (proactive note delivered)
- **Source:** https://mlcommons.org/2026/09/mlperf-training-llm-post-training/

## What it is
MLPerf Training v6.1 added its first LLM post-training benchmark: GRPO on a Qwen 3.5 397B MoE coding agent, OpenHands harness, sandboxed SWE tasks with binary pass/fail rewards, reference implementation in NVIDIA NeMo-RL with Ray (disaggregated Megatron training, vLLM rollouts, agent sandboxes, async RL).

## Why it matters here
Read-only on 1× A100 — but the spec is a production-grade RLVR design doc. Skim: Environment, Implementation Details, "Managing Variance and Evaluation Cost." Reward-hacking notes (hidden test files) and pass@4 eval-stability discussion are production details the path doesn't cover. Phase 3 systems reference.
