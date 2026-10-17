# Notes — codebases & docs (read 2026-09-23, index)

## HuggingFace TRL — https://huggingface.co/docs/trl
- Trainers: SFTTrainer, DPOTrainer, GRPOTrainer, RewardTrainer, KTOTrainer, RLOO; experimental: OnlineDPO, ORPO, CPO, GKD. CLI: trl sft/dpo/grpo/kto/rloo/reward/distillation.
- Quickstart models: Qwen2.5-0.5B; datasets trl-lib/Capybara, DeepMath-103K, ultrafeedback_binarized, Polaris-Dataset-53K, kto-mix-14k. Reasoning models: reasoning_accuracy_reward().

## verl — https://github.com/volcengine/verl (ByteDance Seed)
- Open-source HybridFlow (arXiv:2409.19256, Sep 2024). Hybrid-controller programming model; 3D-HybridEngine zero-redundancy resharding between generate/train.
- Integrates FSDP, Megatron-LM, vLLM, SGLang; flexible per-model device mapping. Algos: PPO, GRPO, DAPO, GSPO, REINFORCE++, PRIME.
- DAPO recipe lives here.

## OpenRLHF — https://github.com/OpenRLHF/OpenRLHF
- Ray + vLLM + ZeRO-3 + HF Transformers. PPO, REINFORCE++, GRPO, RLOO, DAPO, async RL. 70B+, cross-GPU actor/reward separation.

## Open Instruct — https://github.com/allenai/open-instruct
- Tülu 3 reproduction: scripts/train/tulu3/{finetune_8b,dpo_8b}.sh; RLVR: scripts/train/rlvr/tulu_rlvr.sh (+ single-GPU debug script).
- Tülu 3 8B SFT: 64 H100 (8 nodes), DeepSpeed ZeRO-3, lr 5e-6, eff. batch 128, 2 epochs, flash-attn. Decontamination scripts included.

## axolotl — https://github.com/axolotl-ai-cloud/axolotl ; docs https://docs.axolotl.ai
- YAML configs; LoRA/QLoRA, full FT, DPO/IPO/KTO/ORPO, GRPO/GDPO, RM/PRM; multipacking, FlashAttn 2/3/4, Liger, Cut-CE, FSDP1/2, DeepSpeed, sequence parallelism.
- QLoRA example fits ~8.2 GiB VRAM.

## Others (from community lists, index)
- verifiers (https://github.com/willccbb/verifiers) — GRPO in verifiable envs, small-scale.
- understand-r1-zero (https://github.com/sail-sg/understand-r1-zero) — 7B Dr. GRPO recipe.
- NVIDIA NeMo-RL (https://github.com/NVIDIA-NeMo/RL); OAT (https://github.com/sail-sg/oat); ROLL (Alibaba); AReaL; RAGEN; Meta LlamaRL (arXiv:2505.24034v2).
- LLaMA-Factory, Unsloth (2× faster, 60% less VRAM), TorchTune, ms-swift — SFT-side.
