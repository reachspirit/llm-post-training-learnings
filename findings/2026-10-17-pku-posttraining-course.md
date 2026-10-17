# PKU "LLM Post-Training Practice" — a runnable spine for Phase 1 and the Phase 2 on-ramp

- **Date found:** 2026-10-17
- **Source:** Course docs site, pointed out 2026-10-17; full curriculum enumerated from the site (all lecture, lab, papers, and resources pages)
- **Type:** Course / structured lab sequence

## Sources checked
- Course overview: https://posttrain.gaozhijun.me/docs/ — 《大语言模型后训练实践》, Peking University School of Software and Microelectronics, instructor Gao Zhijun, course 01714806, Spring 2026, 6 sessions × 3 h (60% lecture / 40% lab), taught in Chinese.
- Lecture and lab pages: https://posttrain.gaozhijun.me/docs/lecture-1/ through /lecture-6/, each with theory sub-pages, a full-code lab, and a recommended-papers page with reading guides.
- Grading: https://posttrain.gaozhijun.me/docs/grading/ — 5 lab reports (50%: code + training logs + 1–2 page analysis each) + final project (40%, must combine ≥2 techniques).
- Setup: https://posttrain.gaozhijun.me/docs/resources/setup-guide/ — single A100-40G/80G assumed; transformers ≥4.51, trl ≥0.15; all experiments LoRA/QLoRA on Qwen3.

## What it is (own summary)
A one-credit PKU graduate course that walks one stack — Qwen3 (1.7B workhorse, 8B in the compression lecture) + Hugging Face TRL — through the full post-training arc in five lab units: (1) SFT with QLoRA; (2) SFT data engineering (Self-Instruct→MAGPIE, dedup / difficulty grading / decontamination, Tülu 3 mixing) with self-chosen hyperparameters, an ablation, and LLM-as-Judge evaluation; (3) preference alignment as a controlled DPO-vs-SimPO experiment on UltraFeedback, with an explicit length-bias check on the preference data and a refusal-rate safety test; (4) RLHF theory (InstructGPT pipeline, Bradley-Terry RM, PPO) plus GRPO — a "mini DeepSeek-R1-Zero" lab running TRL `GRPOTrainer` on Qwen3-1.7B-Base + GSM8K with hand-written verifiable rewards; (5) quantization (PTQ/QAT, GPTQ/AWQ) at 8B, distillation, and capability extensions (multimodal, tool use). Lecture 6 is a final project combining ≥2 techniques. Paper spine: InstructGPT, DPO, SimPO, Tülu 3, LIMA, MAGPIE, DeepSeekMath, DeepSeek-R1, DAPO, Snell et al. — with per-paper reading guides and time estimates; Lambert's RLHF Book is listed as extended reading.

## Why it matters here
1. **It is, almost exactly, the A100 track with a syllabus.** Labs run 45 min–2 h on one A100 at 1.7B–8B — the same envelope as the PostTrainBench cells and Sessions A/B. Lab 4 is stronger than an analogy: same base model (Qwen3-1.7B-Base), same dataset (GSM8K), same algorithm (GRPO + verifiable rewards) as the sheeted PostTrainBench GSM8K cell and the optional Session C. It works as a rehearsal/reference run before running the cell blind.
2. **It covers Phase 1 in runnable form, in the path's own order.** SFT (with masked loss) → data engineering → DPO + one variant, where the variant choice (SimPO) is settled by experiment rather than by reading. That is the Phase 1 build list, sequenced.
3. **Its gaps are as informative as its coverage.** Reward modeling is derived in theory (§4.1) but never trained in a lab; training systems stop at a TRL/OpenRLHF/veRL survey; evaluation is judge-centric with no repeated-trial or contamination protocol. Those three gaps are precisely where the local materials are stronger (RM build + C2, cluster labs C1–C5, Session A + Phase-2 eval checklist) — so the course slots in as a spine without displacing them.
4. **Cost to note honestly:** all materials are in Chinese, and several labs default to Chinese datasets and a qwen3-max judge API. Content quality is high; pacing depends on reading technical Chinese, and English-data substitutions are the student's own deviation to make.

## Filed as
- Finding (this file). Referenced from `learning-path.md` as the Phase 1 → early Phase 2 course spine.
- Pairs with: `labs/posttrainbench-gsm8k/` (Sessions A/B/C), `labs/cluster-labs-h200.md`, `labs/phase-2-eval-checklist.md`.
