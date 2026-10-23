# LLM Post-Training Learnings

Personal learning tracker for LLM post-training — papers, labs, eval notes, and new findings as they land.

**Context:** A personal learning tracker for LLM post-training — papers, findings, and hands-on labs. Two hardware tracks: a single A100 80GB for quick iteration, and an 8× H200 cluster (1024 GB total) for full-scale labs — rehearse on the A100, scale on the cluster.

## What's here

| Path | What it holds |
|---|---|
| `learning-path.md` | The phased path (Phase 0 → 3) with build targets per phase |
| `reading-list.md` | Prioritized core reading, in rank order, plus later additions |
| `papers/` | One note per paper — what it introduced, why it matters, what to reproduce |
| `labs/` | Hands-on work: checklists, run logs, scoring templates — A100 labs plus the 8× H200 cluster track in `labs/cluster-labs-h200.md` (C1–C5) |
| `findings/` | Dated notes on new results (benchmarks, talks, codebases) as they arrive |
| `notes/` | Topic notes (codebases, resources, career) |
| `templates/` | Copy-paste templates for new entries |

Interactive companion: the LLM Post-Training Learning Path page (built 2026-09-23, slug `llm-post-training-learning-path`).

## How new findings get filed

1. Copy the relevant template from `templates/` into `findings/` (dated) or `papers/`.
2. One finding per file. Date in the filename: `findings/2026-10-06-thinkingbox.md`.
3. Link it from `reading-list.md` if it earns a permanent slot.
4. If it changes how a lab should run, update the lab note and note the date.

## Current state (2026-10-17)

- Learning path researched and built (2026-09-23). Phases 0–3 defined.
- Two-track hardware plan (2026-10-17): A100 track remains the quick-iteration / rehearsal path; a new 8× H200 cluster track (1024 GB) adds labs C1–C5 in `labs/cluster-labs-h200.md` — full-parameter SFT head-to-head (C1), 8B reward model (C2), GRPO at 7–14B (C3), PostTrainBench at native fidelity incl. a 32B cell (C4), and a 32B GRPO capstone (C5). `learning-path.md` carries the matching cluster variant for each phase.
- Phase 2 eval checklist ready (`labs/phase-2-eval-checklist.md`), distilled from ThinkingBox.
- Reading list: 9 core items in the interactive page; NB-LoRA tracked here pending artifact update (see `reading-list.md`).
- PostTrainBench GSM8K lab built out under `labs/posttrainbench-gsm8k/`: Session A eval/decontamination kit (runnable, audited — continuation-truncation, results-path, and short-item decontamination fixes applied) and Session B SFT kit (`prepare_sft_data.py`, TRL LoRA `train_sft.py`, tokenizer-only `preflight.py` with truncation and ending-format fixes applied). Session C (GRPO) is deliberately not pre-built — gated on SFT stability.
- PKU post-training course (2026-10-17) adopted as the Phase 1 → early Phase 2 spine (`learning-path.md`); Session 1 start kit at `labs/pku-course/session-1-start.md` with a frozen 10-prompt baseline set (`labs/pku-course/session-1-baseline-prompts.md`) for the Session 2 base-vs-tuned comparison.
- Still pending a decision: promoting CorrGRPO to the interactive reading list, merging the Sharpening Tax logging addendum (`labs/phase-2-grpo-logging-addendum-draft.md`, still a draft) into the Phase 2 protocol, and adding NB-LoRA to the interactive page.

## License / visibility

Public repository. Content is kept free of personal details.
