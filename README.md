# LLM Post-Training Learnings

Personal learning tracker for LLM post-training — papers, labs, eval notes, and new findings as they land.

**Owner:** Ran ([@reachspirit](https://github.com/reachspirit))  
**Context:** Senior SWE (recommender systems + ML infrastructure) building toward staff-level LLM post-training expertise. 10–15 hrs/week. Guaranteed compute: 1× A100 80GB (8× H100 is aspirational, not assumed).

## What's here

| Path | What it holds |
|---|---|
| `learning-path.md` | The phased path (Phase 0 → 3) with build targets per phase |
| `reading-list.md` | Prioritized core reading, in rank order, plus later additions |
| `papers/` | One note per paper — what it introduced, why it matters, what to reproduce |
| `labs/` | Hands-on work: checklists, run logs, scoring templates |
| `findings/` | Dated notes on new results (benchmarks, talks, codebases) as they arrive |
| `notes/` | Topic notes (codebases, resources, career) |
| `templates/` | Copy-paste templates for new entries |

Interactive companion: the LLM Post-Training Learning Path page (built 2026-09-23, slug `llm-post-training-learning-path`).

## How new findings get filed

1. Copy the relevant template from `templates/` into `findings/` (dated) or `papers/`.
2. One finding per file. Date in the filename: `findings/2026-10-06-thinkingbox.md`.
3. Link it from `reading-list.md` if it earns a permanent slot.
4. If it changes how a lab should run, update the lab note and note the date.

## Current state (2026-10-16)

- Learning path researched and built (2026-09-23). Phases 0–3 defined.
- Phase 2 eval checklist ready (`labs/phase-2-eval-checklist.md`), distilled from ThinkingBox.
- Reading list: 9 core items in the interactive page; NB-LoRA tracked here pending artifact update (see `reading-list.md`).
- PostTrainBench GSM8K lab built out under `labs/posttrainbench-gsm8k/`: Session A eval/decontamination kit (runnable, audited — continuation-truncation, results-path, and short-item decontamination fixes applied) and Session B SFT kit (`prepare_sft_data.py`, TRL LoRA `train_sft.py`, tokenizer-only `preflight.py` with truncation and ending-format fixes applied). Session C (GRPO) is deliberately not pre-built — gated on SFT stability.
- Still pending a decision: promoting CorrGRPO to the interactive reading list, merging the Sharpening Tax logging addendum (`labs/phase-2-grpo-logging-addendum-draft.md`, still a draft) into the Phase 2 protocol, and adding NB-LoRA to the interactive page.

## License / visibility

Public repository — visibility and name confirmed 2026-10-16: `reachspirit/llm-post-training-learnings`. First push is pending completion of the GitHub connection.
