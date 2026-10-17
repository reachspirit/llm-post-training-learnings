# ThinkingBox — grade the database state, not the agent's story

- **Date found:** 2026-10-06
- **Source:** Microsoft / Hugging Face — https://huggingface.co/blog/microsoft/thinkingbox
- **Type:** Agent benchmark (507 stateful workflows)

## What it is
Each workflow run 20 times from a clean backend, scored on the records and side effects left behind — not on the final response text. Worked example: a support agent whose summary says "resolved" while the ticket's required end state is still on hold.

## Why it matters here
Agent-eval analogue of reward hacking: the proxy (fluent summary, well-formed tool calls) diverges from ground truth (system state) — same Goodhart shape as RLHF. Three views separate capability from reliability:
- pass@1 — usual performance
- pass@20 — breadth (can it ever?)
- 20/20 — dependability (can it always?)

Plus cost-per-dependable-success. Failure analysis leaned toward tool handling and error recovery over abstract reasoning.

## Filed as
- Reading list item 09 (`reading-list.md`)
- Distilled into `labs/phase-2-eval-checklist.md` (2026-10-06)
