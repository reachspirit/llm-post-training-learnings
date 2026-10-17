# Phase 2 Eval Checklist

A drop-in checklist for your GRPO / verifier labs. Built on one idea from ThinkingBox: **grade what the system left behind, not what the model said it did.**

Use this before you trust any Phase 2 run — DPO, GRPO, or reward-model eval.

---

## Core principle

- **Trajectory is a claim, state is evidence**: A clean final message and valid tool calls do not prove the task worked. Check the terminal backend state and side effects.

- **One success is not reliability**: A model that passes once and fails the next four times is not dependable. Measure consistency, not just capability.

- **Cost means cost per dependable outcome**: Cheapest per attempt is not cheapest per trustworthy result.

Source idea: [ThinkingBox - Microsoft / Hugging Face](https://huggingface.co/blog/microsoft/thinkingbox) — 507 stateful workflows, each run 20 times from a clean backend, scored on records and side effects.

---

## 1. Define terminal state first

Do this before writing the reward or eval script.

- [ ] **Write the required end state**: List every field, record, and file that must exist after a correct run, with exact values or predicates.

- [ ] **Write forbidden side effects**: List records that must *not* be created, modified, or deleted. Extra effects count as failures.

- [ ] **Write missing-effect checks**: List required effects that are easy to skip silently (e.g., ticket status set, refund issued, file written).

- [ ] **Make checks executable**: Implement as DB assertions, file diffs, API state queries, or unit tests — not LLM-judge text match alone.

- [ ] **Start from clean state**: Every trial resets to an identical backend snapshot. No carry-over between runs.

**Phase 2 tie-in:** This becomes your verifier. If your GRPO reward cannot be expressed as a terminal-state check, you are optimizing a proxy.

---

## 2. Score state, not summaries

- [ ] **Do not reward fluent summaries**: A response that says "resolved" while the ticket is still on hold is a failure, even if the text looks correct.

- [ ] **Check wrong values**: Validate field values, not just that a state-changing call happened.

- [ ] **Check extra effects**: Flag unintended writes. In ThinkingBox-style evals, extra effects are a common hidden failure mode.

- [ ] **Check clean termination separately**: A run can terminate without error and still leave wrong state. Track "terminated cleanly but failed checks" as its own category.

- [ ] **Keep LLM-judge as secondary**: Use it for explanation quality only after executable checks pass.

**Red flag:** If 60%+ of your failures still "looked successful" in logs, your proxy has diverged from ground truth.

---

## 3. Run repeated trials

- [ ] **Run N = 20 per task**: Same prompt, same clean backend, independent runs. Fewer runs hide variance.

- [ ] **Use the same harness for all models**: Same tools, same timeouts, same reset logic. Otherwise you compare harnesses, not models.

- [ ] **Log every trial**: Store prompt, trajectory, tool calls, terminal state, check results, token/cost usage, and failure signature.

- [ ] **Do not cherry-pick**: Report all attempts. Best-of-N is a different metric from reliability.

**A100 note:** For a 7B GRPO lab, 20 trials × 30-50 tasks is more informative than 200 tasks × 1 trial.

---

## 4. Report three views

Report all three. Each answers a different question.

| Metric | What it measures | Question it answers |
| --- | --- | --- |
| pass@1 | Share of all attempts that pass | How does it usually do? |
| pass@20 | Share of tasks passed at least once | Can it ever do this? |
| Observed 20/20 | Tasks passing all 20 attempts | Can it always do this? |

- [ ] **Never report pass@20 alone for deployment decisions**: Breadth is not dependability.

- [ ] **Track retention**: What share of pass@1 survives to 20/20? A large drop means a consistency problem, not a capability problem.

- [ ] **Break out by domain/task type**: A model can lead in one workflow family and collapse in another.

---

## 5. Track cost two ways

- [ ] **Cost per successful attempt**: Total cost for one pass over all tasks divided by number of successful attempts.

- [ ] **Cost per dependable task**: Total cost for the full repeated campaign divided by number of tasks passing 20/20.

- [ ] **Compare both before choosing a checkpoint**: The cheapest model per success is often not cheapest per dependable success.

- [ ] **Log tokens per trial**: Input, output, and cache separately if your stack exposes them. Price consistently across models.

**Use this when comparing:** base vs SFT vs DPO vs GRPO checkpoints, or different reward designs.

---

## 6. Classify failures

Assign one deterministic signature per failed trial.

- [ ] **Tool handling**: Bad arguments, unhandled errors, failed preconditions, empty lookups not recovered.

- [ ] **Wrong state updates**: Tool called correctly but terminal values are wrong.

- [ ] **Incomplete resolution**: Required user-facing outcome not actually delivered.

- [ ] **No state-changing action**: Model explained or planned but never committed the change.

- [ ] **Review the mix before tuning the model**: If most failures are tool handling, fix retries, tool surface, and error recovery first.

**Production pattern from ThinkingBox:** Most failures were retry/recovery problems, not abstract reasoning failures.

---

## 7. Add pre-commit guards

Bring the benchmark lesson into your lab and later into production.

- [ ] **Check terminal state before commit**: Verify invariants before you mark a task done or write a final record.

- [ ] **Retry only recoverable errors**: Classify tool/system errors so retries target transient failures, not logic errors.

- [ ] **Shrink the tool surface**: Expose only tools the workflow needs. Fewer tools, fewer wrong writes.

- [ ] **Gate irreversible actions**: Require explicit approval or a second check for changes you cannot cheaply undo.

- [ ] **Test the guard**: Show that the guard catches a case where the summary says success but state disagrees.

---

## Quick scoring template

Copy this into your lab notebook for each checkpoint.

- **Checkpoint / reward version:**
- **Tasks / trials per task:**
- **pass@1:**
- **pass@20 (breadth):**
- **Observed 20/20 (dependable):**
- **Cost per successful attempt:**
- **Cost per dependable task:**
- **Top failure signature:**
- **Clean-termination-but-wrong-state rate:**
- **Decision**: keep / retrain / fix harness / fix tools

---

## How to use this in Phase 2

1. Write terminal-state checks before you write the GRPO reward.
2. Run your baseline (SFT or DPO checkpoint) through the 20-trial harness first.
3. Train GRPO against the same executable verifier — not a text-match proxy.
4. Re-run the identical harness on the GRPO checkpoint.
5. Compare pass@1, 20/20, and cost per dependable task. Ship the checkpoint that improves dependability, not just best-case.

**Bottom line:** For post-training, your verifier *is* your product spec. If it cannot catch a confident summary that leaves the wrong state, it will teach the model to produce confident summaries that leave the wrong state.
