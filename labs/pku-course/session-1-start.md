# PKU Course — Session 1 start kit (Lecture 1 + Lab 1, part 1)

Prepared 2026-10-17 from the course setup guide and Lecture 1 page
(https://posttrain.gaozhijun.me/docs/resources/setup-guide/ and
https://posttrain.gaozhijun.me/docs/lecture-1/). Adapted to the local
A100 80GB box — the course's AutoDL steps are replaced with the
equivalent local steps. No personal details.

## What Session 1 is

- Lecture 1 (~80 min theory): post-training overview, SFT core
  concepts (ChatML, masked loss), PEFT/LoRA/QLoRA, eval methods
  (LLM-as-Judge, human preference, capability benchmarks), Qwen3's
  four-stage pipeline incl. `/think` vs `/no_think`.
- Lab 1 (~100 min hands-on): first full fine-tune — Qwen3-1.7B base
  → instruction-following assistant via QLoRA SFT.
- Week 1 Session 1 deliverable (per path): L1 theory notes mapping
  Qwen3's pipeline onto the feedback-source × optimizer framework,
  environment verified, QLoRA load working, baseline outputs from
  the base model logged before any training.

## Environment checklist (run on the A100 box, ~15 min)

Course-pinned versions (hard requirements, from the setup guide):

- [ ] Python / PyTorch present (course reference image: PyTorch 2.3.0,
      Python 3.10, CUDA 12.1 — a newer local stack is fine if the
      version checks below pass)
- [ ] `pip install "transformers>=4.51.0" trl peft accelerate bitsandbytes datasets tokenizers`
      - transformers ≥ 4.51.0 is a hard floor — Qwen3 architecture is
        unavailable in older versions
      - trl ≥ 0.15.0 (setup guide recommendation)
      - bitsandbytes ≥ 0.43.0; if the install fails on CUDA mismatch,
        retry with `pip install bitsandbytes --prefer-binary`
- [ ] Eval/viz extras used later in the course (install once now):
      `pip install rouge-score nltk scikit-learn matplotlib seaborn wandb gradio pandas numpy tqdm`
      (vllm and llamafactory are only needed from Lecture 4/5 — skip for now)

Version verification (course's own checks, condensed):

```python
import transformers, trl, peft, bitsandbytes, torch
from packaging import version
assert version.parse(transformers.__version__) >= version.parse("4.51.0")
print(transformers.__version__, trl.__version__, peft.__version__,
      bitsandbytes.__version__, torch.__version__)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## Pre-downloads (do before the session, ~10 min on a fast link)

The course pre-downloads to avoid mid-lab network failures. Only the
Lab 1–2 items are needed for Week 1:

- [ ] `Qwen/Qwen3-1.7B` (~3.4 GB — Lab 1 workhorse, instruct version)
- [ ] `Qwen/Qwen3-1.7B-Base` (~3.4 GB — baseline outputs in Session 1;
      also the Lab 4 / PostTrainBench base, so this download is reused)
- [ ] UltraChat subset for Lab 1:
      `load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft[:10000]')`
      saved to disk

Deferred (not Week 1): Qwen3-8B (Lecture 5), Qwen3-0.6B pair,
Qwen3-VL-2B, DeepSeek-R1-Distill-Qwen-1.5B, UltraFeedback (Lecture 3),
GSM8K (already local via the Session A kit).

## Run order for the session

1. Theory pass (~80 min): L1 pages; write the Qwen3 four-stage
   pipeline vs feedback-source × optimizer mapping as you go — that
   mapping is the session's written deliverable, not an afterthought.
2. Env verification block above — all asserts green before touching
   the model.
3. QLoRA load of Qwen3-1.7B; confirm memory headroom on the 80GB card.
4. Baseline outputs: run the base model on a fixed 10-prompt set and
   save verbatim outputs (this set is reused in Session 2 for the
   base-vs-tuned table — keep prompts and decoding settings fixed
   and logged).
5. Log the session using `templates/lab-log.md`: hardware, resolved
   library versions, baseline prompt set + settings, any install
   deviations from the course pins.

## Known traps carried over from the Session B kit

- max_length / truncation: TRL truncation defaults to `keep_start`,
  which can cut the trailing answer marker and the appended EOS —
  the Session B `preflight.py` lesson applies to Lab 1's SFT config
  verbatim. Check the course notebook's max_length against the
  UltraChat length distribution before the full run.
- EOS / ending format: confirm the training targets end the way the
  eval expects before training, not after flat scores appear.
- The course defaults to Chinese datasets and a qwen3-max judge API
  in places; substituting English data is a deliberate deviation —
  log it in the lab report when it happens rather than silently.

## Source notes

- Setup guide: pinned versions, model list/sizes, dataset
  pre-download snippets, verify script — all from
  https://posttrain.gaozhijun.me/docs/resources/setup-guide/
  (read 2026-10-17).
- Lecture 1 structure, timings, and learning objectives from
  https://posttrain.gaozhijun.me/docs/lecture-1/ (read 2026-10-17).
- Course context and how it maps to the path:
  `findings/2026-10-17-pku-posttraining-course.md`.
