#!/usr/bin/env python3
"""Session A baseline eval: Qwen3-1.7B-Base on a limited GSM8K test slice.

Grades terminal state, not the run story (Phase 2 checklist): the score comes
from exact-match on the extracted final answer vs the GSM8K gold answer,
computed by this script — never from the model's own summary.

Design notes:
- Pure-logic functions (extract_answer, normalize_number, score_samples) have
  no third-party imports and are covered by selftest.py, so the scoring path
  can be verified without a GPU. Model loading/generation is isolated in
  generate() and imports transformers/torch lazily.
- With --samples > 1 the summary also reports pass@K and the never-solved
  fraction (the optional fields already in the run sheet's scoring block).
  Default --samples 1 keeps Session A cheap: pass@1 only.

Usage (on the A100 box):
    python eval_gsm8k.py --model Qwen/Qwen3-1.7B-Base --limit 200 \
        --fewshot 4 --out results/base-fewshot-200.jsonl

GSM8K source: the `gsm8k` HF dataset, config `main`, split `test` (eval only —
never train on this split; see decontaminate.py for the train-side check).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pure logic (no third-party imports — testable anywhere)
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
_NUMBER_RE = re.compile(r"-?[\d,]*\.?\d+")


def normalize_number(s: str) -> str | None:
    """Normalize a numeric string for exact-match comparison.

    Removes thousands separators and trailing zeros after a decimal point so
    '1,000', '1000' and '1000.0' compare equal. Returns None if unparseable.
    """
    try:
        v = float(s.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None
    if v == int(v):
        return str(int(v))
    return str(v)


def truncate_continuation(text: str) -> str:
    """Cut a generation at the start of an invented next problem.

    generate() sets no stop sequence, and a base model prompted with
    'Question: ... Answer:' will often continue past its own '#### N'
    with a fabricated '\\n\\nQuestion: ... #### M' block. Without this
    cut, extraction (which takes the LAST '####') scores the fabricated
    problem's answer instead of this problem's — silently corrupting
    pass@1. Added 2026-10-11 after reproducing exactly that failure.
    """
    idx = text.find("\nQuestion:")
    return text[:idx] if idx != -1 else text


def extract_answer(text: str) -> str | None:
    """Extract the model's final answer.

    Prefers the GSM8K '#### N' convention; falls back to the last number in
    the text (base models often ignore the format on zero-shot). The text
    is first truncated at any invented next 'Question:' block (see
    truncate_continuation), so 'last' means last of THIS problem.
    """
    text = truncate_continuation(text)
    m = _ANSWER_RE.findall(text)
    if m:
        return normalize_number(m[-1])
    nums = _NUMBER_RE.findall(text.replace("$", "").replace("%", ""))
    if nums:
        return normalize_number(nums[-1])
    return None


def gold_answer(answer_field: str) -> str | None:
    """GSM8K's `answer` field ends with '#### N' — that N is gold."""
    m = _ANSWER_RE.findall(answer_field)
    return normalize_number(m[-1]) if m else None


def score_samples(sample_correct: list[list[bool]]) -> dict:
    """Aggregate per-problem correctness across samples.

    sample_correct[i] = list of bools, one per sampled generation for
    problem i. Returns pass@1 (mean over all samples), pass@K (fraction of
    problems solved at least once) and never_solved fraction.
    """
    n_problems = len(sample_correct)
    if n_problems == 0:
        return {"problems": 0, "pass@1": None, "pass@k": None,
                "never_solved_fraction": None}
    total = sum(len(s) for s in sample_correct)
    correct = sum(sum(s) for s in sample_correct)
    solved_once = sum(1 for s in sample_correct if any(s))
    return {
        "problems": n_problems,
        "samples_per_problem": sample_correct[0].__len__(),
        "pass@1": correct / total if total else None,
        "pass@k": solved_once / n_problems,
        "never_solved_fraction": 1 - solved_once / n_problems,
    }


FEWSHOT = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May. In total she sold 48 + 24 = 72 clips altogether.
#### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10.
#### 10

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose. Her grandparents decided to give her twice as much as her parents. How much more money does Betty need to save on her own?
Answer: Betty has 100/2 = 50 dollars. Her grandparents give her 15 * 2 = 30 dollars. So Betty needs 100 - 50 - 15 - 30 = 5 more dollars.
#### 5

Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Answer: He writes each friend 3 * 2 = 6 pages a week. He writes 2 * 6 = 12 pages every week. In a year he writes 12 * 52 = 624 pages.
#### 624

"""


def build_prompt(question: str, fewshot: int) -> str:
    if fewshot > 0:
        shots = FEWSHOT.strip().split("\n\n")[:fewshot]
        return "\n\n".join(shots) + f"\n\nQuestion: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


# ---------------------------------------------------------------------------
# Generation (A100 box only — imports are lazy so scoring stays testable)
# ---------------------------------------------------------------------------

def generate(model_id: str, prompts: list[str], samples: int,
             max_new_tokens: int, temperature: float,
             seed: int = 1234) -> list[list[str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if samples > 1:
        torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    out: list[list[str]] = []
    for prompt in prompts:
        enc = tok(prompt, return_tensors="pt").to(model.device)
        gens = []
        for _ in range(samples):
            with torch.no_grad():
                ids = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=samples > 1,
                    temperature=temperature if samples > 1 else None,
                    pad_token_id=tok.pad_token_id)
            gens.append(tok.decode(ids[0][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
        out.append(gens)
    return out


def load_test_slice(limit: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    return [{"question": r["question"], "gold": gold_answer(r["answer"])}
            for r in ds]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--limit", type=int, default=200,
                    help="eval slice size (Session A: limited slice, not full 1319)")
    ap.add_argument("--fewshot", type=int, default=4)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1234,
                    help="seed for sampled runs (--samples > 1); logged in summary")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Fail fast on an unwritable output path: the results file is only
    # opened AFTER generation, so a missing results/ directory used to
    # surface as a crash with the whole eval's compute already spent.
    # (Reproduced 2026-10-11: README's results/... path, no results/ dir.)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.parent.is_dir():
        print(f"cannot create output directory: {out_path.parent}", file=sys.stderr)
        return 1

    rows = load_test_slice(args.limit)
    prompts = [build_prompt(r["question"], args.fewshot) for r in rows]
    generations = generate(args.model, prompts, args.samples,
                           args.max_new_tokens, args.temperature, args.seed)

    sample_correct: list[list[bool]] = []
    with open(args.out, "w") as f:
        for row, gens in zip(rows, generations):
            preds = [extract_answer(g) for g in gens]
            correct = [p is not None and p == row["gold"] for p in preds]
            sample_correct.append(correct)
            f.write(json.dumps({"question": row["question"],
                                "gold": row["gold"], "preds": preds,
                                "correct": correct,
                                "generations": gens}) + "\n")

    summary = score_samples(sample_correct)
    # Full generation config in the summary: a score without limit /
    # samples / temperature / max_new_tokens / seed is not reproducible
    # from the artifact alone (Phase 2 provenance check).
    summary.update({"model": args.model, "fewshot": args.fewshot,
                    "limit": args.limit, "samples_requested": args.samples,
                    "temperature": args.temperature if args.samples > 1 else None,
                    "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                    "results_file": str(Path(args.out).resolve())})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
