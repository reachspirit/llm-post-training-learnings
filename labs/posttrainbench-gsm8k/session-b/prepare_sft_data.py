#!/usr/bin/env python3
"""Session B data prep: GSM8K *train* split -> decontamination-ready SFT file.

Pipeline position (Session B of the weekend run sheet):
    prepare_sft_data.py  ->  session-a/decontaminate.py  ->  train_sft.py

Output JSONL rows carry three fields the rest of the kit depends on:
  question    raw question text — this is what decontaminate.py checks
              (its TEXT_FIELDS order reads "question" first), so the
              decontamination step audits the actual training questions.
  prompt      exactly the eval harness's format: "Question: {q}\nAnswer:"
              (matches eval_gsm8k.build_prompt with fewshot=0, so SFT
              trains the format the baseline was scored in).
  completion  " {reasoning}\n#### {gold}" — the GSM8K train answer with
              calculator annotations stripped (see strip_annotations).

Pure stdlib for the conversion logic (covered by selftest.py, no GPU /
no downloads). Loading the HF dataset is isolated in load_rows(); a
local JSONL with the same {"question", "answer"} shape works via
--from-jsonl, which is also how the selftest fixtures run.

Usage:
    python prepare_sft_data.py --out sft_candidates.jsonl
    python prepare_sft_data.py --from-jsonl raw_train.jsonl --out sft_candidates.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

# GSM8K train answers embed calculator markup like <<20*2=40>> inline in
# the reasoning ("He has 20*2=<<20*2=40>>40 apples"). Those annotations are
# dataset tooling, not model output — training on them teaches the model
# to emit markup the eval never produces. Strip them, keep the result.
_ANNOTATION_RE = re.compile(r"<<[^>]*>>")
_ANSWER_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
_TRAILING_ANSWER_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)\s*$")
_WS_RE = re.compile(r"\s+")


def strip_annotations(text: str) -> str:
    return _ANNOTATION_RE.sub("", text)


def gold_answer(answer_field: str) -> str | None:
    m = _ANSWER_RE.findall(answer_field)
    if not m:
        return None
    try:
        v = float(m[-1].replace(",", ""))
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


def convert_row(row: dict) -> dict | None:
    """One GSM8K-shaped row -> SFT row, or None if it must be dropped.

    Drops: missing/blank question, or an answer with no parseable
    '#### N' gold (an SFT target whose final answer cannot be verified
    against its own gold has no place in a verifiable-reward pipeline).
    """
    q = (row.get("question") or "").strip()
    ans = row.get("answer") or ""
    gold = gold_answer(ans)
    if not q or gold is None:
        return None
    reasoning = strip_annotations(ans).strip()
    # Canonicalise the trailing '#### N' to the normalised gold
    # (added 2026-10-13): the source answer may write it '#### 1,000'
    # or '#### 72.00' while gold_answer() normalises the gold field to
    # '1000' / '72'. Left verbatim, prep, preflight.py and the eval
    # disagreed on whether such rows end correctly — eval normalises
    # (its selftest covers the comma case), preflight originally did an
    # exact-string check and flagged them. One canonical target format
    # in training data, and preflight now compares normalised values.
    m = _TRAILING_ANSWER_RE.search(reasoning)
    if m:
        reasoning = reasoning[:m.start()] + f"#### {gold}"
    if not reasoning.endswith(f"#### {gold}") and _ANSWER_RE.search(reasoning) is None:
        return None
    return {
        "question": q,
        "prompt": f"Question: {q}\nAnswer:",
        "completion": f" {reasoning}",
        "gold": gold,
    }


def convert(rows: list[dict], limit: int = 0, seed: int = 1234) -> tuple[list[dict], dict]:
    """Convert + dedupe (by normalized question) + deterministic shuffle."""
    out, seen = [], set()
    dropped_unparseable = dropped_dupe = 0
    for row in rows:
        r = convert_row(row)
        if r is None:
            dropped_unparseable += 1
            continue
        key = _WS_RE.sub(" ", r["question"].lower()).strip()
        if key in seen:
            dropped_dupe += 1
            continue
        seen.add(key)
        out.append(r)
    random.Random(seed).shuffle(out)
    if limit > 0:
        out = out[:limit]
    stats = {"input": len(rows), "kept": len(out),
             "dropped_unparseable_or_blank": dropped_unparseable,
             "dropped_duplicate_question": dropped_dupe}
    return out, stats


def load_rows(from_jsonl: str | None) -> list[dict]:
    if from_jsonl:
        with open(from_jsonl) as f:
            return [json.loads(line) for line in f if line.strip()]
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    return [{"question": r["question"], "answer": r["answer"]} for r in ds]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-jsonl", default=None,
                    help="local JSONL with question/answer fields; default: HF gsm8k train split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = keep all")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rows = load_rows(args.from_jsonl)
    out, stats = convert(rows, limit=args.limit, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with open(out_path, "w") as f:
        for r in out:
            line = json.dumps(r) + "\n"
            digest.update(line.encode())
            f.write(line)
    stats.update({"out": str(out_path.resolve()), "sha256": digest.hexdigest(),
                  "seed": args.seed,
                  "source": args.from_jsonl or "HF gsm8k / main / train"})
    print(json.dumps(stats, indent=2))
    print("Next: run session-a/decontaminate.py on this file before training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
