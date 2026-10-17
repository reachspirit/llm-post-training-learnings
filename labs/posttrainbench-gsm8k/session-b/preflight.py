#!/usr/bin/env python3
"""Session B pre-flight: catch truncation damage BEFORE the A100 run.

Why this exists (audit 2026-10-12 of train_sft.py against current TRL docs):

1. train_sft.py sets SFTConfig(max_length=512). TRL's truncation_mode
   defaults to "keep_start": a tokenized prompt+completion longer than
   max_length is cut from the END. For this kit the end is exactly what
   matters — the completion's trailing "#### {gold}" and the EOS token
   TRL appends during tokenization. An over-length row therefore trains
   an answer that never shows its final answer and never terminates.
   Scoring cannot see this (Session A grades the checkpoint, and a few
   poisoned rows just look like slightly worse SFT).

2. The completions in the JSONL carry no EOS themselves. That is NOT a
   bug — TRL appends the processing class's EOS token when tokenizing —
   but it is an assumption about the installed trl version, so this
   script asserts it empirically instead of trusting it.

What this script checks, on the machine that will train (needs the
tokenizer only — no GPU, no torch):

  a. Length distribution of prompt+completion (+1 for the appended EOS)
     over the DECONTAMINATED training file, and how many rows exceed
     --max-length (default 512, matching train_sft.py). Each such row is
     reported with its index and token count.
  b. Every completion ends with a "#### N" whose value — normalised
     the same way the Session A eval normalises it (commas stripped,
     trailing decimal zeros dropped) — equals its own row's gold field.
     The invariant truncation would destroy. (Originally an exact-string
     `endswith("#### {gold}")` check: that flagged "#### 1,000" /
     "#### 72.00" endings that prep keeps and eval scores correct as
     bad — a false blocker, fixed 2026-10-13; see Session B README.)
  c. tokenizer.eos_token_id exists (the token generation will stop on).

Exit code 0 = safe to train as configured. Exit 1 = over-length rows or
broken invariants; fix by raising --max-length in train_sft.py, or by
dropping/shortening the flagged rows — do not just proceed: keep_start
truncation hides the damage in the training loss.

Usage (A100 box, after decontamination, before train_sft.py):
    python preflight.py --data data/sft_clean.jsonl
    python preflight.py --data data/sft_clean.jsonl --max-length 512

Self-test (any machine, stub tokenizer, no downloads):
    python preflight.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import sys

_TRAILING_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)\s*$")


def normalize_number(s: str) -> str | None:
    """Same normalisation as session-a/eval_gsm8k.normalize_number:
    '1,000', '1000' and '1000.0' compare equal."""
    try:
        v = float(str(s).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None
    return str(int(v)) if v == int(v) else str(v)


def ending_ok(completion: str, gold: str) -> bool:
    """True iff the completion ends with '#### N' and N, normalised,
    equals the row's gold, normalised."""
    m = _TRAILING_RE.search(completion.rstrip())
    if not m:
        return False
    return normalize_number(m.group(1)) == normalize_number(gold)


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_rows(rows: list[dict], encode, max_length: int) -> dict:
    """Pure logic (stub-encodable for the selftest).

    encode: text -> list of token ids (no special tokens added).
    A row's trained length is prompt+completion plus one EOS token,
    which TRL appends at tokenization time.
    """
    over, bad_ending = [], []
    lengths = []
    for i, r in enumerate(rows):
        n = len(encode(r["prompt"])) + len(encode(r["completion"])) + 1
        lengths.append(n)
        if n > max_length:
            over.append({"index": i, "tokens": n})
        gold = str(r.get("gold", ""))
        if gold and not ending_ok(r["completion"], gold):
            bad_ending.append(i)
    lengths.sort()
    return {
        "rows": len(rows),
        "max_length": max_length,
        "token_max": lengths[-1] if lengths else 0,
        "token_p95": lengths[int(0.95 * (len(lengths) - 1))] if lengths else 0,
        "over_length_rows": over,
        "bad_ending_rows": bad_ending,
    }


def selftest() -> int:
    # Stub tokenizer: one token per word — enough to exercise counting,
    # the +1 EOS allowance, and the #### ending invariant.
    encode = lambda text: text.split()  # noqa: E731
    rows = [
        {"prompt": "Question: 2+2?\nAnswer:", "completion": " It is 4.\n#### 4",
         "gold": "4"},                                            # 8 tokens, ok
        {"prompt": " ".join(["q"] * 10), "completion": " ".join(["a"] * 10) + "\n#### 1",
         "gold": "1"},                                            # 23 tokens, over at 16
        {"prompt": "Question: x?\nAnswer:", "completion": " reasoning with no marker",
         "gold": "7"},                                            # bad ending
        {"prompt": "Question: c?\nAnswer:", "completion": " Has many.\n#### 1,000",
         "gold": "1000"},                                         # comma format: OK (eval normalises)
        {"prompt": "Question: d?\nAnswer:", "completion": " Costs this.\n#### 72.00",
         "gold": "72"},                                           # decimal format: OK (eval normalises)
        {"prompt": "Question: w?\nAnswer:", "completion": " Wrong total.\n#### 9",
         "gold": "10"},                                           # wrong value: bad ending
    ]
    rep = check_rows(rows, encode, max_length=16)
    failures = []
    if rep["rows"] != 6:
        failures.append(f"rows: {rep['rows']} != 6")
    if [o["index"] for o in rep["over_length_rows"]] != [1]:
        failures.append(f"over_length: {rep['over_length_rows']}")
    if rep["bad_ending_rows"] != [2, 5]:
        failures.append(f"bad_ending: {rep['bad_ending_rows']}")
    if failures:
        print("FAIL")
        for f in failures:
            print(" -", f)
        return 1
    print("OK — length counting (+1 EOS), over-length flag, #### ending invariant "
          "(normalised value vs gold: comma/decimal formats pass, wrong value fails)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="decontaminated prompt/completion JSONL")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--max-length", type=int, default=512,
                    help="must match train_sft.py --max-length (default 512)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.data:
        ap.error("--data is required (or pass --selftest)")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.eos_token_id is None:
        print("FAIL: tokenizer has no eos_token_id — generation has no stop token")
        return 1
    encode = lambda text: tok(text, add_special_tokens=False)["input_ids"]  # noqa: E731

    rows = load_rows(args.data)
    rep = check_rows(rows, encode, args.max_length)
    print(json.dumps({k: v for k, v in rep.items() if k != "over_length_rows"},
                     indent=2))
    if rep["over_length_rows"]:
        print(f"OVER-LENGTH ROWS ({len(rep['over_length_rows'])}): first 10 "
              f"{rep['over_length_rows'][:10]}")
    if rep["bad_ending_rows"]:
        print(f"BAD #### ENDINGS at rows: {rep['bad_ending_rows'][:10]}")
    if rep["over_length_rows"] or rep["bad_ending_rows"]:
        print("NOT SAFE TO TRAIN AS CONFIGURED — see docstring for fixes.")
        return 1
    print("SAFE: every row fits max_length with its #### ending and EOS intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
