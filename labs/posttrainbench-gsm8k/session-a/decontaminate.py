#!/usr/bin/env python3
"""Decontamination check for the PostTrainBench GSM8K cell.

The run sheet's terminal-state check is: 'no GSM8K test items in any train
file (split names are not proof — check contents)'. This script is that check:
word n-gram overlap between a candidate training file and the GSM8K *test*
questions. Any training item sharing an n-gram (default 10 words, normalized)
with any test question is flagged and excluded from the filtered output.

Pure stdlib — runs anywhere, no GPU, no datasets download for the train side.
The test questions come from a local file if --test-file is given, else from
the HF `gsm8k` dataset (test split) if `datasets` is installed.

Short-item fallback: an n-gram check is blind to any text shorter than n
words — it produces zero n-grams, so an exact duplicate of a short test
question can never be flagged. check() therefore also tests containment:
a test text shorter than n words appearing verbatim inside a training
item (or a training item shorter than n words appearing verbatim inside
a test text) is flagged. Added 2026-10-14 after reproducing an 8-word
exact duplicate passing with zero flags.

Input format: JSONL with a "question" field (or "text"/"prompt" — see
TEXT_FIELDS). Output:
  --out         filtered JSONL (flagged items removed)
  --report      JSON report: counts, threshold, flagged indices + matched
                n-gram examples — paste the summary into the dataset card /
                provenance log.

Usage:
    python decontaminate.py --train train_candidates.jsonl \
        --out train_clean.jsonl --report decontamination-report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys

TEXT_FIELDS = ("question", "text", "prompt", "instruction")
_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def item_text(item: dict) -> str:
    for f in TEXT_FIELDS:
        if f in item and isinstance(item[f], str):
            return item[f]
    raise KeyError(f"no text field in {TEXT_FIELDS}: {list(item)[:5]}")


def _contains(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    """True iff needle occurs as a contiguous slice of haystack."""
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1))


def check(train_items: list[dict], test_texts: list[str],
          n: int = 10) -> list[dict]:
    """Return one flag record per contaminated training item."""
    test_words = [tuple(normalize(t)) for t in test_texts]
    test_grams: set[tuple[str, ...]] = set()
    for w in test_words:
        test_grams |= ngrams(list(w), n)
    # Texts shorter than n yield no n-grams (see module docstring): keep
    # them for verbatim-containment checks in both directions.
    short_tests = [w for w in test_words if 0 < len(w) < n]
    flags = []
    for idx, item in enumerate(train_items):
        words = tuple(normalize(item_text(item)))
        grams = ngrams(list(words), n)
        hit = grams & test_grams
        example = " ".join(sorted(hit)[0]) if hit else None
        if not hit and words:
            short_hit = next((st for st in short_tests if _contains(words, st)),
                             None)
            if short_hit is not None:
                hit = {short_hit}
                example = " ".join(short_hit)
            elif len(words) < n and any(_contains(tw, words)
                                         for tw in test_words):
                hit = {words}
                example = " ".join(words)
        if hit:
            flags.append({"index": idx,
                          "matched_ngrams": len(hit),
                          "example": example})
    return flags


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_test_questions(test_file: str | None) -> list[str]:
    if test_file:
        return [item_text(r) for r in load_jsonl(test_file)]
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    return [r["question"] for r in ds]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test-file", default=None,
                    help="local JSONL of test items; default: HF gsm8k test")
    ap.add_argument("--ngram", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    train = load_jsonl(args.train)
    test_texts = load_test_questions(args.test_file)
    flags = check(train, test_texts, n=args.ngram)
    flagged_idx = {f["index"] for f in flags}

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i, item in enumerate(train):
            if i not in flagged_idx:
                f.write(json.dumps(item) + "\n")

    report = {"train_file": args.train, "train_items": len(train),
              "test_items": len(test_texts), "ngram": args.ngram,
              "flagged": len(flags), "kept": len(train) - len(flags),
              "flags": flags}
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"train={len(train)} test={len(test_texts)} ngram={args.ngram} "
          f"flagged={len(flags)} kept={len(train) - len(flags)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
