#!/usr/bin/env python3
"""Self-test for Session B data prep — no GPU, no downloads.

Covers what Session B must not get wrong before any training starts:
1. GSM8K calculator annotations (<<...>>) stripped from SFT targets,
2. eval-format prompt/completion construction,
3. rows with no parseable gold dropped, duplicate questions deduped.

Run: python selftest.py  (exit 0 = all pass)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prepare_sft_data import convert, strip_annotations

failures = []


def expect(name, got, want):
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


expect("annotation strip",
       strip_annotations("He has 20*2=<<20*2=40>>40 apples."),
       "He has 20*2=40 apples.")

rows = [
    {"question": "Sara bought 3 packs of 4 pens. How many pens?",
     "answer": "She bought 3*4=<<3*4=12>>12 pens in all.\n#### 12"},
    {"question": "sara bought 3 packs of 4 pens.   How many pens?",
     "answer": "Duplicate with different case/spacing.\n#### 12"},
    {"question": "Broken row with no final answer?",
     "answer": "Some reasoning with no gold marker."},
    {"question": "  ", "answer": "#### 5"},
]
out, stats = convert(rows, seed=1234)
expect("kept", stats["kept"], 1)
expect("dropped unparseable/blank", stats["dropped_unparseable_or_blank"], 2)
expect("dropped dupe", stats["dropped_duplicate_question"], 1)
if out:
    expect("prompt format", out[0]["prompt"],
           "Question: Sara bought 3 packs of 4 pens. How many pens?\nAnswer:")
    expect("completion clean", out[0]["completion"],
           " She bought 3*4=12 pens in all.\n#### 12")
    expect("gold", out[0]["gold"], "12")

# Trailing-answer canonicalisation (2026-10-13): source answers written
# '#### 1,000' / '#### 72.00' must become '#### 1000' / '#### 72' so the
# training target, the gold field and preflight's ending check all agree.
canon, _canon_stats = convert([
    {"question": "Comma formatting case?", "answer": "Many in total.\n#### 1,000"},
    {"question": "Decimal formatting case?", "answer": "Costs this much.\n#### 72.00"},
], seed=1234)
_by_q = {r["question"]: r for r in canon}
expect("comma gold canonicalised",
       _by_q["Comma formatting case?"]["completion"],
       " Many in total.\n#### 1000")
expect("decimal gold canonicalised",
       _by_q["Decimal formatting case?"]["completion"],
       " Costs this much.\n#### 72")

if failures:
    print("FAIL")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("OK — annotation stripping, format, dedupe, gold filtering, "
      "trailing-answer canonicalisation all pass")
