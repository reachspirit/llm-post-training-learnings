#!/usr/bin/env python3
"""Self-test for the Session A kit's pure logic — no GPU, no downloads.

Covers the three things Session A must not get wrong:
1. answer extraction + numeric normalization (the score itself),
2. pass@1 / pass@K / never-solved aggregation,
3. the decontamination check actually catching a test item planted in a
   training file, and passing clean items through.

Run: python selftest.py  (exit 0 = all pass)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from decontaminate import check
from eval_gsm8k import extract_answer, gold_answer, normalize_number, score_samples

failures = []


def expect(name, got, want):
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


# 1. extraction / normalization
expect("#### format", extract_answer("work...\n#### 72"), "72")
expect("comma gold", gold_answer("reasoning\n#### 1,000"), "1000")
expect("fallback last number", extract_answer("so the total is 42 dollars"), "42")
expect("trailing .0", normalize_number("1000.0"), "1000")
expect("negative", extract_answer("#### -3"), "-3")
expect("no number", extract_answer("no digits here"), None)
# continuation: base model invents the next problem after its own answer —
# extraction must score THIS problem's ####, not the fabricated one (2026-10-11)
expect("continuation ####",
       extract_answer("work...\n#### 72\n\nQuestion: invented?\nAnswer: x\n#### 35"),
       "72")
expect("continuation fallback",
       extract_answer("so the total is 42 dollars\n\nQuestion: invented?\n#### 9"),
       "42")

# 2. aggregation: problem A solved 2/2, B solved 1/2, C solved 0/2
s = score_samples([[True, True], [True, False], [False, False]])
expect("pass@1", s["pass@1"], 0.5)
expect("pass@k", round(s["pass@k"], 4), round(2 / 3, 4))
expect("never-solved", round(s["never_solved_fraction"], 4), round(1 / 3, 4))

# 3. decontamination: one planted near-copy of a test question, one clean item
test_q = ("A robe takes 2 bolts of blue fiber and half that much white fiber. "
          "How many bolts in total does it take?")
train = [
    {"question": test_q},  # planted duplicate — must be flagged
    {"question": "Sara bought 3 packs of pens with 4 pens in each pack. "
                 "How many pens did she buy in all?"},  # clean
]
flags = check(train, [test_q], n=10)
expect("flag count", len(flags), 1)
expect("flagged index", flags[0]["index"] if flags else None, 0)

# 3b. short items (< n words): an n-gram check produces zero grams for
# them, so without the containment fallback an exact duplicate of a
# short test question passes silently (reproduced 2026-10-14).
short_q = "How many pens did Sara buy in all"  # 8 normalized words
short_train = [
    {"question": short_q},  # exact short duplicate — must be flagged
    {"question": "How many apples did Tom eat"},  # short but clean
]
short_flags = check(short_train, [short_q], n=10)
expect("short flag count", len(short_flags), 1)
expect("short flagged index", short_flags[0]["index"] if short_flags else None, 0)

if failures:
    print("FAIL")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("OK — extraction, aggregation, decontamination all pass on synthetic fixtures")
