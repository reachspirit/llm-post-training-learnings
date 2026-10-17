#!/usr/bin/env python3
"""Session B SFT: Qwen3-1.7B-Base + LoRA on decontaminated GSM8K train data.

ONE version per invocation. The run sheet's Session B rule is v1 -> v2 -> v3
with exactly one change per version (data mix, formatting, LR/epochs, LoRA
rank) — every knob here is a CLI flag so the one change is a flag diff, and
the resolved config is written to <out>/train_config.json before training
starts (provenance: a checkpoint whose config cannot be read off disk fails
the Phase 2 terminal-state check).

Input: the DECONTAMINATED JSONL from prepare_sft_data.py +
session-a/decontaminate.py (prompt/completion fields). Training is
completion-only (SFTConfig completion_only_loss=True): the model is not
trained to generate questions, only answers in eval format.

Output: a MERGED checkpoint at --out (adapter merged into the base), so the
Session A harness can score it unchanged:
    python ../session-a/eval_gsm8k.py --model <out> --limit 200 --fewshot 4 ...

⚠️ NOT RUN in prep (no GPU / torch / trl in the prep environment — same
honest status as Session A's generation path). The API shape follows the
current TRL pattern checked 2026-10-11: SFTTrainer(model=..., args=
SFTConfig(..., dataset_text_field / max_length / completion_only_loss on
the config), train_dataset=..., peft_config=LoraConfig(...)). If the A100
box's trl version differs, the import/constructor step is where it will
show — run the --max-steps 2 smoke first.

Usage (A100 box, after decontamination):
    python train_sft.py --data sft_clean.jsonl --out ckpt/sft-v1 --max-steps 2   # smoke
    python train_sft.py --data sft_clean.jsonl --out ckpt/sft-v1                 # v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="decontaminated prompt/completion JSONL")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=-1, help="set 2 for the smoke run")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"effective_batch": args.batch * args.grad_accum}
    (out_path / "train_config.json").write_text(json.dumps(config, indent=2))

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("json", data_files=args.data, split="train")
    sft_args = SFTConfig(
        output_dir=str(out_path / "trainer"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        max_length=args.max_length,
        completion_only_loss=True,
        seed=args.seed,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=args.base,
        args=sft_args,
        train_dataset=ds,
        peft_config=LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM", target_modules="all-linear"),
    )
    trainer.train()

    # Merge so Session A's harness (AutoModelForCausalLM.from_pretrained)
    # can load the result directly — an adapter-only save would not load.
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(out_path))
    trainer.processing_class.save_pretrained(str(out_path))
    print(f"merged checkpoint -> {out_path} (config: {out_path}/train_config.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
