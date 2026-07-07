"""Did distillation damage the encoder's intrinsic language ability?

Distillation only trains the transformer *body* and saves it without the MLM
head. To ask "did the body forget how to model Sorbian?", we take the ORIGINAL
MLM head from the pretrained checkpoint, bolt it onto (a) the original body and
(b) the distilled body, and measure MLM perplexity on held-out Sorbian with
*identical* masking. Same head, same masks -> the only variable is the body.

Read = perplexity/top-1. If distilled >> original, distillation degraded the
language model (catastrophic forgetting). If they match, Sorbian is intact.

Usage:
  uv run python scripts/eval_forgetting.py \
    --pretrained models/dsb/xlmr_morfessor_v1 \
    --distilled  models/dsb/labse_distill_morfessor_v1 \
    --tokenizer-path models/dsb/morfessor_v1 \
    --corpus data/processed/dsb/v1_dev.txt
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import load_dataset
from transformers import (
    AutoModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    XLMRobertaForMaskedLM,
)

from scripts.pretrain import (
    count_chars,
    make_compute_metrics,
    pick_device,
    preprocess_logits_for_metrics,
)
from tokenization.registry import load_tokenizer


def evaluate_model(model, tokenized_eval, collator, tokenizer, compute_metrics,
                   batch_size, seed, device):
    args = TrainingArguments(
        output_dir="/tmp/eval_forgetting",
        per_device_eval_batch_size=batch_size,
        report_to=[],
        dataloader_num_workers=0,
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=tokenized_eval,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if compute_metrics else None,
    )
    # Reset RNG right before eval so the collator draws identical masks for
    # every model we test.
    torch.manual_seed(seed)
    return trainer.evaluate()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", required=True,
                   help="Original MLM checkpoint (XLMRobertaForMaskedLM: body + head).")
    p.add_argument("--distilled", required=True,
                   help="Distilled encoder (XLMRobertaModel: body only).")
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--corpus", required=True, help="Held-out Sorbian text, one sentence per line.")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-eval-sents", type=int, default=0,
                   help="Evaluate on the first N sentences only (0 = all). "
                        "A few thousand is plenty to compare perplexity.")
    p.add_argument("--fp16", action="store_true",
                   help="Half-precision eval: cuts the logits-tensor memory that "
                        "chokes MPS. Perplexity is unchanged to ~2 decimals.")
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full-metrics", action="store_true",
                   help="Also compute top-1/top-5/BPC (slow: topk over vocab). "
                        "Default reports perplexity only (fast).")
    args = p.parse_args()

    device = pick_device("auto")
    print(f"[forgetting] device = {device}")

    tokenizer = load_tokenizer(args.tokenizer_path).to_hf_tokenizer()

    raw = load_dataset("text", data_files={"eval": args.corpus})
    tokenized = raw["eval"].map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=args.seq_len),
        batched=True, remove_columns=["text"],
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )
    compute_metrics = (
        make_compute_metrics(count_chars(args.corpus), args.mlm_probability)
        if args.full_metrics else None
    )

    # Original: body + head straight from the pretrained checkpoint.
    orig = XLMRobertaForMaskedLM.from_pretrained(args.pretrained)

    # Distilled: same architecture, then overwrite the body with the distilled
    # weights, keeping the original head.
    swapped = copy.deepcopy(orig)
    distilled_body = AutoModel.from_pretrained(args.distilled)
    missing, unexpected = swapped.roberta.load_state_dict(
        distilled_body.state_dict(), strict=False
    )
    print(f"[forgetting] body swap: {len(missing)} missing, {len(unexpected)} unexpected "
          f"(pooler-only is expected)")

    orig_m = evaluate_model(orig, tokenized, collator, tokenizer, compute_metrics,
                            args.batch_size, args.seed, device)
    swap_m = evaluate_model(swapped, tokenized, collator, tokenizer, compute_metrics,
                            args.batch_size, args.seed, device)

    def ppl(m):
        return math.exp(m["eval_loss"])

    def row(name, m):
        extra = ""
        if args.full_metrics:
            extra = (f"  top1={m['eval_top1']*100:5.1f}%  "
                     f"top5={m['eval_top5']*100:5.1f}%  bpc={m['eval_bpc']:.3f}")
        print(f"  {name:10s}  loss={m['eval_loss']:.4f}  ppl={ppl(m):7.2f}{extra}")

    print("\n=== intrinsic Sorbian MLM (same head, identical masks) ===")
    row("original", orig_m)
    row("distilled", swap_m)
    dppl = ppl(swap_m) - ppl(orig_m)
    print(f"\n  perplexity change: {dppl:+.2f} ({dppl / ppl(orig_m) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
