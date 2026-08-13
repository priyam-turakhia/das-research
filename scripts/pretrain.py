"""XLM-R base MLM pretraining for dsb using our custom tokenizers.

Smoke mode (--smoke) runs a tiny 2-layer/128-hidden model with batch 2 to
validate the wiring on a laptop. Production mode uses the full XLM-R-base
architecture (12 layers / 768 hidden / 12 heads) shaped to match stock
xlm-roberta-base so the trained encoder is drop-in for cross-lingual
embedding distillation.

Eval metrics on the dev set:
- eval_loss   : mean MLM cross-entropy
- perplexity  : exp(eval_loss)
- top1, top5  : prediction accuracy at masked positions
- bpc         : bits per character (tokenization-invariant; use for cross-
                tokenizer comparison)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    XLMRobertaConfig,
    XLMRobertaForMaskedLM,
)

from tokenization.registry import load_tokenizer

DEFAULT_MAX_POSITION_EMBEDDINGS = 514  # matches stock xlm-roberta-base


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_config(tokenizer, args: argparse.Namespace) -> XLMRobertaConfig:
    common = dict(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.max_position_embeddings,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )
    if args.smoke:
        return XLMRobertaConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            **common,
        )
    return XLMRobertaConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        **common,
    )


def count_chars(corpus_path: str) -> int:
    """Count characters in a text corpus (whitespace stripped per line)."""
    total = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            total += len(line.rstrip("\n"))
    return total


def preprocess_logits_for_metrics(logits, labels):
    """Reduce per-batch logits before Trainer accumulates them across eval set.

    Without this, Trainer accumulates the full (N, L, V) logits tensor — for
    our dev corpus that's ~200 GB and OOMs immediately. We only need
    (top-5 indices, NLL at label position) per token, both tiny.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    top5 = log_probs.topk(5, dim=-1).indices                          # (B, L, 5)
    safe_labels = labels.clone()
    safe_labels[safe_labels == -100] = 0
    nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, L)
    return top5, nll


def make_compute_metrics(eval_char_count: int, mlm_probability: float):
    """Return a compute_metrics fn for MLM eval.

    Reports perplexity, top-1/top-5 accuracy at masked positions, and bits per
    character (BPC). BPC normalizes loss by source-text character count, which
    is tokenization-invariant — use it for cross-tokenizer comparison.
    """

    def compute_metrics(eval_pred):
        (top5, nll), labels = eval_pred
        top5_t = torch.as_tensor(top5)
        nll_t = torch.as_tensor(nll)
        labels_t = torch.as_tensor(labels)
        mask = labels_t != -100
        n_targets = int(mask.sum().item())
        if n_targets == 0:
            return {"perplexity": float("nan"), "top1": 0.0, "top5": 0.0, "bpc": float("nan")}

        target_top5 = top5_t[mask]               # (T, 5)
        target_nll = nll_t[mask]                 # (T,)
        target_labels = labels_t[mask].long()    # (T,)

        top1_acc = (target_top5[:, 0] == target_labels).float().mean().item()
        top5_acc = (target_top5 == target_labels.unsqueeze(-1)).any(-1).float().mean().item()

        mean_nll = target_nll.mean().item()
        perplexity = math.exp(mean_nll)

        # BPC: total NLL on masked positions extrapolated to full text via the
        # 1/p_mask scaling factor (only mlm_probability of tokens are scored),
        # divided by corpus character count, converted nats -> bits.
        total_nll_nats = target_nll.sum().item() / mlm_probability
        bpc = (total_nll_nats / max(eval_char_count, 1)) / math.log(2)

        return {
            "perplexity": perplexity,
            "top1": top1_acc,
            "top5": top5_acc,
            "bpc": bpc,
        }

    return compute_metrics


class NoShuffleTrainer(Trainer):
    """Trainer that reads the training data in file order (no reshuffle).

    Used with an interleaved multilingual corpus so every batch alternates
    languages instead of drawing a random mix. MLM masking stays random (it
    lives in the collator, not the sampler).
    """

    def _get_train_sampler(self, *args, **kwargs):
        from torch.utils.data import SequentialSampler

        return SequentialSampler(self.train_dataset)


def main() -> None:
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--corpus", required=True, help="Train text (one sentence per line)")
    p.add_argument("--eval-corpus", default=None)
    p.add_argument("--output", required=True)
    # Mode
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    # Model shape
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--max-position-embeddings", type=int, default=DEFAULT_MAX_POSITION_EMBEDDINGS)
    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accumulation-steps", type=int, default=4)
    p.add_argument("--num-train-epochs", type=float, default=10.0)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Override epochs with a fixed step count (>0). Smoke defaults to 20.")
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    # Precision
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    # Checkpointing
    p.add_argument("--save-steps", type=int, default=2000)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Path to checkpoint dir; resumes optimizer/scheduler/RNG/data cursor.")
    p.add_argument("--load-best-model-at-end", action="store_true",
                   help="After training, final saved model is the checkpoint with lowest eval_loss.")
    # Logging / eval
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--report-to", default="none",
                   help="Comma-separated: tensorboard,wandb,none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-shuffle", action="store_true",
                   help="Read the training data in file order (no reshuffle). Use with an "
                        "interleaved multilingual corpus so every batch alternates languages.")
    args = p.parse_args()

    if args.smoke and args.max_steps == -1:
        args.max_steps = 20

    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"[pretrain] device = {device}")

    # Tokenizer
    native = load_tokenizer(args.tokenizer_path)
    tokenizer = native.to_hf_tokenizer()
    print(f"[pretrain] tokenizer vocab_size={tokenizer.vocab_size} "
          f"pad={tokenizer.pad_token_id} cls={tokenizer.cls_token_id} "
          f"sep={tokenizer.sep_token_id} mask={tokenizer.mask_token_id}")

    # Model
    config = build_config(tokenizer, args)
    model = XLMRobertaForMaskedLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] model params = {n_params/1e6:.1f}M "
          f"(hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"max_pos={config.max_position_embeddings})")

    # Data
    data_files = {"train": args.corpus}
    if args.eval_corpus:
        data_files["eval"] = args.eval_corpus
    raw = load_dataset("text", data_files=data_files)

    def encode(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.seq_len)

    tokenized = raw.map(encode, batched=True, remove_columns=["text"])
    print(f"[pretrain] tokenized train rows = {len(tokenized['train'])}")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )

    # Eval metrics
    compute_metrics = None
    if args.eval_corpus and not args.smoke:
        n_eval_chars = count_chars(args.eval_corpus)
        print(f"[pretrain] eval corpus chars = {n_eval_chars}")
        compute_metrics = make_compute_metrics(n_eval_chars, args.mlm_probability)

    report_to = [s.strip() for s in args.report_to.split(",") if s.strip() and s.strip() != "none"]

    do_eval = bool(args.eval_corpus) and not args.smoke

    training_args = TrainingArguments(
        output_dir=args.output,
        seed=args.seed,
        # Batch / steps
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        num_train_epochs=args.num_train_epochs if args.max_steps == -1 else 1.0,
        max_steps=args.max_steps,
        # Optimizer
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        # Precision
        bf16=args.bf16,
        fp16=args.fp16,
        # Logging
        logging_steps=args.logging_steps,
        report_to=report_to,
        # Eval
        eval_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps,
        # Save
        save_strategy="steps",
        save_steps=args.save_steps if not args.smoke else max(args.max_steps, 1),
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end and do_eval,
        metric_for_best_model="eval_loss" if args.load_best_model_at_end else None,
        greater_is_better=False if args.load_best_model_at_end else None,
        # Misc
        dataloader_num_workers=0,
    )

    trainer_cls = NoShuffleTrainer if args.no_shuffle else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("eval"),
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if compute_metrics else None,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[pretrain] done. saved to {args.output}")


if __name__ == "__main__":
    main()
