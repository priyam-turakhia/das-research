"""Cross-lingual embedding distillation: Polish student -> German LaBSE anchor.

Turns a trained encoder (any of our tokenizers, e.g. models/dsb/xlmr_morfessor_v1)
into a sentence encoder that maps Slavic input into LaBSE's space. We minimise
MSE(student(polish), LaBSE(german)) on de-pl parallel data — the cross-lingual
term of Reimers & Gurevych (2020). Polish is the measurable stand-in for dsb; the
eventual dsb step is structurally identical (student(dsb) -> LaBSE(german)).

The teacher (LaBSE) is frozen, so its German embeddings are precomputed once and
reused across epochs/runs. The student is mean-pooled (R&G 2019). Selection metric
is bitext retrieval P@1 (pl->de), not training loss, so we don't reward a model
that merely memorises Europarl into LaBSE.

Mirrors scripts/pretrain.py: --smoke local validation, device auto-pick, bf16,
checkpoints, tensorboard, modular flags. On LRZ Slurm + pyxis the NGC container
pre-sets RANK/LOCAL_RANK which makes accelerate try distributed init; run
`unset RANK LOCAL_RANK` once in the shell before launching, as with pretrain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from tokenization.registry import load_tokenizer

TEACHER_MODEL = "sentence-transformers/LaBSE"
SMOKE_TRAIN_PAIRS = 512
SMOKE_EVAL_PAIRS = 128


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SentenceDistiller(nn.Module):
    """Encoder body + mean pooling. Regresses the pooled embedding onto a target.

    Returns the pooled vector under the key "logits" purely so HF Trainer routes it
    to compute_metrics as predictions — it is a sentence embedding, not logits.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
        summed = (out.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        emb = summed / counts
        if labels is not None:
            loss = F.mse_loss(emb, labels.to(emb.dtype))
            return {"loss": loss, "logits": emb}
        return {"logits": emb}


def freeze_bottom_layers(encoder: nn.Module, n: int) -> None:
    """Freeze the embeddings and the bottom `n` transformer layers."""
    for p in encoder.embeddings.parameters():
        p.requires_grad = False
    for layer in encoder.encoder.layer[:n]:
        for p in layer.parameters():
            p.requires_grad = False


def read_lines(path: Path, limit: int | None = None) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return lines[:limit] if limit else lines


_LABSE = {}


def get_labse(device: torch.device):
    if "model" not in _LABSE:
        from sentence_transformers import SentenceTransformer
        print(f"[distill] loading teacher {TEACHER_MODEL} ...")
        _LABSE["model"] = SentenceTransformer(TEACHER_MODEL, device=str(device))
    return _LABSE["model"]


def teacher_embeddings(
    sentences: list[str], device: torch.device, batch_size: int, cache_path: Path | None
) -> np.ndarray:
    """LaBSE-embed German sentences (unit-norm). Cache to fp16 .npy when a path is given."""
    if cache_path is not None and cache_path.exists():
        print(f"[distill] loading cached teacher embeddings: {cache_path}")
        return np.load(cache_path).astype(np.float32)
    labse = get_labse(device)
    emb = labse.encode(
        sentences,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    if cache_path is not None:
        np.save(cache_path, emb.astype(np.float16))
        print(f"[distill] cached teacher embeddings -> {cache_path}")
    return emb


def build_dataset(polish: list[str], teacher: np.ndarray, tokenizer, seq_len: int) -> Dataset:
    """Tokenize Polish once and attach the teacher vector as the regression target."""
    ds = Dataset.from_dict({"text": polish, "labels": teacher.astype(np.float32).tolist()})
    ds = ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=seq_len),
        batched=True,
        remove_columns=["text"],
    )
    return ds


class DistillCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch


def retrieval_metrics(student: np.ndarray, teacher: np.ndarray) -> dict:
    """MSE + bitext retrieval P@1 both directions over the eval pool (cosine argmax)."""
    student = student.astype(np.float32)
    teacher = teacher.astype(np.float32)
    s = student / (np.linalg.norm(student, axis=1, keepdims=True) + 1e-9)
    t = teacher / (np.linalg.norm(teacher, axis=1, keepdims=True) + 1e-9)
    sims = s @ t.T
    n = len(s)
    idx = np.arange(n)
    pl2de = float((sims.argmax(axis=1) == idx).mean())
    de2pl = float((sims.argmax(axis=0) == idx).mean())
    mse = float(((student - teacher) ** 2).mean())
    return {"mse": mse, "p1_pl2de": pl2de, "p1_de2pl": de2pl, "p1_mean": (pl2de + de2pl) / 2}


def compute_metrics(eval_pred) -> dict:
    preds = eval_pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    return retrieval_metrics(np.asarray(preds), np.asarray(eval_pred.label_ids))


def preprocess_logits_for_metrics(logits, labels):
    # `logits` is already the 768-d sentence embedding; pass it straight through.
    return logits


@torch.no_grad()
def embed_pool(distiller, dataset, collator, device, batch_size) -> tuple[np.ndarray, np.ndarray]:
    """Manually embed a pool (for the pre-training baseline and final report)."""
    distiller.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    students, teachers = [], []
    for batch in loader:
        teachers.append(batch.pop("labels").numpy())
        batch = {k: v.to(device) for k, v in batch.items()}
        students.append(distiller(**batch)["logits"].float().cpu().numpy())
    return np.concatenate(students), np.concatenate(teachers)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Models
    p.add_argument("--encoder", required=True, help="Trained encoder dir (MLM head ignored).")
    p.add_argument("--tokenizer-path", required=True, help="Matching tokenizer dir (any of the 5).")
    p.add_argument("--output", required=True)
    # Data
    p.add_argument("--data-dir", default="data/processed/de-pl",
                   help="Holds {train,dev,test}.{src,tgt}. src=German (teacher), tgt=Polish (student).")
    p.add_argument("--src-lang", default="de")
    p.add_argument("--tgt-lang", default="pl")
    p.add_argument("--ood-eval", default=None,
                   help="Prefix of an out-of-domain eval set (<prefix>.<src> / .<tgt>), e.g. Tatoeba. "
                        "When set, selection switches to OOD retrieval P@1.")
    # Mode
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    # Sweep / data size
    p.add_argument("--max-train-pairs", type=int, default=0,
                   help="Subsample TRAIN to this many pairs (0=all). dev/test stay fixed. Sweep knob.")
    p.add_argument("--eval-pool", type=int, default=1000,
                   help="Retrieval pool size for the in-loop + final eval (Tatoeba-standard ~1000).")
    # Teacher
    p.add_argument("--teacher-batch-size", type=int, default=64)
    p.add_argument("--no-cache-teacher", action="store_true", help="Don't write/read the .npy cache.")
    # Model shape
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--freeze-layers", type=int, default=0,
                   help="Freeze embeddings + bottom N encoder layers (retain MLM-init features).")
    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accumulation-steps", type=int, default=1)
    p.add_argument("--num-train-epochs", type=float, default=5.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    # Checkpoint / select
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--load-best-model-at-end", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=3)
    # Logging / eval
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--report-to", default="none")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_pool(polish, teacher, size, seed):
    """Seeded subsample of (polish, teacher) down to `size` (or all if smaller)."""
    if size and len(polish) > size:
        idx = np.random.RandomState(seed).choice(len(polish), size, replace=False)
        return [polish[i] for i in idx], teacher[idx]
    return polish, teacher


def main() -> None:
    args = parse_args()
    if args.smoke and args.max_steps == -1:
        args.max_steps = 20
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    data_dir = Path(args.data_dir)
    src, tgt = args.src_lang, args.tgt_lang
    print(f"[distill] device = {device}")

    # Tokenizer (generic over the 5 tokenizers) + student encoder body.
    tokenizer = load_tokenizer(args.tokenizer_path).to_hf_tokenizer()
    encoder = AutoModel.from_pretrained(args.encoder)
    if args.freeze_layers:
        freeze_bottom_layers(encoder, args.freeze_layers)
        print(f"[distill] froze embeddings + bottom {args.freeze_layers} layers")
    distiller = SentenceDistiller(encoder).to(device)
    n_train_params = sum(p.numel() for p in distiller.parameters() if p.requires_grad)
    print(f"[distill] encoder={args.encoder} trainable params={n_train_params/1e6:.1f}M "
          f"tokenizer={args.tokenizer_path} vocab={tokenizer.vocab_size}")

    train_limit = SMOKE_TRAIN_PAIRS if args.smoke else None
    eval_limit = SMOKE_EVAL_PAIRS if args.smoke else None
    cache_ok = not args.smoke and not args.no_cache_teacher

    def cache_path(name: str) -> Path | None:
        return (data_dir / f"{name}.{src}.labse.npy") if cache_ok else None

    # --- Train ---
    train_pl = read_lines(data_dir / f"train.{tgt}", train_limit)
    train_de = read_lines(data_dir / f"train.{src}", train_limit)
    train_teacher = teacher_embeddings(train_de, device, args.teacher_batch_size, cache_path("train"))
    if args.max_train_pairs and len(train_pl) > args.max_train_pairs:
        idx = np.random.RandomState(args.seed).choice(len(train_pl), args.max_train_pairs, replace=False)
        train_pl = [train_pl[i] for i in idx]
        train_teacher = train_teacher[idx]
    print(f"[distill] train pairs = {len(train_pl)}")
    train_ds = build_dataset(train_pl, train_teacher, tokenizer, args.seq_len)

    # --- Eval pools (in-domain dev, final test, optional OOD) ---
    dev_pl = read_lines(data_dir / f"dev.{tgt}", eval_limit)
    dev_de = read_lines(data_dir / f"dev.{src}", eval_limit)
    dev_teacher = teacher_embeddings(dev_de, device, args.teacher_batch_size, cache_path("dev"))
    dev_pl, dev_teacher = make_pool(dev_pl, dev_teacher, args.eval_pool, args.seed)
    dev_ds = build_dataset(dev_pl, dev_teacher, tokenizer, args.seq_len)

    test_pl = read_lines(data_dir / f"test.{tgt}", eval_limit)
    test_de = read_lines(data_dir / f"test.{src}", eval_limit)
    test_teacher = teacher_embeddings(test_de, device, args.teacher_batch_size, cache_path("test"))
    test_pl, test_teacher = make_pool(test_pl, test_teacher, args.eval_pool, args.seed)
    test_ds = build_dataset(test_pl, test_teacher, tokenizer, args.seq_len)

    ood_ds = None
    if args.ood_eval:
        ood_prefix = Path(args.ood_eval)
        ood_pl = read_lines(ood_prefix.with_suffix(f".{tgt}"), eval_limit)
        ood_de = read_lines(ood_prefix.with_suffix(f".{src}"), eval_limit)
        ood_teacher = teacher_embeddings(
            ood_de, device, args.teacher_batch_size,
            None if not cache_ok else ood_prefix.with_suffix(f".{src}.labse.npy"),
        )
        ood_pl, ood_teacher = make_pool(ood_pl, ood_teacher, args.eval_pool, args.seed)
        ood_ds = build_dataset(ood_pl, ood_teacher, tokenizer, args.seq_len)

    # Free LaBSE before training — targets are materialised.
    _LABSE.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    collator = DistillCollator(tokenizer)
    # Selection runs on OOD when provided, else the held-out in-domain dev pool.
    select_ds = ood_ds if ood_ds is not None else dev_ds
    select_name = "OOD" if ood_ds is not None else "dev"

    # --- Baseline (un-distilled student) ---
    print("[distill] baseline (before training):")
    for name, ds in [("dev", dev_ds), ("test", test_ds)] + ([("OOD", ood_ds)] if ood_ds else []):
        s, t = embed_pool(distiller, ds, collator, device, args.batch_size)
        print(f"  baseline {name:4s}: {retrieval_metrics(s, t)}")

    report_to = [s.strip() for s in args.report_to.split(",") if s.strip() and s.strip() != "none"]
    training_args = TrainingArguments(
        output_dir=args.output,
        seed=args.seed,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        num_train_epochs=args.num_train_epochs if args.max_steps == -1 else 1.0,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        report_to=report_to,
        eval_strategy="steps",
        eval_steps=args.eval_steps if not args.smoke else args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps if not args.smoke else args.max_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="p1_pl2de" if args.load_best_model_at_end else None,
        greater_is_better=True if args.load_best_model_at_end else None,
        label_names=["labels"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    callbacks = []
    if args.load_best_model_at_end:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    print(f"[distill] selection metric: {select_name} retrieval p1_pl2de")
    trainer = Trainer(
        model=distiller,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=select_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # --- Final report (best model) ---
    print("[distill] final (after training):")
    for name, ds in [("dev", dev_ds), ("test", test_ds)] + ([("OOD", ood_ds)] if ood_ds else []):
        s, t = embed_pool(distiller, ds, collator, device, args.batch_size)
        print(f"  final {name:4s}: {retrieval_metrics(s, t)}")

    # --- Save sentence encoder ---
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    distiller.encoder.save_pretrained(out)
    tokenizer.save_pretrained(out)
    with open(out / "sentence_encoder.json", "w", encoding="utf-8") as f:
        json.dump(
            {"pooling": "mean", "tokenizer_path": args.tokenizer_path,
             "base_encoder": args.encoder, "teacher": TEACHER_MODEL},
            f, indent=2,
        )
    print(f"[distill] done. saved to {out}")


if __name__ == "__main__":
    main()
