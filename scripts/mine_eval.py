"""De–dsb parallel sentence mining eval (BUCC F1) for a distilled encoder.

The real downstream test for the distilled sentence encoder: given monolingual
German and Lower Sorbian candidate pools (BUCC-style, one `<id>\\t<sentence>` per
line), mine parallel pairs and score precision / recall / F1 against a gold pair
list. Uses the PaSeMiLL / UnsupPSE protocol — CSLS scoring + a dynamic threshold —
on the PaSeMiLL de–dsb BUCC data.

Cross-model by design: the dsb side is embedded by our distilled **student** (the
only thing that encodes dsb), the German side by **LaBSE** (the distillation
teacher). Both live in LaBSE's space — that is exactly what the distillation
produced — so CSLS between them is meaningful. This differs from PaSeMiLL, which
embeds both sides with one multilingual model.

Pooling here is mean over all non-pad tokens (incl. [CLS]/[SEP]), matching how the
student was distilled in scripts/distill.py — deliberately NOT PaSeMiLL's
special-token-excluding pool, which would mismatch the training objective.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from transformers import AutoModel

from tokenization.registry import load_tokenizer

TEACHER_MODEL = "sentence-transformers/LaBSE"


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_bucc(path: Path) -> tuple[list[str], list[str]]:
    """Read a BUCC-style monolingual file: one `<id>\\t<sentence>` per line."""
    ids, sents = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sid, sent = line.split("\t", 1)
            ids.append(sid.strip())
            sents.append(sent)
    return ids, sents


def read_plain(path: Path) -> list[str]:
    """Read a plain one-sentence-per-line file (strips a leading BOM if present)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def read_gold(path: Path) -> dict[str, str]:
    """Read the gold pair list: `<src_id>\\t<trg_id>` per line (src = dsb, trg = de)."""
    gold = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            src, trg = line.split("\t", 1)
            gold[src.strip()] = trg.strip()
    return gold


@torch.no_grad()
def embed_student(encoder, tokenizer, sents, device, batch_size, seq_len) -> np.ndarray:
    """Mean-pool the student encoder over all non-pad tokens (matches distill.py)."""
    encoder.eval()
    vecs = []
    for i in range(0, len(sents), batch_size):
        enc = tokenizer(
            sents[i : i + batch_size],
            truncation=True,
            max_length=seq_len,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vecs.append(emb.float().cpu().numpy())
    x = np.concatenate(vecs)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def embed_labse(sents, device, batch_size, cache_path: Path | None) -> np.ndarray:
    """LaBSE sentence embeddings (unit-norm), same teacher as distillation. Cacheable."""
    if cache_path is not None and cache_path.exists():
        print(f"[mine] loading cached German embeddings: {cache_path}")
        return np.load(cache_path)
    from sentence_transformers import SentenceTransformer

    print(f"[mine] embedding German with {TEACHER_MODEL} ...")
    labse = SentenceTransformer(TEACHER_MODEL, device=str(device))
    y = labse.encode(
        sents, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    if cache_path is not None:
        np.save(cache_path, y)
        print(f"[mine] cached German embeddings -> {cache_path}")
    return y


def csls_mine(x: np.ndarray, y: np.ndarray, k: int, device, chunk: int = 1024):
    """For each source row, return its best target index and CSLS score.

    CSLS(i,j) = 2·cos(x_i,y_j) − r_k(x_i) − r_k(y_j), where r_k is the mean cosine
    to the k nearest neighbours in the other language (local scaling, the hubness
    fix that plain cosine retrieval lacks). Chunked so large target pools fit.
    """
    # Similarity math: CUDA if available (GPU box has the memory), else CPU.
    # We avoid MPS here — large pools' chunked sim matrices OOM the MPS limit,
    # and the embeddings are small enough (~hundreds of MB) that CPU is fine.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    nx, ny = xt.shape[0], yt.shape[0]
    kx, ky = min(k, ny), min(k, nx)

    r_s = torch.empty(nx, device=device)
    for i in range(0, nx, chunk):
        sims = xt[i : i + chunk] @ yt.T
        r_s[i : i + chunk] = sims.topk(kx, dim=1).values.mean(1)

    r_t = torch.empty(ny, device=device)
    for j in range(0, ny, chunk):
        sims = yt[j : j + chunk] @ xt.T
        r_t[j : j + chunk] = sims.topk(ky, dim=1).values.mean(1)

    best_j = torch.empty(nx, dtype=torch.long, device=device)
    best_score = torch.empty(nx, device=device)
    for i in range(0, nx, chunk):
        csls = 2 * (xt[i : i + chunk] @ yt.T) - r_s[i : i + chunk].unsqueeze(1) - r_t.unsqueeze(0)
        vals, idx = csls.max(dim=1)
        best_j[i : i + chunk] = idx
        best_score[i : i + chunk] = vals
    return best_j.cpu().numpy(), best_score.cpu().numpy()


def bucc_prf(pred_pairs, gold: dict[str, str]):
    """BUCC precision / recall / F1. fn = |gold| − tp (PaSeMiLL convention)."""
    tp = fp = 0
    for src, trg in pred_pairs:
        if src in gold and gold[src] == trg:
            tp += 1
        else:
            fp += 1
    fn = len(gold) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp, fp, fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, help="Distilled sentence encoder dir.")
    p.add_argument("--tokenizer-path", required=True, help="Matching tokenizer dir.")
    p.add_argument("--dsb-file", default="data/raw/bucc-de-dsb/dsb-de.test.dsb",
                   help="BUCC dsb pool (<id>\\t<sentence>); embedded by the student.")
    p.add_argument("--de-file", default="data/raw/bucc-de-dsb/dsb-de.test.de",
                   help="BUCC German pool (<id>\\t<sentence>); embedded by LaBSE.")
    p.add_argument("--gold", default="data/raw/bucc-de-dsb/dsb-de.test.gold",
                   help="Gold pairs: <dsb_id>\\t<de_id>.")
    p.add_argument("--de-cache", default=None,
                   help="Optional .npy cache for the LaBSE German embeddings (reused across tokenizers).")
    p.add_argument("--output", default=None, help="Optional path to write predicted <dsb_id>\\t<de_id> pairs.")
    p.add_argument("--parallel", action="store_true",
                   help="Treat --dsb-file/--de-file as plain, line-aligned parallel text (no IDs, no "
                        "gold): report CSLS retrieval P@1 both directions over the 1:1 pool.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--csls-k", type=int, default=20, help="CSLS neighbourhood (PaSeMiLL uses 20).")
    p.add_argument("--threshold-mode", default="dynamic", choices=["dynamic", "fixed"])
    p.add_argument("--threshold", type=float, default=2.0,
                   help="dynamic: lambda in mean+lambda*std (PaSeMiLL default 2.0). fixed: CSLS cutoff.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"[mine] device = {device}")

    tokenizer = load_tokenizer(args.tokenizer_path).to_hf_tokenizer()
    encoder = AutoModel.from_pretrained(args.encoder).to(device)

    if args.parallel:
        dsb_sents = read_plain(Path(args.dsb_file))
        de_sents = read_plain(Path(args.de_file))
        if len(dsb_sents) != len(de_sents):
            raise ValueError(f"parallel files not line-aligned: {len(dsb_sents)} vs {len(de_sents)}")
        n = len(dsb_sents)
        print(f"[mine] parallel de-dsb pairs = {n}")
        x = embed_student(encoder, tokenizer, dsb_sents, device, args.batch_size, args.seq_len)
        y = embed_labse(de_sents, device, args.batch_size,
                        Path(args.de_cache) if args.de_cache else None)
        best_de, _ = csls_mine(x, y, args.csls_k, device)   # dsb query -> de
        best_dsb, _ = csls_mine(y, x, args.csls_k, device)  # de query  -> dsb
        idx = np.arange(n)
        p_dsb2de = float((best_de == idx).mean())
        p_de2dsb = float((best_dsb == idx).mean())
        print(f"[mine] CSLS retrieval P@1  dsb->de={p_dsb2de:.4f}  de->dsb={p_de2dsb:.4f}  "
              f"mean={(p_dsb2de + p_de2dsb) / 2:.4f}")
        return

    dsb_ids, dsb_sents = read_bucc(Path(args.dsb_file))
    de_ids, de_sents = read_bucc(Path(args.de_file))
    gold = read_gold(Path(args.gold))
    print(f"[mine] dsb pool={len(dsb_sents)} de pool={len(de_sents)} gold pairs={len(gold)}")

    # Only score against gold pairs whose BOTH sentences are actually in the
    # loaded pools — a target that isn't in the candidate pool can never be
    # retrieved, so counting it would unfairly cap recall.
    dsb_set, de_set = set(dsb_ids), set(de_ids)
    in_pool = {s: t for s, t in gold.items() if s in dsb_set and t in de_set}
    if len(in_pool) < len(gold):
        print(f"[mine] WARNING: {len(gold) - len(in_pool)}/{len(gold)} gold pairs reference "
              f"sentences absent from the pools; scoring against the {len(in_pool)} in-pool pairs.")
    gold = in_pool

    x = embed_student(encoder, tokenizer, dsb_sents, device, args.batch_size, args.seq_len)
    y = embed_labse(de_sents, device, args.batch_size,
                    Path(args.de_cache) if args.de_cache else None)

    best_j, scores = csls_mine(x, y, args.csls_k, device)

    # Threshold-free retrieval: for each gold dsb sentence, is its single nearest
    # German (ignoring the cutoff) the true one? Isolates pure retrieval quality
    # from threshold calibration — separates "model can't rank it" from "threshold
    # threw it away".
    dsb_index = {sid: i for i, sid in enumerate(dsb_ids)}
    hits = sum(1 for src, trg in gold.items() if de_ids[best_j[dsb_index[src]]] == trg)
    print(f"[mine] retrieval P@1 (no threshold, pool of {len(de_ids)}): "
          f"{hits}/{len(gold)} = {hits / len(gold):.4f}")

    if args.threshold_mode == "dynamic":
        tau = float(scores.mean() + args.threshold * scores.std())
    else:
        tau = args.threshold
    print(f"[mine] threshold ({args.threshold_mode}) = {tau:.4f} "
          f"[score mean={scores.mean():.4f} std={scores.std():.4f}]")

    pred_pairs = [
        (dsb_ids[i], de_ids[best_j[i]])
        for i in range(len(dsb_ids)) if scores[i] > tau
    ]
    print(f"[mine] predicted pairs = {len(pred_pairs)}")

    p, r, f, tp, fp, fn = bucc_prf(pred_pairs, gold)
    print(f"[mine] BUCC  precision={p:.4f}  recall={r:.4f}  F1={f:.4f}  "
          f"(tp={tp} fp={fp} fn={fn})")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fout:
            for src, trg in pred_pairs:
                fout.write(f"{src}\t{trg}\n")
        print(f"[mine] wrote predictions -> {args.output}")


if __name__ == "__main__":
    main()
