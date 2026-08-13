"""Build a balanced, interleaved multilingual pretraining corpus.

Each non-anchor language is subsampled (seeded) so its whitespace-token count
matches the anchor language's (dsb), then the languages are interleaved
round-robin, one line at a time, so no language dominates and the batches the
trainer forms (read in order via `pretrain.py --no-shuffle`) alternate between
languages. Output: <output>/mix_{train,dev,test}.txt, one sentence per line.

Example:
  uv run python scripts/build_multilingual_corpus.py \
    --langs de dsb --anchor dsb \
    --corpus dsb=data/processed/dsb/v2 de=data/processed/de/news \
    --output data/processed/de-dsb-mono
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenization.evaluate import load_corpus

SPLITS = ("train", "dev", "test")


def token_count(lines: list[str]) -> int:
    return sum(len(line.split()) for line in lines)


def cap_to_tokens(lines: list[str], budget: int, seed: int) -> list[str]:
    """Seeded subsample of `lines` whose total whitespace-token count ~ budget."""
    if token_count(lines) <= budget:
        return lines
    order = list(range(len(lines)))
    random.Random(seed).shuffle(order)
    out: list[str] = []
    total = 0
    for i in order:
        out.append(lines[i])
        total += len(lines[i].split())
        if total >= budget:
            break
    return out


def interleave(lists: list[list[str]]) -> list[str]:
    """Round-robin merge: one line from each list in turn until all are exhausted."""
    iters = [iter(x) for x in lists]
    out: list[str] = []
    active = list(range(len(iters)))
    while active:
        still: list[int] = []
        for k in active:
            try:
                out.append(next(iters[k]))
                still.append(k)
            except StopIteration:
                pass
        active = still
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", required=True, help="languages in the mix, e.g. de dsb")
    p.add_argument("--corpus", nargs="+", required=True,
                   help="lang=prefix pairs; reads <prefix>_{train,dev,test}.txt")
    p.add_argument("--anchor", default="dsb", help="language whose token count sets the budget")
    p.add_argument("--output", required=True, help="output directory for mix_{train,dev,test}.txt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    prefix: dict[str, str] = {}
    for kv in args.corpus:
        k, v = kv.split("=", 1)
        prefix[k] = v
    for lang in args.langs:
        if lang not in prefix:
            p.error(f"no --corpus prefix given for language '{lang}'")
    if args.anchor not in args.langs:
        p.error(f"anchor '{args.anchor}' must be one of --langs")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        per_lang = {lang: load_corpus(f"{prefix[lang]}_{split}.txt") for lang in args.langs}
        budget = token_count(per_lang[args.anchor])
        balanced = []
        for lang in args.langs:
            lines = per_lang[lang] if lang == args.anchor else cap_to_tokens(per_lang[lang], budget, args.seed)
            balanced.append(lines)
            print(f"  [{split}] {lang}: {len(lines):>7,} lines  {token_count(lines):>9,} tokens")
        mixed = interleave(balanced)
        out_path = out_dir / f"mix_{split}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(mixed) + "\n")
        print(f"  [{split}] -> {out_path}  ({len(mixed):,} lines total)\n")


if __name__ == "__main__":
    main()
