#!/usr/bin/env python3
"""Clean and split a parallel (bitext) corpus for cross-lingual distillation.

Pair-aware analogue of scripts/download_data.py: every filter operates on the
*pair*, dropping both sides together so line alignment is preserved. Built for
the de-pl Europarl corpus (German anchor, Polish student input), but the source
and target language tags are flags.

Pipeline, in order:
  1. NFC normalize + strip ASCII control chars (both sides).
  2. Either-side dedup: a sentence may appear at most once on EACH side; drop the
     pair if its source or target sentence was already seen.
  3. GlotLID language filter (both sides must be the expected language).
  4. Terminal-punctuation filter (both sides must end in sentence punctuation).
  5. Length + ratio filter (Moses clean-corpus-n convention).
  6. Optional subsample (--max-pairs), then a deterministic 90/5/5 split.

No Moses pretokenization is applied: the student's tokenizer Moses-pretokenizes
its input internally, and the teacher (LaBSE) wants natural text. Stored files
are cleaned but otherwise natural on both sides.
"""

import argparse
import logging
import random
import unicodedata
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

ASCII_CONTROL_CHARS = "".join(chr(i) for i in range(32)) + chr(127)
ASCII_CONTROL_TRANSLATION = str.maketrans("", "", ASCII_CONTROL_CHARS)

# Shared defaults with download_data.py where they overlap.
GLOTLID_THRESHOLD = 0.5
TERMINAL_PUNCT = ".!?…»\"'"
GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_FILENAME = "model.bin"

# Parallel-specific cleaning (Koehn Moses clean-corpus-n conventions).
MIN_LENGTH = 3
MAX_LENGTH = 80
MAX_RATIO = 3.0

TRAIN_RATIO = 0.90
DEV_RATIO = 0.05
SPLIT_SEED = 42

Pair = tuple[str, str]


def read_pairs(src_path: Path, tgt_path: Path) -> list[Pair]:
    """Read two line-aligned files into a list of (src, tgt) pairs."""
    with open(src_path, "r", encoding="utf-8") as f:
        src_lines = f.read().splitlines()
    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_lines = f.read().splitlines()
    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            f"Misaligned files: {src_path.name} has {len(src_lines)} lines, "
            f"{tgt_path.name} has {len(tgt_lines)}"
        )
    return list(zip(src_lines, tgt_lines))


def _normalize(text: str) -> str:
    """NFC-normalize and strip ASCII control chars."""
    return unicodedata.normalize("NFC", text).translate(ASCII_CONTROL_TRANSLATION).strip()


def normalize_and_clean(pairs: list[Pair]) -> tuple[list[Pair], int]:
    """NFC + control-strip both sides; drop the pair if either side goes empty."""
    logger.info(f"Normalizing and cleaning {len(pairs)} pairs...")
    kept: list[Pair] = []
    dropped = 0
    for src, tgt in pairs:
        s, t = _normalize(src), _normalize(tgt)
        if not s or not t:
            dropped += 1
            continue
        kept.append((s, t))
    logger.info(f"  Kept {len(kept)}, dropped {dropped} (empty after clean)")
    return kept, dropped


def dedup_either_side(pairs: list[Pair]) -> tuple[list[Pair], int]:
    """Drop a pair if its source OR target sentence has already appeared.

    Guarantees every sentence is unique on both sides — required so bitext
    retrieval has exactly one correct target per query.
    """
    logger.info(f"Either-side dedup on {len(pairs)} pairs...")
    seen_src: set[str] = set()
    seen_tgt: set[str] = set()
    kept: list[Pair] = []
    dropped = 0
    for src, tgt in pairs:
        if src in seen_src or tgt in seen_tgt:
            dropped += 1
            continue
        seen_src.add(src)
        seen_tgt.add(tgt)
        kept.append((src, tgt))
    logger.info(f"  Kept {len(kept)}, dropped {dropped} (duplicate on a side)")
    return kept, dropped


def filter_language(
    pairs: list[Pair], threshold: float, src_label: str, tgt_label: str
) -> tuple[list[Pair], int]:
    """Keep pairs whose source is `src_label` and target is `tgt_label`, both >= threshold."""
    from huggingface_hub import hf_hub_download
    import fasttext

    logger.info(f"Loading GlotLID from {GLOTLID_REPO}...")
    model_path = hf_hub_download(repo_id=GLOTLID_REPO, filename=GLOTLID_FILENAME)
    model = fasttext.load_model(model_path)

    logger.info(
        f"Language filter: src=={src_label}, tgt=={tgt_label}, both conf >= {threshold}"
    )

    def ok(text: str, label: str) -> bool:
        pred = model.predict(text.replace("\n", " "), k=1)
        return pred[0][0] == label and float(pred[1][0]) >= threshold

    kept: list[Pair] = []
    dropped = 0
    for src, tgt in pairs:
        if ok(src, src_label) and ok(tgt, tgt_label):
            kept.append((src, tgt))
        else:
            dropped += 1
    logger.info(f"  Kept {len(kept)}, dropped {dropped}")
    return kept, dropped


def filter_terminal_punct(pairs: list[Pair], terminal_chars: str) -> tuple[list[Pair], int]:
    """Keep pairs where both sides end in a terminal-punctuation character."""
    logger.info(f"Terminal-punct filter: both sides must end in {list(terminal_chars)!r}")
    kept: list[Pair] = []
    dropped = 0
    for src, tgt in pairs:
        if src[-1] in terminal_chars and tgt[-1] in terminal_chars:
            kept.append((src, tgt))
        else:
            dropped += 1
    logger.info(f"  Kept {len(kept)}, dropped {dropped}")
    return kept, dropped


def filter_length_ratio(
    pairs: list[Pair], min_length: int, max_length: int, max_ratio: float
) -> tuple[list[Pair], int]:
    """Keep pairs where each side is in [min,max] words and the length ratio <= max_ratio."""
    logger.info(
        f"Length+ratio filter: each side {min_length}<=words<={max_length}, "
        f"max/min words <= {max_ratio}"
    )
    kept: list[Pair] = []
    dropped = 0
    for src, tgt in pairs:
        ns, nt = len(src.split()), len(tgt.split())
        if not (min_length <= ns <= max_length and min_length <= nt <= max_length):
            dropped += 1
            continue
        if max(ns, nt) / min(ns, nt) > max_ratio:
            dropped += 1
            continue
        kept.append((src, tgt))
    logger.info(f"  Kept {len(kept)}, dropped {dropped}")
    return kept, dropped


def split_pairs(
    pairs: list[Pair],
    train_ratio: float = TRAIN_RATIO,
    dev_ratio: float = DEV_RATIO,
    seed: int = SPLIT_SEED,
) -> tuple[list[Pair], list[Pair], list[Pair]]:
    """Deterministic train/dev/test split, shuffling pairs as units."""
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    dev_end = train_end + int(len(shuffled) * dev_ratio)
    return shuffled[:train_end], shuffled[train_end:dev_end], shuffled[dev_end:]


def write_split(out_dir: Path, name: str, pairs: list[Pair], src_lang: str, tgt_lang: str) -> None:
    """Write a split as two aligned files: <name>.<src_lang> and <name>.<tgt_lang>."""
    src_path = out_dir / f"{name}.{src_lang}"
    tgt_path = out_dir / f"{name}.{tgt_lang}"
    with open(src_path, "w", encoding="utf-8") as fs, open(tgt_path, "w", encoding="utf-8") as ft:
        for src, tgt in pairs:
            fs.write(src + "\n")
            ft.write(tgt + "\n")
    logger.info(f"  Wrote {len(pairs)} pairs to {src_path.name} / {tgt_path.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean and split a parallel corpus for distillation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src-file", default=str(DATA_DIR / "raw/de-pl/Europarl.de-pl.de"),
                   help="Source-side (German) line-aligned text file.")
    p.add_argument("--tgt-file", default=str(DATA_DIR / "raw/de-pl/Europarl.de-pl.pl"),
                   help="Target-side (Polish) line-aligned text file.")
    p.add_argument("--src-lang", default="de", help="Source language tag for output filenames.")
    p.add_argument("--tgt-lang", default="pl", help="Target language tag for output filenames.")
    p.add_argument("--src-glotlid-label", default="__label__deu_Latn")
    p.add_argument("--tgt-glotlid-label", default="__label__pol_Latn")
    p.add_argument("--output-dir", default=str(DATA_DIR / "processed/de-pl"))
    p.add_argument("--glotlid-threshold", type=float, default=GLOTLID_THRESHOLD)
    p.add_argument("--min-length", type=int, default=MIN_LENGTH)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--max-ratio", type=float, default=MAX_RATIO)
    p.add_argument("--terminal-punct", default=TERMINAL_PUNCT)
    p.add_argument("--skip-glotlid", action="store_true", help="Skip the GlotLID language filter.")
    p.add_argument("--max-pairs", type=int, default=0,
                   help="Subsample to this many clean pairs before splitting (0 = keep all). "
                        "The knob for the data-size sweep.")
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Source: {args.src_file}")
    logger.info(f"Target: {args.tgt_file}")

    pairs = read_pairs(Path(args.src_file), Path(args.tgt_file))
    n_raw = len(pairs)
    logger.info(f"Read {n_raw} aligned pairs")

    pairs, empty_dropped = normalize_and_clean(pairs)
    pairs, dup_dropped = dedup_either_side(pairs)

    if args.skip_glotlid:
        logger.info("Skipping GlotLID language filter (--skip-glotlid)")
        glotlid_dropped = 0
    else:
        pairs, glotlid_dropped = filter_language(
            pairs, args.glotlid_threshold, args.src_glotlid_label, args.tgt_glotlid_label
        )

    pairs, terminal_dropped = filter_terminal_punct(pairs, args.terminal_punct)
    pairs, length_dropped = filter_length_ratio(
        pairs, args.min_length, args.max_length, args.max_ratio
    )

    n_clean = len(pairs)

    subsampled = 0
    if args.max_pairs and len(pairs) > args.max_pairs:
        random.Random(args.seed).shuffle(pairs)
        subsampled = len(pairs) - args.max_pairs
        pairs = pairs[: args.max_pairs]
        logger.info(f"Subsampled to {len(pairs)} pairs (--max-pairs {args.max_pairs})")

    train, dev, test = split_pairs(pairs, seed=args.seed)

    src_tokens = sum(len(s.split()) for s, _ in pairs)
    tgt_tokens = sum(len(t.split()) for _, t in pairs)

    logger.info("Writing splits...")
    write_split(out_dir, "train", train, args.src_lang, args.tgt_lang)
    write_split(out_dir, "dev", dev, args.src_lang, args.tgt_lang)
    write_split(out_dir, "test", test, args.src_lang, args.tgt_lang)

    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"  Raw pairs:                  {n_raw}")
    logger.info(f"  Dropped empty-after-clean:  {empty_dropped}")
    logger.info(f"  Dropped duplicate-on-side:  {dup_dropped}")
    logger.info(f"  Dropped GlotLID:            {glotlid_dropped}")
    logger.info(f"  Dropped terminal-punct:     {terminal_dropped}")
    logger.info(f"  Dropped length/ratio:       {length_dropped}")
    logger.info(f"  Clean pairs:                {n_clean}")
    if subsampled:
        logger.info(f"  Dropped by --max-pairs:     {subsampled}")
    logger.info(f"  Train/dev/test:             {len(train)} / {len(dev)} / {len(test)}")
    logger.info(f"  Tokens (src/tgt):           {src_tokens} / {tgt_tokens}")
    logger.info(f"  Output dir:                 {out_dir}")


if __name__ == "__main__":
    main()
