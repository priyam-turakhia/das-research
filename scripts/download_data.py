#!/usr/bin/env python3
"""Download and preprocess Upper Sorbian monolingual data."""

import argparse
import gzip
import logging
import os
import random
import sys
import tarfile
import unicodedata
from pathlib import Path

import requests

# Add parent dir to path so we can import the shared Moses helper.
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenization.pretokenize import moses_pretokenize  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

LEIPZIG_URL = "https://downloads.wortschatz-leipzig.de/corpora/hsb_mixed_2012_300K.tar.gz"
WMT22_URL = "https://raw.githubusercontent.com/mariondimarco/WMT22_UnsupVeryLowResMT_Data/main/HSB_monolingual.txt.gz"

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Per-source raw archive paths (URL-downloaded sources also have a URL above).
LEIPZIG_ARCHIVE = RAW_DIR / "hsb_mixed_2012_300K.tar.gz"
WMT22_ARCHIVE = RAW_DIR / "HSB_monolingual.txt.gz"
WIKI_FILE = RAW_DIR / "wiki_hsb_monolingual.txt"
WITAJ_FILE = RAW_DIR / "witaj_hsb_monolingual.txt"

ALL_SOURCES = ("leipzig", "wmt22", "wiki", "witaj")
DEFAULT_SOURCES = ("leipzig", "wmt22")

ASCII_CONTROL_CHARS = "".join(chr(i) for i in range(32)) + chr(127)
ASCII_CONTROL_TRANSLATION = str.maketrans("", "", ASCII_CONTROL_CHARS)
DROP_SUBSTRINGS = (
    "filename",
    "dateiname formatverbinden",
)

# Tunable thresholds — also surfaced as CLI flags. Either edit here for a
# permanent default or pass the flag for a one-off run.
GLOTLID_THRESHOLD = 0.5
MIN_LENGTH = 3
MAX_LENGTH = 100
TERMINAL_PUNCT = ".!?…»\"'"
GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_FILENAME = "model.bin"
GLOTLID_LABEL = "__label__hsb_Latn"

TRAIN_RATIO = 0.90
DEV_RATIO = 0.05
SPLIT_SEED = 42


def download_file(url: str, dest: Path) -> None:
    """Download a file from URL to destination path."""
    if dest.exists():
        logger.info(f"File already exists: {dest}")
        return

    logger.info(f"Downloading {url}...")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                if downloaded % (1024 * 1024) < 8192:
                    logger.info(f"  Downloaded {downloaded / 1024 / 1024:.1f}MB ({pct:.0f}%)")

    logger.info(f"Downloaded to {dest}")


def extract_leipzig(archive_path: Path) -> list[str]:
    """Extract sentences from the Leipzig corpus archive (column 2 of TSV)."""
    logger.info(f"Extracting Leipzig corpus from {archive_path}...")
    sentences = []

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("-sentences.txt"):
                logger.info(f"  Found sentences file: {member.name}")
                f = tar.extractfile(member)
                if f is None:
                    continue

                for line in f:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        sentences.append(parts[1])

    logger.info(f"  Extracted {len(sentences)} sentences from Leipzig")
    return sentences


def extract_plain(path: Path, source_name: str) -> list[str]:
    """Extract sentences from a plain UTF-8 file with one sentence per line."""
    logger.info(f"Extracting {source_name} corpus from {path}...")
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    logger.info(f"  Extracted {len(sentences)} sentences from {source_name}")
    return sentences


def extract_wmt22(archive_path: Path) -> list[str]:
    """Extract sentences from the WMT22 gzipped text file."""
    logger.info(f"Extracting WMT22 corpus from {archive_path}...")
    sentences = []

    with gzip.open(archive_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)

    logger.info(f"  Extracted {len(sentences)} sentences from WMT22")
    return sentences


def normalize_and_clean(sentences: list[str]) -> tuple[list[str], dict[str, int]]:
    """NFC-normalize, strip ASCII control chars, drop boilerplate and empty lines."""
    logger.info(f"Normalizing and cleaning {len(sentences)} lines...")
    result = []
    stats = {
        "control_char_lines_cleaned": 0,
        "boilerplate_lines_dropped": 0,
        "empty_lines_dropped": 0,
    }

    for sent in sentences:
        normalized = unicodedata.normalize("NFC", sent)
        cleaned = normalized.translate(ASCII_CONTROL_TRANSLATION).strip()

        if cleaned != normalized.strip():
            stats["control_char_lines_cleaned"] += 1

        if not cleaned:
            stats["empty_lines_dropped"] += 1
            continue

        lowered = cleaned.lower()
        if any(marker in lowered for marker in DROP_SUBSTRINGS):
            stats["boilerplate_lines_dropped"] += 1
            continue

        result.append(cleaned)

    logger.info(f"  Kept {len(result)} after normalize/clean")
    return result, stats


def filter_language(
    sentences: list[str], threshold: float, label: str
) -> tuple[list[str], int]:
    """Keep only lines where GlotLID's top prediction is `label` with confidence >= threshold."""
    from huggingface_hub import hf_hub_download
    import fasttext

    logger.info(f"Loading GlotLID from {GLOTLID_REPO}...")
    model_path = hf_hub_download(repo_id=GLOTLID_REPO, filename=GLOTLID_FILENAME)
    model = fasttext.load_model(model_path)

    logger.info(f"Language filter: keep '{label}' with confidence >= {threshold}")
    kept = []
    dropped = 0
    for sent in sentences:
        # GlotLID requires single-line input; replace newlines defensively.
        prediction = model.predict(sent.replace("\n", " "), k=1)
        top_label = prediction[0][0]
        confidence = float(prediction[1][0])
        if top_label == label and confidence >= threshold:
            kept.append(sent)
        else:
            dropped += 1

    logger.info(f"  Kept {len(kept)}, dropped {dropped}")
    return kept, dropped


def filter_terminal_punct(
    sentences: list[str], terminal_chars: str
) -> tuple[list[str], int]:
    """Drop lines that don't end in one of the terminal punctuation characters."""
    logger.info(f"Terminal-punct filter: keep lines ending in any of {list(terminal_chars)!r}")
    kept = []
    dropped = 0
    for sent in sentences:
        if sent and sent[-1] in terminal_chars:
            kept.append(sent)
        else:
            dropped += 1

    logger.info(f"  Kept {len(kept)}, dropped {dropped}")
    return kept, dropped


def filter_length(
    sentences: list[str], min_length: int, max_length: int
) -> tuple[list[str], int]:
    """Drop lines outside the [min_length, max_length] whitespace-token range."""
    logger.info(f"Length filter: keep lines with {min_length} <= words <= {max_length}")
    kept = []
    dropped = 0
    for sent in sentences:
        n = len(sent.split())
        if min_length <= n <= max_length:
            kept.append(sent)
        else:
            dropped += 1

    logger.info(f"  Kept {len(kept)}, dropped {dropped}")
    return kept, dropped


def apply_moses(sentences: list[str]) -> list[str]:
    """Apply Moses pretokenization (separates punctuation from words)."""
    logger.info(f"Applying Moses pretokenization to {len(sentences)} lines...")
    return [moses_pretokenize(sent) for sent in sentences]


def deduplicate(sentences: list[str]) -> tuple[list[str], int]:
    """Drop exact duplicate lines, preserve order of first occurrence."""
    logger.info(f"Deduplicating {len(sentences)} lines...")
    seen = set()
    kept = []
    dropped = 0
    for sent in sentences:
        if sent in seen:
            dropped += 1
            continue
        seen.add(sent)
        kept.append(sent)
    logger.info(f"  Kept {len(kept)}, dropped {dropped}")
    return kept, dropped


def split_sentences(
    sentences: list[str],
    train_ratio: float = TRAIN_RATIO,
    dev_ratio: float = DEV_RATIO,
    seed: int = SPLIT_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Create deterministic train/dev/test splits."""
    shuffled = sentences.copy()
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    dev_end = train_end + int(len(shuffled) * dev_ratio)
    return shuffled[:train_end], shuffled[train_end:dev_end], shuffled[dev_end:]


def write_sentences(path: Path, sentences: list[str]) -> None:
    """Write one sentence per line."""
    logger.info(f"Writing {len(sentences)} lines to {path}...")
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(sent + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and preprocess the Upper Sorbian corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--glotlid-threshold",
        type=float,
        default=GLOTLID_THRESHOLD,
        help="Minimum GlotLID confidence to keep a line as Upper Sorbian.",
    )
    p.add_argument(
        "--min-length",
        type=int,
        default=MIN_LENGTH,
        help="Minimum sentence length in whitespace words.",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help="Maximum sentence length in whitespace words.",
    )
    p.add_argument(
        "--terminal-punct",
        type=str,
        default=TERMINAL_PUNCT,
        help="Characters that count as legitimate sentence-final punctuation.",
    )
    p.add_argument(
        "--skip-glotlid",
        action="store_true",
        help="Skip the GlotLID language filter step.",
    )
    p.add_argument(
        "--sources",
        nargs="+",
        choices=ALL_SOURCES,
        default=list(DEFAULT_SOURCES),
        help="Which sources to combine into the corpus.",
    )
    p.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix for output filenames. Empty produces hsb.txt; "
             "'lww' produces hsb_lww.txt, hsb_lww_train.txt, etc.",
    )
    return p.parse_args()


def load_source(name: str) -> list[str]:
    """Download (if needed) and extract sentences for one source."""
    if name == "leipzig":
        download_file(LEIPZIG_URL, LEIPZIG_ARCHIVE)
        return extract_leipzig(LEIPZIG_ARCHIVE)
    if name == "wmt22":
        download_file(WMT22_URL, WMT22_ARCHIVE)
        return extract_wmt22(WMT22_ARCHIVE)
    if name == "wiki":
        if not WIKI_FILE.exists():
            raise FileNotFoundError(f"Wiki file not found: {WIKI_FILE}")
        return extract_plain(WIKI_FILE, "Wiki")
    if name == "witaj":
        if not WITAJ_FILE.exists():
            raise FileNotFoundError(f"Witaj file not found: {WITAJ_FILE}")
        return extract_plain(WITAJ_FILE, "Witaj")
    raise ValueError(f"Unknown source: {name}")


def main() -> None:
    args = parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_file = PROCESSED_DIR / f"hsb{suffix}.txt"
    train_file = PROCESSED_DIR / f"hsb{suffix}_train.txt"
    dev_file = PROCESSED_DIR / f"hsb{suffix}_dev.txt"
    test_file = PROCESSED_DIR / f"hsb{suffix}_test.txt"

    logger.info(f"Sources: {args.sources}")
    logger.info(f"Output suffix: {args.output_suffix or '(none)'}")

    per_source_counts: dict[str, int] = {}
    all_sentences: list[str] = []
    for source in args.sources:
        sents = load_source(source)
        per_source_counts[source] = len(sents)
        all_sentences.extend(sents)

    logger.info("=" * 50)
    for source, count in per_source_counts.items():
        logger.info(f"Lines from {source}: {count}")
    logger.info(f"Total before pipeline: {len(all_sentences)}")

    # 1. NFC + control chars + boilerplate + empties
    sentences, clean_stats = normalize_and_clean(all_sentences)

    # 2. Language filter
    if args.skip_glotlid:
        logger.info("Skipping GlotLID language filter (--skip-glotlid)")
        glotlid_dropped = 0
    else:
        sentences, glotlid_dropped = filter_language(
            sentences, args.glotlid_threshold, GLOTLID_LABEL
        )

    # 3. Terminal-punctuation filter
    sentences, terminal_dropped = filter_terminal_punct(sentences, args.terminal_punct)

    # 4. Length filter (on raw whitespace word counts, before Moses)
    sentences, length_dropped = filter_length(sentences, args.min_length, args.max_length)

    # 5. Moses pretokenization
    sentences = apply_moses(sentences)

    # 6. Dedup (after Moses, so spacing-only differences collapse)
    sentences, dedup_dropped = deduplicate(sentences)

    # 7. Split
    train_sentences, dev_sentences, test_sentences = split_sentences(sentences)

    total_tokens = sum(len(sent.split()) for sent in sentences)

    write_sentences(output_file, sentences)
    write_sentences(train_file, train_sentences)
    write_sentences(dev_file, dev_sentences)
    write_sentences(test_file, test_sentences)

    logger.info("=" * 50)
    logger.info("Summary:")
    for source, count in per_source_counts.items():
        logger.info(f"  Lines from {source}: {count}")
    logger.info(f"  Control-char lines cleaned: {clean_stats['control_char_lines_cleaned']}")
    logger.info(f"  Boilerplate lines dropped: {clean_stats['boilerplate_lines_dropped']}")
    logger.info(f"  Empty lines dropped: {clean_stats['empty_lines_dropped']}")
    logger.info(f"  Language filter dropped: {glotlid_dropped}")
    logger.info(f"  Terminal-punct dropped: {terminal_dropped}")
    logger.info(f"  Length filter dropped: {length_dropped}")
    logger.info(f"  Duplicates dropped (after Moses): {dedup_dropped}")
    logger.info(f"  Final corpus size: {len(sentences)} sentences")
    logger.info(f"  Train/dev/test sizes: {len(train_sentences)} / {len(dev_sentences)} / {len(test_sentences)}")
    logger.info(f"  Total token count: {total_tokens}")
    logger.info(f"  Full corpus: {output_file}")
    logger.info(f"  Train split: {train_file}")
    logger.info(f"  Dev split: {dev_file}")
    logger.info(f"  Test split: {test_file}")


if __name__ == "__main__":
    main()
