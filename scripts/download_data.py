#!/usr/bin/env python3
"""Download and preprocess Upper Sorbian monolingual data."""

import gzip
import logging
import os
import random
import tarfile
import unicodedata
from pathlib import Path

import requests

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
OUTPUT_FILE = PROCESSED_DIR / "hsb.txt"
TRAIN_FILE = PROCESSED_DIR / "hsb_train.txt"
DEV_FILE = PROCESSED_DIR / "hsb_dev.txt"
TEST_FILE = PROCESSED_DIR / "hsb_test.txt"

ASCII_CONTROL_CHARS = "".join(chr(i) for i in range(32)) + chr(127)
ASCII_CONTROL_TRANSLATION = str.maketrans("", "", ASCII_CONTROL_CHARS)
DROP_SUBSTRINGS = (
    "filename",
    "dateiname formatverbinden",
)
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
                if downloaded % (1024 * 1024) < 8192:  # Log every ~1MB
                    logger.info(f"  Downloaded {downloaded / 1024 / 1024:.1f}MB ({pct:.0f}%)")

    logger.info(f"Downloaded to {dest}")


def extract_leipzig(archive_path: Path) -> list[str]:
    """Extract sentences from Leipzig corpus archive.

    Returns list of sentences (column 2 from tab-separated sentences file).
    """
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


def extract_wmt22(archive_path: Path) -> list[str]:
    """Extract sentences from WMT22 gzipped text file."""
    logger.info(f"Extracting WMT22 corpus from {archive_path}...")
    sentences = []

    with gzip.open(archive_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)

    logger.info(f"  Extracted {len(sentences)} sentences from WMT22")
    return sentences


def normalize_clean_and_deduplicate(sentences: list[str]) -> tuple[list[str], dict[str, int]]:
    """Normalize, lightly clean, and deduplicate while preserving order."""
    logger.info("Normalizing, cleaning, and deduplicating...")

    seen = set()
    result = []
    stats = {
        "control_char_lines_cleaned": 0,
        "boilerplate_lines_dropped": 0,
        "empty_lines_dropped": 0,
        "duplicate_lines_dropped": 0,
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

        if cleaned in seen:
            stats["duplicate_lines_dropped"] += 1
            continue

        seen.add(cleaned)
        result.append(cleaned)

    logger.info(f"  After deduplication: {len(result)} sentences")
    return result, stats


def split_sentences(
    sentences: list[str],
    train_ratio: float = TRAIN_RATIO,
    dev_ratio: float = DEV_RATIO,
    seed: int = SPLIT_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Create deterministic train/dev/test splits from a deduplicated corpus."""
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


def count_tokens(sentences: list[str]) -> int:
    """Count total whitespace-separated tokens."""
    return sum(len(sent.split()) for sent in sentences)


def main() -> None:
    """Main entry point for data download and preprocessing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Download files
    leipzig_archive = RAW_DIR / "hsb_mixed_2012_300K.tar.gz"
    wmt22_archive = RAW_DIR / "HSB_monolingual.txt.gz"

    download_file(LEIPZIG_URL, leipzig_archive)
    download_file(WMT22_URL, wmt22_archive)

    # Extract sentences
    leipzig_sentences = extract_leipzig(leipzig_archive)
    wmt22_sentences = extract_wmt22(wmt22_archive)

    logger.info("=" * 50)
    logger.info(f"Lines from Leipzig: {len(leipzig_sentences)}")
    logger.info(f"Lines from WMT22: {len(wmt22_sentences)}")

    # Concatenate
    all_sentences = leipzig_sentences + wmt22_sentences
    logger.info(f"Total before deduplication: {len(all_sentences)}")

    # Normalize, lightly clean, and deduplicate
    final_sentences, cleaning_stats = normalize_clean_and_deduplicate(all_sentences)

    # Deterministic train/dev/test split for honest held-out evaluation
    train_sentences, dev_sentences, test_sentences = split_sentences(final_sentences)

    # Count tokens
    total_tokens = count_tokens(final_sentences)

    # Write outputs
    write_sentences(OUTPUT_FILE, final_sentences)
    write_sentences(TRAIN_FILE, train_sentences)
    write_sentences(DEV_FILE, dev_sentences)
    write_sentences(TEST_FILE, test_sentences)

    logger.info("=" * 50)
    logger.info("Summary:")
    logger.info(f"  Lines from Leipzig: {len(leipzig_sentences)}")
    logger.info(f"  Lines from WMT22: {len(wmt22_sentences)}")
    logger.info(f"  Lines after deduplication: {len(final_sentences)}")
    logger.info(f"  Control-char lines cleaned: {cleaning_stats['control_char_lines_cleaned']}")
    logger.info(f"  Boilerplate lines dropped: {cleaning_stats['boilerplate_lines_dropped']}")
    logger.info(f"  Empty lines dropped: {cleaning_stats['empty_lines_dropped']}")
    logger.info(f"  Duplicate lines dropped: {cleaning_stats['duplicate_lines_dropped']}")
    logger.info(f"  Train/dev/test sizes: {len(train_sentences)} / {len(dev_sentences)} / {len(test_sentences)}")
    logger.info(f"  Total token count: {total_tokens}")
    logger.info(f"  Full corpus: {OUTPUT_FILE}")
    logger.info(f"  Train split: {TRAIN_FILE}")
    logger.info(f"  Dev split: {DEV_FILE}")
    logger.info(f"  Test split: {TEST_FILE}")


if __name__ == "__main__":
    main()
