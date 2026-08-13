#!/usr/bin/env python3
"""Download and preprocess Sorbian monolingual data.

The pipeline is language-agnostic. Per-language configuration lives in
`LANG_REGISTRY` below: source files, download URLs, and the GlotLID label
the language filter keeps. Add a new language by adding a registry entry.
"""

import argparse
import gzip
import logging
import os
import random
import sys
import tarfile
import unicodedata
from dataclasses import dataclass
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

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass(frozen=True)
class SourceSpec:
    """How to obtain and extract one raw source file."""

    filename: str
    extractor: str  # "leipzig" | "wmt22_gz" | "plain"
    url: str | None = None  # None means the file must be placed manually


@dataclass(frozen=True)
class LangSpec:
    """Per-language configuration for the preprocessing pipeline."""

    glotlid_label: str
    sources: dict[str, SourceSpec]
    default_sources: tuple[str, ...]


LANG_REGISTRY: dict[str, LangSpec] = {
    "hsb": LangSpec(
        glotlid_label="__label__hsb_Latn",
        sources={
            "leipzig": SourceSpec(
                filename="hsb_mixed_2012_300K.tar.gz",
                extractor="leipzig",
                url="https://downloads.wortschatz-leipzig.de/corpora/hsb_mixed_2012_300K.tar.gz",
            ),
            "wmt22": SourceSpec(
                filename="HSB_monolingual.txt.gz",
                extractor="wmt22_gz",
                url="https://raw.githubusercontent.com/mariondimarco/WMT22_UnsupVeryLowResMT_Data/main/HSB_monolingual.txt.gz",
            ),
            "wiki": SourceSpec(filename="wiki_hsb_monolingual.txt", extractor="plain"),
            "witaj": SourceSpec(filename="witaj_hsb_monolingual.txt", extractor="plain"),
        },
        default_sources=("leipzig", "wmt22"),
    ),
    "dsb": LangSpec(
        glotlid_label="__label__dsb_Latn",
        sources={
            "witaj": SourceSpec(filename="witaj_dsb_monolingual.txt", extractor="plain"),
            "mt_train": SourceSpec(
                filename="train.de-dsb.dsb",
                extractor="plain",
                url="https://raw.githubusercontent.com/TUM-NLP/llms-limited-resources2025/main/Sorbian/dsb/MT/train.de-dsb.dsb",
            ),
            "mt_dev": SourceSpec(
                filename="dev.de-dsb.dsb",
                extractor="plain",
                url="https://raw.githubusercontent.com/TUM-NLP/llms-limited-resources2025/main/Sorbian/dsb/MT/dev.de-dsb.dsb",
            ),
            "wmt22": SourceSpec(
                filename="66408_DSB_monolingual.txt.gz",
                extractor="wmt22_gz",
            ),
            "wiki": SourceSpec(
                filename="8815_DSB_wikipedia_2021.txt.gz",
                extractor="wmt22_gz",
            ),
            "mono": SourceSpec(
                filename="mono.dsb.gz",
                extractor="wmt22_gz",
            ),
        },
        default_sources=("witaj", "mt_train", "mt_dev"),
    ),
    "de": LangSpec(
        glotlid_label="__label__deu_Latn",
        sources={
            "news": SourceSpec(
                filename="deu_news_2023_300K.tar.gz",
                extractor="leipzig",
                url="https://downloads.wortschatz-leipzig.de/corpora/deu_news_2023_300K.tar.gz",
            ),
            "wiki": SourceSpec(
                filename="deu_wikipedia_2021_300K.tar.gz",
                extractor="leipzig",
                url="https://downloads.wortschatz-leipzig.de/corpora/deu_wikipedia_2021_300K.tar.gz",
            ),
        },
        default_sources=("news",),
    ),
    "pl": LangSpec(
        glotlid_label="__label__pol_Latn",
        sources={
            "news": SourceSpec(
                filename="pol_news_2022_300K.tar.gz",
                extractor="leipzig",
                url="https://downloads.wortschatz-leipzig.de/corpora/pol_news_2022_300K.tar.gz",
            ),
            "wiki": SourceSpec(
                filename="pol_wikipedia_2021_300K.tar.gz",
                extractor="leipzig",
                url="https://downloads.wortschatz-leipzig.de/corpora/pol_wikipedia_2021_300K.tar.gz",
            ),
        },
        default_sources=("news",),
    ),
}

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
        description="Download and preprocess a Sorbian corpus (hsb or dsb).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--lang",
        choices=sorted(LANG_REGISTRY),
        default="hsb",
        help="Language code. Selects sources, GlotLID label, and output directory.",
    )
    p.add_argument(
        "--glotlid-threshold",
        type=float,
        default=GLOTLID_THRESHOLD,
        help="Minimum GlotLID confidence to keep a line as the target language.",
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
        default=None,
        help="Which sources to combine into the corpus. Defaults to the "
             "registry's default_sources for the chosen language.",
    )
    p.add_argument(
        "--output-suffix",
        default="v3",
        help="Dataset tag for output filenames under data/processed/<lang>/. "
             "Default 'v3' produces v3.txt, v3_train.txt, etc.",
    )
    return p.parse_args()


def load_source(name: str, spec: SourceSpec, raw_dir: Path) -> list[str]:
    """Download (if needed) and extract sentences for one source."""
    path = raw_dir / spec.filename
    if spec.url is not None:
        download_file(spec.url, path)
    elif not path.exists():
        raise FileNotFoundError(
            f"Source {name!r} has no download URL and the file is missing: {path}"
        )

    if spec.extractor == "leipzig":
        return extract_leipzig(path)
    if spec.extractor == "wmt22_gz":
        return extract_wmt22(path)
    if spec.extractor == "plain":
        return extract_plain(path, name)
    raise ValueError(f"Unknown extractor {spec.extractor!r} for source {name!r}")


def main() -> None:
    args = parse_args()

    lang_spec = LANG_REGISTRY[args.lang]
    raw_dir = DATA_DIR / "raw" / args.lang
    processed_dir = DATA_DIR / "processed" / args.lang
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sources = tuple(args.sources) if args.sources else lang_spec.default_sources
    unknown = [s for s in sources if s not in lang_spec.sources]
    if unknown:
        valid = sorted(lang_spec.sources)
        raise SystemExit(
            f"Unknown source(s) for --lang {args.lang}: {unknown}. Valid: {valid}"
        )

    # Output files live under data/processed/<lang>/ — the language is the
    # parent directory, so the filename only carries the dataset tag.
    dataset = args.output_suffix or "default"
    output_file = processed_dir / f"{dataset}.txt"
    train_file = processed_dir / f"{dataset}_train.txt"
    dev_file = processed_dir / f"{dataset}_dev.txt"
    test_file = processed_dir / f"{dataset}_test.txt"

    logger.info(f"Language: {args.lang}")
    logger.info(f"Sources: {list(sources)}")
    logger.info(f"Output dataset tag: {dataset}")

    per_source_counts: dict[str, int] = {}
    all_sentences: list[str] = []
    for source in sources:
        sents = load_source(source, lang_spec.sources[source], raw_dir)
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
            sentences, args.glotlid_threshold, lang_spec.glotlid_label
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
