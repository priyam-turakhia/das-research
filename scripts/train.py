#!/usr/bin/env python3
"""Train a tokenizer on a corpus."""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenization.registry import get_tokenizer_classes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    tokenizer_classes = get_tokenizer_classes()

    parser = argparse.ArgumentParser(
        description="Train a tokenizer on a corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=list(tokenizer_classes.keys()),
        help="Tokenization method to use.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to training corpus. Prefer the *_train.txt split when split files exist; e.g. data/processed/<lang>/<dataset>_train.txt.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16000,
        help="Target vocabulary size.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for trained tokenizer.",
    )

    args = parser.parse_args()

    # Validate corpus exists
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        logger.error(f"Corpus not found: {corpus_path}")
        sys.exit(1)

    # Warn if pointing at an unsplit full corpus file (e.g. v3.txt) when the
    # _train/_dev sibling splits exist — training on the full corpus
    # contaminates the dev set.
    stem = corpus_path.stem
    if not (
        stem.endswith("_train") or stem.endswith("_dev") or stem.endswith("_test")
    ):
        sibling_train = corpus_path.with_name(f"{stem}_train.txt")
        sibling_dev = corpus_path.with_name(f"{stem}_dev.txt")
        if sibling_train.exists() and sibling_dev.exists():
            logger.warning(
                "Using the full corpus for training. Prefer %s so %s stays held out.",
                sibling_train,
                sibling_dev,
            )

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Tokenizer Training")
    logger.info("=" * 60)
    logger.info(f"Method: {args.method}")
    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Vocab size: {args.vocab_size}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    # Create tokenizer
    tokenizer = tokenizer_classes[args.method]()

    # Train
    start_time = time.time()
    tokenizer.train(str(corpus_path), args.vocab_size)
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.1f}s")

    # Save
    tokenizer.save(str(output_path))

    # Summary
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Final vocab size: {tokenizer.vocab_size}")
    logger.info(f"Training time: {train_time:.1f}s")
    logger.info(f"Saved to: {output_path}")

    # Quick test
    test_text = "Hornjoserbšćina je zapadosłowjanska rěč."
    logger.info(f"\nTest: '{test_text}'")
    tokens = tokenizer.tokenize(test_text)
    logger.info(f"Tokens: {tokens}")
    ids = tokenizer.encode(test_text)
    logger.info(f"IDs: {ids}")
    decoded = tokenizer.decode(ids)
    logger.info(f"Decoded: '{decoded}'")


if __name__ == "__main__":
    main()
