#!/usr/bin/env python3
"""Train a tokenizer on a corpus."""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenization.morfessor import MorfessorTokenizer
from tokenization.spm_bpe import SPMBPETokenizer
from tokenization.spm_unigram import SPMUnigramTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TOKENIZER_CLASSES = {
    "spm_bpe": SPMBPETokenizer,
    "spm_unigram": SPMUnigramTokenizer,
    "morfessor": MorfessorTokenizer,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tokenizer on a corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=list(TOKENIZER_CLASSES.keys()),
        help="Tokenization method to use.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to training corpus (prefer hsb_train.txt when split files exist).",
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

    if corpus_path.name == "hsb.txt":
        sibling_train = corpus_path.with_name("hsb_train.txt")
        sibling_dev = corpus_path.with_name("hsb_dev.txt")
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
    tokenizer_cls = TOKENIZER_CLASSES[args.method]
    tokenizer = tokenizer_cls()

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
