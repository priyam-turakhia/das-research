#!/usr/bin/env python3
"""Evaluate tokenizer(s) on a corpus."""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenization.base import BaseTokenizer
from tokenization.evaluate import (
    EvaluationResult,
    evaluate_tokenizer,
    print_comparison_table,
    side_by_side_segmentation,
)
from tokenization.registry import TOKENIZER_TYPES, load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate tokenizer(s) on a corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        action="append",
        dest="model_paths",
        help="Path to trained tokenizer directory. Can be specified multiple times.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to held-out evaluation corpus. Prefer the *_dev.txt or *_test.txt split; e.g. data/processed/<lang>/<dataset>_dev.txt.",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=list(TOKENIZER_TYPES),
        help="Override tokenizer type detection.",
    )

    args = parser.parse_args()

    if not args.model_paths:
        logger.error("At least one --model-path is required")
        sys.exit(1)

    # Validate corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        logger.error(f"Corpus not found: {corpus_path}")
        sys.exit(1)

    # Warn if evaluating on an unsplit full corpus file (e.g. v3.txt) while the
    # _dev/_test sibling splits exist — full-corpus metrics overlap training.
    stem = corpus_path.stem
    if not (
        stem.endswith("_train") or stem.endswith("_dev") or stem.endswith("_test")
    ):
        sibling_dev = corpus_path.with_name(f"{stem}_dev.txt")
        sibling_test = corpus_path.with_name(f"{stem}_test.txt")
        if sibling_dev.exists() and sibling_test.exists():
            logger.warning(
                "Using the full corpus for evaluation. Prefer %s or %s for held-out metrics.",
                sibling_dev,
                sibling_test,
            )

    logger.info("=" * 60)
    logger.info("Tokenizer Evaluation")
    logger.info("=" * 60)
    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Models: {args.model_paths}")
    logger.info("=" * 60)

    # Load and evaluate each tokenizer
    results: dict[str, EvaluationResult] = {}
    tokenizers: dict[str, BaseTokenizer] = {}

    for model_path_str in args.model_paths:
        model_path = Path(model_path_str)
        if not model_path.exists():
            logger.warning(f"Model path not found, skipping: {model_path}")
            continue

        try:
            tokenizer = load_tokenizer(model_path, args.type)
            name = model_path.name
            tokenizers[name] = tokenizer

            result = evaluate_tokenizer(tokenizer, str(corpus_path), name)
            results[name] = result
        except Exception as e:
            logger.error(f"Error evaluating {model_path}: {e}")
            continue

    if not results:
        logger.error("No tokenizers were successfully evaluated")
        sys.exit(1)

    # Print comparison
    print_comparison_table(results)

    # If multiple tokenizers, show side-by-side segmentation
    if len(tokenizers) > 1:
        print("\nSide-by-side comparison:")
        segmentations = side_by_side_segmentation(tokenizers)
        for word, segs in list(segmentations.items())[:10]:
            print(f"\n{word}:")
            for name, tokens in segs.items():
                print(f"  {name:>15}: {' | '.join(tokens)}")


if __name__ == "__main__":
    main()
