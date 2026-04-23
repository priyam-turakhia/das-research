#!/usr/bin/env python3
"""Evaluate tokenizer(s) on a corpus."""

import argparse
import json
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
from tokenization.morfessor import MorfessorTokenizer
from tokenization.spm_bpe import SPMBPETokenizer
from tokenization.spm_unigram import SPMUnigramTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def detect_tokenizer_type(model_path: Path) -> str:
    """Auto-detect tokenizer type from saved files."""
    config_file = model_path / "tokenizer_config.json"

    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)
            tokenizer_class = config.get("tokenizer_class", "")

            if "BPE" in tokenizer_class:
                return "spm_bpe"
            elif "Unigram" in tokenizer_class:
                return "spm_unigram"
            elif "Morfessor" in tokenizer_class or config.get("model_type") == "morfessor":
                return "morfessor"

    # Fallback: check for model files
    if (model_path / "model.pkl").exists():
        return "morfessor"
    elif (model_path / "spm.model").exists():
        # Check if we can determine BPE vs Unigram from model
        # Default to BPE if unclear
        return "spm_bpe"

    raise ValueError(f"Cannot detect tokenizer type from {model_path}")


def load_tokenizer(model_path: Path, tokenizer_type: str | None = None) -> BaseTokenizer:
    """Load a tokenizer from a saved directory."""
    if tokenizer_type is None:
        tokenizer_type = detect_tokenizer_type(model_path)

    logger.info(f"Loading {tokenizer_type} tokenizer from {model_path}")

    if tokenizer_type == "spm_bpe":
        tokenizer = SPMBPETokenizer()
    elif tokenizer_type == "spm_unigram":
        tokenizer = SPMUnigramTokenizer()
    elif tokenizer_type == "morfessor":
        tokenizer = MorfessorTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    tokenizer.load(str(model_path))
    return tokenizer


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
        help="Path to held-out evaluation corpus (prefer hsb_dev.txt or hsb_test.txt).",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["spm_bpe", "spm_unigram", "morfessor"],
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

    if corpus_path.name == "hsb.txt":
        sibling_dev = corpus_path.with_name("hsb_dev.txt")
        sibling_test = corpus_path.with_name("hsb_test.txt")
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
