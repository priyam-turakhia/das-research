"""Evaluation utilities for tokenizers.

These metrics are sanity checks, not downstream task evaluation.
They verify tokenizers work correctly but do not indicate which is
best for actual NLP tasks.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from tokenization.base import SPECIAL_TOKENS, BaseTokenizer

logger = logging.getLogger(__name__)

# Fixed sample words for side-by-side comparison
SAMPLE_WORDS = [
    "dźěłaćerjo",  # workers
    "přewodźowanje",  # accompanying
    "najwjetšich",  # largest
    "předstajichu",  # presented
    "wozjewjenje",  # announcement
    "zwjazkoweje",  # federal
    "towaršnosće",  # society
    "předsydstwom",  # chairmanship
    "wuwiće",  # development
    "přinoški",  # contributions
    "słowjanskich",  # Slavic
    "wědomostnym",  # scientific
    "přełožować",  # to translate
    "wukubłanje",  # education
    "młodźinska",  # youth
    "wubědźowanje",  # competition
    "zawěsćić",  # to ensure
    "přewzać",  # to take over
    "skutkowanje",  # activity
    "předewzaće",  # enterprise
]


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    fertility_mean: float
    fertility_std: float
    vocab_size: int
    unique_tokens: int
    vocab_coverage: float
    oov_rate: float
    round_trip_pass: int
    round_trip_fail: int
    hf_match: int
    hf_mismatch: int
    segmentations: Dict[str, List[str]]


def load_corpus(corpus_path: str) -> List[str]:
    """Load a corpus file as a list of non-empty sentences."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def compute_fertility(tokenizer: BaseTokenizer, sentences: List[str]) -> Tuple[float, float]:
    """Compute tokens per whitespace word (mean, std).

    Args:
        tokenizer: Tokenizer to evaluate.
        sentences: List of sentences.

    Returns:
        Tuple of (mean_fertility, std_fertility).
    """
    ratios = []
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        tokens = tokenizer.tokenize(sentence)
        ratio = len(tokens) / len(words)
        ratios.append(ratio)

    if not ratios:
        return 0.0, 0.0

    mean = sum(ratios) / len(ratios)
    variance = sum((r - mean) ** 2 for r in ratios) / len(ratios)
    std = variance ** 0.5

    return mean, std


def compute_unique_tokens(tokenizer: BaseTokenizer, sentences: List[str]) -> int:
    """Count distinct token types actually produced on the evaluation set.

    This measures vocabulary utilization: how many of the learned vocab
    entries the tokenizer actually uses on real text.
    """
    seen = set()
    for sentence in sentences:
        for token in tokenizer.tokenize(sentence):
            seen.add(token)
    return len(seen)


def compute_oov_rate(
    tokenizer: BaseTokenizer, evaluation_sentences: List[str], unk_token: str = "[UNK]"
) -> float:
    """Compute OOV-like rate on an evaluation set.

    Counts words that either emit an unknown token or degrade into
    pure character-level segmentation.

    Args:
        tokenizer: Tokenizer to evaluate.
        evaluation_sentences: Evaluation sentences.
        unk_token: Unknown token string.

    Returns:
        OOV rate as fraction.
    """
    total_words = 0
    oov_words = 0

    for sentence in evaluation_sentences:
        for word in sentence.split():
            total_words += 1
            tokens = tokenizer.tokenize(word)
            normalized_tokens = [
                token[1:] if token.startswith("▁") else token
                for token in tokens
                if token != "▁"
            ]

            # Check if any token is UNK
            if unk_token in normalized_tokens:
                oov_words += 1
            elif normalized_tokens and all(len(t) == 1 for t in normalized_tokens) and len(word) > 1:
                oov_words += 1

    return oov_words / total_words if total_words > 0 else 0.0


def round_trip_test(
    tokenizer: BaseTokenizer, sentences: List[str], n_samples: int = 1000, seed: int = 42
) -> Tuple[int, int, List[Tuple[str, str]]]:
    """Test decode(encode(text)) == text.

    Args:
        tokenizer: Tokenizer to test.
        sentences: Sentences to sample from.
        n_samples: Number of samples to test.
        seed: Random seed.

    Returns:
        Tuple of (pass_count, fail_count, list of failures).
    """
    random.seed(seed)
    samples = random.sample(sentences, min(n_samples, len(sentences)))

    passed = 0
    failed = 0
    failures = []

    for text in samples:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        # Normalize whitespace for comparison
        text_normalized = " ".join(text.split())
        decoded_normalized = " ".join(decoded.split())

        if text_normalized == decoded_normalized:
            passed += 1
        else:
            failed += 1
            if len(failures) < 10:  # Keep first 10 failures
                failures.append((text_normalized, decoded_normalized))

    return passed, failed, failures


def hf_compatibility_test(
    tokenizer: BaseTokenizer, sentences: List[str], n_samples: int = 100, seed: int = 42
) -> Tuple[int, int, List[Tuple[str, List[int], List[int]]]]:
    """Test HuggingFace tokenizer matches native encode().

    Args:
        tokenizer: Tokenizer to test.
        sentences: Sentences to sample from.
        n_samples: Number of samples.
        seed: Random seed.

    Returns:
        Tuple of (match_count, mismatch_count, list of mismatches).
    """
    try:
        hf_tok = tokenizer.to_hf_tokenizer()
    except Exception as e:
        logger.warning(f"Could not create HF tokenizer: {e}")
        return 0, 0, []

    random.seed(seed)
    samples = random.sample(sentences, min(n_samples, len(sentences)))

    matched = 0
    mismatched = 0
    mismatches = []

    for text in samples:
        native_ids = tokenizer.encode(text)

        try:
            hf_ids = hf_tok.encode(text, add_special_tokens=False)
        except Exception:
            hf_ids = []

        if native_ids == hf_ids:
            matched += 1
        else:
            mismatched += 1
            if len(mismatches) < 5:
                mismatches.append((text, native_ids, hf_ids))

    return matched, mismatched, mismatches


def side_by_side_segmentation(
    tokenizers: Dict[str, BaseTokenizer], words: List[str] | None = None
) -> Dict[str, Dict[str, List[str]]]:
    """Get segmentations from multiple tokenizers for comparison.

    Args:
        tokenizers: Dict mapping name to tokenizer.
        words: Words to segment (uses SAMPLE_WORDS if None).

    Returns:
        Dict mapping word to dict of tokenizer name to segments.
    """
    if words is None:
        words = SAMPLE_WORDS

    result = {}
    for word in words:
        result[word] = {}
        for name, tok in tokenizers.items():
            try:
                segments = tok.tokenize(word)
                result[word][name] = segments
            except Exception as e:
                result[word][name] = [f"ERROR: {e}"]

    return result


def evaluate_tokenizer(
    tokenizer: BaseTokenizer,
    corpus_path: str,
    name: str = "tokenizer",
) -> EvaluationResult:
    """Run full evaluation on a tokenizer.

    Args:
        tokenizer: Tokenizer to evaluate.
        corpus_path: Path to corpus.
        name: Name for logging.

    Returns:
        EvaluationResult with all metrics.
    """
    logger.info(f"Evaluating {name}...")

    evaluation_sentences = load_corpus(corpus_path)
    logger.info(f"  Evaluation sentences: {len(evaluation_sentences)}")

    # Fertility
    fert_mean, fert_std = compute_fertility(tokenizer, evaluation_sentences)
    logger.info(f"  Fertility: {fert_mean:.3f} +/- {fert_std:.3f}")

    # Vocab size (excluding special tokens)
    vocab = tokenizer.get_vocab()
    vocab_size = len([t for t in vocab if t not in SPECIAL_TOKENS])
    logger.info(f"  Vocab size (excl. special): {vocab_size}")

    # Unique token types used on eval set
    unique_tokens = compute_unique_tokens(tokenizer, evaluation_sentences)
    total_vocab = tokenizer.vocab_size
    vocab_coverage = unique_tokens / total_vocab if total_vocab > 0 else 0.0
    logger.info(f"  Unique tokens used: {unique_tokens:,} ({vocab_coverage:.1%} of vocab)")

    # OOV rate
    oov_rate = compute_oov_rate(tokenizer, evaluation_sentences)
    logger.info(f"  OOV rate: {oov_rate:.2%}")

    # Round-trip test
    rt_pass, rt_fail, rt_failures = round_trip_test(tokenizer, evaluation_sentences)
    logger.info(f"  Round-trip: {rt_pass} pass, {rt_fail} fail")
    if rt_failures:
        logger.info("  Sample failures:")
        for orig, decoded in rt_failures[:3]:
            logger.info(f"    '{orig[:50]}...' -> '{decoded[:50]}...'")

    # HF compatibility
    hf_match, hf_mismatch, hf_mismatches = hf_compatibility_test(tokenizer, evaluation_sentences)
    logger.info(f"  HF compatibility: {hf_match} match, {hf_mismatch} mismatch")

    # Segmentations
    segmentations = {}
    for word in SAMPLE_WORDS[:10]:
        try:
            segmentations[word] = tokenizer.tokenize(word)
        except Exception:
            segmentations[word] = ["ERROR"]

    return EvaluationResult(
        fertility_mean=fert_mean,
        fertility_std=fert_std,
        vocab_size=vocab_size,
        unique_tokens=unique_tokens,
        vocab_coverage=vocab_coverage,
        oov_rate=oov_rate,
        round_trip_pass=rt_pass,
        round_trip_fail=rt_fail,
        hf_match=hf_match,
        hf_mismatch=hf_mismatch,
        segmentations=segmentations,
    )


def print_comparison_table(results: Dict[str, EvaluationResult]) -> None:
    """Print comparison table for multiple tokenizers.

    Args:
        results: Dict mapping tokenizer name to evaluation results.
    """
    print("\n" + "=" * 80)
    print("TOKENIZER COMPARISON")
    print("=" * 80)
    print(
        "\nNote: These metrics are sanity checks only. They do not indicate "
        "downstream task performance."
    )

    # Header
    names = list(results.keys())
    column_width = max(18, max(len(name) for name in names) + 2)
    header = f"{'Metric':<25}" + "".join(f"{n:>{column_width}}" for n in names)
    print("\n" + header)
    print("-" * len(header))

    # Metrics
    print(
        f"{'Fertility (mean)':<25}"
        + "".join(f"{r.fertility_mean:>{column_width}.3f}" for r in results.values())
    )
    print(
        f"{'Fertility (std)':<25}"
        + "".join(f"{r.fertility_std:>{column_width}.3f}" for r in results.values())
    )
    print(
        f"{'Vocab size':<25}"
        + "".join(f"{r.vocab_size:>{column_width},}" for r in results.values())
    )
    print(
        f"{'Unique tokens used':<25}"
        + "".join(f"{r.unique_tokens:>{column_width},}" for r in results.values())
    )
    print(
        f"{'Vocab coverage':<25}"
        + "".join(f"{r.vocab_coverage:>{column_width - 1}.1%}" for r in results.values())
    )
    print(
        f"{'OOV rate':<25}"
        + "".join(f"{r.oov_rate:>{column_width}.2%}" for r in results.values())
    )
    print(
        f"{'Round-trip pass':<25}"
        + "".join(f"{r.round_trip_pass:>{column_width},}" for r in results.values())
    )
    print(
        f"{'Round-trip fail':<25}"
        + "".join(f"{r.round_trip_fail:>{column_width},}" for r in results.values())
    )
    print(
        f"{'HF match':<25}"
        + "".join(f"{r.hf_match:>{column_width},}" for r in results.values())
    )
    print(
        f"{'HF mismatch':<25}"
        + "".join(f"{r.hf_mismatch:>{column_width},}" for r in results.values())
    )

    # Side-by-side segmentation
    print("\n" + "=" * 80)
    print("SAMPLE SEGMENTATIONS")
    print("=" * 80)

    for word in SAMPLE_WORDS[:10]:
        print(f"\n{word}:")
        for name, result in results.items():
            segs = result.segmentations.get(word, ["N/A"])
            print(f"  {name:>15}: {' | '.join(segs)}")

    print("\n" + "=" * 80)
