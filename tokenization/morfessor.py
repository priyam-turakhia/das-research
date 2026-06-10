"""Morfessor 2.0 tokenizer with HuggingFace integration."""

import json
import logging
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import morfessor

from tokenization.base import SPECIAL_TOKENS, BaseTokenizer
from tokenization.hf_base import BaseHFTokenizer, apply_default_special_tokens
from tokenization.pretokenize import moses_detokenize, moses_pretokenize
from tokenization.registry import write_config

logger = logging.getLogger(__name__)


def load_morfessor_annotations(path: str | Path) -> dict[str, list[tuple[str, ...]]]:
    """Load TSV annotations in the structure expected by Morfessor."""
    annotations: dict[str, list[tuple[str, ...]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            surface, morph_text = line.rstrip("\n").split("\t", 1)
            annotations[surface] = [tuple(morph_text.split())]
    return annotations


def segment_word_with_vocab(
    model: morfessor.BaselineModel,
    vocab: Dict[str, int],
    word: str,
) -> List[str]:
    """Segment a word and fall back to characters when segments are out of vocab."""
    try:
        segments, _ = model.viterbi_segment(word)
    except (KeyError, ValueError):
        logger.debug(f"OOV word, falling back to chars: {word}")
        return list(word)

    result = []
    for segment in segments:
        if segment in vocab:
            result.append(segment)
        else:
            logger.debug(f"Morpheme not in vocab, char fallback: {segment}")
            result.extend(list(segment))
    return result


class MorfessorHFTokenizer(BaseHFTokenizer):
    """HuggingFace-compatible wrapper for Morfessor tokenizer.

    This is a slow tokenizer (not Fast) because Morfessor segmentation
    is done in Python.
    """

    vocab_files_names = {"vocab_file": "vocab.json", "model_file": "model.pkl"}

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        model_file: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        model: Optional[morfessor.BaselineModel] = None,
        **kwargs,
    ):
        apply_default_special_tokens(kwargs)

        # Load vocab and model from files if not provided directly
        if vocab is not None:
            self.vocab = vocab
        elif vocab_file is not None:
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}

        if model is not None:
            self.morfessor_model = model
        elif model_file is not None:
            with open(model_file, "rb") as f:
                self.morfessor_model = pickle.load(f)
        else:
            self.morfessor_model = None

        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using the same ID stream as the native tokenizer."""
        if self.morfessor_model is None:
            raise RuntimeError("Morfessor model not loaded")

        tokens = []
        for word in moses_pretokenize(text).split():
            tokens.extend(segment_word_with_vocab(self.morfessor_model, self.vocab, word))

        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get("[UNK]", 1))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, "[UNK]")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Concatenate morphemes and reverse Moses pretokenization.

        Word boundaries are not preserved in Morfessor's flat ID stream by
        design — Morfessor never crosses word boundaries during segmentation,
        so the original word groupings are recovered approximately by Moses
        detokenization rules.
        """
        return moses_detokenize("".join(tokens))

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str, str]:
        """Save vocabulary and model to directory."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        prefix = f"{filename_prefix}-" if filename_prefix else ""

        vocab_file = os.path.join(save_directory, f"{prefix}vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        model_file = os.path.join(save_directory, f"{prefix}model.pkl")
        if self.morfessor_model is not None:
            with open(model_file, "wb") as f:
                pickle.dump(self.morfessor_model, f)

        return vocab_file, model_file


class MorfessorTokenizer(BaseTokenizer):
    """Morfessor 2.0 tokenizer.

    Uses NumMorphCorpusWeight during training to target a morpheme count that
    fits the vocabulary budget, with character-level fallback for any morpheme
    that ends up outside the final vocabulary.
    """

    tokenizer_type = "morfessor"

    def __init__(self) -> None:
        self.model: morfessor.BaselineModel | None = None
        self.vocab: Dict[str, int] = {}
        self.ids_to_tokens: Dict[int, str] = {}

    def train(
        self,
        corpus_path: str,
        vocab_size: int,
        annotations_path: str | None = None,
        use_num_morph_weight: bool = True,
    ) -> None:
        """Train Morfessor model and build the vocabulary.

        `use_num_morph_weight=True` (default) wires Morfessor's `NumMorphCorpusWeight`
        updater so α auto-tunes during training to hit the target morpheme-type budget.
        Subclasses that combine training-time supervision (`set_annotations`) with a
        weight tuner can pass `False` to keep α fixed and avoid the two updaters
        adapting against each other; the post-hoc cap below still enforces the budget.
        """
        logger.info("Training Morfessor model...")
        logger.info(f"  Corpus: {corpus_path}")
        logger.info(f"  Target vocab size: {vocab_size}")

        # Read corpus and collect character inventory
        logger.info("  Reading corpus and collecting characters...")
        char_inventory = set()
        word_counts: Counter = Counter()

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                for char in line:
                    char_inventory.add(char)
                for word in line.split():
                    word_counts[word] += 1

        logger.info(f"  Found {len(char_inventory)} unique characters")
        logger.info(f"  Found {len(word_counts)} unique words")

        # Target morpheme count: vocab_size minus special tokens and character slots.
        # Morfessor self-tunes its corpus weight (α) during training to hit this budget
        # when use_num_morph_weight is True; otherwise the post-hoc cap below enforces it.
        target_morphs = vocab_size - len(SPECIAL_TOKENS) - len(char_inventory)
        logger.info(f"  Target morpheme types: {target_morphs}")

        # Canonical Morfessor settings:
        # - forcesplit_list=['-']: always split hyphens (CLI default)
        # - NumMorphCorpusWeight: auto-tune α during training to target a morpheme budget
        logger.info("  Training Morfessor model...")
        if use_num_morph_weight:
            self.model = morfessor.BaselineModel(
                forcesplit_list=["-"],
                corpusweight=morfessor.baseline.NumMorphCorpusWeight(
                    num_morph_types=target_morphs
                ),
            )
        else:
            # Fixed α (Morfessor default 1.0). Used when a second updater (e.g. the
            # annotation-weight tuner from set_annotations) needs to adapt alone.
            self.model = morfessor.BaselineModel(forcesplit_list=["-"])

        # count_modifier=lambda c: 1 implements the canonical "ones" count dampening.
        # Every word type contributes equally regardless of frequency, preventing
        # high-frequency words from dominating the lexicon.
        logger.info(f"  Loading {len(word_counts)} word types...")
        word_freq_list = [(count, word) for word, count in word_counts.items()]
        self.model.load_data(word_freq_list, count_modifier=lambda c: 1)

        if annotations_path is not None:
            annotations = load_morfessor_annotations(annotations_path)
            logger.info(f"  Loading {len(annotations)} supervised annotations: {annotations_path}")
            self.model.set_annotations(annotations)

        # Train the model
        self.model.train_batch()

        # Get morphemes from model lexicon
        logger.info("  Extracting morphemes from model lexicon...")
        morpheme_counts: Counter = Counter()

        # Get segmentations and count morphemes
        for word, count in word_counts.items():
            try:
                segments, _ = self.model.viterbi_segment(word)
                for morpheme in segments:
                    morpheme_counts[morpheme] += count
            except (KeyError, ValueError):
                # Word not in model, will use char fallback
                pass

        logger.info(f"  Found {len(morpheme_counts)} unique morphemes")

        # Build vocabulary with proper structure:
        # [special tokens 0-4] + [chars 5 to 5+N-1] + [morphemes 5+N to vocab_size]
        self.vocab = {}

        # 1. Add special tokens (IDs 0-4)
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.vocab[tok] = i

        # 2. Add character inventory using a running counter so IDs stay dense
        # even when a char duplicates a special token. Characters are guaranteed
        # slots so the fallback path can never produce [UNK] for a previously-seen
        # character.
        sorted_chars = sorted(char_inventory)
        next_id = len(SPECIAL_TOKENS)
        for char in sorted_chars:
            if char not in self.vocab:  # Avoid duplicates
                self.vocab[char] = next_id
                next_id += 1

        # 3. Add morphemes. Same running-counter pattern: when a candidate
        # morpheme duplicates an existing char (e.g. single-char morphemes like
        # "o", "a", "n"), it's skipped without leaving an ID gap. Gaps here
        # would make max(vocab.values()) > len(vocab) - 1, breaking the model
        # embedding lookup at runtime.
        if next_id >= vocab_size:
            logger.warning(
                f"  No slots for morphemes! Chars ({len(sorted_chars)}) + "
                f"special ({len(SPECIAL_TOKENS)}) >= vocab_size ({vocab_size})"
            )

        for morpheme, _ in morpheme_counts.most_common():
            if next_id >= vocab_size:
                break
            if morpheme not in self.vocab:
                self.vocab[morpheme] = next_id
                next_id += 1

        # Build reverse mapping
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        logger.info(f"  Final vocab size: {len(self.vocab)}")
        logger.info(f"    Special tokens: {len(SPECIAL_TOKENS)}")
        logger.info(f"    Characters: {len(sorted_chars)}")
        logger.info(f"    Morphemes: {len(self.vocab) - len(SPECIAL_TOKENS) - len(sorted_chars)}")

        # Log example segmentations
        logger.info("  Example segmentations:")
        sample_words = list(word_counts.keys())[:10]
        for word in sample_words:
            try:
                segments, _ = self.model.viterbi_segment(word)
                logger.info(f"    {word} -> {segments}")
            except (KeyError, ValueError):
                logger.info(f"    {word} -> [char fallback]")

        # Calculate mean morphemes per word
        total_morphemes = 0
        total_words = 0
        for word, count in word_counts.items():
            try:
                segments, _ = self.model.viterbi_segment(word)
                total_morphemes += len(segments) * count
                total_words += count
            except (KeyError, ValueError):
                total_morphemes += len(word) * count  # char fallback
                total_words += count

        mean_morphemes = total_morphemes / total_words if total_words > 0 else 0
        logger.info(f"  Mean morphemes per word: {mean_morphemes:.2f}")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into a flat list of morphemes (no word-boundary markers)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")

        tokens = []
        for word in moses_pretokenize(text).split():
            tokens.extend(segment_word_with_vocab(self.model, self.vocab, word))
        return tokens

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        return [self.vocab.get(tok, self.vocab["[UNK]"]) for tok in self.tokenize(text)]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text.

        Word boundaries are not preserved in the flat ID stream — Morfessor
        never crosses word boundaries during segmentation, so the morpheme
        run is concatenated and Moses detokenization places spacing.
        """
        tokens = [self.ids_to_tokens.get(i, "[UNK]") for i in ids]
        return moses_detokenize("".join(tokens))

    def save(self, path: str) -> None:
        """Save tokenizer to directory."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        os.makedirs(path, exist_ok=True)

        model_file = os.path.join(path, "model.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(self.model, f)

        vocab_file = os.path.join(path, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        write_config(
            path,
            tokenizer_type=self.tokenizer_type,
            vocab_size=len(self.vocab),
            auto_map={"AutoTokenizer": "morfessor.MorfessorHFTokenizer"},
        )

        logger.info(f"Saved tokenizer to {path}")

    def load(self, path: str) -> None:
        """Load tokenizer from directory."""
        model_file = os.path.join(path, "model.pkl")
        vocab_file = os.path.join(path, "vocab.json")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        with open(model_file, "rb") as f:
            self.model = pickle.load(f)

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        logger.info(f"Loaded tokenizer from {path}")
        logger.info(f"  Vocab size: {len(self.vocab)}")

    def to_hf_tokenizer(self) -> MorfessorHFTokenizer:
        """Convert to HuggingFace tokenizer."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")

        return MorfessorHFTokenizer(
            vocab=self.vocab.copy(),
            model=self.model,
        )

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary as dict."""
        return self.vocab.copy()
