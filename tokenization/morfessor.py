"""Morfessor 2.0 tokenizer with HuggingFace integration."""

import json
import logging
import os
import pickle
from collections import Counter
from typing import Dict, List, Optional, Tuple

import morfessor
from transformers import PreTrainedTokenizer

from tokenization.base import SPECIAL_TOKENS, BaseTokenizer

logger = logging.getLogger(__name__)


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


class MorfessorHFTokenizer(PreTrainedTokenizer):
    """HuggingFace-compatible wrapper for Morfessor tokenizer.

    This is a slow tokenizer (not Fast) because Morfessor segmentation
    is done in Python.
    """

    vocab_files_names = {"vocab_file": "vocab.json", "model_file": "model.pkl"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        model_file: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        model: Optional[morfessor.BaselineModel] = None,
        **kwargs,
    ):
        # Set special tokens before calling super().__init__
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("mask_token", "[MASK]")
        kwargs.setdefault("clean_up_tokenization_spaces", False)

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
        for word in text.split():
            word_tokens = segment_word_with_vocab(self.morfessor_model, self.vocab, word)
            if word_tokens:
                tokens.append("▁")
                tokens.extend(word_tokens)

        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get("[UNK]", 1))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, "[UNK]")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens back to text using ▁ as an explicit word boundary."""
        words = []
        current_word = []

        for token in tokens:
            if token == "▁":
                if current_word:
                    words.append("".join(current_word))
                current_word = []
                continue

            if token.startswith("▁"):
                if current_word:
                    words.append("".join(current_word))
                current_word = [token[1:]]
                continue

            current_word.append(token)

        if current_word:
            words.append("".join(current_word))

        return " ".join(words)

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

    def __init__(self) -> None:
        self.model: morfessor.BaselineModel | None = None
        self.vocab: Dict[str, int] = {}
        self.ids_to_tokens: Dict[int, str] = {}

    def train(self, corpus_path: str, vocab_size: int) -> None:
        """Train Morfessor model and build the vocabulary."""
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
        # Morfessor self-tunes its corpus weight (α) during training to hit this budget,
        # rather than us post-hoc capping a larger lexicon.
        target_morphs = vocab_size - len(SPECIAL_TOKENS) - len(char_inventory) - 1
        logger.info(f"  Target morpheme types: {target_morphs}")

        # Canonical Morfessor settings:
        # - forcesplit_list=['-']: always split hyphens (CLI default)
        # - NumMorphCorpusWeight: auto-tune α during training to target a morpheme budget
        logger.info("  Training Morfessor model...")
        self.model = morfessor.BaselineModel(
            forcesplit_list=["-"],
            corpusweight=morfessor.baseline.NumMorphCorpusWeight(
                num_morph_types=target_morphs
            ),
        )

        # count_modifier=lambda c: 1 implements the canonical "ones" count dampening.
        # Every word type contributes equally regardless of frequency, preventing
        # high-frequency words from dominating the lexicon.
        logger.info(f"  Loading {len(word_counts)} word types...")
        word_freq_list = [(count, word) for word, count in word_counts.items()]
        self.model.load_data(word_freq_list, count_modifier=lambda c: 1)

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

        # 2. Add character inventory (IDs 5 to 5+len(chars)-1)
        # Include ▁ (word boundary marker) in character set
        char_inventory.add("▁")
        char_start_id = len(SPECIAL_TOKENS)
        sorted_chars = sorted(char_inventory)
        for i, char in enumerate(sorted_chars):
            if char not in self.vocab:  # Avoid duplicates
                self.vocab[char] = char_start_id + i

        # 3. Add morphemes. NumMorphCorpusWeight already targeted ~target_morphs
        # during training, so this top-N cut rarely evicts anything — it's a
        # safety net in case the model overshoots the budget.
        morpheme_start_id = char_start_id + len(sorted_chars)
        morpheme_slots = vocab_size - morpheme_start_id

        if morpheme_slots <= 0:
            logger.warning(
                f"  No slots for morphemes! Chars ({len(sorted_chars)}) + "
                f"special ({len(SPECIAL_TOKENS)}) >= vocab_size ({vocab_size})"
            )
            morpheme_slots = 0

        # Get top morphemes by frequency
        top_morphemes = morpheme_counts.most_common(morpheme_slots)
        for i, (morpheme, _) in enumerate(top_morphemes):
            if morpheme not in self.vocab:  # Avoid char/morpheme overlap
                self.vocab[morpheme] = morpheme_start_id + i

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
        """Tokenize text using Morfessor segmentation.

        Returns tokens with word boundary markers (▁ prefix for first morpheme of each word).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")

        tokens = []
        for word in text.split():
            word_tokens = self._segment_word(word)
            # Mark first token of word with ▁ prefix
            if word_tokens:
                word_tokens[0] = "▁" + word_tokens[0]
            tokens.extend(word_tokens)

        return tokens

    def _segment_word(self, word: str) -> List[str]:
        """Segment a single word, falling back to chars if needed."""
        return segment_word_with_vocab(self.model, self.vocab, word)

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        tokens = self.tokenize(text)
        ids = []
        for tok in tokens:
            if tok.startswith("▁"):
                # Word boundary token: encode ▁ + the rest separately
                ids.append(self.vocab["▁"])
                rest = tok[1:]
                if rest:
                    ids.append(self.vocab.get(rest, self.vocab["[UNK]"]))
            elif tok in self.vocab:
                ids.append(self.vocab[tok])
            else:
                ids.append(self.vocab["[UNK]"])
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = [self.ids_to_tokens.get(i, "[UNK]") for i in ids]
        # Reconstruct with spaces at word boundaries
        result = []
        current_word = []
        for tok in tokens:
            if tok.startswith("▁"):
                # New word starts
                if current_word:
                    result.append("".join(current_word))
                current_word = [tok[1:]]  # Remove ▁ prefix
            else:
                current_word.append(tok)
        if current_word:
            result.append("".join(current_word))
        return " ".join(result)

    def save(self, path: str) -> None:
        """Save tokenizer to directory."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        os.makedirs(path, exist_ok=True)

        # Save Morfessor model
        model_file = os.path.join(path, "model.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(self.model, f)

        # Save vocabulary
        vocab_file = os.path.join(path, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save config for loading and HF compatibility
        config = {
            "tokenizer_class": "MorfessorTokenizer",
            "vocab_size": len(self.vocab),
            "model_type": "morfessor",
            "auto_map": {
                "AutoTokenizer": "morfessor.MorfessorHFTokenizer",
            },
        }
        config_file = os.path.join(path, "tokenizer_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

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
