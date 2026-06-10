"""Hybrid Morfessor + BPE tokenizer.

Morfessor segments words into morphemes (no vocabulary budget), then BPE
trains on the morpheme stream and compresses to the target vocab size.
Canonical literature baseline for Morfessor + BPE hybrids.
"""

import logging
import os
import pickle
import tempfile
from collections import Counter
from typing import Dict, List, Optional, Tuple

import morfessor
import sentencepiece as spm

from tokenization.base import BaseTokenizer
from tokenization.hf_base import SpmBackedHFTokenizer, apply_default_special_tokens
from tokenization.pretokenize import moses_detokenize, moses_pretokenize
from tokenization.registry import write_config
from tokenization.spm_base import train_spm_model

logger = logging.getLogger(__name__)


def _segment_to_morpheme_stream(model: morfessor.BaselineModel, words: List[str]) -> str:
    """Run Morfessor segmentation on each word and join morphemes with spaces."""
    pieces = []
    for word in words:
        try:
            segments, _ = model.viterbi_segment(word)
        except (KeyError, ValueError):
            segments = list(word)
        pieces.extend(segments)
    return " ".join(pieces)


class MorphBPEHFTokenizer(SpmBackedHFTokenizer):
    """HuggingFace-compatible wrapper for the Morfessor + BPE hybrid."""

    vocab_files_names = {"morfessor_file": "morfessor_model.pkl", "spm_file": "spm.model"}

    def __init__(
        self,
        morfessor_file: Optional[str] = None,
        spm_file: Optional[str] = None,
        morfessor_model: Optional[morfessor.BaselineModel] = None,
        spm_model_bytes: Optional[bytes] = None,
        **kwargs,
    ):
        apply_default_special_tokens(kwargs)

        if morfessor_model is not None:
            self.morfessor_model = morfessor_model
        elif morfessor_file is not None:
            with open(morfessor_file, "rb") as f:
                self.morfessor_model = pickle.load(f)
        else:
            raise ValueError("Either morfessor_model or morfessor_file must be provided.")

        self.sp_model = spm.SentencePieceProcessor()
        if spm_model_bytes is not None:
            self.sp_model.load_from_serialized_proto(spm_model_bytes)
            self._spm_proto = spm_model_bytes
        elif spm_file is not None:
            self.sp_model.load(spm_file)
            with open(spm_file, "rb") as f:
                self._spm_proto = f.read()
        else:
            raise ValueError("Either spm_model_bytes or spm_file must be provided.")

        super().__init__(**kwargs)

    def _tokenize(self, text: str) -> List[str]:
        words = moses_pretokenize(text).split()
        stream = _segment_to_morpheme_stream(self.morfessor_model, words)
        return self.sp_model.encode(stream, out_type=str)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str, str]:
        os.makedirs(save_directory, exist_ok=True)
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        morfessor_file = os.path.join(save_directory, f"{prefix}morfessor_model.pkl")
        with open(morfessor_file, "wb") as f:
            pickle.dump(self.morfessor_model, f)
        spm_file = os.path.join(save_directory, f"{prefix}spm.model")
        with open(spm_file, "wb") as f:
            f.write(self._spm_proto)
        return morfessor_file, spm_file


class MorphBPETokenizer(BaseTokenizer):
    """Morfessor + BPE hybrid.

    Pipeline at training: Morfessor (unconstrained) is trained on the corpus,
    each word is segmented into morphemes, and SPM BPE is trained on the
    resulting morpheme stream with the target vocabulary size.
    """

    tokenizer_type = "morph_bpe"

    def __init__(self) -> None:
        self.morfessor_model: morfessor.BaselineModel | None = None
        self.sp_model: spm.SentencePieceProcessor | None = None
        self._spm_bytes: bytes | None = None

    def train(self, corpus_path: str, vocab_size: int) -> None:
        """Train Morfessor (no vocab budget) then BPE on the morpheme stream."""
        logger.info("Training MorphBPE tokenizer...")
        logger.info(f"  Corpus: {corpus_path}")
        logger.info(f"  Target vocab size: {vocab_size}")

        # Step 1: read corpus, build word-frequency counter
        logger.info("  Reading corpus and counting words...")
        word_counts: Counter = Counter()
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                for word in line.strip().split():
                    word_counts[word] += 1
        logger.info(f"  Found {len(word_counts)} unique words")

        # Step 2: train Morfessor with no vocabulary budget
        logger.info("  Training Morfessor (unconstrained)...")
        self.morfessor_model = morfessor.BaselineModel(forcesplit_list=["-"])
        word_freq_list = [(count, word) for word, count in word_counts.items()]
        self.morfessor_model.load_data(word_freq_list, count_modifier=lambda c: 1)
        self.morfessor_model.train_batch()

        # Step 3: pre-segment the corpus into a temp file
        logger.info("  Pre-segmenting corpus with Morfessor...")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as tmp:
            segmented_path = tmp.name
            with open(corpus_path, "r", encoding="utf-8") as src:
                for line in src:
                    words = line.strip().split()
                    if not words:
                        continue
                    tmp.write(_segment_to_morpheme_stream(self.morfessor_model, words))
                    tmp.write("\n")
        logger.info(f"  Wrote segmented corpus to {segmented_path}")

        # Step 4: train SPM BPE on the segmented corpus
        logger.info("  Training SPM BPE on morpheme stream...")
        try:
            self._spm_bytes = train_spm_model(segmented_path, vocab_size, "bpe")
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load_from_serialized_proto(self._spm_bytes)
        finally:
            os.unlink(segmented_path)

        logger.info(f"  Trained BPE vocab size: {self.sp_model.get_piece_size()}")

    def tokenize(self, text: str) -> List[str]:
        if self.morfessor_model is None or self.sp_model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        words = moses_pretokenize(text).split()
        stream = _segment_to_morpheme_stream(self.morfessor_model, words)
        return self.sp_model.encode(stream, out_type=str)

    def encode(self, text: str) -> List[int]:
        if self.morfessor_model is None or self.sp_model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        words = moses_pretokenize(text).split()
        stream = _segment_to_morpheme_stream(self.morfessor_model, words)
        return self.sp_model.encode(stream, out_type=int)

    def decode(self, ids: List[int]) -> str:
        if self.sp_model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        return moses_detokenize(self.sp_model.decode(ids))

    def save(self, path: str) -> None:
        if self.morfessor_model is None or self.sp_model is None or self._spm_bytes is None:
            raise RuntimeError("Model not trained. Call train() first.")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "morfessor_model.pkl"), "wb") as f:
            pickle.dump(self.morfessor_model, f)
        with open(os.path.join(path, "spm.model"), "wb") as f:
            f.write(self._spm_bytes)
        write_config(
            path,
            tokenizer_type=self.tokenizer_type,
            vocab_size=self.sp_model.get_piece_size(),
        )
        logger.info(f"Saved tokenizer to {path}")

    def load(self, path: str) -> None:
        morfessor_file = os.path.join(path, "morfessor_model.pkl")
        spm_file = os.path.join(path, "spm.model")
        if not os.path.exists(morfessor_file):
            raise FileNotFoundError(f"Morfessor model not found: {morfessor_file}")
        if not os.path.exists(spm_file):
            raise FileNotFoundError(f"SPM model not found: {spm_file}")

        with open(morfessor_file, "rb") as f:
            self.morfessor_model = pickle.load(f)
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(spm_file)
        with open(spm_file, "rb") as f:
            self._spm_bytes = f.read()

        logger.info(f"Loaded tokenizer from {path}")
        logger.info(f"  Vocab size: {self.sp_model.get_piece_size()}")

    def to_hf_tokenizer(self) -> MorphBPEHFTokenizer:
        if self.morfessor_model is None or self._spm_bytes is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        return MorphBPEHFTokenizer(
            morfessor_model=self.morfessor_model,
            spm_model_bytes=self._spm_bytes,
        )

    @property
    def vocab_size(self) -> int:
        if self.sp_model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> Dict[str, int]:
        if self.sp_model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        return {
            self.sp_model.id_to_piece(i): i
            for i in range(self.sp_model.get_piece_size())
        }
