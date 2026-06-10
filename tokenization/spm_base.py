"""Base class for SentencePiece tokenizers (shared by BPE and Unigram)."""

import logging
import os
import tempfile
from typing import List

import sentencepiece as spm

from tokenization.base import SPECIAL_TOKENS, BaseTokenizer
from tokenization.hf_base import SpmBackedHFTokenizer, apply_default_special_tokens
from tokenization.pretokenize import moses_detokenize, moses_pretokenize
from tokenization.registry import detect_tokenizer_type, read_config, write_config

logger = logging.getLogger(__name__)


def train_spm_model(corpus_path: str, vocab_size: int, model_type: str) -> bytes:
    """Train a SentencePiece model and return its serialized bytes.

    Shared by `BaseSPMTokenizer.train()` and `MorphBPETokenizer.train()`.
    """
    user_defined_symbols = [t for t in SPECIAL_TOKENS if t != "[UNK]"]
    with tempfile.TemporaryDirectory() as tmpdir:
        model_prefix = os.path.join(tmpdir, "spm")
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            user_defined_symbols=user_defined_symbols,
            pad_id=0,
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
            pad_piece="[PAD]",
            unk_piece="[UNK]",
            train_extremely_large_corpus=False,
            num_threads=os.cpu_count() or 1,
        )
        with open(f"{model_prefix}.model", "rb") as f:
            return f.read()


class SentencePieceHFTokenizer(SpmBackedHFTokenizer):
    """Slow Hugging Face wrapper backed directly by a SentencePiece model."""

    vocab_files_names = {"vocab_file": "spm.model"}

    def __init__(
        self,
        vocab_file: str | None = None,
        model_proto: bytes | None = None,
        **kwargs,
    ) -> None:
        apply_default_special_tokens(kwargs)

        self.sp_model = spm.SentencePieceProcessor()
        if model_proto is not None:
            self.sp_model.load_from_serialized_proto(model_proto)
            self._model_proto = model_proto
        elif vocab_file is not None:
            self.sp_model.load(vocab_file)
            with open(vocab_file, "rb") as f:
                self._model_proto = f.read()
        else:
            raise ValueError("Either vocab_file or model_proto must be provided.")

        super().__init__(**kwargs)

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(moses_pretokenize(text), out_type=str)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        model_file = os.path.join(save_directory, f"{prefix}spm.model")
        with open(model_file, "wb") as f:
            f.write(self._model_proto)
        return (model_file,)


class BaseSPMTokenizer(BaseTokenizer):
    """Base class for SentencePiece-based tokenizers.

    Subclasses set `tokenizer_type` and `model_type` (the SPM algorithm name).
    """

    tokenizer_type: str = ""  # Override in subclass: "spm_bpe" or "spm_unigram"
    model_type: str = ""  # Override in subclass: "bpe" or "unigram"

    def __init__(self) -> None:
        self.sp_model: spm.SentencePieceProcessor | None = None
        self._model_bytes: bytes | None = None

    def train(self, corpus_path: str, vocab_size: int) -> None:
        """Train a SentencePiece model on the corpus."""
        if not self.model_type:
            raise ValueError("model_type must be set by subclass")

        logger.info(f"Training SentencePiece {self.model_type} model...")
        logger.info(f"  Corpus: {corpus_path}")
        logger.info(f"  Target vocab size: {vocab_size}")

        self._model_bytes = train_spm_model(corpus_path, vocab_size, self.model_type)
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load_from_serialized_proto(self._model_bytes)

        logger.info(f"  Trained vocab size: {self.sp_model.get_piece_size()}")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subword pieces."""
        if self.sp_model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        return self.sp_model.encode(moses_pretokenize(text), out_type=str)

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        if self.sp_model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        return self.sp_model.encode(moses_pretokenize(text), out_type=int)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if self.sp_model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        return moses_detokenize(self.sp_model.decode(ids))

    def save(self, path: str) -> None:
        """Save the tokenizer to a directory."""
        if self.sp_model is None or self._model_bytes is None:
            raise RuntimeError("Model not loaded. Call train() first.")

        os.makedirs(path, exist_ok=True)
        model_file = os.path.join(path, "spm.model")
        with open(model_file, "wb") as f:
            f.write(self._model_bytes)

        write_config(
            path,
            tokenizer_type=self.tokenizer_type,
            vocab_size=self.sp_model.get_piece_size(),
        )

        logger.info(f"Saved tokenizer to {path}")

    def load(self, path: str) -> None:
        """Load a tokenizer from a directory."""
        model_file = os.path.join(path, "spm.model")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_file)

        with open(model_file, "rb") as f:
            self._model_bytes = f.read()

        # Sanity check: if a config is present, it should match this class.
        config = read_config(path)
        if config:
            saved_type = config.get("tokenizer_type") or detect_tokenizer_type(path)
            if self.tokenizer_type and saved_type != self.tokenizer_type:
                logger.warning(
                    "Loading %s tokenizer from a %s save directory.",
                    self.tokenizer_type,
                    saved_type,
                )

        logger.info(f"Loaded tokenizer from {path}")
        logger.info(f"  Vocab size: {self.sp_model.get_piece_size()}")

    def to_hf_tokenizer(self) -> SentencePieceHFTokenizer:
        """Convert to a HuggingFace tokenizer."""
        if self.sp_model is None or self._model_bytes is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")

        return SentencePieceHFTokenizer(model_proto=self._model_bytes)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.sp_model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> dict:
        """Get vocabulary as dict mapping tokens to IDs."""
        if self.sp_model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        return {
            self.sp_model.id_to_piece(i): i
            for i in range(self.sp_model.get_piece_size())
        }
