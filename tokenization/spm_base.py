"""Base class for SentencePiece tokenizers (shared by BPE and Unigram)."""

import json
import logging
import os
import tempfile
from typing import List

import sentencepiece as spm
from transformers import PreTrainedTokenizer

from tokenization.base import SPECIAL_TOKENS, BaseTokenizer

logger = logging.getLogger(__name__)


class SentencePieceHFTokenizer(PreTrainedTokenizer):
    """Slow Hugging Face wrapper backed directly by a SentencePiece model."""

    vocab_files_names = {"vocab_file": "spm.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str | None = None,
        model_proto: bytes | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("mask_token", "[MASK]")
        kwargs.setdefault("clean_up_tokenization_spaces", False)

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

    @property
    def vocab_size(self) -> int:
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> dict[str, int]:
        return {
            self.sp_model.id_to_piece(i): i
            for i in range(self.sp_model.get_piece_size())
        }

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        if 0 <= index < self.sp_model.get_piece_size():
            return self.sp_model.id_to_piece(index)
        return self.unk_token

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.sp_model.decode_pieces(tokens)

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

    Subclasses set `model_type` to 'bpe' or 'unigram'.
    """

    model_type: str = ""  # Override in subclass

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

        # Train in a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            model_prefix = os.path.join(tmpdir, "spm")

            # Define user symbols (excluding UNK which SPM handles)
            user_defined_symbols = SPECIAL_TOKENS.copy()
            user_defined_symbols.remove("[UNK]")

            spm.SentencePieceTrainer.train(
                input=corpus_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type=self.model_type,
                character_coverage=1.0,
                user_defined_symbols=user_defined_symbols,
                pad_id=0,
                unk_id=1,
                bos_id=-1,  # Disable BOS
                eos_id=-1,  # Disable EOS
                pad_piece="[PAD]",
                unk_piece="[UNK]",
                train_extremely_large_corpus=False,
                num_threads=os.cpu_count() or 1,
            )

            # Load the trained model
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(f"{model_prefix}.model")

            # Store model bytes for later saving
            with open(f"{model_prefix}.model", "rb") as f:
                self._model_bytes = f.read()

        logger.info(f"  Trained vocab size: {self.sp_model.get_piece_size()}")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subword pieces."""
        if self.sp_model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        return self.sp_model.encode(text, out_type=str)

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        if self.sp_model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        return self.sp_model.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if self.sp_model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        return self.sp_model.decode(ids)

    def save(self, path: str) -> None:
        """Save the tokenizer to a directory."""
        if self.sp_model is None:
            raise RuntimeError("Model not loaded. Call train() first.")

        os.makedirs(path, exist_ok=True)

        # Save the model file
        model_file = os.path.join(path, "spm.model")
        with open(model_file, "wb") as f:
            f.write(self._model_bytes)

        # Save config for loading
        config = {
            "model_type": self.model_type,
            "vocab_size": self.sp_model.get_piece_size(),
            "tokenizer_class": self.__class__.__name__,
        }
        config_file = os.path.join(path, "tokenizer_config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

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
