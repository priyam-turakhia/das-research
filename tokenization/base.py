"""Abstract base class for tokenizers."""

from abc import ABC, abstractmethod
from typing import List

from transformers import PreTrainedTokenizerBase

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizer implementations.

    All tokenizers must implement these methods to ensure a consistent interface.
    Special tokens are assigned IDs 0-4 in the order defined in SPECIAL_TOKENS.
    """

    @abstractmethod
    def train(self, corpus_path: str, vocab_size: int) -> None:
        """Train the tokenizer on a corpus.

        Args:
            corpus_path: Path to the training corpus (one sentence per line).
            vocab_size: Target vocabulary size (including special tokens).
        """
        ...

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of token strings.

        Args:
            text: Input text to tokenize.

        Returns:
            List of token strings.
        """
        ...

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs.

        Args:
            text: Input text to encode.

        Returns:
            List of token IDs.
        """
        ...

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded text string.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the tokenizer to a directory.

        Args:
            path: Directory path to save tokenizer files.
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a tokenizer from a directory.

        Args:
            path: Directory path containing tokenizer files.
        """
        ...

    @abstractmethod
    def to_hf_tokenizer(self) -> PreTrainedTokenizerBase:
        """Convert to a HuggingFace tokenizer.

        Returns:
            A HuggingFace PreTrainedTokenizer or PreTrainedTokenizerFast.
        """
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Get the vocabulary size.

        Returns:
            Total vocabulary size including special tokens.
        """
        ...

    def get_vocab(self) -> dict:
        """Get the vocabulary as a dict mapping tokens to IDs.

        Default implementation raises NotImplementedError.
        Subclasses should override this method.
        """
        raise NotImplementedError("Subclass must implement get_vocab()")
