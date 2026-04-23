"""SentencePiece BPE tokenizer."""

from tokenization.spm_base import BaseSPMTokenizer


class SPMBPETokenizer(BaseSPMTokenizer):
    """SentencePiece tokenizer using BPE (Byte Pair Encoding) algorithm."""

    model_type = "bpe"
