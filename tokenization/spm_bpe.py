"""SentencePiece BPE tokenizer."""

from tokenization.spm_base import BaseSPMTokenizer


class SPMBPETokenizer(BaseSPMTokenizer):
    """SentencePiece tokenizer using BPE (Byte Pair Encoding) algorithm."""

    tokenizer_type = "spm_bpe"
    model_type = "bpe"
