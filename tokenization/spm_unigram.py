"""SentencePiece Unigram tokenizer."""

from tokenization.spm_base import BaseSPMTokenizer


class SPMUnigramTokenizer(BaseSPMTokenizer):
    """SentencePiece tokenizer using Unigram language model algorithm."""

    tokenizer_type = "spm_unigram"
    model_type = "unigram"
