"""SentencePiece Unigram tokenizer."""

from tokenization.spm_base import BaseSPMTokenizer


class SPMUnigramTokenizer(BaseSPMTokenizer):
    """SentencePiece tokenizer using Unigram language model algorithm."""

    model_type = "unigram"
