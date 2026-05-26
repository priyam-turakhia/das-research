"""Moses pretokenization shared by all tokenizers.

Splits punctuation off words (`slovo,druhe.` -> `slovo , druhe .`) and reverses
the operation. Upper Sorbian is not a registered Moses language; we use Czech
('cs'), the closest registered Slavic relative.
"""

from sacremoses import MosesDetokenizer, MosesTokenizer

_TOKENIZER = MosesTokenizer(lang="cs")
_DETOKENIZER = MosesDetokenizer(lang="cs")


def moses_pretokenize(text: str) -> str:
    return _TOKENIZER.tokenize(text, return_str=True, escape=False)


def moses_detokenize(text: str) -> str:
    return _DETOKENIZER.detokenize(text.split())
