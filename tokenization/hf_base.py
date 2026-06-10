"""Shared base classes for HuggingFace tokenizer wrappers.

Eliminates the special-token setdefault boilerplate every wrapper repeated and
factors out the SentencePiece-backed methods shared by the two SPM-based
wrappers (plain SPM and MorphBPE).
"""

from typing import Dict, List

from transformers import PreTrainedTokenizer

DEFAULT_SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "clean_up_tokenization_spaces": False,
}


def apply_default_special_tokens(kwargs: dict) -> None:
    """Populate the standard special-token kwargs in-place if absent."""
    for key, value in DEFAULT_SPECIAL_TOKENS.items():
        kwargs.setdefault(key, value)


class BaseHFTokenizer(PreTrainedTokenizer):
    """Common base for all HF tokenizer wrappers in this project.

    Subclasses should call `apply_default_special_tokens(kwargs)` before
    `super().__init__(**kwargs)`.

    Adds BERT-style `[CLS] X [SEP]` wrapping and the matching special-token
    mask + segment IDs. Required for: MLM data collators (which read
    `get_special_tokens_mask` to avoid masking specials) and downstream tasks
    that consume a `[CLS]` sentence embedding.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: List[int] | None = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: List[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: List[int] | None = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return [0] * (len(cls) + len(token_ids_0) + len(sep))
        return (
            [0] * (len(cls) + len(token_ids_0) + len(sep))
            + [1] * (len(token_ids_1) + len(sep))
        )


class SpmBackedHFTokenizer(BaseHFTokenizer):
    """Base for HF wrappers whose vocabulary is owned by a SentencePiece model.

    Subclasses must set `self.sp_model` (a loaded SentencePieceProcessor) before
    `super().__init__()` and implement `_tokenize` and `save_vocabulary`.
    """

    @property
    def vocab_size(self) -> int:
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> Dict[str, int]:
        return {
            self.sp_model.id_to_piece(i): i
            for i in range(self.sp_model.get_piece_size())
        }

    def _convert_token_to_id(self, token: str) -> int:
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        if 0 <= index < self.sp_model.get_piece_size():
            return self.sp_model.id_to_piece(index)
        return self.unk_token

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        from tokenization.pretokenize import moses_detokenize

        return moses_detokenize(self.sp_model.decode_pieces(tokens))
