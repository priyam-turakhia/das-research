"""Semi-supervised Morfessor tokenizer using metadix morphology annotations."""

from pathlib import Path

from tokenization.morfessor import MorfessorTokenizer


DEFAULT_ANNOTATIONS_PATH = Path("data/processed/dsb/metadix_morph_annotations_500.tsv")


class SemiSupervisedMorfessorTokenizer(MorfessorTokenizer):
    """Morfessor with `set_annotations` supervision and a fixed corpus weight.

    Differs from `MorfessorTokenizer` in two ways:
    1. `train()` passes an annotations TSV through to `model.set_annotations(...)`.
    2. `use_num_morph_weight=False` so `corpusweight` stays at Morfessor's default 1.0.
       The annotation-weight tuner from `set_annotations` then adapts alone instead of
       fighting `NumMorphCorpusWeight`. The 16k vocabulary budget is enforced by the
       parent's post-hoc top-N morpheme cap.
    """

    tokenizer_type = "morfessor_semi"

    def __init__(self, annotations_path: str | Path | None = None) -> None:
        super().__init__()
        self.annotations_path = Path(annotations_path or DEFAULT_ANNOTATIONS_PATH)

    def train(self, corpus_path: str, vocab_size: int) -> None:
        super().train(
            corpus_path,
            vocab_size,
            annotations_path=str(self.annotations_path),
            use_num_morph_weight=False,
        )
