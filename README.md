# Upper Sorbian Tokenization Pipeline

Three subword tokenizers — SentencePiece BPE, SentencePiece Unigram, and Morfessor 2.0 — trained on the same Upper Sorbian corpus at the same vocabulary budget (16,000 tokens). The setup is an ablation: same data, same target size, same evaluation, so any downstream performance difference is attributable to the tokenization algorithm itself.

Current version: **v3**. The pipeline cleans the corpus with language identification, length and punctuation filters, applies Moses pretokenization, and trains Morfessor without a special word-boundary marker.

## Quick start

```
uv sync
uv run python scripts/download_data.py
uv run python scripts/train.py --method spm_bpe --corpus data/processed/hsb_train.txt --vocab-size 16000 --output models/hsb_spm_bpe_v3
uv run python scripts/train.py --method spm_unigram --corpus data/processed/hsb_train.txt --vocab-size 16000 --output models/hsb_spm_unigram_v3
uv run python scripts/train.py --method morfessor --corpus data/processed/hsb_train.txt --vocab-size 16000 --output models/hsb_morfessor_v3
uv run python scripts/evaluate.py --model-path models/hsb_spm_bpe_v3 --model-path models/hsb_spm_unigram_v3 --model-path models/hsb_morfessor_v3 --corpus data/processed/hsb_dev.txt
```

End-to-end wall-clock time is dominated by Morfessor training (~13 min on this corpus). Everything else finishes in under a minute combined.

## Documentation

- [docs/PROJECT.md](docs/PROJECT.md) — full description of the project as it stands today: directory layout, data pipeline, library code file by file, scripts and flags, modularity, caveats. Self-contained.
- [docs/METRICS.md](docs/METRICS.md) — what each evaluation metric calculates, exactly how the code computes it, what the numbers mean in plain terms, and what a typical good score looks like.
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — round-by-round history of changes to the pipeline (v1 → v2 → v3) with the reasoning behind each change.
- [docs/EVALUATIONS.md](docs/EVALUATIONS.md) — current evaluation table, comparison against earlier rounds, sample segmentations.

Raw evaluation outputs live in `results/` (one file per round).
