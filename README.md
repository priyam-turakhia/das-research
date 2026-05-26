# Upper Sorbian Tokenization Pipeline

Four subword tokenizers — SentencePiece BPE, SentencePiece Unigram, Morfessor 2.0, and a Morfessor + BPE hybrid (MorphBPE) — trained on the same Upper Sorbian corpus at the same vocabulary budget (16,000 tokens). The setup is an ablation: same data, same target size, same evaluation, so any downstream performance difference is attributable to the tokenization algorithm itself.

Current corpus: **lww** (Leipzig + Wiki + Witaj, 1.29M sentences, 21M tokens). An earlier v3 corpus (Leipzig + WMT22) is retained as historical record.

## Quick start

```
uv sync
uv run python scripts/download_data.py --sources leipzig wiki witaj --output-suffix lww
uv run python scripts/train.py --method spm_bpe     --corpus data/processed/hsb_lww_train.txt --vocab-size 16000 --output models/hsb_spm_bpe_lww
uv run python scripts/train.py --method spm_unigram --corpus data/processed/hsb_lww_train.txt --vocab-size 16000 --output models/hsb_spm_unigram_lww
uv run python scripts/train.py --method morfessor   --corpus data/processed/hsb_lww_train.txt --vocab-size 16000 --output models/hsb_morfessor_lww
uv run python scripts/train.py --method morph_bpe   --corpus data/processed/hsb_lww_train.txt --vocab-size 16000 --output models/hsb_morph_bpe_lww
uv run python scripts/evaluate.py \
    --model-path models/hsb_spm_bpe_lww \
    --model-path models/hsb_spm_unigram_lww \
    --model-path models/hsb_morfessor_lww \
    --model-path models/hsb_morph_bpe_lww \
    --corpus data/processed/hsb_lww_dev.txt
```

End-to-end wall-clock time is dominated by Morfessor training and MorphBPE training (~20 min and ~16 min respectively on the lww corpus). SPM BPE and Unigram each finish in under a minute.

## Documentation

- [docs/PROJECT.md](docs/PROJECT.md) — full description of the project as it stands today: directory layout, data pipeline, library code file by file, scripts and flags, modularity, caveats. Self-contained.
- [docs/METRICS.md](docs/METRICS.md) — what each evaluation metric calculates, exactly how the code computes it, what the numbers mean in plain terms, and what a typical good score looks like.
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — round-by-round history of changes to the pipeline (v1 → v2 → v3) with the reasoning behind each change.
- [docs/EVALUATIONS.md](docs/EVALUATIONS.md) — current evaluation table, comparison against earlier rounds, sample segmentations.

Raw evaluation outputs live in `results/` (one file per round).
