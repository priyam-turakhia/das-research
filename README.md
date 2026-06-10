# Sorbian Tokenization Pipeline

Four subword tokenizers — SentencePiece BPE, SentencePiece Unigram, Morfessor 2.0, and a Morfessor + BPE hybrid (MorphBPE) — trained per language at the same vocabulary budget (16,000 tokens). Same data, same target size, same evaluation, so downstream comparisons can attribute differences to the tokenization algorithm itself.

Languages currently supported: **Upper Sorbian (`hsb`)** and **Lower Sorbian (`dsb`)**. Each language is a module under `data/<lang>/`, `models/<lang>/`, `results/<lang>/`; the tokenization library and CLI scripts are language-agnostic and dispatch on `--lang`.

Datasets currently in the repo:
- **hsb lww** — Leipzig + Wiki + Witaj, 1.29M sentences, 21M tokens (current main).
- **hsb v3** — Leipzig + WMT22, 704k sentences (historical Leipzig+WMT22 corpus).
- **dsb v1** — Witaj + MT (de↔dsb train+dev), 239k sentences, 3.77M tokens.

## Quick start (hsb, lww corpus)

```
uv sync
uv run python scripts/download_data.py --lang hsb --sources leipzig wiki witaj --output-suffix lww
uv run python scripts/train.py --method spm_bpe     --corpus data/processed/hsb/lww_train.txt --vocab-size 16000 --output models/hsb/spm_bpe_lww
uv run python scripts/train.py --method spm_unigram --corpus data/processed/hsb/lww_train.txt --vocab-size 16000 --output models/hsb/spm_unigram_lww
uv run python scripts/train.py --method morfessor   --corpus data/processed/hsb/lww_train.txt --vocab-size 16000 --output models/hsb/morfessor_lww
uv run python scripts/train.py --method morph_bpe   --corpus data/processed/hsb/lww_train.txt --vocab-size 16000 --output models/hsb/morph_bpe_lww
uv run python scripts/evaluate.py \
    --model-path models/hsb/spm_bpe_lww \
    --model-path models/hsb/spm_unigram_lww \
    --model-path models/hsb/morfessor_lww \
    --model-path models/hsb/morph_bpe_lww \
    --corpus data/processed/hsb/lww_dev.txt
```

End-to-end wall-clock time on lww is dominated by Morfessor (~20 min) and MorphBPE (~16 min) training. SPM BPE and Unigram each finish in under a minute.

For Lower Sorbian (`dsb`), substitute `--lang dsb --output-suffix v1` in step 1, `data/processed/dsb/v1_train.txt` and `models/dsb/<method>_v1` in the train/evaluate commands, and `data/processed/dsb/v1_dev.txt` for the eval corpus. See [docs/PROJECT.md §9](docs/PROJECT.md) for the explicit command sequence.

## MLM pretraining (XLM-R-base)

`scripts/pretrain.py` trains an XLM-RoBERTa-base-shaped encoder from scratch using any of the trained tokenizers above. Same script runs as a smoke test on a laptop (tiny model, MPS / CPU) and as the real training run on a CUDA GPU; only flags differ. Smoke takes ~10 s. GPU full run is per-tokenizer, scale depending on hardware.

```
# Local smoke (tiny model, ~10 s)
uv run python scripts/pretrain.py \
    --tokenizer-path models/dsb/spm_bpe_v1 \
    --corpus data/processed/dsb/v1_train.txt \
    --output /tmp/smoke --smoke --batch-size 2 --seq-len 128 --max-steps 20

# GPU run (full XLM-R base, bf16, with eval + checkpoints + perplexity/top-k/BPC)
uv run python scripts/pretrain.py \
    --tokenizer-path models/dsb/spm_bpe_v1 \
    --corpus data/processed/dsb/v1_train.txt \
    --eval-corpus data/processed/dsb/v1_dev.txt \
    --output models/dsb/xlmr_spm_bpe_v1 \
    --seq-len 256 --batch-size 64 --grad-accumulation-steps 4 \
    --num-train-epochs 10 --learning-rate 5e-4 \
    --warmup-ratio 0.06 --weight-decay 0.01 --bf16 \
    --eval-steps 500 --save-steps 2000 --save-total-limit 3 \
    --load-best-model-at-end --report-to tensorboard
```

See [docs/PROJECT.md §9](docs/PROJECT.md) for the script's full flag set and the MLM eval metrics (loss, perplexity, top-1/top-5 accuracy at masked positions, bits per character).

## Documentation

- [docs/PROJECT.md](docs/PROJECT.md) — full description of the project as it stands today: directory layout, data pipeline, library code file by file, scripts and flags, modularity, caveats. Self-contained.
- [docs/METRICS.md](docs/METRICS.md) — what each evaluation metric calculates, exactly how the code computes it, what the numbers mean in plain terms, and what a typical good score looks like.
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — history of changes to the pipeline (v1 → v2 → v3, then lww, MorphBPE, the internal refactor, and the dsb module) with the reasoning behind each change.
- [docs/EVALUATIONS.md](docs/EVALUATIONS.md) — current evaluation tables for every run, comparison against earlier rounds, sample segmentations.

Raw evaluation outputs live in `results/<lang>/` — one file per run.
