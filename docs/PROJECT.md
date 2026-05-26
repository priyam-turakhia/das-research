# Project description

This document describes the project as it stands today (version v3). It is self-contained: a new reader does not need to consult any earlier document to understand the current state. For round-by-round history of how things got here, see [CHANGELOG.md](CHANGELOG.md). For metric definitions and what typical scores look like, see [METRICS.md](METRICS.md). For evaluation results across rounds, see [EVALUATIONS.md](EVALUATIONS.md).

The writing rules: plain language, no jargon, every sentence purposeful.

---

## 1. What this project is

The goal is to train and compare three different subword tokenizers for Upper Sorbian — a West Slavic language spoken by roughly 30,000 people in eastern Germany.

A tokenizer is the component that takes raw text and breaks it into small pieces ("tokens") that a language model consumes. Different tokenizers break words apart differently, and the choice can affect downstream model quality. This project sets up three tokenizers on the same corpus, at the same target vocabulary size (16,000), so that any downstream comparison can attribute differences to the tokenization algorithm itself rather than other variables.

The four tokenizers:

1. **SentencePiece BPE** — a frequency-driven method that starts from single characters and repeatedly merges the most common adjacent pair until the vocabulary reaches the target size. "BPE" stands for Byte Pair Encoding.
2. **SentencePiece Unigram** — a probabilistic method that starts with a large pool of candidate pieces and prunes the least useful ones until the target size is reached. At inference, it picks the most likely segmentation under a trained unigram language model.
3. **Morfessor 2.0** — a linguistically motivated method that tries to discover real morphemes (prefixes, stems, suffixes) by minimizing a description-length objective. Not originally designed for fixed-vocab tokenization; we adapted it using Morfessor's built-in vocabulary-size controller.
4. **MorphBPE** — a hybrid baseline. Morfessor is trained with no vocabulary budget (its natural larger morpheme inventory), used as a fixed pre-segmenter, and BPE is trained on the resulting morpheme stream to compress to the target vocabulary size. Canonical literature setup for Morfessor + BPE hybrids.

All four share the same five reserved tokens — `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]` — at IDs 0 through 4.

---

## 2. Directory layout

```
das-research/
├── data/
│   ├── raw/                   Downloaded archives, untouched
│   └── processed/             Cleaned, filtered, Moses-pretokenized corpus
│       ├── hsb.txt            Full corpus
│       ├── hsb_train.txt      90% split
│       ├── hsb_dev.txt        5% split
│       └── hsb_test.txt       5% split
├── tokenization/              Library code
│   ├── __init__.py
│   ├── base.py                Abstract interface all tokenizers implement
│   ├── pretokenize.py         Moses pretokenize/detokenize helper
│   ├── spm_base.py            Shared SentencePiece code + HuggingFace wrapper
│   ├── spm_bpe.py             BPE variant
│   ├── spm_unigram.py         Unigram variant
│   ├── morfessor.py           Morfessor tokenizer + HuggingFace wrapper
│   ├── morph_bpe.py           Hybrid Morfessor + BPE tokenizer + HF wrapper
│   └── evaluate.py            Metric functions
├── scripts/                   Command-line entry points
│   ├── download_data.py       Downloads, filters, and preprocesses the corpus
│   ├── train.py               Trains one tokenizer
│   └── evaluate.py            Evaluates one or more trained tokenizers
├── models/                    Trained tokenizer artifacts (versioned)
├── results/                   Raw evaluation output text files
├── docs/                      Documentation (this folder)
├── pyproject.toml             Python project metadata and dependencies
└── README.md                  Entry point with reproduction commands
```

---

## 3. Data pipeline — `scripts/download_data.py`

### Sources

1. **Leipzig Corpora** — `hsb_mixed_2012_300K.tar.gz` from `downloads.wortschatz-leipzig.de`. A curated mixed-domain Upper Sorbian corpus with ~300,000 sentences. Distributed as a tar archive; the file named `*-sentences.txt` inside contains tab-separated data, where column 2 is the sentence text (column 1 is an ID).
2. **WMT22 monolingual** — `HSB_monolingual.txt.gz` from the Dimarco/WMT22 GitHub mirror. A plain-text corpus used in the 2022 Workshop on Machine Translation shared task for low-resource translation. Distributed as a gzip file with one sentence per line.

### Pipeline, in order

1. **Download** both archives to `data/raw/` if not already present. Skips re-download if the file is there.
2. **Extract** sentences. Leipzig: column 2 of the tab-separated sentences file. WMT22: plain gzipped text.
3. **Concatenate** both sources.
4. **Normalize to NFC**. Unicode has multiple equivalent ways to encode the same character — for example, `š` can be a single codepoint or `s` followed by a combining caron. NFC ("Normalization Form C") consistently picks the single-codepoint version. Important for Upper Sorbian's many diacritics (`š č ž ě ł ć ń ó`).
5. **Strip ASCII control characters** (codepoints 0–31 and 127 — tabs, form feeds, bells). Invisible junk that would otherwise become tokens.
6. **Drop boilerplate lines**. Lines containing `"filename"` or `"dateiname formatverbinden"` (a column header phrase from Leipzig's raw export) are removed. These are export artifacts, not real Sorbian text.
7. **Language filter (GlotLID)**. Loads the GlotLID model from HuggingFace and asks it to classify each line. Keeps the line only if the top prediction is `__label__hsb_Latn` with confidence at or above the threshold.
8. **Terminal-punctuation filter**. Keeps only lines whose final non-whitespace character is one of `.`, `!`, `?`, `…`, `»`, `"`, `'`. Drops sentence fragments.
9. **Length filter**. Drops lines with fewer than the minimum or more than the maximum number of whitespace-separated words. Counts the natural words *before* Moses, so the limits stay intuitive.
10. **Moses pretokenization**. Walks each line and inserts spaces around punctuation. Implementation: `sacremoses.MosesTokenizer(lang='cs')` (Czech is the closest registered Slavic language; the tag affects only the abbreviation list, which we don't depend on).
11. **Deduplicate** by exact string match. Order of first occurrence is preserved. Runs after Moses so two lines that differ only in punctuation spacing collapse correctly.
12. **Deterministic train/dev/test split** at 90% / 5% / 5%. Fixed random seed (42). Same seed always produces the same splits.
13. **Write four UTF-8 files**: `hsb.txt` (full corpus) plus the three splits, one sentence per line.

### Tunable thresholds — exposed two ways

Every threshold below has both a module-level constant near the top of `scripts/download_data.py` and a CLI flag. Edit the constant for a permanent default, or pass the flag for a one-off run.

| Threshold | Constant in code | CLI flag | Default | What it does |
|---|---|---|---|---|
| GlotLID confidence | `GLOTLID_THRESHOLD` | `--glotlid-threshold` | `0.5` | Minimum confidence for GlotLID's top prediction. Lower means looser filter, more lines kept. |
| Min sentence length | `MIN_LENGTH` | `--min-length` | `3` | Minimum number of whitespace-separated words. |
| Max sentence length | `MAX_LENGTH` | `--max-length` | `100` | Maximum number of whitespace-separated words. |
| Terminal punctuation | `TERMINAL_PUNCT` | `--terminal-punct` | `.!?…»"'` | Characters that count as legitimate sentence endings. |
| Skip language filter | (none) | `--skip-glotlid` | off | Disables GlotLID entirely. Useful when the model download is undesired or for fast reruns. |
| Source selection | `DEFAULT_SOURCES` | `--sources` | `leipzig wmt22` | Which sources to combine. Choose any subset of `leipzig`, `wmt22`, `wiki`, `witaj`. Wiki and Witaj expect plain-text files at `data/raw/wiki_hsb_monolingual.txt` and `data/raw/witaj_hsb_monolingual.txt`. |
| Output filename suffix | (none) | `--output-suffix` | `""` | Appended to processed-corpus filenames. Empty produces `hsb.txt`; passing `lww` produces `hsb_lww.txt`, `hsb_lww_train.txt`, etc. Lets you build parallel corpora without overwriting existing artifacts. |

### Logging

Every filter logs its input size, drop count, and output size at the INFO level. The summary at the end lists every drop count individually so you can see exactly where each line went.

---

## 4. Library code — `tokenization/`

### 4.1 `base.py` — the abstract interface

Defines `SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]` and an abstract class `BaseTokenizer` with these required methods:

- `train(corpus_path, vocab_size)` — learn from a corpus on disk.
- `tokenize(text) -> list[str]` — split text into human-readable pieces.
- `encode(text) -> list[int]` — same result, but as integer IDs.
- `decode(ids) -> str` — reverse of encode.
- `save(path)` / `load(path)` — serialize to / from a directory.
- `to_hf_tokenizer()` — return a HuggingFace-compatible wrapper.
- `vocab_size` property.
- `get_vocab() -> dict[str, int]`.

The point of this base class is that every tokenizer in the project honors the same contract, so callers don't have to branch on algorithm type.

### 4.2 `pretokenize.py` — Moses helper

A small shared module with one Moses tokenizer instance and one detokenizer instance. Two functions: `moses_pretokenize` and `moses_detokenize`. Both are imported by every tokenizer that needs them, so the behavior is guaranteed identical across SentencePiece BPE, SentencePiece Unigram, and Morfessor.

### 4.3 `spm_base.py` — SentencePiece shared logic

Contains two classes:

**`SentencePieceHFTokenizer`** (subclass of `PreTrainedTokenizer`).
HuggingFace-compatible wrapper that holds a `sentencepiece.SentencePieceProcessor` and delegates tokenize/encode/decode to it. The `_tokenize` method applies `moses_pretokenize` before the SPM call. The `convert_tokens_to_string` method applies `moses_detokenize` after the SPM call. Other methods forward directly to the SPM processor. This is a "slow" tokenizer in HuggingFace terminology — Python-based, not the Rust-backed fast variant.

**`BaseSPMTokenizer`** (subclass of `BaseTokenizer`).
Training and inference logic. `train()` calls `SentencePieceTrainer.train()` with:

- `vocab_size` passed through from the caller.
- `model_type` set by the subclass to `"bpe"` or `"unigram"`.
- `character_coverage=1.0` — every character in the corpus is in the vocabulary. Without this, SPM defaults to ~99.95% and drops rare characters. Important for Upper Sorbian's diacritics.
- `pad_id=0, unk_id=1, bos_id=-1, eos_id=-1` — pins special-token IDs to match our scheme; disables begin/end-of-sentence tokens.
- `user_defined_symbols=["[PAD]", "[CLS]", "[SEP]", "[MASK]"]` — forces these into the vocabulary intact.
- `num_threads=os.cpu_count()` — uses all available cores.

Training happens in a temporary directory; the resulting `.model` file is read into memory as raw bytes (`_model_bytes`) so it can be re-written on `save()`.

`tokenize`, `encode`, `decode` apply Moses around the SPM call (`tokenize` and `encode` pretokenize first; `decode` detokenizes the SPM output). `save()` writes `spm.model` plus `tokenizer_config.json`. `load()` reads them back. `to_hf_tokenizer()` constructs a `SentencePieceHFTokenizer` using the stored model bytes.

### 4.4 `spm_bpe.py` and `spm_unigram.py`

Each is a one-line subclass of `BaseSPMTokenizer` setting `model_type = "bpe"` or `"unigram"`. All actual work is in the base class.

### 4.5 `morfessor.py` — Morfessor tokenizer

Three pieces:

**`segment_word_with_vocab(model, vocab, word)` — shared helper function.**
Runs Morfessor's `viterbi_segment` on a word to get a list of morphemes. For each morpheme, checks whether it is in the vocabulary. If yes, keeps it as-is; if no, replaces it with the list of its characters. If `viterbi_segment` itself fails (word outside the Morfessor model), the whole word falls back to characters. Used by both the native tokenizer and the HuggingFace wrapper, so their output is guaranteed consistent.

**`MorfessorHFTokenizer`** (subclass of `PreTrainedTokenizer`).
HuggingFace wrapper. `_tokenize` applies Moses pretokenization, splits on whitespace, runs `segment_word_with_vocab` per word, returns the flat list of morphemes (no special boundary markers). `_convert_token_to_id` looks up the vocabulary (falls back to `[UNK]`'s ID). `convert_tokens_to_string` concatenates the morphemes and applies Moses detokenization.

**`MorfessorTokenizer`** (subclass of `BaseTokenizer`) — the main class.

Training does this, in order:

1. Read the corpus once. Collect (a) the set of every character seen, (b) a word-frequency counter.
2. Compute `target_morphs = vocab_size − len(SPECIAL_TOKENS) − len(char_inventory)`. Number of morphemes the model should produce so that specials + chars + morphemes fit in the budget.
3. Construct `BaselineModel` with two key settings:
   - `forcesplit_list=["-"]` — always splits on hyphens (Morfessor CLI default).
   - `corpusweight=NumMorphCorpusWeight(num_morph_types=target_morphs)` — Morfessor self-tunes its α parameter during training to converge on the target morpheme count. This makes Morfessor respect the budget natively, instead of post-hoc truncating.
4. Call `load_data(word_freq_list, count_modifier=lambda c: 1)`. The `count_modifier=lambda c: 1` is the canonical "ones" dampening — every unique word contributes weight 1 regardless of frequency. Prevents extremely common words from dominating the lexicon.
5. Call `train_batch()` with default `algorithm='recursive'`. Iterates until cost stops decreasing; typically 6–10 epochs. The corpus-weight updater adjusts α at the end of each epoch.
6. Re-segment every corpus word with `viterbi_segment` and count morpheme frequencies.
7. Build the final vocabulary: special tokens (IDs 0–4), characters (next N IDs), morphemes (remaining IDs up to the budget). The morpheme cap rarely evicts anything because `NumMorphCorpusWeight` already converged near the target — it acts as a safety net.

At inference:

- `tokenize(text)` applies Moses pretokenization, splits on whitespace, runs `segment_word_with_vocab` on each word, returns the flat morpheme list.
- `encode(text)` tokenizes, then maps each morpheme to its vocabulary ID. One-to-one.
- `decode(ids)` maps IDs back to morpheme strings, concatenates them, runs Moses detokenization.

`save()` writes `model.pkl` (pickled Morfessor model), `vocab.json`, and a `tokenizer_config.json`. `load()` reads them back.

**Documented limitation.** Morfessor's flat ID stream does not carry word-boundary information. This is a deliberate design choice (the `▁` marker was removed in v3 because it caused a vocabulary-coverage reporting artifact). Consequence: `decode` cannot perfectly reconstruct word boundaries. `tokenize` produces clean morpheme lists for inspection (so segmentation quality can be evaluated normally), and the encoded ID stream is correct and deterministic — but `decode(encode(text))` for Morfessor produces concatenated morphemes joined by Moses spacing rules, not the original text. SentencePiece is unaffected. See [CHANGELOG.md](CHANGELOG.md) for the full reasoning.

### 4.6 `morph_bpe.py` — hybrid Morfessor + BPE tokenizer

Contains `MorphBPETokenizer` (the main class) and `MorphBPEHFTokenizer` (HuggingFace wrapper). Both hold two underlying models: a `morfessor.BaselineModel` for pre-segmentation and a `sentencepiece.SentencePieceProcessor` for the final BPE encoding.

Training does this, in order:

1. Read the corpus, build a word-frequency counter.
2. Train Morfessor with no vocabulary budget: `BaselineModel(forcesplit_list=["-"])`, `load_data(..., count_modifier=lambda c: 1)`, `train_batch()`. Morfessor produces its natural larger morpheme inventory (typically 30k to 80k morphemes), no eviction, no character fallback at the segmentation layer.
3. Pre-segment the corpus into a temp file: for each line, split on whitespace, call `viterbi_segment` per word, write morphemes joined by single space.
4. Train SPM BPE on the temp file: `model_type="bpe"`, `vocab_size=16000`, `character_coverage=1.0`, `pad_id=0, unk_id=1, bos_id=-1, eos_id=-1`, `user_defined_symbols=["[PAD]", "[CLS]", "[SEP]", "[MASK]"]`.

At inference, `tokenize` and `encode` apply Moses pretokenization, then Morfessor segmentation per word, then SPM encoding. `decode` is `moses_detokenize(sp_model.decode(ids))` — lossy in the same way as standalone Morfessor (the flat ID stream does not preserve original word boundaries).

`save()` writes `morfessor_model.pkl` (pickled Morfessor), `spm.model` (raw SPM bytes), and `tokenizer_config.json`. `load()` reads them back.

The final vocabulary (returned by `get_vocab()` and used for ID conversions) is the SPM BPE vocabulary. Morfessor's morpheme inventory is intermediate, not part of the final vocab. BPE handles unknown sub-pieces via character-level subword decomposition (so OOV rate is effectively zero).

### 4.7 `evaluate.py` — metric functions

Defines `EvaluationResult` (a dataclass holding every metric value) and the metric functions: `compute_fertility`, `compute_unique_tokens`, `compute_oov_rate`, `round_trip_test`, `hf_compatibility_test`, `side_by_side_segmentation`, `evaluate_tokenizer`, `print_comparison_table`. Each metric is documented in detail in [METRICS.md](METRICS.md).

---

## 5. Command-line scripts — `scripts/`

### 5.1 `download_data.py`

Described in §3. Run with no arguments to use defaults; flags listed in the §3 table.

```
uv run python scripts/download_data.py
```

### 5.2 `train.py`

Trains one tokenizer on a corpus.

```
uv run python scripts/train.py \
    --method {spm_bpe|spm_unigram|morfessor|morph_bpe} \
    --corpus data/processed/hsb_train.txt \
    --vocab-size 16000 \
    --output models/my_tokenizer
```

| Flag | Default | What it does |
|---|---|---|
| `--method` | (required) | Which tokenizer to train. One of `spm_bpe`, `spm_unigram`, `morfessor`, `morph_bpe`. |
| `--corpus` | (required) | Path to the training corpus, one sentence per line. Should normally be `data/processed/hsb_train.txt`. |
| `--vocab-size` | `16000` | Target vocabulary size, including the five special tokens. |
| `--output` | (required) | Directory to save the trained tokenizer into. Created if it doesn't exist. |

If you point `--corpus` at the full `hsb.txt` while `hsb_train.txt` and `hsb_dev.txt` exist, the script logs a warning telling you to train on the train split. It still runs if you insist.

Approximate training times on the lww corpus:
- SPM BPE: ~50 seconds.
- SPM Unigram: ~50 seconds.
- Morfessor: ~20 minutes (single-threaded Python, 6–10 epochs).
- MorphBPE: ~16 minutes (Morfessor unconstrained training + pre-segmentation + BPE).

### 5.3 `evaluate.py`

Runs evaluation on one or more trained tokenizers. Repeat `--model-path` for each model.

```
uv run python scripts/evaluate.py \
    --model-path models/hsb_spm_bpe_v3 \
    --model-path models/hsb_spm_unigram_v3 \
    --model-path models/hsb_morfessor_v3 \
    --corpus data/processed/hsb_dev.txt
```

| Flag | Default | What it does |
|---|---|---|
| `--model-path` | (required, repeatable) | Path to a trained tokenizer directory. Repeat once per model to compare multiple at once. |
| `--corpus` | (required) | Path to the evaluation corpus. Should normally be `data/processed/hsb_dev.txt`. |
| `--type` | (auto-detected) | Override tokenizer type detection. Rarely needed. |

Auto-detects tokenizer type from `tokenizer_config.json`. Same held-out-corpus warning as `train.py` if pointed at the full corpus while a held-out split exists. Prints the comparison table and side-by-side segmentations of 10 sample words.

---

## 6. Modularity

The design intent: callers should only ever see `BaseTokenizer`. Algorithm-specific details stay inside the subclasses.

- `scripts/train.py` looks up the class via `TOKENIZER_CLASSES[args.method]`. It does not know or care which algorithm.
- `scripts/evaluate.py` auto-detects the class from the saved config. Every call site uses the base-class interface.
- `tokenization/evaluate.py` takes `BaseTokenizer` instances and measures them identically.
- The two HuggingFace wrappers (`SentencePieceHFTokenizer`, `MorfessorHFTokenizer`) both subclass `PreTrainedTokenizer`, so downstream model code that expects a HuggingFace tokenizer accepts any of them.

Shared code is lifted where reasonable:

- BPE and Unigram share `BaseSPMTokenizer`; each subclass is one line.
- Morfessor's character-fallback logic lives in a single module-level function used by both the native class and the HuggingFace wrapper, so they never diverge.
- Moses pretokenization lives in one shared module imported everywhere.
- All evaluation metrics apply uniformly to all three tokenizers — no per-algorithm special cases.

What this buys: adding a fourth tokenizer (e.g., a Morfessor → BPE hybrid) means writing one new subclass with its own `train()` method. The scripts, evaluation module, and HuggingFace integration all work without changes.

---

## 7. Caveats

1. **Proxy metrics do not equal downstream quality.** All numbers in this project are sanity checks. Which tokenizer makes a better language model is a separate question that requires a training run.
2. **Corpus size.** ~700,000 sentences after filtering is small by modern NLP standards but substantial for Upper Sorbian specifically. Results are specific to this corpus.
3. **Morfessor decode is lossy.** See §4.5 and [CHANGELOG.md](CHANGELOG.md) for the design reasoning. Inspection via `tokenize` is unaffected; downstream training (which consumes IDs) is unaffected; only human-readable reconstruction of full sentences from a flat ID stream is affected.
4. **Moses uses the Czech language tag** because Upper Sorbian is not a registered `sacremoses` language. The tag affects only the abbreviation list, which we don't depend on.
5. **`numpy<2` is pinned** because of an upstream incompatibility in `fasttext`. One-line workaround. Removable once `fasttext` is fixed upstream.
6. **GlotLID adds a one-time download (~1 GB)** when the preprocessing script first runs. Subsequent runs use the HuggingFace cache.
7. **Morfessor training is slow** (~13 min). Budget accordingly when iterating.

---

## 8. Reproduction

```
uv sync

# 1. Build the cleaned, filtered, Moses-pretokenized corpus
uv run python scripts/download_data.py

# 2. Train each tokenizer on the training split
uv run python scripts/train.py --method spm_bpe \
    --corpus data/processed/hsb_train.txt \
    --vocab-size 16000 --output models/hsb_spm_bpe_v3

uv run python scripts/train.py --method spm_unigram \
    --corpus data/processed/hsb_train.txt \
    --vocab-size 16000 --output models/hsb_spm_unigram_v3

uv run python scripts/train.py --method morfessor \
    --corpus data/processed/hsb_train.txt \
    --vocab-size 16000 --output models/hsb_morfessor_v3

# 3. Evaluate all three on the dev split
uv run python scripts/evaluate.py \
    --model-path models/hsb_spm_bpe_v3 \
    --model-path models/hsb_spm_unigram_v3 \
    --model-path models/hsb_morfessor_v3 \
    --corpus data/processed/hsb_dev.txt
```

End-to-end wall-clock time is dominated by Morfessor training (~13 min). Everything else finishes in under a minute combined.
