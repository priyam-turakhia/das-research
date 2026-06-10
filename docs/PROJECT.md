# Project description

This document describes the project as it stands today. It is self-contained: a new reader does not need to consult any earlier document to understand the current state. For round-by-round history of how things got here, see [CHANGELOG.md](CHANGELOG.md). For metric definitions and what typical scores look like, see [METRICS.md](METRICS.md). For evaluation results across rounds, see [EVALUATIONS.md](EVALUATIONS.md).

The writing rules: plain language, no jargon, every sentence purposeful.

---

## 1. What this project is

The goal is to train and compare four subword tokenizers for Sorbian — a pair of related West Slavic languages spoken in eastern Germany. The project covers both **Upper Sorbian (`hsb`)**, the larger of the two with roughly 30,000 speakers, and **Lower Sorbian (`dsb`)**, with around 7,000 speakers. Each language is a separate module with its own raw data, processed corpora, trained tokenizers, and evaluation results, but they share the same pipeline code and the same tokenization algorithms.

A tokenizer is the component that takes raw text and breaks it into small pieces ("tokens") that a language model consumes. Different tokenizers break words apart differently, and the choice can affect downstream model quality. This project sets up four tokenizers per language on the same corpus, at the same target vocabulary size (16,000), so that any downstream comparison can attribute differences to the tokenization algorithm itself rather than other variables.

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
│   ├── raw/                       Downloaded archives, untouched
│   │   ├── hsb/                   Upper Sorbian source files
│   │   └── dsb/                   Lower Sorbian source files
│   └── processed/                 Cleaned, filtered, Moses-pretokenized corpora
│       ├── hsb/
│       │   ├── v3.txt             Full corpus (default dataset tag "v3")
│       │   ├── v3_train.txt       90% split
│       │   ├── v3_dev.txt         5% split
│       │   ├── v3_test.txt        5% split
│       │   └── lww{,_train,_dev,_test}.txt   Leipzig+Wiki+Witaj alternative
│       └── dsb/
│           ├── v1{,_train,_dev,_test}.txt    Witaj+MT corpus
│           └── metadix_morph_annotations_{1000,full}.tsv  Apertium-derived word-level segmentations (semi-supervised Morfessor input)
├── apertium-dsb.dsb.metadix       Apertium Lower Sorbian morphological dictionary (raw source for the annotations TSV)
├── tokenization/                  Library code (language-agnostic)
│   ├── __init__.py
│   ├── base.py                    Abstract interface all tokenizers implement
│   ├── pretokenize.py             Moses pretokenize/detokenize helper
│   ├── hf_base.py                 Shared HuggingFace wrapper bases
│   ├── registry.py                Tokenizer dispatch + tokenizer_config.json schema
│   ├── spm_base.py                Shared SentencePiece code + HF wrapper
│   ├── spm_bpe.py                 BPE variant
│   ├── spm_unigram.py             Unigram variant
│   ├── morfessor.py               Morfessor tokenizer + HuggingFace wrapper
│   ├── morfessor_semi.py          Semi-supervised Morfessor variant (dsb default)
│   ├── morph_bpe.py               Hybrid Morfessor + BPE tokenizer + HF wrapper
│   └── evaluate.py                Metric functions
├── scripts/                       Command-line entry points (language-agnostic)
│   ├── download_data.py           Downloads, filters, preprocesses one language
│   ├── train.py                   Trains one tokenizer
│   ├── evaluate.py                Evaluates one or more trained tokenizers
│   ├── pretrain.py                XLM-R-base MLM pretraining using any trained tokenizer
│   └── extract_dsb_morph_annotations.py   Apertium metadix → semi-supervised annotation TSV
├── models/
│   ├── hsb/                       Upper Sorbian trained tokenizers
│   └── dsb/                       Lower Sorbian trained tokenizers
├── results/
│   ├── hsb/                       Upper Sorbian evaluation outputs
│   └── dsb/                       Lower Sorbian evaluation outputs
├── docs/                          Documentation (this folder)
├── pyproject.toml                 Python project metadata and dependencies
└── README.md                      Entry point with reproduction commands
```

### Naming conventions for artifacts

Everything language-specific lives under a language-coded directory (`hsb/` or `dsb/`). Inside that directory, names carry only the dataset tag and (for models) the algorithm — the parent dir conveys the language.

**Dataset tags currently in use:**

| Lang | Tag | Sources | Pipeline |
|---|---|---|---|
| hsb | `v2` | Leipzig + WMT22 | partial v2 cleaning (historical) |
| hsb | `v3` | Leipzig + WMT22 | full v3 pipeline (current default) |
| hsb | `lww` | Leipzig + Wiki + Witaj | full v3 pipeline (alternative) |
| dsb | `v1` | Witaj + MT (de↔dsb train + dev) | full v3 pipeline |

**Artifact name patterns:**

| Artifact | Pattern | Example |
|---|---|---|
| Processed corpus file | `data/processed/<lang>/<dataset>.txt` | `data/processed/hsb/v3.txt`, `data/processed/dsb/v1.txt` |
| Split file | `data/processed/<lang>/<dataset>_<split>.txt` | `data/processed/hsb/v3_train.txt`, `data/processed/dsb/v1_dev.txt` |
| Trained tokenizer directory | `models/<lang>/<method>_<dataset>` | `models/hsb/spm_bpe_v3`, `models/dsb/morph_bpe_v1` |
| Evaluation output file | `results/<lang>/eval_<split>_<dataset>[_<tag>].txt` | `results/hsb/eval_dev_v3.txt`, `results/hsb/eval_dev_lww_4way.txt` |

Methods are: `spm_bpe`, `spm_unigram`, `morfessor`, `morfessor_semi`, `morph_bpe`. Splits are: `train`, `dev`, `test`. The optional `<tag>` on evaluation files captures one-off run characteristics (e.g. `4way` for the 4-tokenizer comparison including MorphBPE, or `morfessor_semi` for the dsb semi-supervised Morfessor attempt).

Adding a new language means: a new `LANG_REGISTRY` entry in [scripts/download_data.py](../scripts/download_data.py) (sources, GlotLID label, defaults) and the `data/raw/<lang>/`, `data/processed/<lang>/`, `models/<lang>/`, `results/<lang>/` subdirectories. The pipeline functions, the tokenization library, and `train.py` / `evaluate.py` need no changes.

---

## 3. Data pipeline — `scripts/download_data.py`

The script is language-agnostic. Per-language configuration lives in `LANG_REGISTRY` at the top of the file: which sources exist, where each one comes from (URL or manual placement), how to extract sentences from it, the GlotLID label to keep, and the default source combination. Selecting the language with `--lang` is the only switch the user needs.

### Sources by language

**Upper Sorbian (`--lang hsb`)**:

1. **Leipzig Corpora** — `hsb_mixed_2012_300K.tar.gz` from `downloads.wortschatz-leipzig.de`. A curated mixed-domain corpus with ~300,000 sentences. Distributed as a tar archive; the file named `*-sentences.txt` inside contains tab-separated data, where column 2 is the sentence text.
2. **WMT22 monolingual** — `HSB_monolingual.txt.gz` from the Dimarco/WMT22 GitHub mirror. A plain-text corpus used in the 2022 Workshop on Machine Translation shared task. Distributed as a gzip file with one sentence per line.
3. **Wiki** — `wiki_hsb_monolingual.txt`, Upper Sorbian Wikipedia dump (~48k sentences). Placed manually under `data/raw/hsb/`.
4. **Witaj** — `witaj_hsb_monolingual.txt`, Witaj educational publisher monolingual data (~1.07M sentences). Placed manually.

Default source combination: `leipzig + wmt22` (produces the `v3` dataset). Alternative combination: `leipzig + wiki + witaj` (produces `lww`).

**Lower Sorbian (`--lang dsb`)**:

1. **Witaj** — `witaj_dsb_monolingual.txt` (~120k sentences). Placed manually under `data/raw/dsb/`.
2. **MT train** — `train.de-dsb.dsb` from the TUM-NLP `llms-limited-resources2025` repo, the dsb side of a de↔dsb MT training set (~172k sentences).
3. **MT dev** — `dev.de-dsb.dsb` from the same repo (~4k sentences). Pooled with the other sources and re-split, not treated as a separate held-out set.

Default source combination: `witaj + mt_train + mt_dev` (produces the `v1` dataset).

### Pipeline, in order

The pipeline below applies identically to every language; only the GlotLID label (looked up from the registry) changes.

1. **Download** any source that has a URL to `data/raw/<lang>/` if not already present. Skips re-download if the file is there. Manually-placed sources (Wiki, Witaj, etc.) must already exist or the script errors out.
2. **Extract** sentences using the per-source extractor. Leipzig: column 2 of the tab-separated sentences file. WMT22: plain gzipped text. Plain: one sentence per line.
3. **Concatenate** all selected sources.
4. **Normalize to NFC**. Unicode has multiple equivalent ways to encode the same character — for example, `š` can be a single codepoint or `s` followed by a combining caron. NFC ("Normalization Form C") consistently picks the single-codepoint version. Important for Sorbian's many diacritics (`š č ž ě ł ć ń ó` and dsb-specific spellings).
5. **Strip ASCII control characters** (codepoints 0–31 and 127 — tabs, form feeds, bells). Invisible junk that would otherwise become tokens.
6. **Drop boilerplate lines**. Lines containing `"filename"` or `"dateiname formatverbinden"` (a column header phrase from Leipzig's raw export) are removed. These are export artifacts, not real Sorbian text.
7. **Language filter (GlotLID)**. Loads the GlotLID model from HuggingFace and asks it to classify each line. Keeps the line only if the top prediction is `__label__<lang>_Latn` with confidence at or above the threshold.
8. **Terminal-punctuation filter**. Keeps only lines whose final non-whitespace character is one of `.`, `!`, `?`, `…`, `»`, `"`, `'`. Drops sentence fragments.
9. **Length filter**. Drops lines with fewer than the minimum or more than the maximum number of whitespace-separated words. Counts the natural words *before* Moses, so the limits stay intuitive.
10. **Moses pretokenization**. Walks each line and inserts spaces around punctuation. Implementation: `sacremoses.MosesTokenizer(lang='cs')` (Czech is the closest registered Slavic language; the tag affects only the abbreviation list, which we don't depend on). Applies to both Sorbian variants.
11. **Deduplicate** by exact string match. Order of first occurrence is preserved. Runs after Moses so two lines that differ only in punctuation spacing collapse correctly.
12. **Deterministic train/dev/test split** at 90% / 5% / 5%. Fixed random seed (42). Same seed always produces the same splits.
13. **Write four UTF-8 files** tagged with the dataset suffix (default `v3` for hsb): `<dataset>.txt` (full corpus) plus the three splits `<dataset>_train.txt` / `<dataset>_dev.txt` / `<dataset>_test.txt` under `data/processed/<lang>/`, one sentence per line.

### Tunable thresholds — exposed two ways

Every threshold below has both a module-level constant near the top of `scripts/download_data.py` and a CLI flag. Edit the constant for a permanent default, or pass the flag for a one-off run.

| Threshold | Constant in code | CLI flag | Default | What it does |
|---|---|---|---|---|
| Language | (none) | `--lang` | `hsb` | Picks the `LANG_REGISTRY` entry: which sources exist, the GlotLID label, and the output directory. Currently supports `hsb` and `dsb`. |
| GlotLID confidence | `GLOTLID_THRESHOLD` | `--glotlid-threshold` | `0.5` | Minimum confidence for GlotLID's top prediction. Lower means looser filter, more lines kept. |
| Min sentence length | `MIN_LENGTH` | `--min-length` | `3` | Minimum number of whitespace-separated words. |
| Max sentence length | `MAX_LENGTH` | `--max-length` | `100` | Maximum number of whitespace-separated words. |
| Terminal punctuation | `TERMINAL_PUNCT` | `--terminal-punct` | `.!?…»"'` | Characters that count as legitimate sentence endings. |
| Skip language filter | (none) | `--skip-glotlid` | off | Disables GlotLID entirely. Useful when the model download is undesired or for fast reruns. |
| Source selection | per-language `default_sources` in `LANG_REGISTRY` | `--sources` | hsb: `leipzig wmt22`, dsb: `witaj mt_train mt_dev` | Which sources to combine. Valid names depend on `--lang`; the registry entry lists what's available. |
| Dataset tag | (none) | `--output-suffix` | `"v3"` | Tag appended to processed-corpus filenames as the dataset identifier. Default `v3` produces `v3.txt`, `v3_train.txt`, etc. under `data/processed/<lang>/`. Passing `lww` (hsb) or `v1` (dsb) builds parallel corpora without overwriting existing artifacts. |

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

A small shared module with one Moses tokenizer instance and one detokenizer instance. Two functions: `moses_pretokenize` and `moses_detokenize`. Both are imported by every tokenizer that needs them, so the behavior is guaranteed identical across SentencePiece BPE, SentencePiece Unigram, Morfessor, and MorphBPE.

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

#### Semi-supervised variant — `morfessor_semi.py`

`SemiSupervisedMorfessorTokenizer` is a thin subclass of `MorfessorTokenizer` (`tokenizer_type = "morfessor_semi"`). It overrides `train()` to pass two extra arguments to the parent:

1. `annotations_path` — TSV of word-level segmentations with rows `surface\tmorph1 morph2 ...`. The parent's `train()` loads it and calls `model.set_annotations(annotations)` after `load_data()` and before `train_batch()`.
2. `use_num_morph_weight=False` — the parent skips `NumMorphCorpusWeight` and constructs `BaselineModel(forcesplit_list=["-"])` with Morfessor's default fixed `corpusweight=1.0`, so the annotation-weight tuner from `set_annotations` adapts alone during training instead of competing with the morph-budget tuner. The 16k vocabulary budget is enforced by the parent's post-hoc top-N morpheme cap.

Default annotations path: `data/processed/dsb/metadix_morph_annotations_500.tsv` — a paradigm-balanced 500-row sample extracted from the Apertium Lower Sorbian metadix (`apertium-dsb.dsb.metadix` at the repo root) by `scripts/extract_dsb_morph_annotations.py`. The choice of 500 is from a 13-configuration tuning sweep documented in [CHANGELOG.md](CHANGELOG.md); larger sample sizes produced no meaningful gain on this corpus. The extraction script also writes a 31,601-row `metadix_morph_annotations_full.tsv` for reference and runs a Morfessor smoke test on its output before exiting so format breakage is caught early. The class is dsb-specific by default; using it for hsb would require overriding the path at construction.

Result on dsb v1: vocab coverage +9.6 pp, OOV −0.16 pp, fertility mean slightly shorter, fertility std widened (more bimodal length distribution). Full numbers in [EVALUATIONS.md §8](EVALUATIONS.md); the history of the first failed attempt, the weight-tuner fix, and the sample-size sweep is in [CHANGELOG.md](CHANGELOG.md). The saved tokenizer at `models/dsb/morfessor_semi_v1/` loads, tokenizes, and round-trips correctly.

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

Described in §3. Pick the language with `--lang` and use defaults for the rest, or override any flag. The script writes to `data/processed/<lang>/`.

```
uv run python scripts/download_data.py --lang hsb
uv run python scripts/download_data.py --lang dsb --output-suffix v1
```

### 5.2 `train.py`

Trains one tokenizer on a corpus.

```
uv run python scripts/train.py \
    --method {spm_bpe|spm_unigram|morfessor|morph_bpe} \
    --corpus data/processed/hsb/v3_train.txt \
    --vocab-size 16000 \
    --output models/hsb/my_tokenizer
```

| Flag | Default | What it does |
|---|---|---|
| `--method` | (required) | Which tokenizer to train. One of `spm_bpe`, `spm_unigram`, `morfessor`, `morph_bpe`. |
| `--corpus` | (required) | Path to the training corpus, one sentence per line. Should normally be `data/processed/<lang>/<dataset>_train.txt`. |
| `--vocab-size` | `16000` | Target vocabulary size, including the five special tokens. |
| `--output` | (required) | Directory to save the trained tokenizer into. Created if it doesn't exist. Convention: `models/<lang>/<method>_<dataset>`. |

If you point `--corpus` at an unsplit full corpus file (e.g. `v3.txt`) while the corresponding `_train.txt` and `_dev.txt` splits exist, the script logs a warning telling you to train on the train split. It still runs if you insist.

Approximate training times:
- SPM BPE: well under a minute on either hsb or dsb corpora.
- SPM Unigram: well under a minute.
- Morfessor: ~20 minutes on hsb lww (1.16M sentences), shorter on smaller corpora. Single-threaded Python.
- MorphBPE: ~16 minutes on hsb lww, similar scaling.

### 5.3 `evaluate.py`

Runs evaluation on one or more trained tokenizers. Repeat `--model-path` for each model.

```
uv run python scripts/evaluate.py \
    --model-path models/hsb/spm_bpe_v3 \
    --model-path models/hsb/spm_unigram_v3 \
    --model-path models/hsb/morfessor_v3 \
    --corpus data/processed/hsb/v3_dev.txt
```

| Flag | Default | What it does |
|---|---|---|
| `--model-path` | (required, repeatable) | Path to a trained tokenizer directory. Repeat once per model to compare multiple at once. |
| `--corpus` | (required) | Path to the evaluation corpus. Should normally be `data/processed/<lang>/<dataset>_dev.txt`. |
| `--type` | (auto-detected) | Override tokenizer type detection. Rarely needed. |

Auto-detects tokenizer type from `tokenizer_config.json`. Same held-out-corpus warning as `train.py` if pointed at the full corpus while a held-out split exists. Prints the comparison table and side-by-side segmentations of 10 sample words.

### 5.4 `pretrain.py`

Trains an XLM-RoBERTa-base-shaped encoder from scratch using one of the trained tokenizers. The encoder is intended to become the **student** in a later cross-lingual embedding distillation step (teacher: stock `FacebookAI/xlm-roberta-base`); MLM here is the priming step. Full description of model shape, eval metrics, and flag set is in §7.

```
uv run python scripts/pretrain.py \
    --tokenizer-path models/dsb/spm_bpe_v1 \
    --corpus data/processed/dsb/v1_train.txt \
    --eval-corpus data/processed/dsb/v1_dev.txt \
    --output models/dsb/xlmr_spm_bpe_v1 \
    --bf16
```

A `--smoke` flag swaps in a tiny 2-layer / 128-hidden model so the entire pipeline can be validated on a laptop in seconds before committing to a GPU run.

---

## 6. Modularity

The design intent: callers should only ever see `BaseTokenizer`. Algorithm-specific details stay inside the subclasses.

- `scripts/train.py` looks up the class via `get_tokenizer_classes()` from `tokenization/registry.py`. It does not know or care which algorithm.
- `scripts/evaluate.py` calls `load_tokenizer(path)` from the same registry; the registry auto-detects the class from the saved `tokenizer_config.json`. Every call site uses the base-class interface.
- `tokenization/evaluate.py` takes `BaseTokenizer` instances and measures them identically.
- The three HuggingFace wrappers (`SentencePieceHFTokenizer`, `MorfessorHFTokenizer`, `MorphBPEHFTokenizer`) all subclass `PreTrainedTokenizer` (via the small bases in `tokenization/hf_base.py`), so downstream model code that expects a HuggingFace tokenizer accepts any of them.

Shared code is lifted where reasonable:

- BPE and Unigram share `BaseSPMTokenizer`; each subclass is one line.
- `train_spm_model()` in `spm_base.py` carries the shared SentencePiece training call used by both `BaseSPMTokenizer` and `MorphBPETokenizer`.
- `BaseHFTokenizer` and `SpmBackedHFTokenizer` in `hf_base.py` carry the special-token boilerplate and the SentencePiece-backed methods reused by the SPM and MorphBPE HF wrappers.
- Morfessor's character-fallback logic lives in a single module-level function used by both the native class and the HuggingFace wrapper, so they never diverge.
- Moses pretokenization lives in one shared module imported everywhere.
- All evaluation metrics apply uniformly to every tokenizer — no per-algorithm special cases.

What this buys: adding a fifth tokenizer means writing one new subclass (with its own `train()` method) plus declaring a `tokenizer_type` class attribute and adding it to `get_tokenizer_classes()`. The scripts, evaluation module, and HuggingFace integration all work without changes. Adding a new language means adding a `LANG_REGISTRY` entry in `download_data.py` and creating the matching `<lang>/` subdirectories — the tokenization library and the CLI scripts need no changes.

---

## 7. MLM pretraining — `scripts/pretrain.py`

A downstream step that uses the trained tokenizers, not part of the tokenizer evaluation. Trains an XLM-R-base-shaped encoder from random init with masked-language-modeling on a dsb (or hsb) corpus. The resulting encoder is intended to be the **student** in cross-lingual embedding distillation against stock `FacebookAI/xlm-roberta-base` as the teacher; MLM here is a sensible init, not the final training step.

### Model shape

Built from `XLMRobertaConfig` with `vocab_size` taken from the loaded tokenizer. Two presets:

- **Smoke (`--smoke`)**: 2 layers, hidden 128, 4 attention heads, FFN 256. ~2.4 M params. For pipeline validation only.
- **Production (default)**: 12 layers, hidden 768, 12 heads, FFN 3072. With a 16k vocabulary this is ~110 M params total (vs. ~270 M for stock XLM-R-base, where the 250k SPM vocab dominates the parameter count). `max_position_embeddings` defaults to 514, matching stock XLM-R-base, so the model architecturally supports inputs up to 512 tokens regardless of the training `seq_len`.

### Tokenizer wiring

The trained tokenizer is loaded via `tokenization/registry.py:load_tokenizer()` and converted to the HuggingFace interface via `to_hf_tokenizer()`. The `BaseHFTokenizer` parent (in `tokenization/hf_base.py`) wraps every input with `[CLS] … [SEP]` and exposes a correct `get_special_tokens_mask`, which the MLM data collator uses to exclude special tokens from masking. All five tokenizer types are supported uniformly.

### Training loop

HuggingFace `Trainer` with `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)`. The collator marks 15% of non-special tokens per sequence as prediction targets and replaces 80% of those with `[MASK]`, 10% with a random token, 10% unchanged. Standard BERT/RoBERTa recipe. Loss is cross-entropy over the marked positions.

Optimizer is AdamW with linear warmup and linear decay; defaults are `--warmup-ratio 0.06`, `--weight-decay 0.01`, `--learning-rate 5e-4`. Effective batch size is `batch_size × grad_accumulation_steps`.

### Eval metrics

When `--eval-corpus` is set and `--smoke` is off, eval runs every `--eval-steps` and reports four metrics in addition to `eval_loss`:

| Metric | What it is |
|---|---|
| `perplexity` | `exp(eval_loss)`. Standard LM metric. **Depends on tokenization granularity** — not directly comparable across the 5 tokenizers. |
| `top1` | Fraction of masked positions where the model's top-1 prediction equals the true token. |
| `top5` | Same, but for top-5. |
| `bpc` | Bits per character. Total NLL on masked positions, scaled up by `1 / mlm_probability` to extrapolate to full-text NLL, divided by the source-text character count, converted nats → bits. **Tokenization-invariant** — use this for cross-tokenizer comparison. |

Implementation detail: `preprocess_logits_for_metrics` reduces each batch from full `(B, L, V)` logits to top-5 indices + per-token NLL *before* Trainer accumulates them across the eval set. Without this the accumulator would OOM on any device for a 16k-vocab × seq-256 × 12k-sentence dev split.

### Flag reference

| Flag | Default | What it does |
|---|---|---|
| `--tokenizer-path` | (required) | A trained tokenizer directory from `models/<lang>/<method>_<dataset>`. |
| `--corpus` | (required) | Training text, one sentence per line. |
| `--eval-corpus` | none | If set and `--smoke` off, runs eval every `--eval-steps`. |
| `--output` | (required) | Where to save checkpoints and the final model. |
| `--smoke` | off | Tiny 2-layer model, 20 steps, eval disabled — laptop sanity check. |
| `--device` | `auto` | `auto` picks CUDA → MPS → CPU; or set explicitly. |
| `--seq-len` | `256` | Training sequence length (longer = O(L²) cost). |
| `--max-position-embeddings` | `514` | Architectural ceiling on input length. Decoupled from `seq_len` on purpose. |
| `--batch-size` | `64` | Per-device batch size. |
| `--grad-accumulation-steps` | `4` | Effective batch is `batch_size × grad_accumulation_steps`. |
| `--num-train-epochs` | `10` | Passes over the train corpus (overridden if `--max-steps` > 0). |
| `--max-steps` | `-1` | Hard step cap; `-1` means use epochs. Smoke defaults to 20. |
| `--learning-rate` | `5e-4` | Peak LR after warmup. |
| `--warmup-ratio` | `0.06` | Fraction of total steps spent ramping LR from 0 to peak. |
| `--weight-decay` | `0.01` | AdamW weight decay. |
| `--mlm-probability` | `0.15` | Fraction of tokens marked as prediction targets per batch. |
| `--bf16`, `--fp16` | off | Mixed-precision. `--bf16` is the GPU default; both flaky on MPS. |
| `--save-steps` | `2000` | Checkpoint every N steps. |
| `--save-total-limit` | `3` | Keep only the last K checkpoints. |
| `--resume-from-checkpoint` | none | Resume optimizer/scheduler/RNG/data cursor from a checkpoint dir. |
| `--load-best-model-at-end` | off | Final saved model is the lowest-`eval_loss` checkpoint, not the last. |
| `--logging-steps` | `50` | Console log frequency during training. |
| `--eval-steps` | `500` | Eval frequency during training. |
| `--report-to` | `none` | Comma-separated: `tensorboard,wandb,none`. |
| `--seed` | `42` | Torch RNG seed. |

### Dependencies

`torch`, `datasets`, and `accelerate` are pulled in for this script (declared in `pyproject.toml`; installed by `uv sync`). The tokenizer training pipeline does not need them.

### Caveat

The saved tokenizer in each checkpoint directory cannot be reloaded via HuggingFace's `AutoTokenizer.from_pretrained` because our custom tokenizer classes are not registered with HF's auto-discovery. Workaround: reload the tokenizer separately via `tokenization/registry.py:load_tokenizer(...).to_hf_tokenizer()` and load the model via `AutoModelForMaskedLM.from_pretrained(checkpoint)`. The model checkpoint itself is a standard HF artifact and reloads normally.

---

## 8. Caveats

1. **Proxy metrics do not equal downstream quality.** All numbers in this project are sanity checks. Which tokenizer makes a better language model is a separate question that requires a training run.
2. **Corpus size varies a lot across datasets.** hsb v3 is ~700k sentences, hsb lww is 1.29M, dsb v1 is 239k. All are small by modern NLP standards; lww is the largest and dsb is roughly a third of hsb v3. Per-tokenizer numbers should be read against the dataset they were trained on, not directly compared across.
3. **Morfessor decode is lossy.** See §4.5 and [CHANGELOG.md](CHANGELOG.md) for the design reasoning. Inspection via `tokenize` is unaffected; downstream training (which consumes IDs) is unaffected; only human-readable reconstruction of full sentences from a flat ID stream is affected.
4. **Moses uses the Czech language tag** because neither Upper nor Lower Sorbian is a registered `sacremoses` language. The tag affects only the abbreviation list, which we don't depend on.
5. **`numpy<2` is pinned** because of an upstream incompatibility in `fasttext`. One-line workaround. Removable once `fasttext` is fixed upstream.
6. **GlotLID adds a one-time download (~1 GB)** when the preprocessing script first runs. Subsequent runs use the HuggingFace cache.
7. **Morfessor and MorphBPE training scale with corpus size.** Roughly: ~3–5 minutes on dsb v1 (215k train sentences), ~13 minutes on hsb v3 (633k), ~20 minutes on hsb lww (1.16M). SPM BPE and Unigram finish in well under a minute regardless. Budget accordingly when iterating.
8. **The dsb semi-supervised Morfessor variant (`morfessor_semi`) improves coverage and OOV but widens fertility variance.** Vocab coverage rose 9.6 pp and OOV dropped ~18% relative on dsb v1; fertility (std) went from 0.166 to 0.270 because the length distribution became more bimodal. Implementation detail in §4.5 ("Semi-supervised variant"), numbers in [EVALUATIONS.md §8](EVALUATIONS.md), history (including the tuning sweep that selected the 500-row annotation set) in [CHANGELOG.md](CHANGELOG.md). Two minor follow-ups remain: record the annotations path in the saved `tokenizer_config.json` for reproducibility, and try a richer-than-two-piece annotation source if one becomes available.

---

## 9. Reproduction

```
uv sync
```

### Upper Sorbian (hsb)

```
# 1. Build the cleaned, filtered, Moses-pretokenized corpus
uv run python scripts/download_data.py --lang hsb

# 2. Train each tokenizer on the training split
uv run python scripts/train.py --method spm_bpe \
    --corpus data/processed/hsb/v3_train.txt \
    --vocab-size 16000 --output models/hsb/spm_bpe_v3

uv run python scripts/train.py --method spm_unigram \
    --corpus data/processed/hsb/v3_train.txt \
    --vocab-size 16000 --output models/hsb/spm_unigram_v3

uv run python scripts/train.py --method morfessor \
    --corpus data/processed/hsb/v3_train.txt \
    --vocab-size 16000 --output models/hsb/morfessor_v3

uv run python scripts/train.py --method morph_bpe \
    --corpus data/processed/hsb/v3_train.txt \
    --vocab-size 16000 --output models/hsb/morph_bpe_v3

# 3. Evaluate all four on the dev split
uv run python scripts/evaluate.py \
    --model-path models/hsb/spm_bpe_v3 \
    --model-path models/hsb/spm_unigram_v3 \
    --model-path models/hsb/morfessor_v3 \
    --model-path models/hsb/morph_bpe_v3 \
    --corpus data/processed/hsb/v3_dev.txt
```

### Lower Sorbian (dsb)

The Witaj plain-text file must be placed at `data/raw/dsb/witaj_dsb_monolingual.txt` before running; the MT train and dev files are auto-downloaded from the TUM-NLP GitHub mirror.

```
# 1. Preprocess
uv run python scripts/download_data.py --lang dsb --output-suffix v1

# 2. Train
uv run python scripts/train.py --method spm_bpe \
    --corpus data/processed/dsb/v1_train.txt \
    --vocab-size 16000 --output models/dsb/spm_bpe_v1

uv run python scripts/train.py --method spm_unigram \
    --corpus data/processed/dsb/v1_train.txt \
    --vocab-size 16000 --output models/dsb/spm_unigram_v1

uv run python scripts/train.py --method morfessor \
    --corpus data/processed/dsb/v1_train.txt \
    --vocab-size 16000 --output models/dsb/morfessor_v1

uv run python scripts/train.py --method morph_bpe \
    --corpus data/processed/dsb/v1_train.txt \
    --vocab-size 16000 --output models/dsb/morph_bpe_v1

# 3. Evaluate
uv run python scripts/evaluate.py \
    --model-path models/dsb/spm_bpe_v1 \
    --model-path models/dsb/spm_unigram_v1 \
    --model-path models/dsb/morfessor_v1 \
    --model-path models/dsb/morph_bpe_v1 \
    --corpus data/processed/dsb/v1_dev.txt
```

Optional (experimental — see caveat 8): the semi-supervised Morfessor variant on dsb. Requires `apertium-dsb.dsb.metadix` at the repo root.

```
# Extract paradigm-balanced 1,000-row annotation TSV from the Apertium metadix
uv run python scripts/extract_dsb_morph_annotations.py

# Train the semi-supervised variant (uses the TSV produced above by default)
uv run python scripts/train.py --method morfessor_semi \
    --corpus data/processed/dsb/v1_train.txt \
    --vocab-size 16000 --output models/dsb/morfessor_semi_v1

# Evaluate the variant alongside the unsupervised baseline
uv run python scripts/evaluate.py \
    --model-path models/dsb/morfessor_v1 \
    --model-path models/dsb/morfessor_semi_v1 \
    --corpus data/processed/dsb/v1_dev.txt
```

End-to-end wall-clock time per language is dominated by Morfessor and MorphBPE training (Morfessor scales with corpus size, so dsb is faster than hsb). SPM BPE and Unigram each finish in under a minute.
