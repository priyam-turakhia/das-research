# Upper Sorbian Tokenization Pipeline — Project Summary

This document walks through the entire project as it stands today, file by file. Nothing is skipped. The language is kept plain and the tone neutral.

---

## 1. What this project is

The goal is to train and compare three different "tokenizers" for Upper Sorbian — a West Slavic language spoken by roughly 30,000 people in eastern Germany.

A tokenizer is the component that takes raw text and breaks it into small pieces ("tokens") that a language model consumes. Different tokenizers break words apart differently, and the choice can affect downstream model quality. This project sets up three tokenizers on the same corpus, at the same target vocabulary size (16,000), so that any downstream comparison can attribute differences to the tokenization algorithm itself rather than other variables.

The three tokenizers:

1. **SentencePiece BPE** — a frequency-driven method that starts from single characters and repeatedly merges the most common adjacent pair until the vocabulary reaches the target size. "BPE" stands for Byte Pair Encoding.
2. **SentencePiece Unigram** — a probabilistic method that starts with a large pool of candidate pieces and prunes the least useful ones until the target size is reached. At inference, it picks the most likely segmentation under a trained unigram language model.
3. **Morfessor 2.0** — a linguistically motivated method that tries to discover real morphemes (prefixes, stems, suffixes) by minimizing a description-length objective. Not originally designed for fixed-vocab tokenization; we adapted it using Morfessor's built-in vocabulary-size controller.

All three share the same five reserved tokens — `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]` — at IDs 0 through 4.

---

## 2. Directory layout

```
das-research/
├── data/
│   ├── raw/                   Downloaded archives, untouched
│   └── processed/             Cleaned, split corpus files
│       ├── hsb.txt            Full deduped corpus
│       ├── hsb_train.txt      90% split
│       ├── hsb_dev.txt        5% split
│       └── hsb_test.txt       5% split
├── tokenization/              Library code
│   ├── __init__.py
│   ├── base.py                Abstract interface all tokenizers implement
│   ├── spm_base.py            Shared SentencePiece code
│   ├── spm_bpe.py             BPE variant
│   ├── spm_unigram.py         Unigram variant
│   ├── morfessor.py           Morfessor tokenizer + HuggingFace wrapper
│   └── evaluate.py            Metric functions
├── scripts/                   Command-line entry points
│   ├── download_data.py       Downloads and preprocesses the corpus
│   ├── train.py               Trains one tokenizer
│   └── evaluate.py            Evaluates one or more trained tokenizers
├── models/                    Trained tokenizer artifacts
├── results/                   Evaluation output files
├── pyproject.toml             Python project metadata + dependencies
├── requirements.txt           Same dependencies, pip-style (redundant)
└── main.py                    Unused stub from uv init
```

---

## 3. Data pipeline — `scripts/download_data.py`

### Sources

1. **Leipzig Corpora** — `hsb_mixed_2012_300K.tar.gz` from `downloads.wortschatz-leipzig.de`. A curated mixed-domain Upper Sorbian corpus with ~300,000 sentences. Distributed as a tar archive; the file named `*-sentences.txt` inside contains tab-separated data, where column 2 is the sentence text (column 1 is an ID).
2. **WMT22 monolingual** — `HSB_monolingual.txt.gz` from the Dimarco/WMT22 GitHub mirror. A plain-text corpus used in the 2022 Workshop on Machine Translation shared task for low-resource translation. Distributed as a gzip file with one sentence per line.

Note on source #2: the original spec pointed to a path inside an `HSB/monolingual/` subdirectory that didn't exist. The file actually lives at the repository root. This was verified with an HTTP HEAD request before updating the URL.

### Processing steps, in order

1. **Download** both archives to `data/raw/` if not already present. The script skips re-download if the file is there.
2. **Extract sentences**: Leipzig gets column 2 of the tab-separated sentences file; WMT22 is read as plain gzipped text.
3. **Concatenate** both sources into a single list.
4. **Normalize to NFC**. Unicode has multiple equivalent ways to encode the same character — for example, `š` can be a single codepoint or `s` followed by a combining caron. NFC ("Normalization Form C") consistently picks the single-codepoint version. This matters for Upper Sorbian because of its many diacritics (`š č ž ě ł ć ń ó`).
5. **Strip ASCII control characters** (codepoints 0–31 and 127 — tabs, form feeds, bells, etc.). These are invisible junk that would otherwise become tokens.
6. **Drop boilerplate lines**. Lines containing either `"filename"` or `"dateiname formatverbinden"` (a German phrase meaning "filename format-connect" that appears as a column header in Leipzig's raw data) are removed. These are export artifacts, not real Sorbian text.
7. **Deduplicate** by exact string match. Order of first occurrence is preserved.
8. **Deterministic train/dev/test split** at 90% / 5% / 5%. With a fixed random seed (42), the deduped sentences are shuffled once and split. The same seed always produces the same splits.
9. **Write four UTF-8 files**: `hsb.txt` (full corpus) plus the three splits, one sentence per line.

### What gets logged

Line counts at every step: Leipzig sentences, WMT22 sentences, total before dedup, after dedup, control-character lines cleaned, boilerplate lines dropped, empty lines dropped, duplicate lines dropped, final train/dev/test sizes, total whitespace tokens. This makes the preprocessing auditable.

### Caveats

- **The split is done on the deduped corpus**, so the same sentence never appears in both train and a held-out split.
- **No language filtering**. If the source corpora contain non-Sorbian text (e.g., quoted English), it is kept. One of the round-trip failures noted later is literally a Beatles lyric.
- **No length or token-count filtering**. Very short or very long lines are not filtered.
- **No case normalization**.

---

## 4. Library code — `tokenization/`

### 4.1 `base.py` — the abstract interface

Defines `SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]` and an abstract class `BaseTokenizer` with these required methods:

- `train(corpus_path, vocab_size)` — learn from a corpus on disk
- `tokenize(text) -> list[str]` — split text into human-readable pieces
- `encode(text) -> list[int]` — same result, but as integer IDs
- `decode(ids) -> str` — reverse of encode
- `save(path)` / `load(path)` — serialize to / from a directory
- `to_hf_tokenizer()` — return a HuggingFace-compatible wrapper
- `vocab_size` property
- `get_vocab() -> dict[str, int]`

The entire point of this base class is that every tokenizer in the project honors the same contract, so callers don't have to branch on algorithm type.

### 4.2 `spm_base.py` — SentencePiece shared logic

Contains two classes:

**`SentencePieceHFTokenizer`** (subclass of `PreTrainedTokenizer`).
A HuggingFace-compatible wrapper that holds a `sentencepiece.SentencePieceProcessor` and delegates tokenize/encode/decode to it. The `_tokenize`, `_convert_token_to_id`, `_convert_id_to_token`, and `convert_tokens_to_string` methods all forward directly to the SPM processor. This is a "slow" tokenizer in HuggingFace terminology — Python-based, not the Rust-backed fast variant.

This class exists because an earlier attempt to convert the SPM model into HuggingFace's native BPE format (via `tokenizers.models.BPE`) produced character-level output since the merge rules couldn't be extracted from the SPM model. The current approach skips the format conversion entirely.

**`BaseSPMTokenizer`** (subclass of `BaseTokenizer`).
The training logic. `train()` calls `SentencePieceTrainer.train()` with:

- `vocab_size` passed through from the caller
- `model_type` set by the subclass to `"bpe"` or `"unigram"`
- `character_coverage=1.0` — every character in the corpus is in the vocabulary. Without this, SPM defaults to ~99.95% and drops rare characters. Important for Upper Sorbian's diacritics.
- `pad_id=0, unk_id=1, bos_id=-1, eos_id=-1` — pins special-token IDs to match our scheme, disables begin/end-of-sentence tokens (we don't need them)
- `user_defined_symbols=["[PAD]", "[CLS]", "[SEP]", "[MASK]"]` — forces these into the vocabulary intact
- `num_threads=os.cpu_count()` — uses all available cores

Training happens in a temporary directory; the resulting `.model` file is read into memory as raw bytes (stored in `_model_bytes`) so it can be re-written on `save()`.

`tokenize`, `encode`, `decode` delegate to the SPM processor.

`save()` writes `spm.model` plus `tokenizer_config.json` (model type, vocab size, class name).

`load()` reads `spm.model` and reconstructs the processor.

`to_hf_tokenizer()` constructs a `SentencePieceHFTokenizer` using the stored model bytes.

### 4.3 `spm_bpe.py` and `spm_unigram.py`

Each is a one-line subclass of `BaseSPMTokenizer` setting `model_type = "bpe"` or `model_type = "unigram"`. All the actual work is in the base class.

### 4.4 `morfessor.py` — Morfessor tokenizer

The most complex file in the project. Contains three pieces:

**`segment_word_with_vocab(model, vocab, word)` — shared helper function.**
Runs Morfessor's `viterbi_segment` on a word to get a list of morphemes. For each morpheme, checks whether it is in the vocabulary dictionary. If yes, keeps it as-is; if no, replaces it with the list of its characters. If `viterbi_segment` itself fails (word outside the Morfessor model), the whole word falls back to characters. This function is called both by the native tokenizer and by the HuggingFace wrapper, so their output is guaranteed consistent.

**`MorfessorHFTokenizer`** (subclass of `PreTrainedTokenizer`).
HuggingFace wrapper. Its `_tokenize` method runs `segment_word_with_vocab` for each word and emits a standalone `▁` token in front, then the morphemes. `_convert_token_to_id` looks up the vocabulary (falls back to `[UNK]` id). `convert_tokens_to_string` treats `▁` as a word boundary and rejoins the pieces with spaces.

**`MorfessorTokenizer`** (subclass of `BaseTokenizer`) — the main class.

Training does this, in order:

1. Read the corpus once. Collect (a) the set of every character seen, (b) a word-frequency counter.
2. Compute `target_morphs = vocab_size − len(SPECIAL_TOKENS) − len(char_inventory) − 1`. This is the number of morphemes the Morfessor model should produce so that specials + chars + morphemes fit into the 16k budget.
3. Construct `BaselineModel` with two key settings:
   - `forcesplit_list=["-"]` — always splits on hyphens, matching the Morfessor CLI default.
   - `corpusweight=NumMorphCorpusWeight(num_morph_types=target_morphs)` — Morfessor has built-in "corpus-weight updaters" that can adjust its internal α parameter during training. `NumMorphCorpusWeight` specifically tunes α to converge toward a target number of morpheme types. This makes Morfessor respect a vocabulary budget natively, instead of us post-hoc truncating.
4. Call `load_data(word_freq_list, count_modifier=lambda c: 1)`. The `count_modifier=lambda c: 1` is the canonical "ones" dampening — every unique word contributes weight 1 regardless of its actual frequency. This prevents extremely common words (`je`, `a`, `so`) from dominating the lexicon and biasing the model toward keeping frequent words whole.
5. Call `train_batch()` with the default `algorithm='recursive'`. The model iterates until the cost stops decreasing; typically 6–8 epochs on this corpus. The corpus-weight updater adjusts α at the end of each epoch.
6. Re-segment every corpus word with `viterbi_segment` and count morpheme frequencies across the corpus.
7. Build the final vocabulary:
   - IDs 0–4: special tokens
   - Next N IDs: the character inventory plus `▁`, so the character-fallback path can never produce an `[UNK]` as long as only previously-seen characters appear
   - Remaining IDs up to `vocab_size`: morphemes sorted by frequency

Because `NumMorphCorpusWeight` already drove the model close to the target during training, the post-hoc frequency cap in step 7 rarely evicts anything — it functions as a safety net rather than the primary sizing mechanism.

At inference:

- `tokenize(text)` splits on whitespace. For each word it runs `segment_word_with_vocab`, then prefixes the first morpheme of the word with `▁`.
- `encode(text)` tokenizes, then walks the token list. When a token starts with `▁`, it emits two IDs: the standalone `▁` id, then the id of the bare morpheme. This is why the HuggingFace wrapper emits `▁` as a separate token — the two ID streams must match for the HF compatibility check to pass.
- `decode(ids)` maps IDs back to strings, treats `▁` as a word boundary, joins pieces with single spaces.

`save()` writes `model.pkl` (pickled Morfessor model), `vocab.json`, and a `tokenizer_config.json`.
`load()` reads those three files back.

### 4.5 `evaluate.py` — metric functions

Defines `EvaluationResult` (a dataclass holding every metric value) and the following functions, each taking a tokenizer and a list of sentences:

- **`compute_fertility`** — mean and standard deviation of tokens-per-word across sentences. A fertility of 1.0 means the tokenizer leaves words intact on average. 2.0 means each word becomes two pieces on average. Lower is a shorter sequence for the model to process, but lower is not automatically better.
- **`compute_unique_tokens`** — number of distinct token strings actually emitted on the eval set. Compared to vocab size this tells you how much of the learned vocabulary is actually used.
- **`compute_oov_rate`** — fraction of words where the tokenizer either emits `[UNK]` or collapses to single characters. Single-character words (like `je`) are not counted as OOV. For Morfessor, the `▁` prefix is stripped before the check.
- **`round_trip_test`** — samples 1,000 sentences, runs `decode(encode(text))`, and compares to the original after whitespace normalization. A correct tokenizer passes all 1,000.
- **`hf_compatibility_test`** — samples 100 sentences, runs `tokenizer.encode(text)` both natively and via the HuggingFace wrapper, and checks the integer ID streams match exactly.
- **`side_by_side_segmentation`** — runs multiple tokenizers on a fixed list of 20 morphologically interesting Upper Sorbian words and returns their segmentations, for human inspection.
- **`evaluate_tokenizer`** — orchestrates all of the above and returns an `EvaluationResult`.
- **`print_comparison_table`** — formats results for multiple tokenizers as a text table.

**Caveat.** Every one of these metrics is a sanity check — it confirms the tokenizer works and gives a rough feel for efficiency and coverage. None of them predicts how well a downstream language model or translation system will perform with a given tokenization. That requires actually training a model.

---

## 5. Command-line scripts — `scripts/`

### 5.1 `download_data.py`

Described in section 3. Run with no arguments:

```
uv run python scripts/download_data.py
```

### 5.2 `train.py`

Trains one tokenizer on a corpus.

```
uv run python scripts/train.py \
    --method {spm_bpe|spm_unigram|morfessor} \
    --corpus data/processed/hsb_train.txt \
    --vocab-size 16000 \
    --output models/my_tokenizer
```

Internally it looks up the right class from `TOKENIZER_CLASSES`, calls `.train()` and `.save()`, times the training, and runs a test `tokenize → encode → decode` on a fixed Sorbian sentence so the output is eyeball-verifiable in the logs.

If you point `--corpus` at the full `hsb.txt` while `hsb_train.txt` and `hsb_dev.txt` exist, the script logs a warning telling you to train on the train split so the dev/test splits stay held out. It still runs if you insist.

Approximate training times on this corpus:
- SPM BPE: ~1 minute
- SPM Unigram: ~1 minute
- Morfessor: ~15–18 minutes (single-threaded Python, 6–8 epochs)

### 5.3 `evaluate.py`

Runs evaluation on one or more trained tokenizers. Repeat `--model-path` for each model.

```
uv run python scripts/evaluate.py \
    --model-path models/hsb_spm_bpe_cleaned \
    --model-path models/hsb_spm_unigram_cleaned \
    --model-path models/hsb_morfessor_v2 \
    --corpus data/processed/hsb_dev.txt
```

Features:

- Auto-detects tokenizer type from `tokenizer_config.json`
- Same held-out-corpus warning as `train.py`: if you point it at `hsb.txt` while split files exist, it logs that you should prefer a held-out split
- Prints the comparison table and side-by-side segmentation of the 10 sample words

---

## 6. Modularity — how the pieces fit together

The design intent: **callers should only ever see `BaseTokenizer`**. Algorithm-specific details stay inside the subclasses.

- `scripts/train.py` looks up the class via `TOKENIZER_CLASSES[args.method]`. It does not know or care whether the returned object is BPE, Unigram, or Morfessor.
- `scripts/evaluate.py` auto-detects the class from the saved config. Every call site uses the base-class interface.
- `tokenization/evaluate.py` takes `BaseTokenizer` instances and measures them identically.
- The two HuggingFace wrappers (`SentencePieceHFTokenizer`, `MorfessorHFTokenizer`) both subclass `PreTrainedTokenizer`, so downstream model code that expects a HuggingFace tokenizer accepts any of them.

Shared code is lifted where reasonable:

- BPE and Unigram share `BaseSPMTokenizer`; each subclass is literally one line.
- Morfessor's character-fallback logic lives in a single module-level function used by both the native class and the HuggingFace wrapper, so they never diverge.

What this buys: adding a fourth tokenizer (for example, a Morfessor→BPE hybrid) means writing one new subclass with its own `train()` method. The scripts, the evaluation module, and the HuggingFace integration all work without changes.

---

## 7. Current evaluation results

Evaluated on `hsb_dev.txt` (36,770 held-out sentences). From [results/eval_dev_v2.txt](results/eval_dev_v2.txt):

| Metric | SPM BPE | SPM Unigram | Morfessor v2 |
|---|---|---|---|
| Fertility (mean) | 1.596 | 1.604 | 1.733 |
| Fertility (std)  | 0.310 | 0.334 | 0.247 |
| Vocab size       | 15,995 | 15,995 | 15,825 |
| Unique tokens used | 15,831 | 15,888 | 20,699 |
| Vocab coverage   | 98.9% | 99.3% | 130.8% (see note) |
| OOV rate         | 0.38% | 0.44% | 0.30% |
| Round-trip pass  | 997 / 1000 | 997 / 1000 | 1000 / 1000 |
| HF compatibility | 100 / 100 | 100 / 100 | 100 / 100 |

### Reading the numbers

- **Fertility.** All three sit between 1.60 and 1.73 tokens per word on average. BPE is shortest. Morfessor has the lowest standard deviation (0.247), meaning it is the most consistent in sequence length — this comes from segmenting more predictably along morpheme boundaries.
- **OOV rate.** All under 0.5%. Morfessor is actually lowest at 0.30%.
- **Round-trip.** BPE and Unigram both fail the same 3 out of 1000. The cause is SentencePiece's internal Unicode normalization: it converts the `…` ellipsis character (one codepoint) into `...` (three separate periods) on decode. This is inside SPM and is purely cosmetic — a human sees the same text but a byte comparison fails. Morfessor passes all 1000.
- **Vocab coverage for Morfessor (130.8%).** This looks like excess but it's a reporting artifact. Morfessor's `tokenize()` adds the `▁` prefix to the first morpheme of each word *after* vocabulary lookup. So `přinoški` and `▁přinoški` both appear in the output as distinct token strings even though only `přinoški` is actually stored in the vocabulary. A fair coverage number would strip the `▁` prefix before counting uniques.
- **HF compatibility.** All three match perfectly. Previously BPE was 0/100 (a bug in the earlier attempt to convert SPM to HuggingFace's BPE format) and Morfessor was 0/100 (the HF wrapper wasn't emitting `▁` markers). Both fixed.

### Sample segmentations

`najwjetšich` ("largest", genitive plural):
- BPE: `▁najwjetšich` (kept whole)
- Unigram: `▁najwjetši | ch` (stem + case ending)
- Morfessor v2: `▁naj | wjetši | ch` (superlative prefix + stem + case ending — the linguistically correct analysis)

`předsydstwom` ("by the chairmanship"):
- BPE: `▁předsyd | stwom`
- Unigram: `▁předsydstwo | m`
- Morfessor v2: `▁předsyd | stwom` (identical to BPE)

Morfessor tends to produce linguistically recognizable pieces. BPE and Unigram tend to keep common whole words intact and split only when a word is rare or novel.

### Which is best?

None of these numbers alone tells you. Each tokenizer wins a different metric. The only real answer to "best" is to train a downstream model on each and compare perplexity or BLEU. That experiment is a future step.

---

## 8. The Morfessor change — what happened and why

The Morfessor tokenizer went through two versions because the first one was using the library in a non-standard way.

**v1 (original, now superseded — artifacts in `models/hsb_morfessor/` and `models/hsb_morfessor_cleaned/`).**
Trained with `BaselineModel()` defaults and `load_data(word_freq_list)`. The model learned ~30–50k morphemes, which we post-hoc capped at 16k by frequency with a character fallback for evicted morphemes. Results on dev: OOV 20.28%, fertility 2.76 ± 1.03. Words like `předsydstwom` collapsed to 12 single characters because their morphemes had been evicted from the capped vocabulary.

**What was wrong with v1:**
- Raw word frequencies were passed into `load_data` without the count-dampening (`count_modifier=lambda c: 1`) that the Morfessor CLI applies by default. This biased the lexicon toward keeping frequent whole words.
- Post-hoc capping of Morfessor's lexicon is not how the library is meant to be used. Morfessor has built-in vocabulary-size controllers that work during training. Published work uses those, not post-hoc truncation.

**v2 (current — `models/hsb_morfessor_v2/`).**
- `BaselineModel(corpusweight=NumMorphCorpusWeight(num_morph_types=...))`. Morfessor self-tunes its α parameter during training to land near the target number of morpheme types.
- `load_data(..., count_modifier=lambda c: 1)` — canonical "ones" count dampening.
- `forcesplit_list=["-"]` — always split on hyphens (Morfessor CLI default).
- The post-hoc cap is still in the code as a safety net but rarely evicts anything, because the model already converges near the target on its own.

Results on dev after the change: OOV 0.30%, fertility 1.73 ± 0.25. Morfessor went from clearly worse than SPM on proxy metrics to competitive, while producing more linguistically interpretable splits.

This is also a cleaner comparison for the ablation: each tokenizer now uses its own native mechanism to hit the 16,000 budget, rather than forcing Morfessor into a frequency-cap framework that it wasn't designed for.

---

## 9. Caveats

1. **Proxy metrics do not equal downstream quality.** Every number in this project is a sanity check. Which tokenizer makes a better language model is a separate question that requires a training run.
2. **Corpus size.** ~735,000 sentences is small by modern NLP standards but substantial for Upper Sorbian specifically. All results are specific to this corpus.
3. **The three SPM round-trip failures are an SPM normalization behavior**, not a bug in our code. They are cosmetic.
4. **Morfessor training is slow** (~15–18 min per run, single-threaded Python). Budget accordingly when iterating on settings.
5. **The `▁` boundary marker** is SPM's convention, adopted for Morfessor so all three tokenizers produce consistent output. This inflates Morfessor's "unique tokens used" metric (see section 7).
6. **No language filter in the data pipeline.** Non-Sorbian lines in the source corpora (quoted English, etc.) are kept.

---

## 10. How to reproduce from scratch

```
uv sync

# 1. Download and preprocess the corpus
uv run python scripts/download_data.py

# 2. Train each tokenizer on the training split
uv run python scripts/train.py --method spm_bpe \
    --corpus data/processed/hsb_train.txt \
    --vocab-size 16000 --output models/hsb_spm_bpe_cleaned

uv run python scripts/train.py --method spm_unigram \
    --corpus data/processed/hsb_train.txt \
    --vocab-size 16000 --output models/hsb_spm_unigram_cleaned

uv run python scripts/train.py --method morfessor \
    --corpus data/processed/hsb_train.txt \
    --vocab-size 16000 --output models/hsb_morfessor_v2

# 3. Evaluate all three on the dev split
uv run python scripts/evaluate.py \
    --model-path models/hsb_spm_bpe_cleaned \
    --model-path models/hsb_spm_unigram_cleaned \
    --model-path models/hsb_morfessor_v2 \
    --corpus data/processed/hsb_dev.txt
```

Total wall-clock time is dominated by Morfessor training (~15–18 min). Everything else finishes in a few minutes combined.
