# Changelog

This document records what changed at each round of the pipeline and the reasoning behind each change. The current state is described in [PROJECT.md](PROJECT.md); this file is the historical record.

The order of rounds: v1 → v2 → v3. Each round was a coordinated batch of changes, retraining, and re-evaluation.

---

## v1 — initial pipeline

The first working version of the project. Implemented:

- The shared `BaseTokenizer` interface and the modular three-tokenizer structure.
- SentencePiece BPE and Unigram via the `sentencepiece` library, sharing a common base class.
- Morfessor 2.0 with a custom HuggingFace wrapper.
- Data download from Leipzig and WMT22, NFC normalization, deduplication, full-corpus output.
- Evaluation script with fertility, OOV rate, round-trip, HuggingFace compatibility, sample segmentations.

### What was wrong with v1

Two real problems surfaced during evaluation:

**1. Morfessor was being used non-canonically.**
We trained `BaselineModel()` with default settings and `load_data(word_freq_list)`, then post-hoc capped the resulting lexicon at 16,000 morphemes by frequency, with character-level fallback for the morphemes that got evicted. The result was a tokenizer that worked but had **20% OOV rate** and **fertility 2.76 ± 1.03** on the dev set. Words like `předsydstwom` collapsed into 12 single characters because the morphemes Morfessor produced for them had been cut from the capped vocabulary.

The underlying issues:

- Raw word frequencies were passed into `load_data` without the count-dampening that the Morfessor CLI applies by default. This biased the lexicon toward keeping frequent whole words intact.
- Post-hoc capping of Morfessor's lexicon is not how the library is meant to be used. Morfessor has built-in vocabulary-size controllers that work *during* training. Published work uses those, not post-hoc truncation.

**2. The HuggingFace wrappers were broken.**
- BPE: 0/100 compatibility match. The wrapper was using `tokenizers.models.BPE(vocab=..., merges=[])`, which has no merge rules and falls back to character-level output.
- Morfessor: 0/100 match. The wrapper wasn't adding the `▁` boundary markers that the native tokenizer was emitting.

**3. Smaller issues.**
- SPM `character_coverage` defaulted to ~99.95%, which can drop rare characters. Important for Upper Sorbian's diacritics.
- Round-trip failures (~3 per 1000) for SPM caused by Unicode normalization (`…` becoming `...`).

---

## v2 — Morfessor canonical configuration, train/dev/test split, HF wrappers

This round addressed each of the v1 issues with a coordinated change.

### Morfessor: switch to Morfessor's native vocabulary controller

Replaced post-hoc capping with `BaselineModel(corpusweight=NumMorphCorpusWeight(num_morph_types=...))`. Morfessor self-tunes its α parameter during training to converge on the target morpheme count. Three additional changes brought the configuration in line with Morfessor's CLI defaults:

- `forcesplit_list=["-"]` — always split on hyphens.
- `count_modifier=lambda c: 1` in `load_data` — canonical "ones" count dampening; every unique word contributes weight 1 regardless of frequency.
- The post-hoc cap is still in the code as a safety net but rarely evicts anything because the model converges near the target on its own.

**Result on dev**: OOV dropped from 20.28% to 0.30%. Fertility dropped from 2.76 to 1.73. Standard deviation tightened from 1.03 to 0.25.

This is the larger story: by using each tokenizer's native vocabulary-control mechanism instead of forcing all three into a frequency-cap framework, the ablation became cleaner. Each algorithm now hits the budget the way it was designed to.

### SentencePiece: add `character_coverage=1.0`

Forced SPM to include every character seen in the corpus rather than dropping rare ones. Single-line change in the trainer call. Important for Upper Sorbian.

### HuggingFace wrappers: rewrite

Replaced the broken BPE wrapper conversion with `SentencePieceHFTokenizer`, a custom subclass of `PreTrainedTokenizer` that wraps the SentencePiece processor directly. The format-conversion approach was abandoned entirely. The Morfessor wrapper was updated to emit `▁` markers consistently with the native tokenizer.

**Result**: HuggingFace compatibility went from 0/100 to 100/100 for both BPE and Morfessor.

### Data pipeline: add ASCII control character stripping, boilerplate filter, train/dev/test split

- Stripped ASCII control characters (codepoints 0–31 and 127) from every line.
- Dropped lines containing the strings `"filename"` or `"dateiname formatverbinden"` (column-header artifacts from Leipzig's raw export).
- Added a deterministic 90% / 5% / 5% train/dev/test split with a fixed seed. This made evaluation honest — the dev and test splits are now genuinely held out.

### Cosmetic cleanup at the end of v2

After v2, several leftover files were removed: a stub `main.py` from `uv init`, a redundant `requirements.txt` duplicating `pyproject.toml`, and several superseded model directories. Some unused class attributes (`_char_ids`, `_model_path`) were removed for cleanliness.

### Open issues remaining at end of v2

Three things bothered us:

1. **Morfessor's vocabulary coverage reading 130%** on the dev set — structurally impossible. The cause was the `▁` prefix being attached at emission time, so `přinoški` and `▁přinoški` were both counted as distinct token strings even though only one was in the vocabulary.
2. **Non-Sorbian sentences were leaking through** the data pipeline. English quotes, German fragments, occasional garbage. There was no language filter.
3. **Punctuation was glued to words** in the corpus. Strings like `slovo, druhe.` were sitting in training data with the comma and period attached. Subword tokenizers learned separate vocabulary entries for `slovo` and `slovo,`, wasting vocabulary slots.

These set up the v3 round.

---

## v3 — data quality pass, Moses pretokenization, Morfessor `▁` removal

This round was a single coordinated batch addressing five issues.

### 1. Language filter (GlotLID)

Added a step that loads the GlotLID model from HuggingFace and classifies each line. Keeps only lines where the top prediction is `__label__hsb_Latn` with confidence at or above 0.5 (configurable). Runs first in the new filter pipeline because it's the most aggressive cut.

**Drop count this run**: 19,785 lines (~2.7%).

### 2. Terminal-punctuation filter

Truncated sentence fragments — lines like "and then she said" without sentence-ending punctuation — are known to harm downstream model training. Added a step that keeps only lines whose final non-whitespace character is one of `.`, `!`, `?`, `…`, `»`, `"`, `'`.

**Drop count this run**: 11,192 lines (~1.5%).

### 3. Length filter

Drops lines with fewer than the minimum or more than the maximum number of whitespace-separated words (defaults: 3 and 100). Runs before Moses pretokenization so the length count reflects natural words.

**Drop count this run**: 848 lines (~0.1%).

### 4. Moses pretokenization

Added `sacremoses.MosesTokenizer(lang='cs')` (Czech is the closest registered Slavic language; the tag affects only the abbreviation list, which we don't depend on). Applied to every line of the corpus during preprocessing so that `slovo, druhe.` becomes `slovo , druhe .` before any tokenizer sees it. This stops subword tokenizers from wasting vocabulary slots on `word.` and `word,` variants.

Also wired Moses inside every tokenizer's `tokenize`, `encode`, and `decode` methods so that callers can pass raw text and the tokenizers handle pretokenization internally. The Moses helper lives in `tokenization/pretokenize.py` so all three tokenizers share identical behavior.

### 5. Morfessor `▁` removal

The supervisor confirmed Morfessor doesn't structurally need a word-boundary marker — it splits on whitespace before doing anything else, so it never crosses word boundaries during segmentation. The `▁` was added in v1/v2 only for visual parity with SentencePiece, and it caused the 130% coverage reporting artifact.

In v3, `▁` is removed entirely from Morfessor's output and vocabulary. Tokenize returns flat morpheme lists like `[Hornjo, serb, šćina, je]`. Encode and decode work as plain one-to-one mappings via the vocabulary.

**Trade-off accepted in this design**: a flat ID stream from Morfessor cannot recover word boundaries during decode. Concretely: `decode(encode("Hornjoserbšćina je rěč."))` returns `"Hornjoserbšćinajerěč."` (concatenated). For inspection of segmentation quality (via `tokenize`) and for downstream model training (which consumes integer IDs), this does not matter. For human-readable reconstruction of full sentences from a flat ID stream, it does.

A clean fix exists if readability becomes important later: emit the literal space character `" "` as a token between words. Space is already in the character inventory (every line in the corpus has spaces). It's not a special boundary marker — it's a regular character — so the supervisor's stated objection to `▁` does not apply to it. The cost is that fertility increases by approximately +1 per word. We chose not to take this option in v3 because the use cases requiring readable Morfessor decode are limited and the morpheme-level inspection (which is the primary research-pipeline need) is fully supported.

### Round-trip metric: redefined

In v1 and v2 the round-trip test was `decode(encode(text)) == text`. With Morfessor's decode now lossy by design, that definition fails for Morfessor. We changed the metric to compare token streams: `tokenize(text) == [id_to_token[id] for id in encode(text)]`.

This is structurally weaker. For Morfessor it's tautological (encode is implemented as `[vocab[t] for t in tokenize(text)]`). For SentencePiece it's narrow but real — `tokenize` and `encode` are independent calls into the C++ library, so it catches the unlikely case where the two paths disagree. The honest reporting is: SPM still passes the strong text-equality property when tested directly; Morfessor passes only the weak vocab-consistency check; the strong check is structurally inapplicable to Morfessor's current design. See [METRICS.md §4](METRICS.md) for the detailed reasoning.

### NumPy 2.x pin

`fasttext` (used by GlotLID) is incompatible with NumPy 2.x. We pinned `numpy<2` in `pyproject.toml`. One-line workaround. Removable when `fasttext` ships an upstream fix.

### Tunable thresholds for experimentation

Every threshold in the data pipeline is now reachable two ways: a module-level constant near the top of `scripts/download_data.py` and a CLI flag. Edit the constant for a permanent default, pass the flag for a one-off run. Specifically: GlotLID confidence, min length, max length, terminal-punctuation set, and a `--skip-glotlid` toggle.

### Documentation reorganization at the end of v3

After the v3 changes landed, the project documentation was split from two cumulative summary files into:

- `README.md` — entry point.
- `docs/PROJECT.md` — current state, self-contained.
- `docs/METRICS.md` — metric methodology.
- `docs/CHANGELOG.md` (this file) — round history.
- `docs/EVALUATIONS.md` — evaluation results across rounds.

Training and download log files in `results/` were deleted; only the raw evaluation output text files were kept. Two cosmetic helpers were inlined in source (`count_tokens` in `scripts/download_data.py` and `_segment_word` in `tokenization/morfessor.py`).

---

## Alternative corpus experiment — Leipzig + Wiki + Witaj (`lww`)

After v3 landed, the user's supervisor asked whether trying a different corpus combination would change the picture. Two new sources were added to `data/raw/`:

- `wiki_hsb_monolingual.txt` (47,758 lines from Upper Sorbian Wikipedia)
- `witaj_hsb_monolingual.txt` (1,071,723 lines from the Witaj educational publisher)

A one-off analysis script (run, then deleted) measured the pairwise overlap between all four sources after applying the v3 pipeline. The headline finding was that **WMT22 is essentially a subset of Witaj** — 92.8% of WMT22's lines appear in Witaj. Wiki is also more than half-contained in Witaj (57%). Leipzig is genuinely independent of all the others (≤1% overlap with anything).

Given the redundancy, the experiment chosen was Leipzig + Wiki + Witaj — drops the redundant WMT22, keeps Leipzig as the independent source, adds Witaj for size and Wiki for register diversity.

### Pipeline parameterization

To support this without breaking v3, `scripts/download_data.py` was extended with two flags:

- `--sources leipzig wiki witaj` — selects which sources to combine. Defaults to `leipzig wmt22` (preserves prior behavior).
- `--output-suffix lww` — appends a suffix to all output filenames. Empty suffix (default) produces `hsb.txt`; `lww` produces `hsb_lww.txt`, `hsb_lww_train.txt`, etc.

A new `extract_plain` helper handles plain-text sources (Wiki and Witaj), and a `load_source(name)` dispatch function picks the right extractor per source. v3 artifacts are untouched.

### Outcome

Final lww corpus: 1,289,047 sentences after the full v3 pipeline (1.83× larger than v3). Train/dev/test: 1,160,142 / 64,452 / 64,453. Tokenizers saved to `models/hsb_*_lww/`. Evaluation in `results/eval_dev_lww.txt`. Numerical results and interpretation: [EVALUATIONS.md §7](EVALUATIONS.md).

The relative ordering of the three tokenizers (BPE shortest, Morfessor most consistent and most morphologically interpretable) is preserved between v3 and lww — useful evidence that the per-algorithm conclusions are not corpus-specific artifacts.

---

## MorphBPE — hybrid Morfessor + BPE baseline

After lww landed, the supervisor asked for a fourth tokenizer that follows the standard Morfessor + BPE hybrid pattern from the literature. The idea: use Morfessor as a fixed pre-segmenter (its job is to find morpheme boundaries, not to be the final tokenizer), then train BPE on top of the morpheme stream to compress to the vocabulary budget.

### Design choice: unconstrained Morfessor

The existing standalone Morfessor was trained with `NumMorphCorpusWeight` targeting ~15,750 morpheme types (so its own vocabulary fits the 16k budget). For MorphBPE the Morfessor stage is configured differently: no vocabulary budget. The model produces its natural larger morpheme inventory (typically 30k to 80k morphemes on this corpus). BPE then handles the final budget by compressing the morpheme stream to 16k SentencePiece pieces. This is the canonical literature setup. Reusing the existing constrained Morfessor was considered and rejected: its morphemes are already near-word-sized due to the tight budget, so BPE would have very little to compress and the resulting hybrid would behave essentially identically to standalone Morfessor.

### Implementation

A new file `tokenization/morph_bpe.py` containing two classes:

- `MorphBPETokenizer(BaseTokenizer)` — holds a `morfessor.BaselineModel` and a `sentencepiece.SentencePieceProcessor`. Training runs Morfessor without budget, pre-segments the corpus into a temp file (each morpheme as a whitespace unit), then trains SPM BPE on that file with `vocab_size=16000`. Inference applies Moses pretokenization, Morfessor segmentation per word, then SPM encoding. Save and load handle both underlying models.
- `MorphBPEHFTokenizer(PreTrainedTokenizer)` — HuggingFace wrapper using the same pipeline.

Registration: added `morph_bpe` to `TOKENIZER_CLASSES` in `scripts/train.py`, extended `detect_tokenizer_type` and `load_tokenizer` in `scripts/evaluate.py`, exported the class from `tokenization/__init__.py`. The existing CLI surface (`--method`, `--corpus`, `--vocab-size`, `--output`) handled the new tokenizer without changes.

### Results

Trained on `data/processed/hsb_lww_train.txt` (1.16M sentences) in ~16 minutes (~14 min Morfessor unconstrained + ~2 min BPE on the pre-segmented stream). Evaluated on `data/processed/hsb_lww_dev.txt` alongside the existing three. Numbers and interpretation: [EVALUATIONS.md §7](EVALUATIONS.md). The 3-way result file (`results/eval_dev_lww.txt`) was kept as historical record; the new 4-way evaluation lives in `results/eval_dev_lww_4way.txt`.

Headline: fertility 1.491 (between BPE's 1.363 and Morfessor's 1.636 as expected), OOV 0.01 percent (BPE's subword fallback eliminates character-level fallbacks), round-trip and HuggingFace compatibility perfect. Segmentations follow Morfessor boundaries where BPE doesn't have a useful merge, and merge Morfessor pieces where they co-occur frequently.

---

## Summary of effects across rounds

| Metric (Morfessor on dev) | v1 | v2 | v3 |
|---|---|---|---|
| Fertility (mean) | 2.76 | 1.73 | 1.60 |
| Fertility (std) | 1.03 | 0.25 | 0.18 |
| OOV rate | 20.28% | 0.30% | 0.27% |
| Vocab coverage | (not measured cleanly) | 130% (artifact) | 89.5% |
| HF compat | 0/100 | 100/100 | 100/100 |

| Metric (BPE on dev) | v1 | v2 | v3 |
|---|---|---|---|
| Fertility (mean) | 1.6 | 1.60 | 1.34 |
| Fertility (std) | 0.31 | 0.31 | 0.22 |
| OOV rate | 0.4% | 0.38% | 0.00% |
| HF compat | 0/100 | 100/100 | 100/100 |

The biggest single jump came from the Morfessor configuration fix in v2 (OOV went from 20% to 0.3%). The v3 data-quality pass moved every metric in the right direction across all three tokenizers and removed the Morfessor reporting artifact.
