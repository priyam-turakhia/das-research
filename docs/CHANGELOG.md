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
- `--output-suffix lww` — sets the dataset tag for all output filenames. Default `v3` produces `hsb_v3.txt`, `hsb_v3_train.txt`, etc.; `lww` produces `hsb_lww.txt`, `hsb_lww_train.txt`, etc.

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

---

## Internal refactor — tokenizer registry, shared config schema, HF wrapper deduplication

After MorphBPE landed, a cleanup pass made it cheaper to add a fifth tokenizer in the future. None of the evaluation numbers changed — verified by re-running the lww 4-way evaluation and getting bit-for-bit identical fertility, OOV, vocab coverage, round-trip, and HF-compat scores to the prior 2026-05-19 run.

### Tokenizer registry

`tokenization/registry.py` is now the single source of truth for the name → class mapping, the saved-config schema, type detection from a saved directory, and the load entry point. Previously this logic was duplicated in `scripts/train.py` and `scripts/evaluate.py` and would drift if a new tokenizer were added. Adding a fifth tokenizer now means writing the class, declaring its `tokenizer_type` attribute, and adding it to `get_tokenizer_classes()` — the CLI scripts pick it up without changes.

### `tokenizer_config.json` schema

Standardized on `{tokenizer_type, vocab_size, version: 1}` written by every tokenizer's `save()`. Tokenizer-specific HuggingFace integration keys (like Morfessor's `auto_map`) pass through as extras. The legacy v2/v3 saves used different combinations of `tokenizer_class` and `model_type`; `detect_tokenizer_type` keeps a legacy fallback so existing model directories under `models/` continue to load unchanged.

### SentencePiece training helper

The `SentencePieceTrainer.train(...)` call with its dozen parameters lived identically in `spm_base.py` and `morph_bpe.py`. Extracted into a `train_spm_model(corpus_path, vocab_size, model_type) -> bytes` helper in `spm_base.py`. Both call sites now share one configuration surface.

### HuggingFace wrapper deduplication

`tokenization/hf_base.py` adds two small bases:
- `BaseHFTokenizer` carries `model_input_names` and `apply_default_special_tokens(kwargs)` (formerly six `setdefault` calls duplicated in every wrapper).
- `SpmBackedHFTokenizer` carries the `sp_model`-keyed `vocab_size`, `get_vocab`, `_convert_token_to_id`, `_convert_id_to_token`, and `convert_tokens_to_string` methods that were identical between `SentencePieceHFTokenizer` and `MorphBPEHFTokenizer`.

`MorfessorHFTokenizer` inherits from `BaseHFTokenizer` only (different vocab structure — a Python dict, not an SPM model).

### Artifact rename: `cleaned` → `v2`

The two SPM tokenizers trained at the end of v2 had been named `hsb_spm_bpe_cleaned` and `hsb_spm_unigram_cleaned` because they were the first SPM models trained on the v2 cleaned corpus. The `cleaned` tag was ambiguous (it described data, not pipeline version) and visually mixed with the proper dataset tags. Renamed to `hsb_spm_bpe_v2` and `hsb_spm_unigram_v2` to align with `hsb_morfessor_v2`. The eval log at `results/eval_dev_v2.txt` references the old paths in its captured log lines but the numbers stand.

### Default data filenames now carry a dataset tag

The processed-corpus files for v3 were previously `hsb.txt`, `hsb_train.txt`, `hsb_dev.txt`, `hsb_test.txt` — the dataset tag was implicit. Renamed to `hsb_v3.txt`, `hsb_v3_train.txt`, etc. so every file's identity is visible at a glance. The download script's `--output-suffix` default changed from `""` to `v3` to match. `data/` is gitignored so this is local-only state; rerun `scripts/download_data.py` to regenerate from sources.

### Naming convention now documented

The artifact naming scheme — dataset tags, method names, splits — is enumerated in [PROJECT.md §2](PROJECT.md). The intent: adding a new tokenizer method or a new dataset version follows the pattern with no special-casing.

---

## Multi-language module structure — Lower Sorbian (`dsb`) added

The project originally targeted Upper Sorbian (`hsb`) only. Adding Lower Sorbian motivated rearranging the repository so each language is a separate **module** that shares the same pipeline code, the same tokenization library, and the same training/evaluation scripts. None of the algorithm code changed — only the layout and the data-loading layer.

### Per-language directory layout

Everything language-specific now lives under a language-coded subdirectory:

- `data/raw/<lang>/`, `data/processed/<lang>/`
- `models/<lang>/`
- `results/<lang>/`

The redundant `hsb_` prefix that used to appear on every artifact (e.g. `hsb_spm_bpe_v3`, `hsb_v3_train.txt`) was stripped — the parent directory already conveys the language. So `models/hsb_spm_bpe_v3` is now `models/hsb/spm_bpe_v3`, and `data/processed/hsb_v3_train.txt` is now `data/processed/hsb/v3_train.txt`. All 10 existing hsb tokenizers were moved without retraining and still load and evaluate identically; the SPM and Morfessor pickles inside don't care about the directory path.

### `download_data.py` is now language-agnostic

The script gained a `--lang` flag and a `LANG_REGISTRY` at the top of the file. The registry maps each language code to its source list (filename, optional download URL, extractor), the GlotLID label to keep during language filtering, and a default source combination. Picking a language switches all three. Pipeline functions (`normalize_and_clean`, `filter_language`, `filter_terminal_punct`, `filter_length`, `apply_moses`, `deduplicate`, `split_sentences`) are unchanged — they were already language-agnostic. Output goes to `data/processed/<lang>/<dataset>.txt` and its splits.

### `train.py` and `evaluate.py` are unchanged in behavior

The CLI scripts already operated on `BaseTokenizer` interfaces, so they didn't need language awareness. The held-out-corpus warning was generalized to look for sibling `_train`/`_dev`/`_test` files for any stem, not just files starting with `hsb_`.

### dsb sources and `v1` dataset

Two sources are wired in for Lower Sorbian:

- **Witaj** (`witaj_dsb_monolingual.txt`, ~120k sentences) — manually placed under `data/raw/dsb/`.
- **MT train + dev** (`train.de-dsb.dsb` and `dev.de-dsb.dsb`, ~172k + ~4k sentences) — auto-downloaded from the TUM-NLP `llms-limited-resources2025` GitHub repo. These are the dsb side of a de↔dsb MT training set; pooled with Witaj and re-split, not treated as a separate held-out set.

The Leipzig portal lists a `dsb_wikipedia_2021` corpus but it is not downloadable (browseable only); the only Lower Sorbian Wikipedia archive on the Leipzig download endpoint is the 2016 10K snapshot, which was tried briefly and then discarded in favor of the MT data above.

Running `scripts/download_data.py --lang dsb --output-suffix v1` produced the first dsb dataset, tagged `v1`:

| Step | Lines |
|---|---|
| Raw total (Witaj + MT train + MT dev) | 296,460 |
| After GlotLID @ 0.5 (`__label__dsb_Latn`) | 277,378 (–19,082) |
| After terminal-punct filter | 247,579 (–29,799) |
| After length filter (3–100 words) | 247,045 (–534) |
| After Moses + dedup | **239,316** |

Splits: 215,384 train / 11,965 dev / 11,967 test. Total tokens: 3,770,578.

For scale comparison: hsb v3 is 703,595 sentences, hsb lww is 1,289,047. dsb v1 sits at roughly one-third of hsb v3 by sentences (and tokens). The two source corpora available for Lower Sorbian don't get us to hsb scale; this is accepted as a corpus-availability constraint, not a pipeline issue. Tokenizer comparisons within dsb are unaffected; cross-language comparisons should note the size asymmetry.

### Source-overlap check (dsb)

Exact-line intersection between Witaj and MT train: **0 lines**. The two sources are independent — unlike on the hsb side, where Witaj subsumes ~93% of WMT22.

### Tokenizer training and 4-way evaluation (dsb)

All four tokenizers were trained on `data/processed/dsb/v1_train.txt` (215,384 sentences) at the same 16,000 vocab budget and evaluated on `v1_dev.txt` (11,965 sentences). Output: [results/dsb/eval_dev_v1_4way.txt](../results/dsb/eval_dev_v1_4way.txt). Headline: fertility 1.289 / 1.514 / 1.571 / 1.491 (BPE / Unigram / Morfessor / MorphBPE), OOV 0.00% for BPE and MorphBPE, 0.88% for Morfessor (character fallback rises on the smaller corpus), round-trip 1000/1000 and HF compatibility 100/100 for all four. The four-tokenizer ordering (BPE shortest, Morfessor lowest fertility std and most morphologically interpretable, MorphBPE between) matches the hsb runs — useful evidence that the per-algorithm conclusions generalize across the two Sorbian variants. Full table and interpretation: [EVALUATIONS.md §8](EVALUATIONS.md).

---

## Semi-supervised Morfessor for dsb — Apertium metadix annotations

The dsb Morfessor baseline has noticeably higher OOV (0.88%) and lower vocab coverage (70.9%) than the hsb runs, both driven by the smaller training corpus. The supervisor pointed out that Lower Sorbian has an Apertium morphological dictionary (`apertium-dsb.dsb.metadix` at the repo root) usable as a small gold-segmentation source. The plan: extract word-level `surface\tstem ending` annotations from the metadix, feed them to Morfessor 2.0's semi-supervised training (`model.set_annotations`), and check whether grounded morphology guidance lifts the dsb Morfessor tokenizer.

### Annotation extraction

`scripts/extract_dsb_morph_annotations.py` walks the metadix XML, recursively expands paradigm definitions, and emits `surface\tstem ending` rows for every surface form (skipping `r="LR"` analysis-only entries). Outputs two TSVs under `data/processed/dsb/`:

- `metadix_morph_annotations_full.tsv` — 31,601 rows (every distinct surface form).
- `metadix_morph_annotations_1000.tsv` — 1,000 rows, paradigm-balanced (sqrt-weighted quotas per paradigm so small closed-class paradigms keep representation), seed 42.

980 of the 1,000 rows are two-part `stem ending`; the remaining 20 are one-part rows for closed-class words where the lemma equals the surface form. The script ends with a Morfessor smoke test (`set_annotations` + one-epoch `train_batch`) so format breakage is caught at extraction time.

### `SemiSupervisedMorfessorTokenizer`

A thin subclass of `MorfessorTokenizer` in `tokenization/morfessor_semi.py` (`tokenizer_type = "morfessor_semi"`). `MorfessorTokenizer.train()` was extended with a keyword-only `annotations_path` argument that, when set, calls `model.set_annotations(annotations)` after `load_data()` and before `train_batch()`. Registry, `__init__.py`, and `detect_tokenizer_type` were extended for the new type. The change is backward-compatible — the existing `MorfessorTokenizer` and `MorphBPETokenizer` callers are untouched.

The first attempt simply added `set_annotations` on top of the unchanged baseline configuration (`NumMorphCorpusWeight`, "ones" count dampening, hyphen forcesplit). Numbers barely moved: fertility 1.571 → 1.591, vocab coverage 70.9% → 70.4%, OOV unchanged at 0.88%. Eight of ten sample segmentations were identical to the baseline. Plumbing was correct, but the supervision was being neutralized.

The cause was two adaptive weight updaters running simultaneously inside `train_batch` and pulling against each other:

1. **`NumMorphCorpusWeight`** auto-tunes the main `corpusweight` toward a target morpheme-type count (~15,750 on this corpus).
2. **`set_annotations(..., annotatedcorpusweight=None)`** auto-tunes a second weight to enforce the annotations.

Compounding this, the apertium annotations are by construction two-piece `stem | ending`, while Morfessor's natural dsb output averages 3–5 pieces. The annotation signal pulled toward fewer, longer pieces; the budget tuner pulled toward enough pieces to fill the budget.

### Fix: fixed corpus weight, annotation tuner adapts alone

The semi-supervised subclass now passes `use_num_morph_weight=False` to the parent's `train()`. The parent constructs `BaselineModel(forcesplit_list=["-"])` with Morfessor's default `corpusweight=1.0` instead of wiring `NumMorphCorpusWeight`, so only the `set_annotations` tuner adapts during training. The 16k vocabulary budget is enforced by the existing post-hoc top-N morpheme cap, which previously sat as a safety net behind the NumMorph tuner. (`MorfessorTokenizer.train()` gained a `use_num_morph_weight: bool = True` kwarg so the baseline behavior is unchanged for everything else.)

### Results

Trained on `data/processed/dsb/v1_train.txt` with a 500-row paradigm-balanced annotation set (the choice of 500 is from the sweep documented below), evaluated on `v1_dev.txt` alongside the baseline `morfessor_v1`. Output: [results/dsb/eval_dev_v1_morfessor_semi.txt](../results/dsb/eval_dev_v1_morfessor_semi.txt).

| Metric | `morfessor_v1` | `morfessor_semi_v1` |
|---|---|---|
| Fertility (mean) | 1.571 | **1.548** |
| Fertility (std)  | **0.166** | 0.270 |
| Vocab size | 15,762 | 15,866 |
| Unique tokens used | 11,184 | **12,769** |
| Vocab coverage | 70.9% | **80.5%** |
| OOV rate | 0.88% | **0.72%** |
| Round-trip / HF | 1000/1000, 100/100 | 1000/1000, 100/100 |

Four of five non-tied metrics improved, three meaningfully:

- **Vocab coverage jumped 9.6 pp.** The annotations pushed Morfessor toward morphemes that actually appear in real text — substantially more of the learned vocabulary is exercised by the dev set.
- **OOV dropped ~18% relative** (0.88% → 0.72%). Fewer character fallbacks, matching the coverage gain.
- **Fertility (mean) is slightly shorter.** Dropping NumMorph did not blow up the budget — the post-hoc cap held.
- **Fertility (std) widened** (0.166 → 0.270). The one tradeoff: with fewer character fallbacks the length distribution is more bimodal (common dsb words stay short, rare/foreign words still fragment).
- **Round-trip and HF compatibility tied at perfect.**

The side-by-side table in the eval output shows more visible fragmentation than the baseline on the 10 hsb sample words used by the evaluator. This is the dsb tokenizer being asked to handle out-of-domain hsb morphology; it does not reflect dsb dev-set behavior, which is captured by the numbers above.

### Remaining follow-ups

- **Record the annotations path in the saved `tokenizer_config.json`.** Currently `save()` does not capture it, so a future reader cannot tell which annotation file produced the model. One-line addition to the `write_config(...)` `**extras`.
- **Allow richer (deeper-than-two-piece) annotations** if a source becomes available. Apertium can only give `stem ending`; a source that exposes inner morpheme structure could improve the result further.

### Tuning sweep — sample size, sampling strategy, annotation weight

Swept 13 configurations to test whether the canonical `morfessor_semi_v1` setup is near-optimal: sample sizes {100, 500, 5000, 10000, 20000} crossed with two sampling strategies (paradigm-balanced, corpus-frequency-weighted), plus a weight sweep {auto-tune, 1, 10, 100} at the best-performing sample size. Morfessor settings, training corpus, and dev split unchanged. All trial artifacts (models, generated TSVs, logs) lived in `/tmp/dsb_semi_sweep/` and were cleaned up afterwards — no new files in the project.

Pre-flight finding: of apertium's 31,601 surface forms, only **3,663 appear in `v1_train`** (~12% overlap). The frequency-weighted strategy capped at 3,663; larger requested sizes were skipped.

Result: only `balanced_500` met all three primary constraints relative to the 1,000-row variant the sweep was run against (coverage ≥ 80.3%, OOV ≤ 0.71%, fertility mean ≤ 1.550), landing at fertility 1.542 / std 0.269 / coverage 80.6% / OOV 0.70%. Deltas vs the 1,000-row variant are within ~0.5% on fertility and within the noise on OOV and coverage; fertility std is a hair worse.

Observations:

1. OOV across the 13 configurations is bimodal — values landed near 0.70% or near 1.30–1.50%, with nothing in between. Only `balanced_500`, `freq_500`, and the 1,000-row variant reached the lower band.
2. Balanced sample sizes 5000 / 10000 / 20000 all landed at OOV ≈ 1.5% despite coverage holding at ~81%.
3. Every explicit `annotatedcorpusweight` tested (1, 10, 100) at `balanced_500` landed at OOV 1.31–1.36%, vs 0.70% with the auto-tuned default.
4. Corpus-frequency-weighted sampling did not beat paradigm-balanced at any tested size.

**Decision: `balanced_500` was promoted as the canonical `morfessor_semi_v1`.** The numerical deltas vs the 1,000-row variant are within noise on every primary metric (the canonical retrain landed at 1.548 / 0.270 / 80.5% / 0.72%), so the choice was not driven by absolute improvement; the smaller annotation set is easier to defend as a default, sits within the range of annotation counts used in published low-resource Morfessor work, and trains slightly faster. `DEFAULT_ANNOTATIONS_PATH` in `tokenization/morfessor_semi.py` now points at `data/processed/dsb/metadix_morph_annotations_500.tsv`; the 1,000-row TSV was removed.

---

## MLM pretraining pipeline (`scripts/pretrain.py`)

Added a script to pretrain an XLM-RoBERTa-base-shaped encoder from random init on any of the trained tokenizers' output. Motivation: the encoder is intended to become the **student** in a cross-lingual embedding distillation step against stock `FacebookAI/xlm-roberta-base` (the multilingual teacher); MLM here is a sensible init, not the final training step. The parallel data for the distillation step is not yet loaded.

### Design choices

- **From scratch, not continued pretraining.** Our 16k vocabulary is entirely different from stock XLM-R's 250k SPM, so the embedding matrix can't be reused — there is no win to loading the rest of the pretrained weights either, since the encoder's input distribution would be wildly mismatched.
- **`XLMRobertaForMaskedLM(XLMRobertaConfig(...))`.** Production preset is the standard BERT-base shape (12 layers, hidden 768, 12 heads, FFN 3072), matching the teacher exactly so the student output space is comparable at distillation time without a projection layer. Smoke preset (`--smoke`) is 2 layers / hidden 128 for laptop validation.
- **`max_position_embeddings=514` by default.** Decoupled from training `seq_len`. An earlier draft tied them together (`seq_len + 2`) which would have capped the architectural input length to whatever flag the user trained with; reverted because the saving is ~0.2 MB out of 110 MB and the downside is real.
- **MLM only**, RoBERTa-style: `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)`. No NSP.

### Tokenizer wrapper fix: `[CLS] … [SEP]`

Discovered during this work: `BaseHFTokenizer` in `tokenization/hf_base.py` did not override `build_inputs_with_special_tokens`, `get_special_tokens_mask`, or `create_token_type_ids_from_sequences`. Default `PreTrainedTokenizer` behavior is no-op, so calling `tokenizer("…")` returned raw content IDs with no `[CLS]` / `[SEP]` wrapping. Two consequences:

- The data collator's `get_special_tokens_mask` returned an empty mask, which means it might have masked `[CLS]` and `[SEP]` had they been there (they weren't).
- Downstream code that reads a `[CLS]` sentence embedding would have gotten garbage (the position is never seen in training).
- The student's input format diverges from the teacher's at distillation time — a problem we caught now rather than later.

Fix: added the three overrides to `BaseHFTokenizer`. All five wrappers inherit, so one change covers everything. Verified on all 5 tokenizers: native `encode()` still matches HF `encode(add_special_tokens=False)` — so the existing `hf_compatibility_test` is unaffected. HF `encode(add_special_tokens=True)` now produces `[CLS] + content + [SEP]`. Special-token mask correctly returns 1 at both endpoints and 0 elsewhere.

### Eval metrics

`compute_metrics` reports `eval_loss`, `perplexity`, top-1 / top-5 accuracy at masked positions, and **bits per character (BPC)**. BPC normalizes the masked-position NLL by the source-text character count, making it **invariant to tokenization granularity** — the right metric for comparing the 5 tokenizers head-to-head, since perplexity by itself depends on each tokenizer's fertility. Implementation detail: `preprocess_logits_for_metrics` reduces each batch from `(B, L, V)` logits to top-5 indices + per-token NLL *before* Trainer accumulates across the eval set. Without that step the full-logits accumulator OOMs even at modest dev-set sizes — would have crashed any device, not just MPS. Standard pattern documented in HuggingFace's own `run_mlm.py`.

### Checkpoints

Standard HF behavior: every `--save-steps` steps, a `checkpoint-<N>/` directory is written containing model + optimizer/scheduler state + tokenizer files + RNG state. `--save-total-limit K` keeps only the K most recent; older auto-deleted. `--resume-from-checkpoint PATH` restores everything including the dataset cursor for true mid-epoch resume. `--load-best-model-at-end` saves the lowest-`eval_loss` checkpoint as the final model.

One nuance: `AutoTokenizer.from_pretrained(checkpoint_path)` does **not** work because our custom tokenizer classes aren't in HF's auto-discovery registry. Workaround: reload the tokenizer separately via `tokenization/registry.py:load_tokenizer(...).to_hf_tokenizer()`. The model itself reloads fine through `AutoModelForMaskedLM`. Inconvenience, not corruption — flagged in [PROJECT.md §7](PROJECT.md).

### Verification on M1

The full XLM-R-base architecture (~110 M params) was test-run on a 16 GB M1 Pro at `seq_len=256, batch_size=1, fp32, MPS`, 3 steps + a full-dev eval. Pipeline ran end-to-end. All metrics finite and consistent with random init (perplexity ≈ vocab size, top-1 ≈ 0.06%, BPC ≈ 3.1). One nit: `eval_loss` came back NaN while the independently-computed perplexity was finite — almost certainly an MPS-specific numerical hiccup at batch 1 with no batch averaging, not a real bug. Expected not to recur on CUDA with bf16 and batch 64.

### New dependencies

`torch>=2.2.0`, `datasets>=2.16.0`, `accelerate>=1.13.0` added to `pyproject.toml`. The first two were declared up-front; `accelerate` was added when Trainer demanded it (transformers 5.x requires `accelerate>=1.1.0` even on single-device runs).

### Remaining follow-ups

- Load the parallel data and implement the cross-lingual embedding distillation step.

---

## Morfessor vocab fix — dense IDs

While running the GPU MLM pretraining (next section), the morfessor and morfessor_semi runs crashed inside the embedding lookup with `vectorized_gather_kernel: index out of bounds`. Root cause was in `MorfessorTokenizer.train()` in `tokenization/morfessor.py`: the char and morpheme loops assigned IDs as `start_id + i`, but skipped candidates that already existed in the vocab (e.g. a single-character morpheme like `o`, `a`, `n` that was already added in the char phase). The loop index `i` still advanced, leaving **gaps** in the assigned ID range — so `len(vocab) < target_vocab_size` while `max(vocab.values()) == target_vocab_size − 1`.

Downstream effect: HF code (data collator's random-token replacement, and the model embedding size we set from `tokenizer.vocab_size`) read `len(vocab)`, but the tokenizer could still emit IDs up to the old target. The embedding was sized to ~15,765 entries, the encoder fed an ID of ~15,990 into it, CUDA gather went out of bounds. SPM-based wrappers were fine because SPM guarantees a dense piece-ID range; only the two pure-Morfessor wrappers had the gap.

Fix: switched both loops to a running counter that advances only when a token is actually inserted. IDs are now contiguous from 0 to `len(vocab) − 1`, which is the assumption the rest of the system makes everywhere else. `morfessor_v1` and `morfessor_semi_v1` were retrained on `data/processed/dsb/v1_train.txt`; both produced dense vocabs (15,765 and 16,000 entries respectively, with no missing IDs) and trained the encoder end-to-end without further changes. Other tokenizers were unaffected.

### Why this hadn't surfaced before

The intrinsic evaluator (`scripts/evaluate.py`) never embeds anything — it tokenizes, looks up IDs, and computes string-level metrics. The bug only triggers when an ID is used as an embedding index, which happens for the first time in the MLM pretraining pipeline. The intrinsic round-trip and HF-compatibility tests both passed because they don't touch an embedding layer.

---

## MLM pretraining — first dsb v1 round (all 5 tokenizers)

All 5 tokenizers were trained from random init for 10 epochs each on `v1_train.txt`, with the XLM-R-base architecture and identical hyperparameters (batch 64 × grad-accum 4, LR 5e-4, warmup 0.06, weight decay 0.01, bf16). One H100, ~21 minutes per tokenizer. Full results table and discussion in [EVALUATIONS.md §9](EVALUATIONS.md).

### Headline

| Tokenizer | BPC | rank |
|---|---|---|
| morfessor | 0.920 | 1 |
| morph_bpe | 0.986 | 2 |
| morfessor_semi | 0.993 | 3 |
| spm_unigram | ~1.06 | 4 |
| spm_bpe | 1.073 | 5 |

Unsupervised Morfessor beats every alternative by a clear margin (0.15 BPC over BPE). The BPE-on-morphemes hybrid (`morph_bpe`) and the annotation-supervised variant (`morfessor_semi`) both regress from plain Morfessor — adding complexity hurt on this corpus / vocab budget. Mechanistic reading in EVALUATIONS.md.

### Why this is the right comparison

`scripts/evaluate.py`'s intrinsic metrics rank tokenizers on compactness, coverage, and OOV — they don't say anything about *how well a downstream model can learn from each*. BPC, computed by `scripts/pretrain.py`'s `compute_metrics`, normalizes the masked-position NLL by source-text character count, which is tokenization-invariant. It's the metric that lets the 5 tokenizers be ranked head-to-head. Definition and reading guide in [METRICS.md §7](METRICS.md).

### Caveats

- The hsb equivalents (v3 and lww) have not been run, so the dsb ordering is not yet known to generalize across languages within Sorbian.
- The encoder is an init for cross-lingual distillation, not a final downstream model. Whether the BPC advantage propagates through distillation is the open question for the next step.

### Remaining follow-ups

- Same MLM comparison on hsb v3 and hsb lww.

---

## Cross-lingual embedding distillation (`scripts/prepare_parallel.py`, `scripts/distill.py`)

The downstream step the MLM encoders were built for. Teacher changed from the originally-planned `xlm-roberta-base` to **LaBSE** — a sentence-embedding model that already aligns translations across 109 languages (incl. German and Polish, but not dsb), so it gives a ready target space for bitext retrieval. The student is distilled to map **Polish** (a measurable West-Slavic stand-in for Lower Sorbian) onto LaBSE's German embeddings; the eventual dsb step is identical in form.

### Parallel data prep — `scripts/prepare_parallel.py`

A pair-aware analogue of `download_data.py`: every filter drops the *pair* so de–pl line alignment is preserved. Pipeline: NFC + control-strip → **either-side dedup** → GlotLID both sides (`deu_Latn` / `pol_Latn`) → terminal-punct both sides → length 3–80 + ratio ≤3 (Moses `clean-corpus-n` convention) → 90/5/5 split (seed 42). No Moses is written into the files — the student tokenizer applies it to Polish internally, and the teacher wants natural German.

The dedup is deliberately **either-side**, not tuple: a sentence may appear at most once on *each* side, so bitext retrieval has exactly one correct target per query. Run on Europarl de–pl: 579,166 → 529,522 clean pairs (biggest cut is the 35,801 either-side duplicates — Europarl procedural boilerplate). Output `data/processed/de-pl/{train,dev,test}.{de,pl}` = 476,569 / 26,476 / 26,477. Verified: alignment preserved, every sentence unique on both sides, zero train↔dev/test leakage.

### Distillation — `scripts/distill.py`

Reimers & Gurevych (2020) MSE distillation, cross-lingual term only: `MSE(student(polish), LaBSE(german))`. Design choices:

- **Mean pooling** for the student sentence embedding (R&G 2019 — mean > CLS, and an MLM-only model has no trained CLS sentence representation). 768-dim student = 768-dim LaBSE → no projection.
- **Cross-lingual term only.** The original method also keeps `student(source) → teacher(source)` to preserve the source language, but we never need the student to encode German, and its dsb tokenizer would fragment German badly — so German is left entirely to the teacher and the student only ever sees Slavic input. No collapse risk because the per-pair targets are distinct fixed vectors.
- **Teacher embeddings cached.** LaBSE is frozen, so `LaBSE(german)` is precomputed once to `.npy` (fp16) and reused across epochs and sweep runs — cheaper than re-running a 470 M-param forward each step, and frees VRAM. `--no-cache-teacher` opts out.
- **Selection on retrieval, not loss.** `metric_for_best_model` is bitext retrieval P@1 (pl→de) over a held-out pool, so early stopping doesn't reward a Europarl-memorizing LaBSE-clone. Optional `--ood-eval` (Tatoeba) switches selection to out-of-domain retrieval; the in-domain-vs-OOD gap is the overfitting diagnostic. `--max-train-pairs` runs a data-size sweep (train only; dev/test fixed).
- **Pretrain parity.** Same `--smoke` / device-auto / bf16 / checkpoint / tensorboard / Slurm-`unset RANK LOCAL_RANK` structure as `pretrain.py`; generic over all five tokenizers via the registry.

Validated end-to-end with `--smoke` on the M1 (LaBSE loads + embeds, MSE drops 0.32 → 0.12 over 20 steps, retrieval eval + baseline run, encoder + `sentence_encoder.json` saved and reloads via `AutoModel`). New dependency: `sentence-transformers`.

### Fix: normalize the student embedding before MSE (mean-collapse)

The first GPU distillation run collapsed: `eval_mse` crashed to ≈ 1/768 (the floor for predicting the target centroid) within a fraction of an epoch while retrieval P@1 stayed pinned at chance and grad-norm went to ~0. Cause: MSE onto **unit-norm** LaBSE targets with an **un-normalized** student is minimized by outputting one constant vector (the mean embedding) — low MSE, zero per-sentence discrimination. Fix: L2-normalize the pooled student embedding before the MSE, so `MSE(ŝ, t) = 2 − 2·cos(ŝ, t)` is exactly the retrieval objective and the centroid solution is no longer reachable (output is forced to unit norm). One-line change in `SentenceDistiller.forward`; also makes train consistent with the eval/inference path, which already normalizes. The cached teacher embeddings are unaffected, so the rerun skips the LaBSE precompute.

### Remaining follow-ups

- Add a Tatoeba de–pl OOD probe under `data/raw/tatoeba-de-pl/` and switch selection to it.
- Repeat distillation for the other four tokenizers.

---

## Morfessor distillation run + de–dsb mining eval (`scripts/mine_eval.py`)

Ran the morfessor distillation on Europarl de–pl (H100, early-stopped epoch 3.29). In-domain retrieval P@1 went 0 → 0.94 (pl→de) / 0.96 mean, dev ≈ test — the normalization fix above was required to get there. Full table in [EVALUATIONS.md §10](EVALUATIONS.md).

Added `scripts/mine_eval.py` to evaluate a distilled encoder on **de–dsb** (the real target): the student embeds dsb, LaBSE embeds German (cross-model — both live in LaBSE's space; how LaBSE-distillation is standardly evaluated). Two modes — `--parallel` (CSLS retrieval P@1 over a 1:1 set) and BUCC (CSLS + dynamic threshold + precision/recall/F1, the PaSeMiLL protocol) — plus a **threshold-free retrieval P@1** that separates retrieval failure from threshold mis-calibration. Metrics in [METRICS.md §9](METRICS.md).

**dsb result (zero-shot — dsb never in distillation):** chance → 0.136 on the clean 1,352-pair parallel set, but 0.005 F1 / 0.021 retrieval P@1 on the BUCC test (44k × 67k pool). The transfer signal is real but too weak to survive a realistic mining pool; the threshold-free number confirms it's a retrieval-quality limit, not just threshold calibration. A reviewer hypothesis that this was a pooling/normalization bug was ruled out: the identical embedding code gives 0.94 on Polish, so extraction is correct; the dsb weakness is genuine zero-shot transfer.

### Notes

- **CSLS similarity runs on CPU/CUDA, not MPS** — large pools (67k targets) OOM the MPS memory limit; CPU has the RAM and the embeddings are small.

### Remaining follow-ups

- Run the parallel + BUCC dsb eval across the other four distilled tokenizers (the tokenizer comparison on the real target).
- Direct de–dsb distillation (train on de–dsb parallel data instead of zero-shot via Polish) — the path to a non-trivial dsb encoder.

---

## Training-length ablation — eval-loss elbow undertrains retrieval

Supervisor question on seeing the loss curves: the MSE loss elbows ~1 epoch in, so does the long tail of training earn its keep, or could we stop at the elbow? Ran a dedicated 1-epoch distillation (`labse_distill_morfessor_1ep`, own LR schedule) and evaluated it on the same Polish test + dsb parallel/BUCC sets.

**Result:** at the elbow the MSE is already at its floor (0.00078 vs 0.00054 full), but every retrieval metric is only 20–70% of the full-run value — Polish test pl→de 0.484 vs 0.941, dsb parallel 0.040 vs 0.136, dsb BUCC F1 0.000 vs 0.005. The distillation MSE and downstream retrieval are **decoupled**: MSE converges ~1 epoch in, retrieval keeps climbing across the whole schedule. So selecting/early-stopping on eval loss undertrains retrieval by 2–5×; select on retrieval P@1 and run the full schedule. Full table in [EVALUATIONS.md §10](EVALUATIONS.md).

### `scripts/eval_forgetting.py` — intrinsic-capability (MLM) probe

Added to check whether distillation damages the encoder's pretrained Sorbian competence. Loads the original MLM head onto a distilled body and measures held-out Sorbian perplexity. **Caveat learned (important):** this frozen-head test is *not* a valid forgetting measure — distillation fine-tunes the whole body, so its token representations rotate and the old head can no longer read them, blowing up perplexity (measured 10.1 → 4400) even when nothing is forgotten. The non-zero dsb retrieval numbers already prove the Sorbian signal survives. The correct standard test is a **probe**: freeze the body, train a *fresh* head, and compare distilled-body vs original-body — not yet implemented. Kept only as a secondary diagnostic; retrieval P@1 is the primary capability metric.

---

## Distillation data source — WikiMatrix beats Europarl for dsb transfer

Re-distilled morfessor on WikiMatrix de–pl (`labse_distill_morfessor_wiki`) instead of Europarl, same training budget (37,235 steps — matched on gradient updates, not epochs, since WikiMatrix is smaller at 232k vs 476k pairs). Only the data source differs.

**Result:** WikiMatrix wins every dsb transfer metric — parallel mean 0.202 vs 0.136 (+49%), BUCC retrieval P@1 47/902 vs 19/902 (~2.5×), BUCC F1 0.0093 vs 0.005 — while scoring *lower* on in-domain Polish (test mean 0.795 vs 0.960). The inversion is the finding: Europarl's narrow parliamentary domain lets the student overfit its Polish distribution (high Polish retrieval, weaker generalization), while WikiMatrix's broad encyclopedic text transfers to Sorbian better. WikiMatrix is the source for the remaining tokenizers. Absolute numbers stay low (zero-shot-via-Polish ceiling); direct de–dsb distillation is still the real lever. Full table in [EVALUATIONS.md §10](EVALUATIONS.md).

---

## Tokenizer comparison on dsb transfer — all five distilled on WikiMatrix

Distilled all five encoders on WikiMatrix de–pl under the identical 37,235-step budget (`labse_distill_<tokenizer>_v2`), evaluated on the de–dsb parallel + BUCC test sets. Ranking by dsb transfer: **spm_unigram ≈ morfessor** (top; tied on the parallel set at 0.200/0.202, unigram ahead on BUCC F1 0.0142 vs 0.0093) > morph_bpe > spm_bpe > morfessor_semi (last). This differs from the intrinsic MLM/BPC ranking — the tokenizer that wins the MLM stage isn't the one that wins the distilled retrieval task. Full table in [EVALUATIONS.md §10](EVALUATIONS.md).

---

## Off-the-shelf ceilings — why the de–dsb scores are low

Ran the same CSLS eval with both sides embedded by one off-the-shelf model, to separate "task is hard" from "student is weak." **LaBSE** (retrieval-trained; no official dsb) scores parallel 0.630 / BUCC F1 0.341 — ~35× the best student's F1 — so the task is feasible and the student is undertrained. **Glot500** (covers dsb, but a raw masked-LM) scores near the floor (parallel 0.048), because raw MLM embeddings aren't retrieval-aligned — the determinant is cross-lingual retrieval training, not dsb coverage. LaBSE's 0.63 is the reference the student's distillation targets; motivates direct de–dsb distillation. Scoring note: `mine_eval.py` uses CSLS (subtract, k=20) + dynamic threshold per **PaSeMiLL** (Okabe et al., ComputEL 2025); the broader BUCC/LaBSE standard is ratio-margin (divide, k=4) — a sibling, not plain cosine. Full table in [EVALUATIONS.md §10](EVALUATIONS.md).

---

## Direct de–dsb distillation — all four tokenizers

Distilled each encoder directly on de–dsb parallel data (`data/processed/de-dsb`, 113,971 train pairs, `--tgt-lang dsb`) instead of the Polish proxy, same 37,235-step budget. PaSeMiLL parallel mean / BUCC F1: **morph_bpe 0.809 / 0.441**, spm_unigram 0.796 / 0.427, spm_bpe 0.696 / 0.287, morfessor 0.589 / 0.187. Large gain over zero-shot (~30×); three of four beat the LaBSE both-sides reference (0.630 / 0.341) on parallel — expected, since LaBSE is zero-shot on dsb while the students train on 113k de–dsb pairs. **Key finding: the tokenizer ranking is regime-dependent — no tokenizer is robustly best.** morfessor wins intrinsic MLM and is ~top zero-shot but is *last* on direct; morph_bpe is mid-pack earlier but *1st* on direct; spm_unigram is the only consistently strong one. So the MLM-stage winner is not the downstream winner, and the two distilled regimes disagree. Caveat: single seed; morph_bpe vs unigram (0.809 vs 0.796) is within likely noise, but morfessor's drop to last is a clear inversion. In-domain de–dsb test higher (morfessor 0.961); PaSeMiLL lower by domain shift. Supervised pairs verified leak-free (unsupervised pretraining-text overlap ~70%, constant across encoders — see §12). Full table in [EVALUATIONS.md §10](EVALUATIONS.md).

---

## Multilingual pretraining — data, tokenizers, metrics (encoders pending)

Set up the "does a related/paired language in pretraining help the dsb encoder" experiment. Added German
and Polish Leipzig-news sources to `LANG_REGISTRY` (`scripts/download_data.py`); new
`scripts/build_multilingual_corpus.py` builds **balanced** (each non-dsb language capped to dsb's
whitespace-token count) and **interleaved** (alternating lines) mixes → `data/processed/{de-dsb,pl-dsb,
de-pl-dsb}-mono/`. `scripts/pretrain.py` gained `--no-shuffle` (SequentialSampler) so the alternation is
preserved (masking stays random). Trained 12 **unified** tokenizers (4 types × 3 mixes, vocab = 16k ×
n-languages). Added three tokenizer metrics to `tokenization/evaluate.py` (chars-per-token, average-rank,
and cross-language `jsd_overlap`, the last via `scripts/evaluate.py --overlap`).

**Tokenizer-level findings (pre-training):** cross-language overlap JSD shows **Polish+dsb shares far
more than German+dsb** on all four types (e.g. unigram 0.468 vs 0.662) — predicting pl+dsb should help
dsb more; and Sorbian **kept its allocation** in the 32k/48k unified tokenizers vs the 16k monolingual
(validating "16k per language"). Motivating papers: Limisiewicz et al. (2023) — overlap helps same-script
sentence retrieval; Hämmerl et al. (2025) — literal overlap is valid for same-script (all our langs are
Latin), so no alignability metric needed. Note: the dsb pretraining side (`dsb/v2`) overlaps ~70–75% of
the PaSeMiLL Sorbian eval *sentences* (unsupervised only; supervised pairs clean; constant across
encoders). Full tables in [EVALUATIONS.md §12](EVALUATIONS.md). Encoders + distillation next.
