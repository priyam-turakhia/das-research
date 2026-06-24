# Project summary

## Setup

Same code, same 16,000 vocabulary budget, same v3 pipeline (NFC, control char strip, boilerplate drop, GlotLID `__label__<lang>_Latn` ≥ 0.5, terminal punctuation `.!?…»"'`, length 3 to 100 whitespace words, Moses pretokenization with `lang='cs'`, dedup, 90 / 5 / 5 split with seed 42). Per-language configuration (sources, GlotLID label, output dir) lives in `LANG_REGISTRY` at the top of `scripts/download_data.py`; selecting the language is `--lang {hsb,dsb}`.

| Lang | Run | Sources | Sentences | Tokens | Train | Dev | Test |
|---|---|---|---|---|---|---|---|
| hsb | v3 | Leipzig + WMT22 | 703,595 | 11,416,063 | 633,235 | 35,179 | 35,181 |
| hsb | lww | Leipzig + Wiki + Witaj | 1,289,047 | 21,066,230 | 1,160,142 | 64,452 | 64,453 |
| dsb | v1 | Witaj + MT (de↔dsb train + dev) | 239,316 | 3,770,578 | 215,384 | 11,965 | 11,967 |

WMT22 is 92.8 % a subset of Witaj on hsb, hence the swap rather than addition. On dsb the two sources (Witaj, MT) have 0 exact-line overlap.

## Tokenizer training

* **SentencePiece BPE** and **Unigram**: `sentencepiece.SentencePieceTrainer.train` with `vocab_size=16000`, `model_type` set per subclass, `character_coverage=1.0`, `pad_id=0, unk_id=1, bos_id=-1, eos_id=-1`, `user_defined_symbols=["[PAD]","[CLS]","[SEP]","[MASK]"]`. Implemented in `tokenization/spm_base.py:BaseSPMTokenizer`.
* **Morfessor 2.0**: `BaselineModel(forcesplit_list=["-"], corpusweight=NumMorphCorpusWeight(num_morph_types=target))` where `target = 16000 − 5 special tokens − len(char inventory)`. `load_data(word_freq_list, count_modifier=lambda c: 1)` (canonical "ones" dampening). `train_batch()` with default `algorithm='recursive'`. Vocabulary built as `[5 specials] + [chars] + [top morphemes by frequency]`. No `▁` boundary marker. Implemented in `tokenization/morfessor.py:MorfessorTokenizer`.
* **MorphBPE** (hybrid): `BaselineModel(forcesplit_list=["-"])` with no vocabulary budget, `load_data(..., count_modifier=lambda c: 1)`, `train_batch()`. Morfessor's natural larger inventory is used as a fixed pre-segmenter. Corpus is pre-segmented into a temp file (each morpheme a whitespace unit), then SPM BPE trained on it with `vocab_size=16000`, `character_coverage=1.0`, same special-token settings as standalone BPE. Final vocabulary is the BPE vocabulary; Morfessor's morphemes are intermediate. Implemented in `tokenization/morph_bpe.py:MorphBPETokenizer`.

All four apply `moses_pretokenize` inside `tokenize` and `encode`, and `moses_detokenize` inside `decode`. Helper in `tokenization/pretokenize.py` (Moses with `lang='cs'`).

## Evaluation

`scripts/evaluate.py` loads every model via `tokenizer.load(path)` and calls `tokenization/evaluate.py:evaluate_tokenizer(tokenizer, corpus_path, name)` against the corresponding dev split.

| Metric | Function | Sample size | What it measures |
|---|---|---|---|
| Fertility (mean, std) | `compute_fertility` | full dev split | `len(tokenize(sentence)) / len(sentence.split())` averaged across sentences |
| Unique tokens, vocab coverage | `compute_unique_tokens` | full dev split | distinct token strings emitted; coverage is `unique / vocab_size` |
| OOV rate | `compute_oov_rate` | full dev split | fraction of words emitting `[UNK]` or collapsing to single chars (multi-char words only) |
| Round-trip | `round_trip_test` | 1000 sentences, seed 42 | `tokenize(text) == [id_to_token[i] for i in encode(text)]` |
| HF compatibility | `hf_compatibility_test` | 100 sentences, seed 42 | native `encode(text)` equals `hf_tok.encode(text, add_special_tokens=False)` |
| Sample segmentations | `side_by_side_segmentation` | 20 fixed Upper Sorbian words | qualitative inspection |

## Results

All tokenizers pass round-trip 1000 / 1000 and HF compatibility 100 / 100 in every run.

### hsb v3 (35,179 dev sentences)

| Metric | SPM BPE | SPM Unigram | Morfessor |
|---|---|---|---|
| Fertility mean | 1.343 | 1.509 | 1.598 |
| Fertility std | 0.223 | 0.254 | 0.184 |
| Vocab size | 15,995 | 15,995 | 15,758 |
| Unique tokens used | 15,762 | 15,843 | 14,104 |
| Vocab coverage | 98.5 % | 99.0 % | 89.5 % |
| OOV rate | 0.00 % | 0.03 % | 0.27 % |

### hsb lww (64,452 dev sentences)

| Metric | SPM BPE | SPM Unigram | Morfessor | MorphBPE |
|---|---|---|---|---|
| Fertility mean | 1.363 | 1.517 | 1.636 | 1.491 |
| Fertility std | 0.232 | 0.265 | 0.189 | 0.227 |
| Vocab size | 15,995 | 15,995 | 15,150 | 15,995 |
| Unique tokens used | 15,349 | 15,444 | 14,007 | 14,861 |
| Vocab coverage | 95.9 % | 96.5 % | 92.4 % | 92.9 % |
| OOV rate | 0.01 % | 0.03 % | 0.48 % | 0.01 % |

### dsb v1 (11,965 dev sentences)

| Metric | SPM BPE | SPM Unigram | Morfessor | MorphBPE |
|---|---|---|---|---|
| Fertility mean | 1.289 | 1.514 | 1.571 | 1.507 |
| Fertility std | 0.214 | 0.240 | 0.166 | 0.195 |
| Vocab size | 15,995 | 15,995 | 15,762 | 15,995 |
| Unique tokens used | 15,063 | 14,211 | 11,184 | 12,720 |
| Vocab coverage | 94.1 % | 88.8 % | 70.9 % | 79.5 % |
| OOV rate | 0.00 % | 0.01 % | 0.88 % | 0.00 % |

Raw output: `results/hsb/eval_dev_v3.txt`, `results/hsb/eval_dev_lww.txt` (3-way historical), `results/hsb/eval_dev_lww_4way.txt` (current with MorphBPE), `results/dsb/eval_dev_v1_4way.txt`. BPE has the shortest sequences, Morfessor the lowest fertility std and the most morphologically motivated splits (e.g. `najwjetšich` as `naj | wjetši | ch`), MorphBPE sits in between. The four-tokenizer ordering is preserved across hsb v3, hsb lww, and dsb v1 — the per-algorithm conclusions generalize across both corpus choice and language. Morfessor coverage drops sharply on dsb v1 (70.9 %) because the smaller dev set exercises a smaller fraction of the fixed 16k vocabulary.

### dsb v1 — semi-supervised Morfessor variant

A `SemiSupervisedMorfessorTokenizer` (`tokenization/morfessor_semi.py`) was trained on the same `v1_train.txt` corpus with 500 `stem ending` annotations extracted from the Apertium Lower Sorbian metadix (`apertium-dsb.dsb.metadix`) by `scripts/extract_dsb_morph_annotations.py`. `model.set_annotations(annotations)` is called before `train_batch`. An initial attempt kept `NumMorphCorpusWeight` and produced no meaningful change because the two adaptive tuners fought each other; the current variant drops `NumMorphCorpusWeight`, uses Morfessor's default fixed `corpusweight=1.0`, and enforces the 16k budget with the existing post-hoc cap. Sample size 500 was selected by a 13-configuration tuning sweep (see CHANGELOG). Output: `results/dsb/eval_dev_v1_morfessor_semi.txt`.

| Metric | morfessor_v1 | morfessor_semi_v1 |
|---|---|---|
| Fertility mean | 1.571 | **1.548** |
| Fertility std | **0.166** | 0.270 |
| Vocab size | 15,762 | 15,866 |
| Unique tokens used | 11,184 | **12,769** |
| Vocab coverage | 70.9 % | **80.5 %** |
| OOV rate | 0.88 % | **0.72 %** |
| Round-trip / HF | 1000/1000, 100/100 | 1000/1000, 100/100 |

Coverage +9.6 pp, OOV −18% relative, fertility mean slightly shorter. Fertility std widened (the one tradeoff: with fewer character fallbacks the length distribution becomes more bimodal — common dsb words stay short, rare/foreign words still fragment). Open follow-up: record the annotations path in the saved `tokenizer_config.json` for reproducibility, and try a richer-than-two-piece annotation source if one becomes available.

## Code layout

| File | Contents |
|---|---|
| `scripts/download_data.py` | Source download, full filter pipeline, splitter. `LANG_REGISTRY` at the top maps `--lang` to sources, GlotLID label, defaults. Flags: `--lang`, `--sources`, `--output-suffix`, `--glotlid-threshold`, `--min-length`, `--max-length`, `--terminal-punct`, `--skip-glotlid`. |
| `scripts/train.py` | Single tokenizer training entry point. Flags: `--method`, `--corpus`, `--vocab-size`, `--output`. |
| `scripts/evaluate.py` | Multi-tokenizer evaluation entry point. Flags: `--model-path` (repeatable), `--corpus`, `--type`. |
| `tokenization/base.py` | `BaseTokenizer` abstract class, `SPECIAL_TOKENS`. |
| `tokenization/pretokenize.py` | `moses_pretokenize`, `moses_detokenize`. |
| `tokenization/spm_base.py` | `BaseSPMTokenizer`, `SentencePieceHFTokenizer`. |
| `tokenization/spm_bpe.py`, `spm_unigram.py` | One-line subclasses. |
| `tokenization/morfessor.py` | `MorfessorTokenizer`, `MorfessorHFTokenizer`, `segment_word_with_vocab`. `train()` accepts an optional `annotations_path` for semi-supervised use. |
| `tokenization/morfessor_semi.py` | `SemiSupervisedMorfessorTokenizer` — thin subclass that wires a default annotations TSV through to the parent's `train()`. dsb-specific by default. |
| `tokenization/morph_bpe.py` | `MorphBPETokenizer`, `MorphBPEHFTokenizer` (hybrid Morfessor + BPE). |
| `scripts/extract_dsb_morph_annotations.py` | Extracts `surface\tstem ending` rows from an Apertium metadix; produces full and paradigm-balanced sampled TSVs. |
| `scripts/prepare_parallel.py` | Pair-aware cleaning + 90/5/5 split of a parallel (bitext) corpus for distillation. Either-side dedup, GlotLID both sides, terminal-punct, length+ratio. Outputs `data/processed/de-pl/{train,dev,test}.{de,pl}`. |
| `scripts/pretrain.py` | XLM-R-base MLM pretraining using any trained tokenizer. `--smoke` for laptop validation. Eval reports loss, perplexity, top-1/top-5 accuracy, bits per character. Encoder is intended as the student in a later cross-lingual embedding distillation step. |
| `scripts/distill.py` | Cross-lingual embedding distillation: `MSE(student(polish), LaBSE(german))`. Mean pooling, cached LaBSE teacher, retrieval-P@1 selection. Generic over tokenizer/encoder; `--smoke` + GPU parity with pretrain. |
| `tokenization/hf_base.py` | `BaseHFTokenizer` and `SpmBackedHFTokenizer` — shared special-token handling (`[CLS]…[SEP]` wrapping, special-tokens mask, segment IDs) inherited by every HF wrapper. |
| `tokenization/registry.py` | `TOKENIZER_TYPES`, `load_tokenizer()`, `detect_tokenizer_type()`, `write_config()`, `read_config()` — single source of truth for tokenizer dispatch and the `tokenizer_config.json` schema. |
| `tokenization/evaluate.py` | All metric functions, `EvaluationResult`, `print_comparison_table`. |

## MLM pretraining

`scripts/pretrain.py` trains an XLM-RoBERTa-base-shaped encoder from random init using any one of the 5 trained tokenizers. Same code path runs as a `--smoke` test on a laptop (tiny 2-layer model, MPS, ~10 s) and as the real training run on a CUDA GPU (12-layer / 768-hidden / ~98 M params after dropping the 250k → 16k vocab embedding, bf16). Eval reports `eval_loss`, perplexity, top-1/top-5 accuracy at masked positions, and **bits per character** — the last being the tokenization-invariant metric for comparing the 5 tokenizers at downstream LM quality. The resulting encoder is intended as the student in a later cross-lingual embedding distillation step against stock `FacebookAI/xlm-roberta-base`; MLM here is a sensible init, not the final training step. Bringing this online required fixing `BaseHFTokenizer` to wrap inputs with `[CLS]…[SEP]` and produce a correct special-tokens mask, plus fixing a vocab-construction bug in `MorfessorTokenizer.train()` that left ID gaps and broke the embedding lookup (both Morfessor variants had to be retrained — see CHANGELOG).

### dsb v1 results (10 epochs, identical hyperparameters per tokenizer, ~21 min each on a single H100)

| Tokenizer | eval_loss | perplexity | top-1 | top-5 | BPC |
|---|---|---|---|---|---|
| spm_bpe | 3.33 | 28.0 | 41.5 % | 58.2 % | 1.073 |
| spm_unigram (epoch 8.3 snapshot) | 2.75 | 15.7 | 50.7 % | 66.1 % | 1.058 |
| morph_bpe | 2.58 | 13.3 | 52.8 % | 69.5 % | 0.986 |
| morfessor_semi | 2.53 | 12.6 | 52.6 % | 70.3 % | 0.993 |
| **morfessor** | **2.32** | **10.3** | **56.1 %** | **73.2 %** | **0.920** |

Headline: unsupervised Morfessor wins by 0.15 BPC over BPE (the field's default), and beats both the BPE-on-morphemes hybrid (`morph_bpe`) and the annotation-supervised variant (`morfessor_semi`). Adding complexity hurt on this corpus and budget. Discussion in `docs/EVALUATIONS.md §9`; metric definitions in `docs/METRICS.md §7`.

## Cross-lingual embedding distillation (LaBSE teacher)

`scripts/distill.py` turns a pretrained encoder into a sentence encoder in LaBSE's space by minimizing `MSE(student(polish), LaBSE(german))` on de–pl Europarl (cross-lingual term of Reimers & Gurevych 2020). Polish is the measurable West-Slavic stand-in for Lower Sorbian (LaBSE covers Polish and German, not dsb); the eventual dsb step is identical in form. Mean pooling (768-dim student = 768-dim LaBSE, no projection); the frozen teacher's German embeddings are precomputed and cached. Selection is on **bitext retrieval P@1** (pl→de) over a held-out pool — not training loss — with an optional out-of-domain (Tatoeba) probe and a `--max-train-pairs` data-size sweep guarding against the student memorizing Europarl into a LaBSE-clone. Same `--smoke`/GPU parity as pretrain; generic over all five tokenizers.

Parallel data is prepped by `scripts/prepare_parallel.py` (pair-aware: either-side dedup, GlotLID deu/pol, terminal-punct, length 3–80 + ratio ≤3): Europarl de–pl 579,166 → 529,522 clean → `data/processed/de-pl/{train,dev,test}.{de,pl}` = 476,569 / 26,476 / 26,477, every sentence unique on both sides. Smoke-validated end-to-end; GPU results pending (see `docs/EVALUATIONS.md §10`, metrics in `docs/METRICS.md §8`).
