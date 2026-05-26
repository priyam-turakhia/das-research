# Project summary

## Setup

Two parallel runs, same code, same 16,000 vocabulary budget, same v3 pipeline (NFC, control char strip, boilerplate drop, GlotLID `__label__hsb_Latn` ≥ 0.5, terminal punctuation `.!?…»"'`, length 3 to 100 whitespace words, Moses pretokenization, dedup, 90 / 5 / 5 split with seed 42).

| Run | Sources | Sentences | Tokens | Train | Dev | Test |
|---|---|---|---|---|---|---|
| v3 | Leipzig + WMT22 | 703,595 | 11,416,063 | 633,235 | 35,179 | 35,181 |
| lww | Leipzig + Wiki + Witaj | 1,289,047 | 21,066,230 | 1,160,142 | 64,452 | 64,453 |

WMT22 is 92.8 % a subset of Witaj, hence the swap rather than addition.

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

All three tokenizers pass round-trip 1000 / 1000 and HF compatibility 100 / 100 in both runs.

### v3 (35,179 dev sentences)

| Metric | SPM BPE | SPM Unigram | Morfessor |
|---|---|---|---|
| Fertility mean | 1.343 | 1.509 | 1.598 |
| Fertility std | 0.223 | 0.254 | 0.184 |
| Vocab size | 15,995 | 15,995 | 15,758 |
| Unique tokens used | 15,762 | 15,843 | 14,104 |
| Vocab coverage | 98.5 % | 99.0 % | 89.5 % |
| OOV rate | 0.00 % | 0.03 % | 0.27 % |

### lww (64,452 dev sentences)

| Metric | SPM BPE | SPM Unigram | Morfessor | MorphBPE |
|---|---|---|---|---|
| Fertility mean | 1.363 | 1.517 | 1.636 | 1.491 |
| Fertility std | 0.232 | 0.265 | 0.189 | 0.227 |
| Vocab size | 15,995 | 15,995 | 15,150 | 15,995 |
| Unique tokens used | 15,349 | 15,444 | 14,007 | 14,861 |
| Vocab coverage | 95.9 % | 96.5 % | 92.4 % | 92.9 % |
| OOV rate | 0.01 % | 0.03 % | 0.48 % | 0.01 % |

Raw output: `results/eval_dev_v3.txt`, `results/eval_dev_lww.txt` (3-way historical), `results/eval_dev_lww_4way.txt` (current with MorphBPE). BPE has the shortest sequences, Morfessor the lowest fertility std and the most morphologically motivated splits (e.g. `najwjetšich` as `naj | wjetši | ch`), MorphBPE sits in between (fertility 1.491, OOV matching BPE at 0.01% via BPE's subword fallback).

## Code layout

| File | Contents |
|---|---|
| `scripts/download_data.py` | Source download, full filter pipeline, splitter. Flags: `--sources`, `--output-suffix`, `--glotlid-threshold`, `--min-length`, `--max-length`, `--terminal-punct`, `--skip-glotlid`. |
| `scripts/train.py` | Single tokenizer training entry point. Flags: `--method`, `--corpus`, `--vocab-size`, `--output`. |
| `scripts/evaluate.py` | Multi-tokenizer evaluation entry point. Flags: `--model-path` (repeatable), `--corpus`, `--type`. |
| `tokenization/base.py` | `BaseTokenizer` abstract class, `SPECIAL_TOKENS`. |
| `tokenization/pretokenize.py` | `moses_pretokenize`, `moses_detokenize`. |
| `tokenization/spm_base.py` | `BaseSPMTokenizer`, `SentencePieceHFTokenizer`. |
| `tokenization/spm_bpe.py`, `spm_unigram.py` | One-line subclasses. |
| `tokenization/morfessor.py` | `MorfessorTokenizer`, `MorfessorHFTokenizer`, `segment_word_with_vocab`. |
| `tokenization/morph_bpe.py` | `MorphBPETokenizer`, `MorphBPEHFTokenizer` (hybrid Morfessor + BPE). |
| `tokenization/evaluate.py` | All metric functions, `EvaluationResult`, `print_comparison_table`. |
