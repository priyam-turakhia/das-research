# Evaluation metrics

This document describes every metric the evaluation script reports. For each metric there are four parts:

1. **What it counts** — plain-language description of the quantity.
2. **Exactly how the code computes it** — function name, file location, and a walkthrough of the logic.
3. **What the numbers mean** — concrete interpretation of typical values.
4. **What's typical** — empirical ranges from low-resource subword tokenization for context.

A general caveat at the top: every metric in this document is a sanity check. They confirm the tokenizer is functioning correctly and give a rough feel for efficiency and coverage. None of them predicts how well a downstream language model or translation system will perform with the tokenization. That requires actually training a downstream model on each, which is out of scope for this pipeline.

All metric functions live in [tokenization/evaluate.py](../tokenization/evaluate.py).

---

## 1. Fertility (mean and standard deviation)

### What it counts

For every sentence in the evaluation set, fertility is the number of tokens the tokenizer produces divided by the number of whitespace-separated words in that sentence. "Fertility (mean)" is the average ratio across all sentences. "Fertility (std)" is the standard deviation of the same ratio.

### Exactly how the code computes it

Function: `compute_fertility`.

For each sentence, the function calls `sentence.split()` to count whitespace-separated words and `tokenizer.tokenize(sentence)` to get the token list, then stores the ratio `len(tokens) / len(words)`. After processing all sentences, it averages the ratios and computes the standard deviation by hand.

One subtlety worth knowing: every sentence contributes one ratio regardless of length. A 3-word sentence and a 50-word sentence weight the average equally, even though the longer sentence's ratio is statistically more reliable. This is standard practice but it means a small number of weird short sentences can pull the average around.

### What the numbers mean

- 1.0 → tokenizer leaves every word intact on average. One word, one token.
- 1.5 → about half the words get split into two pieces.
- 2.0 → every word becomes two pieces on average.
- 3.0 → words are aggressively fragmented.

Standard deviation tells you how variable the output length is from sentence to sentence. A std of 0.2 means most sentences sit close to the mean. A std of 1.0 means some sentences have much higher fertility than others — typically because some words were unfamiliar and got heavily fragmented while common ones stayed whole.

### What's typical

- Subword tokenizers on morphologically rich languages (Slavic, Finnish, Turkish): mean fertility 1.4 to 2.2 is normal at vocab sizes around 16k–32k.
- English: 1.2 to 1.6.
- Below 1.0 only happens with weird "word" definitions (e.g., counting punctuation differently).
- Above 2.5 usually means the vocabulary is too small for the language, or the training data doesn't match the evaluation domain.
- Standard deviation of 0.2 to 0.4 is the typical range. Above 0.6 means the tokenizer is inconsistent.

### Why it matters

Lower fertility means shorter token sequences for the downstream model — faster training, less memory, fewer position embeddings consumed. Lower standard deviation means more predictable sequence lengths, which helps with batching efficiency.

---

## 2. Unique tokens used and vocab coverage

### What it counts

"Unique tokens used" is the number of distinct token strings the tokenizer actually emits when run on the entire evaluation set. "Vocab coverage" is that number divided by the total vocabulary size, expressed as a percentage.

### Exactly how the code computes it

Function: `compute_unique_tokens`.

Maintains a Python `set`. Iterates over every sentence in the evaluation corpus. For each sentence, calls `tokenizer.tokenize(sentence)` and adds every emitted token string to the set. After all sentences are processed, the size of the set is the unique-token count. Coverage is then `unique_tokens / tokenizer.vocab_size`.

### What the numbers mean

- 99% coverage → the tokenizer uses almost every entry in its vocabulary on this evaluation set. The vocabulary is well-matched to the data.
- 90% coverage → about 10% of vocabulary entries don't appear here. They may still be useful elsewhere in the corpus — they're just not exercised on this split.
- 70% or lower → a meaningful chunk of vocabulary is unused on this set. Often points to a vocabulary that's too large for the data, or a mismatch between training and evaluation distributions.

### What's typical

- BPE and Unigram on representative dev sets: 95% to 99%. These algorithms explicitly select vocabulary entries to maximize corpus coverage, so this is by construction.
- Morfessor: typically 80% to 95%. Its vocabulary contains some morphemes selected for linguistic value rather than frequency, so a fraction will be too rare to appear in any given sample.
- Below 70% usually means something is off — wrong corpus, mismatched train/eval split, or undertrained model.

### Historical note

In versions of the pipeline before v3, Morfessor reported 130% coverage, which is structurally impossible. The cause was a `▁` prefix being attached at emission time, so `přinoški` and `▁přinoški` were both counted as distinct token strings even though only one was in the vocabulary. With `▁` removed in v3, coverage now reads naturally. See [CHANGELOG.md](CHANGELOG.md) for full context.

---

## 3. OOV rate

### What it counts

"OOV" stands for out-of-vocabulary. The function walks through every word in the evaluation set and counts the fraction of words where the tokenizer either emits an `[UNK]` token or collapses entirely to single-character pieces (the character-fallback path).

### Exactly how the code computes it

Function: `compute_oov_rate`.

Iterates over every word produced by `sentence.split()` for every sentence in the evaluation set. For each word, calls `tokenizer.tokenize(word)` and checks two conditions:

1. The literal string `[UNK]` appears anywhere in the token list. This is the explicit unknown-token signal.
2. Every token in the list is one character long *and* the original word was more than one character. This catches the character-fallback path: a word like `předsydstwom` becoming `[p, ř, e, d, s, y, d, s, t, w, o, m]`.

Either condition flags the word as OOV. Note the `len(word) > 1` guard — it stops legitimately one-character words (like `a`, `i`, `je`) from being counted as OOV when they tokenize to one character.

### What the numbers mean

- 0% to 0.5% → the tokenizer handles essentially all real text on this evaluation set. Genuine unknowns are rare.
- 1% to 5% → occasional unknown words. Usually proper names, foreign vocabulary, or technical terms not seen during training.
- 10% or higher → the tokenizer is failing on a meaningful fraction of input. Vocabulary too small, training data too narrow, or a setup problem (an early version of Morfessor in this project had 20% OOV due to incorrect post-hoc vocabulary capping — see [CHANGELOG.md](CHANGELOG.md)).

### What's typical

- Modern subword tokenizers on in-domain dev sets: under 1% is the norm.
- Cross-domain evaluation (train on news, test on social media): 2% to 5%.
- Wrong language or out-of-domain entirely: 10% or higher.

---

## 4. Round-trip

### What it counts

The fraction of sample sentences for which the token strings produced by `tokenize` exactly match the token strings produced by `encode` followed by per-ID vocabulary lookup.

### Exactly how the code computes it

Function: `round_trip_test`.

Samples 1,000 sentences from the evaluation set with a fixed random seed. Builds an ID-to-token reverse map from `tokenizer.get_vocab()`. For each sampled sentence:

```python
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)
round_tripped = [id_to_token.get(i, "[UNK]") for i in ids]
```

Counts the sentence as a pass if `tokens == round_tripped`.

### What the numbers mean

A passing test means: the two paths from text to token-string list (the direct `tokenize` path, and the `encode` → vocabulary lookup path) agree. A failing test means: the encode-then-lookup path produces different strings than direct tokenization, which would indicate a vocabulary or implementation inconsistency.

### Honest caveat about strength

This metric is structurally weaker than the alternative "encode then decode then compare to text" round-trip. For Morfessor specifically, `encode` is implemented as `[vocab[t] for t in tokenize(text)]` — meaning `tokenize` and `[vocab_lookup[id] for id in encode(text)]` are tautologically equal as long as the vocabulary map itself is correct. The test cannot fail unless the vocab is broken. For SentencePiece the test is meaningful but narrow: `tokenize` and `encode` are independent calls into the C++ library (one with `out_type=str`, the other with `out_type=int`), so the test catches the unlikely case where those two paths disagree.

What this metric does NOT verify is the original semantic round-trip — that `decode(encode(text))` recovers the original text. SentencePiece still passes that stronger property in practice; Morfessor cannot, by design (its flat ID stream does not carry word-boundary information; see [CHANGELOG.md](CHANGELOG.md)). When the eval reports 1000/1000 across all four tokenizers, the SPM and MorphBPE rows reflect both checks (MorphBPE's final encoding is SPM-BPE); the Morfessor row reflects only the weak vocab-consistency check.

The right way to report this honestly:
- **SPM**: passes the strong text-equality round-trip.
- **Morfessor**: passes the weak vocab-consistency check; the strong check is structurally inapplicable in the current design.

### What's typical

- Under the strong definition (text equality): 99%+ for SentencePiece. The 1% failures are usually Unicode normalization edge cases (e.g., the `…` ellipsis character normalized to three dots).
- Under the weak (current) definition: 100% always, unless there's an actual bug in the vocabulary or tokenizer plumbing.

---

## 5. HuggingFace compatibility

### What it counts

The fraction of sample sentences for which the integer ID stream produced by the native tokenizer exactly matches the integer ID stream produced by the HuggingFace wrapper.

### Exactly how the code computes it

Function: `hf_compatibility_test`.

Calls `tokenizer.to_hf_tokenizer()` to construct the wrapper. Samples 100 random sentences from the evaluation set with a fixed random seed. For each sentence, computes the ID list two ways: `tokenizer.encode(text)` (native) and `hf_tok.encode(text, add_special_tokens=False)` (wrapper). Counts how many of the 100 sentences have identical lists.

### What the numbers mean

100/100 is the only acceptable result. Anything less is a wrapper bug. Common ways this fails:

- The wrapper applies its own preprocessing (whitespace stripping, normalization) that the native side doesn't.
- The underlying library mishandles special tokens.
- The wrapper drops or adds tokens at sentence boundaries.

### Why it matters

Downstream model training pipelines typically use a HuggingFace tokenizer object. If the wrapper produces different tokens than the native side, the model will be trained on different inputs than your sanity-checking expects.

---

## 6. Side-by-side segmentation

### What it counts

This is not a numeric metric. It runs each tokenizer on a fixed list of 20 morphologically interesting Upper Sorbian words and prints the segmentations side-by-side, for human inspection. The same word list is used for the dsb tokenizers — these are hsb words, so for dsb tokenizers the table also shows how the language-agnostic algorithms handle out-of-domain morphology.

### Exactly how the code computes it

Function: `side_by_side_segmentation`. The list of words is defined as `SAMPLE_WORDS` at the top of `tokenization/evaluate.py` and includes terms like `dźěłaćerjo` (workers), `přewodźowanje` (accompanying), `najwjetšich` (largest), `předsydstwom` (chairmanship), and so on. For each word and each tokenizer, the function calls `tokenizer.tokenize(word)` and stores the result.

### What it's for

Linguistic sanity-checking. The numerical metrics tell you the tokenizer is fast, has low OOV, and is internally consistent — but they don't tell you whether the splits make sense morphologically. Looking at `najwjetšich → naj | wjetši | ch` (correct: superlative prefix + stem + case ending) versus `najwjetšich → najwjet | šich` (mostly arbitrary) is the kind of judgment that requires eyes on output, not numbers.

This is the primary way to evaluate whether Morfessor is producing morphologically meaningful segmentations. The sample segmentation table in the evaluation output is what you read for that.

---

## Reading the metrics together

No single metric tells you which tokenizer is best. Each captures a different property:

- Fertility tells you how compact the output is.
- Vocab coverage tells you how much of the learned vocabulary is exercised.
- OOV rate tells you how often the tokenizer fails to handle input.
- Round-trip tells you the encoding is internally consistent (in the weak version) or fully lossless (in the strong version, for SPM).
- HF compatibility tells you the production wrapper matches native behavior.
- Side-by-side segmentation tells you whether the splits make linguistic sense.

A tokenizer can win one metric and lose another. The "best" tokenizer for a downstream task is the one whose output a real model can learn from most efficiently — and that requires training a real model.

---

## 7. MLM pretraining metrics

These come from `scripts/pretrain.py`, not `scripts/evaluate.py`. They measure how well a from-scratch XLM-R-base-shaped encoder learns the language under each tokenization, on the same dev split, with everything else held fixed. They are what closes the gap that §1–§6 leave open: ranking the tokenizers by *downstream* quality, not just intrinsic properties. Implementation in `scripts/pretrain.py:make_compute_metrics`.

### `eval_loss`

Mean cross-entropy over the 15% of dev-token positions the collator masked. This is what HF Trainer optimizes against; `--load-best-model-at-end` picks the checkpoint with the lowest value. **Token-space** — not comparable across tokenizers, since each one has a different vocabulary and per-token prediction difficulty.

### `perplexity`

`exp(eval_loss)`. Same information in a more intuitive unit ("model's effective uncertainty per masked token, in vocab-size equivalents"). Reading guide:

- Random init at vocab 16,000: ≈ 16,000.
- Strong low-resource monolingual MLM at end of training: 10 – 50.
- English RoBERTa-base on full Wikipedia + BookCorpus: ≈ 4.

Same caveat as `eval_loss`: token-space, not cross-tokenizer fair.

### `top-1`, `top-5`

Fraction of masked positions where the true token is the argmax (`top-1`) or in the model's top 5 predictions (`top-5`). Easy to read but **inflated for tokenizers that produce fewer, more semantically chunked pieces** — Unigram and Morfessor look better here partly because their per-token prediction problem is statistically easier than BPE's, not only because the model learned more. Use these for sanity, not the cross-tokenizer ranking.

### `bpc` — bits per character

The cross-tokenizer fair metric. `(total NLL on masked positions / mlm_probability) / corpus_char_count / log(2)`.

The denominator is **source-text character count**, which is the same number regardless of which tokenizer you used — so BPC normalizes away each tokenizer's fertility. The `1 / mlm_probability` factor extrapolates the NLL from "the 15% of positions actually scored" to the whole text, assuming the masked sample is representative. Result is in bits per character.

Reading guide:

- 2.0+ — early training or essentially random.
- 1.2–1.5 — model has learned real linguistic structure; in the range of low-resource published monolingual MLMs (AfriBERTa, TigrinyaBERT etc.).
- 0.9–1.1 — strong low-resource result.
- ≤ 0.9 — saturated for this corpus size, or you have a lot of data.

For the comparison this project is set up to do — *which tokenization makes a better language model at fixed vocab budget on the same corpus* — **BPC is the metric to read.** Differences in `eval_loss` and perplexity across tokenizers can be entirely explained by tokenization-induced difficulty; differences in BPC cannot.

### Implementation note

`preprocess_logits_for_metrics` in the script reduces each batch from `(B, L, V)` logits to `(top-5 indices, per-token NLL)` *before* Trainer accumulates across the eval set. Without this step, accumulating full logits exhausts memory on any dev set of nontrivial size. The top-5 indices and the NLL at the label position are everything the compute function needs.

---

## 8. Cross-lingual distillation metrics

From `scripts/distill.py`. They measure how well the distilled sentence encoder places one language's sentences next to their translations in LaBSE's space. Evaluated over a fixed **retrieval pool** of `--eval-pool` pairs (default 1000 — the Tatoeba-standard difficulty; a larger pool is a harder retrieval problem because there are more distractors). Implementation in `scripts/distill.py:retrieval_metrics`.

### `mse`

Mean squared error between the student's pooled sentence embedding and the teacher's (LaBSE's) embedding of the parallel sentence — the training objective itself. Lower = the student reproduces LaBSE more faithfully. On its own it's a fidelity number, not a quality one: a model can drive MSE down by memorizing the training domain without learning a transferable mapping. Read it alongside retrieval, not instead of it.

### `p1_pl2de`, `p1_de2pl` — bitext retrieval precision@1

The standard cross-lingual sentence-embedding metric (Reimers & Gurevych's "Accuracy"; the Tatoeba/BUCC family). Embed every Polish sentence with the student and every German sentence with LaBSE, build the cosine-similarity matrix over the pool, and ask: for each query, is its true translation the single nearest neighbour?

- `p1_pl2de` — Polish query → German index. **This is the selection metric**, because it matches the eventual dsb→de use (Slavic input retrieved against the German anchor).
- `p1_de2pl` — German query → Polish index (the reverse).
- `p1_mean` — average of the two.

Reading guide (P@1 over a ~1000-pair pool):

- ~0.001 (≈ 1/pool) — chance; an un-distilled random-init student sits here.
- 0.3–0.6 — the student has learned a real cross-lingual mapping.
- 0.8+ — strong alignment; LaBSE itself is in this range on clean Tatoeba pairs.

### Why retrieval, not loss, is the selection metric

Selecting the checkpoint with the lowest training MSE rewards a student that has collapsed into a LaBSE-clone *on the training domain* — which need not transfer to dsb and would wash out tokenizer differences. Retrieval P@1 on a held-out pool measures whether translations actually end up nearest each other. The strongest guard is the **out-of-domain** probe (`--ood-eval`, e.g. Tatoeba de–pl): when present, selection runs on OOD retrieval and the **in-domain-minus-OOD P@1 gap** is the overfitting diagnostic — a large gap means the model memorized Europarl rather than learning a general Slavic→LaBSE map.
