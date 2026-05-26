# Evaluation results

This document collects the evaluation numbers across runs, contextualizes them, and records sample segmentations. For raw evaluator output, see the corresponding files in `results/`. For metric definitions, see [METRICS.md](METRICS.md). For the reasoning behind each change, see [CHANGELOG.md](CHANGELOG.md).

There are two kinds of artifacts in this document:

- **Versioned rounds (v1, v2, v3)** — successive iterations of the *same* corpus (Leipzig + WMT22) with progressive pipeline improvements. v3 is the current main run.
- **Alternative corpus experiments (e.g. `lww`)** — parallel runs with *different* source combinations. Useful for measuring how much the choice of training data affects each tokenizer.

All evaluations are on the held-out dev split. Sentence counts vary because each run uses different source data and filter settings:

- v2 dev: 36,770 sentences (Leipzig + WMT22, less aggressive filtering)
- v3 dev: 35,179 sentences (Leipzig + WMT22, full v3 pipeline)
- lww dev: 64,452 sentences (Leipzig + Wiki + Witaj, full v3 pipeline)

Raw evaluator output:
- [results/eval_dev_v2.txt](../results/eval_dev_v2.txt)
- [results/eval_dev_v3.txt](../results/eval_dev_v3.txt)
- [results/eval_dev_lww.txt](../results/eval_dev_lww.txt) — initial 3-way lww run
- [results/eval_dev_lww_4way.txt](../results/eval_dev_lww_4way.txt) — current 4-way lww run including MorphBPE

---

## 1. Current results — v3

Source: [results/eval_dev_v3.txt](../results/eval_dev_v3.txt). Evaluated on 35,179 dev sentences.

| Metric | SPM BPE | SPM Unigram | Morfessor |
|---|---|---|---|
| Fertility (mean) | **1.343** | 1.509 | 1.598 |
| Fertility (std)  | 0.223 | 0.254 | **0.184** |
| Vocab size       | 15,995 | 15,995 | 15,758 |
| Unique tokens used | 15,762 | 15,843 | 14,104 |
| Vocab coverage   | 98.5% | **99.0%** | 89.5% |
| OOV rate         | **0.00%** | 0.03% | 0.27% |
| Round-trip pass  | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 |
| HF compatibility | 100 / 100 | 100 / 100 | 100 / 100 |

### Reading the numbers

- **Fertility.** All three sit between 1.34 and 1.60 tokens per word. BPE is shortest. Morfessor has the lowest standard deviation (0.184), meaning it is the most consistent in sequence length — this comes from segmenting predictably along morpheme boundaries.
- **OOV rate.** All under 0.3%. BPE doesn't produce a single OOV in 35,000 dev sentences. Morfessor's 0.27% is rare proper nouns falling back to character-level segmentation.
- **Vocab coverage.** SentencePiece variants are at 98–99% because their algorithms explicitly select vocabulary entries to maximize corpus coverage. Morfessor at 89.5% is lower because its morphemes are linguistically driven rather than frequency-driven, so a fraction of vocabulary entries are too rare to appear in any given sample. This is normal for Morfessor.
- **Round-trip and HF compatibility.** All three pass perfectly. The round-trip "100%" reading needs the caveat from [METRICS.md §4](METRICS.md): for SPM it reflects the strong text-equality property; for Morfessor it reflects only the weak vocab-consistency check, because Morfessor's decode is lossy by design.

### Which is best?

None of these numbers alone tells you. Each tokenizer wins a different metric: BPE has the shortest sequences, Morfessor has the most consistent sequence lengths, BPE has the lowest OOV, Morfessor has the most linguistically interpretable splits. The actual question — which tokenizer makes a better downstream language model — requires training a model on each and comparing perplexity or BLEU. That experiment is the next major step.

---

## 2. v2 results — for comparison

Source: [results/eval_dev_v2.txt](../results/eval_dev_v2.txt). Evaluated on 36,770 dev sentences.

| Metric | SPM BPE | SPM Unigram | Morfessor |
|---|---|---|---|
| Fertility (mean) | 1.596 | 1.604 | 1.733 |
| Fertility (std)  | 0.310 | 0.334 | 0.247 |
| Vocab size       | 15,995 | 15,995 | 15,825 |
| Unique tokens used | 15,831 | 15,888 | 20,699 |
| Vocab coverage   | 98.9% | 99.3% | 130.8% (artifact) |
| OOV rate         | 0.38% | 0.44% | 0.30% |
| Round-trip pass  | 997 / 1000 | 997 / 1000 | 1000 / 1000 |
| HF compatibility | 100 / 100 | 100 / 100 | 100 / 100 |

### Caveats specific to v2

- **The 130% Morfessor coverage** was a reporting artifact caused by the `▁` prefix being attached at emission time. Removed in v3.
- **The 3 SPM round-trip failures** were caused by SentencePiece normalizing `…` (one codepoint) to `...` (three periods) on decode. The cleaner v3 corpus removed those edge-case lines, and the metric definition changed in v3 — see [METRICS.md §4](METRICS.md).

---

## 3. v1 — Morfessor failure mode

The first version of the pipeline trained Morfessor non-canonically and produced a tokenizer that worked but had severe issues. We did not preserve a clean evaluation file from v1, but the headline numbers (recorded at the time) were:

| Metric (Morfessor v1) | Value |
|---|---|
| Fertility (mean) | 2.76 |
| Fertility (std) | 1.03 |
| OOV rate | 20.28% |
| HuggingFace compatibility | 0 / 100 |

Words like `předsydstwom` collapsed into 12 single characters because the morphemes Morfessor produced for them had been evicted from the post-hoc-capped vocabulary. See [CHANGELOG.md §v2](CHANGELOG.md) for the full story of what was wrong and how it was fixed.

The v1 SPM tokenizers (BPE and Unigram) were close to v2 numbers on fertility and OOV, but had **0/100 HuggingFace compatibility** because the wrapper conversion approach produced character-level output. This was also fixed in v2.

---

## 4. v2 → v3 contextual comparison

The v3 changes (GlotLID, terminal-punctuation filter, length filter, Moses pretokenization, Morfessor `▁` removal) shifted every metric in the same direction.

| Metric | BPE v2 → v3 | Unigram v2 → v3 | Morfessor v2 → v3 |
|---|---|---|---|
| Fertility (mean) | 1.596 → **1.343** | 1.604 → **1.509** | 1.733 → **1.598** |
| Fertility (std) | 0.310 → 0.223 | 0.334 → 0.254 | 0.247 → 0.184 |
| OOV rate | 0.38% → 0.00% | 0.44% → 0.03% | 0.30% → 0.27% |
| Vocab coverage | 98.9% → 98.5% | 99.3% → 99.0% | 130.8% → **89.5%** |

### What the changes mean

- **Fertility dropped sharply for BPE (-16%)** because Moses pretokenization stopped wasting vocabulary slots on `word.` and `word,` variants. Same root cause for Unigram and Morfessor, just less dramatic because their algorithms were less sensitive to the punctuation-fusion problem.
- **OOV is essentially zero** because the language filter dropped genuinely foreign content that used to cause character fallbacks. Morfessor's residual 0.27% is rare proper nouns.
- **Standard deviation dropped across the board** because the cleaned corpus has more uniform sentence quality (no fragments, no garbage, no mixed-language noise).
- **Morfessor's coverage going from 130% to 89.5%** reflects the removal of the `▁` reporting artifact. The 89.5% is the honest number.
- **Round-trip: SPM moved from 997/1000 to 1000/1000.** Two effects mixed: (a) the cleaned corpus removed the SPM-normalization edge cases that caused v2 failures, and (b) the round-trip metric definition changed in v3 from text-equality to token-stream-equality, which is a weaker check. SPM still passes the strong check when tested directly.

---

## 5. Sample segmentations

These come from the side-by-side segmentation step of the evaluator. They show how each tokenizer splits the same word and let you judge whether the splits are linguistically meaningful.

### v3 (current Leipzig + WMT22 run)

`dźěłaćerjo` (workers, dative/locative case):
- BPE: `▁dźěłaćerjo` (kept whole)
- Unigram: `▁dźěłaćerjo`
- Morfessor: `dźěłaćer | jo` (stem + case ending)

`přewodźowanje` (accompanying, nominalized verb):
- BPE: `▁přewo | dźowanje`
- Unigram: `▁přewod | ź | owanje`
- Morfessor: `přewodź | owanje` (verb stem + nominalizing suffix)

`najwjetšich` (largest, genitive plural):
- BPE: `▁najwjetšich`
- Unigram: `▁najwjetši | ch`
- Morfessor: `naj | wjetši | ch` (superlative prefix + stem + case ending — the linguistically correct analysis)

`předsydstwom` (by the chairmanship):
- BPE: `▁předsyd | stwom`
- Unigram: `▁předsydstwo | m`
- Morfessor: `předsyd | stwom` (matches BPE's split exactly)

### lww 4-way (current)

`dźěłaćerjo`:
- BPE: `▁dźěłaćerjo`
- Unigram: `▁dźěłaćerjo`
- Morfessor: `dźěłaćer | jo`
- MorphBPE: `▁dźěłaćer | ▁jo` (matches Morfessor's split, BPE confirms each piece is whole)

`přewodźowanje`:
- BPE: `▁přewo | dźowanje`
- Unigram: `▁pře | wodź | owanje`
- Morfessor: `přewodź | owanje`
- MorphBPE: `▁přewodź | ▁owanje` (matches Morfessor)

`najwjetšich`:
- BPE: `▁najwjetšich`
- Unigram: `▁najwjetši | ch`
- Morfessor: `naj | wjetši | ch`
- MorphBPE: `▁najwjetši | ▁ch` (BPE merged Morfessor's `naj` and `wjetši` because they co-occur)

The MorphBPE pieces all carry `▁` because Morfessor pre-segmentation produces a stream where each morpheme is a separate whitespace unit; SPM marks every such unit. The interesting hybrid behavior is the merging step: BPE compresses morpheme sequences that frequently co-occur, producing fewer pieces than standalone Morfessor while keeping morpheme boundaries where compression doesn't help.

### Pattern

SentencePiece (BPE and Unigram) keeps common whole words intact and splits only when a word is rare or novel. Morfessor splits more aggressively along morpheme boundaries even for common words. Both behaviors are "correct" — they reflect different philosophies. SPM optimizes for compression and reuse; Morfessor optimizes for linguistic decomposition.

For inspecting whether a tokenizer's splits make morphological sense, the side-by-side segmentation table is the right tool. It's the qualitative counterpart to the numeric metrics.

---

## 6. Metric notes specific to this dev set

A few observations worth recording:

- **The dev split is honest.** It is held out from training (90% train, 5% dev, 5% test, fixed seed). No data leakage between training and evaluation.
- **Morfessor training time scales with corpus size.** v3 corpus is ~5% smaller than v2 (after stricter filtering), and Morfessor training was correspondingly faster (~13 min vs ~18 min in v2).
- **The cleaned corpus is more uniform.** Filtering noise (fragments, foreign quotes, control characters) makes the dev set easier — fertility standard deviations dropped meaningfully across all three tokenizers.

---

## 7. Alternative corpus experiment — Leipzig + Wiki + Witaj (`lww`)

This is a parallel experiment, not a successor to v3. Same v3 pipeline, same vocabulary budget, same evaluation methodology — only the source corpora differ. Motivation: the analysis in §4 of [CHANGELOG.md](CHANGELOG.md) showed that WMT22 is essentially a subset of Witaj (92.8% overlap), so swapping WMT22 out for Witaj yields a much larger and more diverse training corpus.

### Corpus composition

- **Leipzig** (Leipzig Corpora Collection, mixed-domain): 300,000 raw lines.
- **Wiki** (Upper Sorbian Wikipedia dump): 47,758 raw lines.
- **Witaj** (Witaj educational publisher monolingual data): 1,071,723 raw lines.

After the full v3 pipeline (NFC, control chars, boilerplate, GlotLID at threshold 0.5, terminal-punct, length 3–100, Moses, dedup):

- Combined corpus: **1,289,047 sentences** (vs 703,595 in v3 — about 1.83× larger).
- Train / dev / test: 1,160,142 / 64,452 / 64,453.

Pairwise corpus overlaps (after pipeline, before merging):
- Leipzig ∩ Witaj: 0.3%
- Wiki ∩ Witaj: 2.7%
- Leipzig ∩ Wiki: ~0%

### Results — 4-way (current)

Source: [results/eval_dev_lww_4way.txt](../results/eval_dev_lww_4way.txt). Evaluated on 64,452 dev sentences.

| Metric | SPM BPE | SPM Unigram | Morfessor | MorphBPE |
|---|---|---|---|---|
| Fertility (mean) | **1.363** | 1.517 | 1.636 | 1.491 |
| Fertility (std)  | 0.232 | 0.265 | **0.189** | 0.227 |
| Vocab size       | 15,995 | 15,995 | 15,150 | 15,995 |
| Unique tokens used | 15,349 | 15,444 | 14,007 | 14,861 |
| Vocab coverage   | 95.9% | **96.5%** | 92.4% | 92.9% |
| OOV rate         | **0.01%** | 0.03% | 0.48% | **0.01%** |
| Round-trip pass  | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 |
| HF compatibility | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |

MorphBPE sits exactly where you'd expect: fertility between BPE (1.363) and Morfessor (1.636), at 1.491. Standard deviation similar to BPE (0.227 vs 0.232), since BPE's compression behavior dominates the variance pattern. OOV matches BPE at 0.01 percent because BPE's subword fallback handles anything Morfessor doesn't recognize. Vocab coverage 92.9 percent, slightly above standalone Morfessor — BPE compresses some morpheme sequences into shared tokens, so the same dev set exercises fewer distinct vocabulary entries than for SPM BPE.

The earlier 3-way run (without MorphBPE) is preserved at [results/eval_dev_lww.txt](../results/eval_dev_lww.txt) as historical record.

### v3 vs lww comparison

| Metric | BPE v3 → lww | Unigram v3 → lww | Morfessor v3 → lww |
|---|---|---|---|
| Fertility (mean) | 1.343 → 1.363 | 1.509 → 1.517 | 1.598 → 1.636 |
| Fertility (std)  | 0.223 → 0.232 | 0.254 → 0.265 | 0.184 → 0.189 |
| OOV rate         | 0.00% → 0.01% | 0.03% → 0.03% | 0.27% → 0.48% |
| Vocab coverage   | 98.5% → 95.9% | 99.0% → 96.5% | 89.5% → 92.4% |

### What the lww numbers tell us

- **Fertility went up slightly across the board** (about +1.5% to +2.4%). This is the expected outcome of training on more diverse text: the corpus contains more rare words and more domain variety, so a fixed 16k vocabulary can't keep as many high-frequency wholes intact. The increase is small enough that the tokenizers are still operating in the same regime.
- **OOV stayed low.** BPE went from 0.00% to 0.01%, Unigram unchanged. Morfessor went from 0.27% to 0.48% — still under half a percent, but proportionally larger. This reflects the dev set having a wider tail of rare proper nouns and technical terms.
- **Vocab coverage dropped for SPM** (98.5% → 95.9% for BPE) because the larger corpus produces a vocabulary tuned to a wider distribution; the dev set exercises slightly less of it. **Morfessor's coverage actually went up** (89.5% → 92.4%) because its final vocabulary is smaller (15,150 vs 15,758) — `NumMorphCorpusWeight` converged to a tighter morpheme set on the larger corpus, so a higher fraction of it appears in dev.
- **Standard deviation barely moved.** All three tokenizers retain their consistency profile: Morfessor still has the lowest std (0.189), SPM still cluster in the 0.23–0.27 range.
- **Round-trip and HF compatibility are perfect across the board** for both runs.

### Practical implication

The choice between v3 and lww as the "main" tokenizer set depends on what downstream task is being trained:

- **For an Upper Sorbian language model or embedding model**: lww is probably the better starting point. Bigger and more diverse training corpus, marginal fertility cost, better-rounded vocabulary. The 1.83× more training data also gives Morfessor more morpheme statistics to work with.
- **For an ablation comparison of the three algorithms**: either works. The relative ordering of the three tokenizers (BPE shortest, Morfessor most consistent and most morphologically interpretable) is preserved across both runs.

The fact that the relative behavior of the three tokenizers is stable across two quite different training corpora is itself useful evidence — it suggests the per-algorithm strengths and weaknesses we observed in v3 are not artifacts of one particular corpus.

---

## 8. What's missing — and what would close the gap

Everything in this document is a sanity check. None of these metrics tell you which tokenizer makes a better downstream model. The closing experiment, when someone wants to do it, is:

1. Train a small Transformer language model on each tokenization (same architecture, same training budget, same hyperparameters — so the tokenizer is the only variable).
2. Evaluate held-out perplexity on `hsb_test.txt`.
3. Compare.

A tokenizer with slightly worse fertility might still produce a better model if its tokens are more learnable. The proxy metrics here can't tell you that.
