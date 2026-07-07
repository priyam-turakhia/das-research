# Evaluation results

This document collects the evaluation numbers across runs, contextualizes them, and records sample segmentations. For raw evaluator output, see the corresponding files in `results/`. For metric definitions, see [METRICS.md](METRICS.md). For the reasoning behind each change, see [CHANGELOG.md](CHANGELOG.md).

There are three kinds of artifacts in this document:

- **Versioned rounds (v1, v2, v3)** — successive iterations of the *same* hsb corpus (Leipzig + WMT22) with progressive pipeline improvements. v3 is the current main hsb run.
- **Alternative corpus experiments (e.g. `lww`)** — parallel runs with *different* source combinations within the same language. Useful for measuring how much the choice of training data affects each tokenizer.
- **Cross-language runs (`dsb v1`)** — the Lower Sorbian module's first dataset under the same pipeline. Useful for checking whether per-algorithm conclusions generalize across the two Sorbian variants.

All evaluations are on the held-out dev split. Sentence counts vary because each run uses different source data and filter settings:

- hsb v2 dev: 36,770 sentences (Leipzig + WMT22, less aggressive filtering)
- hsb v3 dev: 35,179 sentences (Leipzig + WMT22, full v3 pipeline)
- hsb lww dev: 64,452 sentences (Leipzig + Wiki + Witaj, full v3 pipeline)
- dsb v1 dev: 11,965 sentences (Witaj + MT, full v3 pipeline)

Raw evaluator output:
- [results/hsb/eval_dev_v2.txt](../results/hsb/eval_dev_v2.txt)
- [results/hsb/eval_dev_v3.txt](../results/hsb/eval_dev_v3.txt)
- [results/hsb/eval_dev_lww.txt](../results/hsb/eval_dev_lww.txt) — initial 3-way lww run
- [results/hsb/eval_dev_lww_4way.txt](../results/hsb/eval_dev_lww_4way.txt) — current 4-way lww run including MorphBPE
- [results/dsb/eval_dev_v1_4way.txt](../results/dsb/eval_dev_v1_4way.txt) — dsb 4-way run

---

## 1. Current results — v3

Source: [results/hsb/eval_dev_v3.txt](../results/hsb/eval_dev_v3.txt). Evaluated on 35,179 dev sentences.

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

Source: [results/hsb/eval_dev_v2.txt](../results/hsb/eval_dev_v2.txt). Evaluated on 36,770 dev sentences.

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

Source: [results/hsb/eval_dev_lww_4way.txt](../results/hsb/eval_dev_lww_4way.txt). Evaluated on 64,452 dev sentences.

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

The earlier 3-way run (without MorphBPE) is preserved at [results/hsb/eval_dev_lww.txt](../results/hsb/eval_dev_lww.txt) as historical record.

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

## 8. Lower Sorbian (`dsb`) — `v1` corpus

A parallel run for Lower Sorbian. Same v3 pipeline, same vocabulary budget (16,000), same evaluation methodology — different language, different (smaller) corpus.

### Corpus composition

- **Witaj** (dsb monolingual): 120,500 raw lines.
- **MT train + dev** (`train.de-dsb.dsb` + `dev.de-dsb.dsb`, dsb side of de↔dsb MT pair from TUM-NLP `llms-limited-resources2025`): 171,963 + 4,000 raw lines.

After the full v3 pipeline (NFC, control chars, GlotLID at threshold 0.5 with `__label__dsb_Latn`, terminal-punct, length 3–100, Moses, dedup):

- Combined corpus: **239,316 sentences** (3.77M tokens). About one-third the size of hsb v3.
- Train / dev / test: 215,384 / 11,965 / 11,967.

Exact-line overlap between Witaj and MT train was 0 — the two sources are independent (unlike the hsb side, where Witaj subsumes ~93% of WMT22).

### Results — 4-way

Source: [results/dsb/eval_dev_v1_4way.txt](../results/dsb/eval_dev_v1_4way.txt). Evaluated on 11,965 dev sentences.

| Metric | SPM BPE | SPM Unigram | Morfessor | MorphBPE |
|---|---|---|---|---|
| Fertility (mean) | **1.289** | 1.514 | 1.571 | 1.507 |
| Fertility (std)  | 0.214 | 0.240 | **0.166** | 0.195 |
| Vocab size       | 15,995 | 15,995 | 15,762 | 15,995 |
| Unique tokens used | 15,063 | 14,211 | 11,184 | 12,720 |
| Vocab coverage   | **94.1%** | 88.8% | 70.9% | 79.5% |
| OOV rate         | **0.00%** | 0.01% | 0.88% | **0.00%** |
| Round-trip pass  | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 |
| HF compatibility | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |

### hsb lww vs dsb v1

| Metric | BPE hsb lww → dsb v1 | Morfessor hsb lww → dsb v1 |
|---|---|---|
| Fertility (mean) | 1.363 → 1.289 | 1.636 → 1.571 |
| OOV rate | 0.01% → 0.00% | 0.48% → 0.88% |
| Vocab coverage | 95.9% → 94.1% | 92.4% → **70.9%** |

### What the dsb numbers tell us

- **Fertility dropped across the board** (BPE 1.36 → 1.29, Morfessor 1.64 → 1.57). The smaller corpus has more high-frequency whole words that fit a 16k vocab intact, plus dsb morphology trends shorter on average than hsb in this sample. Not a corpus-quality effect — round-trip and HF compat are perfect.
- **Morfessor OOV roughly doubled** (0.48% → 0.88%). Fewer training tokens per morpheme means more rare morphemes get evicted from the final vocabulary, so the character-fallback path triggers more often on dev. Still under 1%.
- **Morfessor vocab coverage dropped sharply** (92.4% → 70.9%). Expected: with a smaller dev set the fixed 16k vocab has many entries that never appear. Coverage scales with dev-set size as much as with corpus quality.
- **BPE and MorphBPE both hit 0.00% OOV.** BPE's subword fallback eliminates character-level fallbacks at the cost of slightly longer sequences. Same pattern as in hsb.
- **The four-tokenizer ordering is preserved.** BPE shortest, Morfessor lowest std and most morphologically interpretable, MorphBPE between. Same conclusion the hsb v3 / lww comparison reached, now across a language boundary as well as a corpus boundary.
- **Round-trip and HF compatibility are perfect** for all four tokenizers. The language-agnostic pipeline produced clean tokenizers without per-language adjustment.

### Sample segmentations (test words also seen in the hsb side-by-side table)

The dsb tokenizers were inspected on the same 20 fixed words used for hsb (these are hsb words; some are also valid dsb forms, others are out-of-domain for the dsb training set). Selected examples from [results/dsb/eval_dev_v1_4way.txt](../results/dsb/eval_dev_v1_4way.txt):

`předsydstwom`:
- BPE: `▁p | ř | ed | sy | d | stwom`
- Unigram: `▁p | ř | ed | sy | d | stwo | m`
- Morfessor: `před | sy | d | stwom`
- MorphBPE: `▁p | ř | ed | ▁s | ▁y | d | ▁stwom`

`zwjazkoweje`:
- BPE: `▁z | wja | z | koweje`
- Unigram: `▁z | wja | z | k | oweje`
- Morfessor: `zwjazk | oweje`
- MorphBPE: `▁zwja | zk | ▁oweje`

The fragmentation on hsb-only words is visible in the BPE and Unigram rows (more single-char pieces than for native dsb words) — the words aren't in the dsb training distribution, so the subword fallback kicks in. This is the expected behavior, not a defect.

### Caveat

Cross-language comparisons (hsb vs dsb) are observational, not controlled. The corpora differ in size (~5×), source mix, and underlying language. Within-language tokenizer comparisons are unaffected.

### Semi-supervised Morfessor variant

A semi-supervised Morfessor variant (`morfessor_semi_v1`) was trained on the same `data/processed/dsb/v1_train.txt` corpus with 500 word-level `stem ending` annotations extracted from the Apertium Lower Sorbian metadix (`apertium-dsb.dsb.metadix`). `model.set_annotations(annotations)` is called before `train_batch`. An initial attempt kept the baseline's `NumMorphCorpusWeight` setup and barely moved any metric (fertility 1.571 → 1.591, coverage 70.9% → 70.4%); the two adaptive weight updaters (`NumMorphCorpusWeight` and the annotation-weight tuner from `set_annotations`) were fighting each other during training. The current variant drops `NumMorphCorpusWeight`, uses Morfessor's default fixed `corpusweight=1.0`, and lets the annotation tuner adapt alone. The 16k vocabulary budget is enforced by the existing post-hoc top-N morpheme cap. The sample size of 500 was selected by a 13-configuration tuning sweep (sample sizes 100/500/5k/10k/20k × paradigm-balanced/frequency-weighted strategies, plus a weight sub-sweep at 500); see [CHANGELOG.md](CHANGELOG.md) for the sweep observations.

Source: [results/dsb/eval_dev_v1_morfessor_semi.txt](../results/dsb/eval_dev_v1_morfessor_semi.txt). Evaluated on `v1_dev.txt` (11,965 sentences).

| Metric | morfessor_v1 | morfessor_semi_v1 |
|---|---|---|
| Fertility (mean) | 1.571 | **1.548** |
| Fertility (std)  | **0.166** | 0.270 |
| Vocab size | 15,762 | 15,866 |
| Unique tokens used | 11,184 | **12,769** |
| Vocab coverage   | 70.9% | **80.5%** |
| OOV rate         | 0.88% | **0.72%** |
| Round-trip / HF compat | 1000/1000, 100/100 | 1000/1000, 100/100 |

### What the dsb semi-supervised numbers tell us

- **Vocab coverage jumped 9.6 pp** (70.9% → 80.5%). The annotations pushed Morfessor toward morphemes that actually appear in real text — substantially more of the learned vocabulary is exercised by the dev set.
- **OOV dropped ~18% relative** (0.88% → 0.72%). Fewer character fallbacks, matching the coverage gain.
- **Fertility (mean) is slightly shorter** (1.571 → 1.548). Dropping `NumMorphCorpusWeight` did not blow up the budget — the post-hoc cap held.
- **Fertility (std) widened** (0.166 → 0.270). The tradeoff: with fewer character fallbacks the length distribution is more bimodal — common dsb words stay short, rare/foreign words still fragment heavily. Whether this hurts downstream depends on the model.
- **Round-trip and HF compatibility tied at perfect.**

The side-by-side table in the raw eval output shows visibly more fragmentation than the baseline on the 10 hsb sample words used by the evaluator. This is a dsb tokenizer being asked to handle out-of-domain hsb morphology; the in-distribution dsb performance is captured by the numbers above.

### Remaining follow-ups

- Record the annotations path in the saved `tokenizer_config.json` so a future reader can tell which annotation file produced the model.
- If a richer-than-two-piece annotation source becomes available, the result is likely to improve further.

---

## 9. dsb MLM pretraining (v1) — closing the proxy-metric gap

The experiment §10 of earlier versions used to flag as missing has now been run on dsb v1. For each of the 5 trained tokenizers, an XLM-RoBERTa-base-shaped encoder (12 layers, 768 hidden, 12 heads, FFN 3072, ~98 M params after vocab embedding) was trained from random init on `data/processed/dsb/v1_train.txt` (215,384 sentences) for 10 epochs with `--batch-size 64 --grad-accumulation-steps 4 --learning-rate 5e-4 --warmup-ratio 0.06 --weight-decay 0.01 --bf16`. Eval every 500 steps on `v1_dev.txt`. Same training protocol for every tokenizer — tokenization is the only variable. Each run was ~21 min on a single H100. Script: `scripts/pretrain.py`. Metric definitions in [METRICS.md §7](METRICS.md).

### Final eval results (best checkpoint by `eval_loss`)

| Tokenizer | eval_loss | perplexity | top-1 | top-5 | BPC |
|---|---|---|---|---|---|
| spm_bpe | 3.33 | 28.0 | 41.5 % | 58.2 % | 1.073 |
| spm_unigram (epoch 8.3 snapshot) | 2.75 | 15.7 | 50.7 % | 66.1 % | 1.058 |
| morph_bpe | 2.58 | 13.3 | 52.8 % | 69.5 % | 0.986 |
| morfessor_semi | 2.53 | 12.6 | 52.6 % | 70.3 % | 0.993 |
| **morfessor** | **2.32** | **10.3** | **56.1 %** | **73.2 %** | **0.920** |

The Unigram row is the latest snapshot the conversation log retained — the run completed to 10 epochs but only the mid-run number is preserved. The final BPC is within ~0.02 of the snapshot based on trajectory.

### Reading the numbers

**BPC is the column to read for the cross-tokenizer ranking.** Token-space metrics (loss, perplexity, top-k) are inflated for tokenizations that produce fewer, coarser pieces — Unigram and Morfessor look disproportionately good in those columns partly because their per-token prediction problem is easier than BPE's. BPC normalizes that out by dividing by source-text character count.

- **Morfessor wins by 0.15 bits/character over BPE** — a large gap. In compression terms, the Morfessor model encodes the dev set about 14 % more efficiently. Cleanly attributable to the tokenization, since training was identical.
- **MorphBPE and morfessor_semi are roughly tied at second place**, both clearly behind plain Morfessor. The BPE-on-morphemes merging step undoes some of the morpheme-boundary signal Morfessor provides; the supervised annotations push segmentation toward finer pieces than the unsupervised MDL optimum, making the LM job harder.
- **Unigram beats BPE by 0.015 BPC**. Known result in the tokenizer literature for morphologically rich, small-vocab settings (Kudo 2018) — Unigram's probabilistic segmentation handles morphological variation more gracefully than BPE's greedy merges.
- **BPE last**. Frequency-driven merges shred morpheme boundaries; the LM has to relearn what the tokenizer destroyed.

### Headline finding

For Lower Sorbian at a 16,000-token vocabulary budget on this corpus, **unsupervised Morfessor segmentation produces the best LM**. Hybridizing with BPE (MorphBPE) and adding annotation supervision (morfessor_semi) both make things worse. This is a clean inversion of the assumption that more sophisticated pipelines should win.

### What this does not establish

- **Generalization to hsb.** Same experiment hasn't been run on hsb yet. The intrinsic-metric ordering is preserved across hsb v3, hsb lww, and dsb v1 — but intrinsic ordering and MLM ordering are not the same question.
- **Generalization to downstream tasks.** A better MLM init is necessary, not sufficient, for downstream performance. The encoder is intended to become the student in cross-lingual embedding distillation against stock `xlm-roberta-base`; that step will test whether the BPC advantage propagates into the distilled space.
- **Annotation supervision in general.** morfessor_semi underperforming here is specific to (a) the 500-row paradigm-balanced metadix sample and (b) the unsupervised MDL optimum being already strong on this corpus. A richer-than-two-piece annotation source could change the answer.

Raw training logs (per-step train loss, eval at every 500 steps, BPC trajectory) live in each model's `models/dsb/xlmr_<tokenizer>_v1/runs/` directory as TensorBoard event files.

---

## 10. Cross-lingual embedding distillation — morfessor (first run)

The downstream step (`scripts/distill.py`): distill the dsb morfessor encoder into a sentence encoder in LaBSE's space by minimizing `MSE(student(polish), LaBSE(german))` on de–pl Europarl. Polish is the measurable West-Slavic stand-in for Lower Sorbian. Teacher = `sentence-transformers/LaBSE` (768-dim, frozen, cached). Protocol and metric definitions in [METRICS.md §8–9](METRICS.md). One H100, early-stopped at epoch 3.29 (of 5) on the dev retrieval metric.

### In-domain (Europarl de–pl) retrieval P@1, ~1000-pair held-out pool

| | pl→de | de→pl | mean | MSE |
|---|---|---|---|---|
| baseline (un-distilled student) | 0.001 | 0.002 | 0.0015 | 0.317 |
| distilled — dev | 0.945 | 0.983 | 0.964 | 0.00054 |
| distilled — test | 0.941 | 0.979 | 0.960 | 0.00054 |

Chance → 0.94 on the hard direction; dev ≈ test (no overfitting). This proves the method works. **One required fix:** the student embedding must be L2-normalized before the MSE, or the loss collapses to predicting the target centroid (MSE ≈ 1/768, chance retrieval) — see [CHANGELOG.md](CHANGELOG.md). `de→pl > pl→de` throughout is the standard hubness asymmetry (cancelled by CSLS in the mining eval below).

### dsb transfer (zero-shot — dsb was never in distillation)

dsb retrieval leans entirely on the dsb MLM-pretraining + Polish↔dsb similarity; the distillation never saw a dsb sentence. Evaluated with `scripts/mine_eval.py` (student embeds dsb, LaBSE embeds German — both in LaBSE space; CSLS scoring).

**(a) Clean parallel retrieval** — PaSeMiLL `parallel_de-dsb_*`, 1,352 aligned pairs, CSLS P@1:

| | dsb→de | de→dsb | mean |
|---|---|---|---|
| un-distilled student | 0.000 | 0.001 | 0.0004 (chance) |
| distilled | 0.107 | 0.164 | **0.136** |

Distillation moved dsb from chance to a real (if modest) signal — confirming zero-shot transfer happens.

**(b) BUCC test mining** — PaSeMiLL benchmark, 44,615 dsb × 67,512 de pool, 901 gold pairs:

| metric | value |
|---|---|
| retrieval P@1 (no threshold) | 19 / 902 = **0.021** |
| BUCC F1 (CSLS + dynamic threshold) | **0.005** (P 0.004, R 0.007, tp 6) |

Over the realistic 67k-sentence pool the signal mostly evaporates: only 2.1% of gold dsb sentences rank their true German first even with no threshold, and the dynamic threshold further cuts that to 6 confirmed pairs. The threshold-free number isolates the cause — it's primarily a **retrieval-quality** limit (the model can't rank well in a big pool), not just threshold mis-calibration.

### Reading it

Zero-shot-via-Polish gives dsb a weak but genuine alignment (chance → 13.6% on a 1.3k pool, 2.1% on a 67k pool), **far too weak for real bitext mining**. The numbers are consistent across pool sizes — the weak signal simply doesn't survive 50× more distractors. The clear next lever is **distilling on actual de–dsb parallel data** rather than zero-shot transfer.

### Training length ablation — the eval-loss elbow undertrains retrieval

The MSE loss curve elbows early (steep drop to ~1 epoch, then a long slow tail). Natural question: is the tail worth it, or could we stop at the elbow? Answer: **stop at the elbow and every retrieval metric collapses, even though the loss barely differs.** A dedicated 1-epoch run (`labse_distill_morfessor_1ep`, own LR warmup+decay) vs the full run:

| metric | 1 epoch (elbow) | full run | elbow / full |
|---|---|---|---|
| eval MSE (Polish test) | 0.00078 | 0.00054 | ~same (near floor) |
| Polish test pl→de | 0.484 | 0.941 | ≈ ½ |
| Polish test mean | 0.675 | 0.960 | ≈ 0.7× |
| dsb parallel mean (1.3k pool) | 0.040 | 0.136 | ≈ ⅓ |
| dsb BUCC retrieval P@1 (67k pool) | 4/902 = 0.0044 | 19/902 = 0.021 | ≈ ⅕ |
| dsb BUCC F1 | 0.000 (tp 0) | 0.005 (tp 6) | → 0 |

At the elbow the MSE is already at its floor, but retrieval — Polish *and* dsb, at every pool size — is only 20–70% of the fully-trained value and keeps climbing for the rest of the schedule. **The distillation MSE and downstream retrieval are decoupled:** MSE convergence happens ~1 epoch in, retrieval convergence takes the full run. So early-stopping / checkpoint-selecting on eval loss undertrains retrieval by 2–5×. **Train the full schedule and select the checkpoint on retrieval P@1, not on eval loss.**

## 11. What's still missing

- Run the same distillation + dsb eval across the other four tokenizers — same dsb test, only the tokenizer differs, so it answers "which tokenizer transfers best to dsb" (morfessor's bar: parallel mean 0.136). Whether the morfessor BPC advantage carries into the distilled space is the open question.
- **Distill on real de–dsb parallel data** (not zero-shot via Polish) — the path to a genuinely useful dsb encoder and a non-trivial BUCC F1.
- The same MLM comparison on hsb v3 / hsb lww would establish whether the dsb tokenizer ordering is language-general.
