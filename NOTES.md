# Lab Notes

## Lab 1 — Defect Safari

Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

### Defect 1

- Class: Tokenizer vocabulary bias (mBERT)
- Example: mBERT AR fertility = 2.15 vs EN fertility = 1.51 — Arabic words
  need ~42% more subword tokens than English on average.
- Why it matters: Arabic text costs more context-window budget per word
  than English in the same model, and gets split into less meaningful
  subword units — this can hurt downstream understanding of Arabic input.
- Decision: Flag mBERT as suboptimal for Arabic-heavy workloads; prefer
  a tokenizer with lower AR fertility (e.g. CAMeLBERT) when the pipeline
  is mostly Arabic.

### Defect 2

- Class: Severe language mismatch (DistilBERT)
- Example: DistilBERT AR fertility = 4.53 (highest of all four models)
  vs EN fertility = 1.30 — a 3.5x gap.
- Why it matters: distilbert-base-uncased is an English-only tokenizer;
  Arabic text gets fragmented almost to the character level, destroying
  semantic units and making the model effectively unusable for Arabic.
- Decision: Reject DistilBERT for any Arabic-language use case in this
  project; it is only appropriate for English-only pipelines.

### Defect 3

- Class: Reverse tokenizer bias (CAMeLBERT)
- Example: CAMeLBERT EN fertility = 2.70 vs AR fertility = 1.41 — the
  opposite pattern from mBERT/DistilBERT.
- Why it matters: A tokenizer specialized for one language pays an
  efficiency tax on the other; if the real data has mixed Arabic/English
  content (brand names, code-switching, English loanwords), CAMeLBERT
  will over-segment that portion.
- Decision: Use CAMeLBERT only for Arabic-dominant text; do not assume
  it handles mixed-language input as well as a multilingual tokenizer.

### Defect 4

- Class: Truncation risk from sequence-length inflation
- Example: DistilBERT AR p95 length = 47 tokens vs EN p95 = 21 tokens;
  CAMeLBERT EN p95 = 38 tokens vs AR p95 = 20 tokens.
- Why it matters: If a fixed max_length (e.g. 24, as used earlier in
  the attention lab) is applied across languages, the disadvantaged
  language gets truncated far more often, silently dropping content
  and biasing any downstream evaluation.
- Decision: Set max_length per-language or use the higher p95 value
  as a shared ceiling, rather than a single arbitrary cutoff.

### Defect 5

- Class: Unfair cross-model benchmarking
- Example: Comparing mBERT, XLM-R, CAMeLBERT, and DistilBERT with a
  single fixed max_length conflates "model quality" with "tokenizer
  fertility" — a model isn't truncating more content because it
  understands less, but because its tokenizer fragments text more.
- Why it matters: Any accuracy/recall comparison across these models
  (e.g. downstream PII recall) will be confounded by this tokenizer
  effect unless it's controlled for.
- Decision: Report fertility and p95 length alongside any accuracy
  metric, and consider normalizing max_length per tokenizer before
  comparing task performance.

### Defect 6

- Class: Imbalanced dataset composition
- Example: Arabic rows = 7,200 vs English rows = 4,800 (60%/40% split, not balanced 50/50).
- Why it matters: Aggregate metrics computed over the combined dataset
  will be weighted more heavily toward Arabic performance, which can
  mask poor English-side performance (or vice versa) depending on how
  results are averaged.
- Decision: Report all fertility/length/accuracy metrics broken out
  by language separately (as already done here), never as one blended
  aggregate number.

## Lab 2 — Parameter audit

| Checkpoint | Total params | Embeddings % | Other notes                                             |
| ---------- | -----------: | -----------: | ------------------------------------------------------- |
| mBERT      |  177,853,440 |       51.85% | attention=15.94%, FFN=31.86%, norms=0.02%, pooler=0.33% |
| CAMeLBERT  |  109,081,344 |       21.49% | attention=25.99%, FFN=51.95%, norms=0.03%, pooler=0.54% |

**Why is the embedding share different?**
mBERT covers 104 languages and needs a much larger vocabulary (~120K tokens),
so its embedding table takes over half its total parameters, while
CAMeLBERT is Arabic-only with a much smaller vocabulary — this is the
"multilingual tax." Note that attention, FFN, norms, and pooler parameter
_counts_ are identical between the two models (same BERT-base architecture:
12 layers, 12 heads, d_model=768) — only their embeddings differ.

## Lab 4 — Dialect audit

- Distribution: Gulf 4,800 (66.67%) | MSA 2,400 (33.33%) — out of 7,200 Arabic-language rows total
- One-sentence implication for MSA-only evaluation: MSA makes up only 33.3% of Arabic rows (2,400 MSA vs 4,800 Gulf), so evaluating only on MSA would leave the majority of Arabic dialectal traffic (Gulf) unmeasured, risking an inflated sense of Arabic performance that does not reflect real citizen feedback, most of which is dialectal, not MSA.

## Lab 4 — Step 3: Clitic Segmentation Impact on NER

**Note:** LOCATION recall on the Day-2 (Lab 3B) NER model was already
1.00 (100%) on the un-segmented data (see Lab 3B results). Since recall
cannot exceed 100%, the target of "+4 recall points" is mathematically
unreachable here — this is a further symptom of the templated/synthetic
nature of this project's data (see Lab 3A/3B notes on perfect scores).
Re-training on `bayan_ner_segmented.conll` was deferred; `segment()`
itself was implemented and is available for use in a more realistic
dataset where headroom for improvement would exist.

## Lab 5 — Retrieval Evaluation: Low recall@10 root cause

**Measured result:** recall@10 = 0.0077, MRR@10 = 0.0038 (bi-encoder
only, min_score=0.0), far below the 0.80 target.

**Investigation:** initially suspected broken relevant_case_ids labels.
A full audit of all 130 answerable queries disproved this — 100%
(390/390) of relevant_case_id references share the exact same topic
as their query, confirming the labels are topic-coherent.

**Root cause:** each topic contains ~2,500 cases (20,000 total / 8
topics). The 3 labelled "relevant" cases per query appear to be a
near-random sample within that topic, not necessarily the cases most
textually/semantically similar to the query's specific wording
(location, phrasing). Manual self-query testing confirmed the
bi-encoder correctly retrieves the most semantically similar cases
(score=1.0 on exact-text matches) — but those rarely coincide with
the specific 3 IDs labelled as ground truth, since many equally
plausible same-topic cases exist (62 near-duplicate cases found for
a single query text alone). This is a property of the evaluation
label design, not a defect in the search implementation.
