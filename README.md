# Bayan
### SDA-AIE-211 — Natural Language Processing with Transformers

A training project built as part of **SDAIA Academy**, delivering a bilingual (Arabic/English) AI service for analyzing citizen complaints and feedback — from raw text preprocessing, through fine-tuned Transformer models, semantic search, full evaluation, and a latency-optimised serving API.

---

##  What Was Built

### Lab 1 — Bilingual Preprocessing and Tokenisation
- Built a text-cleaning pipeline (Unicode normalization, PII masking, sentence segmentation)
- Benchmarked four Arabic/English tokenizers and selected the best fit based on measured evidence
- **Results:** 25/25 preprocessing golden tests passed, PII recall = 100% (60/60)

### Lab 2 — Anatomy of a Transformer
- Implemented Scaled Dot-Product Attention and Multi-Head Attention from scratch
- Verified numerical equivalence with PyTorch's reference implementation (`atol=1e-6`)
- Audited parameter counts for two models (mBERT vs. CAMeLBERT) and explained the embedding-share difference
- Implemented a causal mask and diagnosed attention leakage onto padding tokens (PAD leak)

### Lab 3A — Topic Classification
- Built a TF-IDF + LinearSVC baseline
- Implemented a leakage-safe grouped train/validation/test split (no citizen appears across multiple splits)
- Fine-tuned a topic classifier on top of CAMeLBERT

### Lab 3B — NER + Extractive QA
- Correctly aligned word-level BIO labels to subword tokens (including Arabic clitic edge cases)
- Fine-tuned a Named Entity Recognition (NER) model, evaluated at entity level
- Implemented constrained best-span selection for extractive QA, including honest "no answer" handling

### Lab 4 — Arabic Pipeline and Dialect-Aware Fine-tuning
- Implemented Arabic text normalization with two configurable profiles, matching 30 supplied golden pairs
- Audited dialect distribution (Gulf vs. MSA) in the dataset and its implications for evaluation
- Implemented Arabic clitic segmentation via CAMeL Tools
- Ran a bake-off comparing three Arabic-centric checkpoints across language/dialect slices

### Lab 5 — Bilingual Semantic Search
- Built an L2-normalised FAISS index (20,000 cases) with versioned metadata and a manifest pinning the exact model/preprocessing used
- Implemented two-stage retrieval: bi-encoder first-stage search plus optional multilingual cross-encoder reranking
- Evaluated retrieval quality (recall@10, MRR@10) on 150 labelled queries, sliced by query language
- Tuned an honest no-answer threshold, verified against 20 deliberately unanswerable queries
- Verified, with measured metrics, the effect of a deliberately planted un-normalised-vector failure on retrieval quality

### Lab 6 — The Evaluation Report
- Implemented bootstrap confidence intervals and paired bootstrap significance testing from scratch
- Built a sliced evaluation report (language, dialect, topic, text length — 14 slices) with per-slice confidence intervals and small-sample flagging
- Implemented and ran an invariance / directional / minimum-functionality behavioural test suite
- Since the topic classifier reaches 100% accuracy on the clean test set, generated 120 real model errors via realistic input noise (truncation, word-drop, typos) and manually taxonomised the root causes, uncovering a strong "attractor-class" bias in the model's failure mode
- Produced model cards (with hand-written "known limitations" sections) for the topic classifier, NER model, and the Arabic dialect-aware checkpoint

### Lab 7 — Hitting the Latency Budget
- Benchmarked baseline fp32 latency, then applied dynamic padding as a free win (**4.47x** p99 speed-up)
- Exported the topic classifier to ONNX and quantised to INT8, with a paired accuracy/latency comparison (**25.6x** cumulative speed-up, **0 measured quality tax**)
- Repeated ONNX export + INT8 quantisation for the NER model, with an evidence-based quantise/keep-fp32 decision (**2.15x** speed-up, 0 quality tax → quantise)
- Wired the winning INT8 ONNX artefact into a FastAPI service (`/health`, `/v1/classify`) with startup canaries that fail fast on artefact/version mismatches
- Ran an HTTP load test with 16 concurrent clients: **16x** reduction in end-to-end p99 latency (1529ms → 93ms) after wiring in the optimised artefact — honestly documented as not fully meeting the ≤40ms HTTP / ≤25ms bare-model targets, with root causes (CPU-only inference, Python GIL under concurrency, single-worker uvicorn) recorded for follow-up

---

##  Notes on the Data

Three recurring, evidence-based findings shaped how results from this project should be read:

1. **Near-perfect scores across most trained models.** From the simple TF-IDF baseline, to fully fine-tuned BERT-based classifiers, to NER, most models reached macro-F1/entity-F1 ≈ 1.0. This strongly suggests the dataset (explicitly flagged as `synthetic` in its columns) is highly templated, making the task easier than it would be on real, messy data.

2. **Low semantic-search recall traced to label design, not a search bug.** In Lab 5, recall@10 was very low (0.0077) despite manual spot-checks confirming the search pipeline retrieves highly relevant results. Root-cause analysis showed the labelled "relevant" cases per query are a near-random sample within the correct topic (not the most textually/semantically similar cases), and that recall scales strongly with how many same-topic "distractor" cases exist in the corpus — confirmed by re-testing on a smaller case subset, where recall rose ~16x.

3. **Manually generated model errors reveal a systemic "attractor-class" bias.** Since the classifier has no organic errors, 120 errors were generated via realistic input noise (Lab 6). These show the model collapses toward 3 of 8 topics (roads/water/parks, 79% of all errors) under low-signal input, and never mispredicts two other classes at all — suggesting overconfidence rather than calibrated uncertainty.

All three observations are documented in detail in `NOTES.md` and `BENCHMARKS.md` for each relevant lab.

---

##  Tech Stack

Built with Python 3.12, using `transformers`, `torch`, `datasets`, `scikit-learn`, `pandas`, `camel-tools`, `faiss-cpu`, `sentence-transformers`, `optimum[onnxruntime]`, and `fastapi`. Training was run on a mix of local CPU and Google Colab GPU runtimes, depending on task size; the final serving benchmarks were run entirely on CPU.

---

##  Key Artefacts

- `artifacts/topic_classifier/` — fine-tuned topic classifier (fp32)
- `artifacts/ner/` — fine-tuned NER model (fp32)
- `artifacts/onnx/classifier_int8/`, `artifacts/onnx/ner_int8/` — quantised serving artefacts (winning configuration, wired into the API)
- `artifacts/search/` — FAISS index, metadata, and manifest
- `BENCHMARKS.md`, `NOTES.md`, `DECISIONS.md`, `EVALUATION_REPORT.md` — measured evidence and written decisions for every lab
- `model_card_topic_classifier.md`, `model_card_ner.md`, `model_card_arabic_da.md` — model cards with hand-written limitations

---

## Author

Dana alsaidan— [GitHub](https://github.com/danaxy0-art)

This repository was built as part of a training project with [**SDAIA Academy**](https://github.com/SDAIAAcademy).
