# Bayan
### SDA-AIE-211 — Natural Language Processing with Transformers

A training project built as part of **SDAIA Academy**, aiming to build a bilingual (Arabic/English) AI service for analyzing citizen complaints and feedback — starting from raw text preprocessing, through fine-tuned Transformer models, semantic search, and full evaluation.

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

### Lab 5 — Bilingual Semantic Search *(in progress)*
- Built an L2-normalised FAISS index with accompanying metadata and a version-pinned manifest

---

##  A Note on the Data

Throughout the project, nearly every trained model reached near-perfect or perfect scores (macro-F1 ≈ 1.0) — from the simple TF-IDF baseline, to fully fine-tuned BERT-based classifiers, to the NER model. This recurring pattern strongly suggests that the project's dataset (explicitly flagged as `synthetic` in its columns) is highly templated, making the task easier than it would be on real, messy data — rather than reflecting genuinely exceptional, generalizable model performance. This observation is documented in detail in `NOTES.md` and `BENCHMARKS.md` for each lab.

---

##  Tech Stack

Built with Python 3.12, using `transformers`, `torch`, `datasets`, `scikit-learn`, `pandas`, `camel-tools`, `faiss-cpu`, and `sentence-transformers`. Training was run on a mix of local CPU and Google Colab GPU runtimes, depending on task size.

---

##  Author

Dana Alsaidan — [GitHub](https://github.com/danaxy0-art)

This repository was built as part of a training project with [**SDAIA Academy**](https://github.com/SDAIAAcademy).
