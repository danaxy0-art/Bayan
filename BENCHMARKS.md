# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit

| Tokenizer  | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
| ---------- | -----------: | -----------: | ---------: | ---------: | ----------: |
| mBERT      |         2.15 |         1.51 |         27 |         25 |         N/A |
| XLM-R      |         1.67 |         1.43 |         21 |         23 |         N/A |
| CAMeLBERT  |         1.41 |         2.70 |         20 |         38 |         N/A |
| DistilBERT |         4.53 |         1.30 |         47 |         21 |         N/A |

- Golden preprocessing: 25 / 25 passed
- PII masking recall: 60 / 60 = 100%

## Lab 3 — Models

| Model              | Metric          | Validation |                        Frozen test | Train time |
| ------------------ | --------------- | ---------: | ---------------------------------: | ---------: |
| TF-IDF + LinearSVC | macro-F1        |     1.0000 |                             1.0000 |        N/A |
| Topic classifier   | macro-F1        |     1.0000 |                             1.0000 |      ~230s |
| NER                | entity-F1       |     1.0000 |                             1.0000 |     ~2540s |
| QA                 | span/null smoke |          — | 12/12 answerable, 0/0 unanswerable |          — |

## Lab 4 - Arabic Model Bake-off

| Model         | All macro-F1 | Gulf macro-F1 | MSA macro-F1 |
| ------------- | -----------: | ------------: | -----------: |
| CAMeLBERT-mix |       1.0000 |        1.0000 |       1.0000 |
| CAMeLBERT-DA  |       0.9992 |        1.0000 |       1.0000 |
| MARBERT       |       1.0000 |        1.0000 |       1.0000 |

## Lab 5 — Search

| Configuration          | recall@10 | MRR@10 | p50 latency/query |
| ---------------------- | --------: | -----: | ----------------: |
| bi-encoder only        |           |        |                   |
| + cross-encoder rerank |           |        |                   |
| cross-lingual slice    |           |        |                   |

- no-answer empty-correct: \_\_\_ / 20
- cross-lingual gap: \_\_\_

## Lab 6 — Evaluation

| Model            | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
| ---------------- | ----------------------- | --------- | --------------: | -------: |
| topic classifier |                         |           |                 |          |
| dialect-aware    |                         |           |                 |          |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder

| Rung                    | p50 | p99 | quality metric / paired Δ | Artefact size |
| ----------------------- | --: | --: | ------------------------- | ------------: |
| fp32 torch @512 padded  |     |     |                           |               |
| fp32 torch @128 dynamic |     |     |                           |               |
| ONNX fp32 @128          |     |     |                           |               |
| ONNX INT8 @128          |     |     |                           |               |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:

## Lab 3A - TF-IDF Baseline

| Model              | Metric   | Validation | Frozen Test |
| ------------------ | -------- | ---------: | ----------: |
| TF-IDF + LinearSVC | macro-F1 |     1.0000 |      1.0000 |
