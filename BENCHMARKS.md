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

## Lab 7 - Latency Optimisation Ladder

| Configuration | p50 (ms) | p99 (ms) | Threads |
|---|---:|---:|---:|
| fp32 torch @512 padded | 1286.77 | 1348.48 | 1 |
| fp32 torch dynamic padding | 176.10 | 301.88 | 1 |

## Lab 7 - ONNX / INT8 Export

| Configuration | macro-F1 | mean (ms) | p99 (ms) |
|---|---:|---:|---:|
| fp32 torch (original) | 1.0000 | 103.87 | 134.23 |
| fp32 ONNX | 1.0000 | 28.86 | 58.64 |
| INT8 ONNX | 1.0000 | 23.77 | 51.40 |

Quantisation quality tax: +0.0000 (within 0.01 target)

INT8 ONNX p99 speed-up vs fp32 torch: 2.61x

## Lab 7 - ONNX / INT8 Export

| Configuration | macro-F1 | mean (ms) | p99 (ms) |
|---|---:|---:|---:|
| fp32 torch (original) | 1.0000 | 99.65 | 112.89 |
| fp32 ONNX | 1.0000 | 29.38 | 55.03 |
| INT8 ONNX | 1.0000 | 24.07 | 52.77 |

Quantisation quality tax: +0.0000 (within 0.01 target)

INT8 ONNX p99 speed-up vs fp32 torch: 2.14x

## Lab 7 - HTTP Load Test (16 concurrent clients)

- Total requests: 200
- Successful: 200, Errors: 0
- Throughput: 12.55 req/s
- p50: 1264.94 ms
- p99: 1529.00 ms
- Target (p99 <= 40ms): FAIL

## Lab 7 - HTTP Load Test (16 concurrent clients)

- Total requests: 200
- Successful: 200, Errors: 0
- Throughput: 198.94 req/s
- p50: 77.71 ms
- p99: 93.19 ms
- Target (p99 <= 40ms): FAIL

## Lab 7 - NER ONNX / INT8 Export

| Configuration | entity-F1 | mean (ms) | p99 (ms) |
|---|---:|---:|---:|
| fp32 torch (original) | 1.0000 | 115.67 | 103.75 |
| fp32 ONNX | 1.0000 | 38.32 | 48.95 |
| INT8 ONNX | 1.0000 | 34.58 | 48.36 |

Quantisation quality tax: +0.0000
INT8 speed-up vs fp32 torch: 2.15x

**NER quantisation decision:** quantise
(quality tax=+0.0000 vs 0.01 limit, speed-up=2.15x)
