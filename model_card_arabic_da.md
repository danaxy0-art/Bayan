# Model Card — Bayan Arabic Dialect-Aware Classifier (CAMeLBERT-DA)

## Intended use

Topic classification for Gulf-dialect and MSA Arabic citizen feedback, evaluated as a dialect-aware alternative to the general-purpose CAMeLBERT-mix checkpoint used in production (see DECISIONS.md#arabic-model).

## Artefact / data versions

- Model/checkpoint: CAMeL-Lab/bert-base-arabic-camelbert-da (fine-tuned, 2 epochs, Lab 4 bake-off)
- Preprocessing version: bayan_ar_v1
- Data version/snapshot: bayan_feedback.csv (synthetic), frozen test split

## Metrics

| Metric   |    All |   Gulf |    MSA |
| -------- | -----: | -----: | -----: |
| macro-F1 | 0.9992 | 1.0000 | 1.0000 |

## Slice metrics

| Slice type | Slice value | macro-F1 |
| ---------- | ----------- | -------: |
| dialect    | all         |   0.9992 |
| dialect    | Gulf        |   1.0000 |
| dialect    | MSA         |   1.0000 |

(See BENCHMARKS.md, Lab 4 Arabic Model Bake-off, for the full comparison against CAMeLBERT-mix and MARBERT.)

## Behavioural tests

Not run for this model (behavioural suite was run on the topic classifier only; see EVALUATION_REPORT.md).

## Known limitations

This model was fine-tuned and evaluated entirely on synthetic,
templated data (see NOTES.md), reaching 100% test accuracy — a
result that should NOT be taken as evidence of real-world readiness.
Manual error analysis (120 noise-perturbed examples) revealed the
model collapses to a small set of "attractor" topics (roads, water,
parks — 79% of errors) under low-signal input, and never mispredicts
billing or digital_services, suggesting systemic overconfidence
rather than calibrated uncertainty. Before production use, this model
needs evaluation on real, non-synthetic citizen feedback.

## Contact / owner

Dana — [GitHub](https://github.com/danaxy0-art)
