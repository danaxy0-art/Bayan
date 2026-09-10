# Model Card — Bayan Arabic Dialect-Aware Classifier (CAMeLBERT-DA)

## Intended use
Topic classification for Gulf-dialect and MSA Arabic citizen feedback, evaluated as a dialect-aware alternative to the general-purpose CAMeLBERT-mix checkpoint used in production (see DECISIONS.md#arabic-model).

## Artefact / data versions
- Model/checkpoint: CAMeL-Lab/bert-base-arabic-camelbert-da (fine-tuned, 2 epochs, Lab 4 bake-off)
- Preprocessing version: bayan_ar_v1
- Data version/snapshot: bayan_feedback.csv (synthetic), frozen test split

## Metrics
| Metric | All | Gulf | MSA |
|---|---:|---:|---:|
| macro-F1 | 0.9992 | 1.0000 | 1.0000 |


## Slice metrics
| Slice type | Slice value | macro-F1 |
|---|---|---:|
| dialect | all | 0.9992 |
| dialect | Gulf | 1.0000 |
| dialect | MSA | 1.0000 |

(See BENCHMARKS.md, Lab 4 Arabic Model Bake-off, for the full comparison against CAMeLBERT-mix and MARBERT.)

## Behavioural tests
Not run for this model (behavioural suite was run on the topic classifier only; see EVALUATION_REPORT.md).

## Known limitations
<!-- Lab 6: write this section by hand. Do not auto-generate it. -->
TODO

## Contact / owner
TODO