# Model Card — Bayan NER Model

## Intended use

Extracts SERVICE, LOCATION, DATE, and REFERENCE entities from bilingual (Arabic/English) citizen feedback text.

## Artefact / data versions

- Model/checkpoint: artifacts/ner
- Preprocessing version: bayan_ar_v1
- Data version/snapshot: bayan_ner.conll (synthetic)

## Metrics

| Metric                        |                   Value |
| ----------------------------- | ----------------------: |
| Entity-level F1 (seqeval)     |                  1.0000 |
| Sentence exact-match accuracy | 1.0000 [1.0000, 1.0000] |

## Slice metrics

Not computed for this model (see EVALUATION_REPORT.md for topic classifier slices).

## Behavioural tests

Not run for this model.

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
