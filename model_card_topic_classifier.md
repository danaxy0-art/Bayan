# Model Card — Bayan Topic Classifier

## Intended use
Classifies bilingual (Arabic/English) citizen feedback text into one of 8 municipal service topics.

## Artefact / data versions
- Model/checkpoint: artifacts/topic_classifier
- Preprocessing version: bayan_ar_v1
- Data version/snapshot: bayan_feedback.csv (synthetic)

## Metrics
| Metric | Value | 95% CI |
|---|---:|---|
| Accuracy | 1.0000 | [1.0000, 1.0000] |


## Slice metrics
| Slice type | Slice value | n | Accuracy | 95% CI | Flag |
|---|---|---:|---:|---|---|
| language | ar | 1450 | 1.0000 | [1.0000, 1.0000] |  |
| language | en | 958 | 1.0000 | [1.0000, 1.0000] |  |
| dialect | Gulf | 991 | 1.0000 | [1.0000, 1.0000] |  |
| dialect | MSA | 459 | 1.0000 | [1.0000, 1.0000] |  |
| class | billing | 309 | 1.0000 | [1.0000, 1.0000] |  |
| class | digital_services | 285 | 1.0000 | [1.0000, 1.0000] |  |
| class | licensing | 286 | 1.0000 | [1.0000, 1.0000] |  |
| class | lighting | 314 | 1.0000 | [1.0000, 1.0000] |  |
| class | parks | 344 | 1.0000 | [1.0000, 1.0000] |  |
| class | roads | 300 | 1.0000 | [1.0000, 1.0000] |  |
| class | waste | 289 | 1.0000 | [1.0000, 1.0000] |  |
| class | water | 281 | 1.0000 | [1.0000, 1.0000] |  |
| length | medium (40-100 chars) | 1888 | 1.0000 | [1.0000, 1.0000] |  |
| length | short (<40 chars) | 520 | 1.0000 | [1.0000, 1.0000] |  |

## Behavioural tests
| Suite | Pass rate |
|---|---:|
| invariance | 100.00% |
| directional | 100.00% |
| mft | 100.00% |

## Known limitations
<!-- Lab 6: write this section by hand. Do not auto-generate it. -->
TODO

## Contact / owner
TODO