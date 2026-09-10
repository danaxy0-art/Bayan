"""Lab 7/capstone starter: startup skew and behaviour canaries."""

import json
from pathlib import Path


CLASSIFIER_DIR = "artifacts/topic_classifier"
EXPECTED_PREPROC_VERSION = "bayan_ar_v1"


def run_startup_canaries() -> None:
    """Fail fast at startup if the serving environment is inconsistent
    with what training/evaluation assumed, rather than silently
    serving wrong predictions.

    Checks:
      1. The classifier artefact directory exists and has the expected
         files (config.json, tokenizer files).
      2. The number of output labels matches the expected topic count.
      3. A pinned smoke input produces the expected topic (behaviour
         canary — catches silent corruption / wrong checkpoint).
    """

    classifier_path = Path(CLASSIFIER_DIR)

    # ------------------------------------------------------------
    # 1) Artefact presence check
    # ------------------------------------------------------------

    if not classifier_path.exists():
        raise RuntimeError(
            f"Startup canary FAILED: classifier artefact directory "
            f"not found at {CLASSIFIER_DIR}. Run "
            f"scripts/train_classifier.py before starting the service."
        )

    config_path = classifier_path / "config.json"
    if not config_path.exists():
        raise RuntimeError(
            f"Startup canary FAILED: {config_path} is missing. "
            f"The classifier artefact appears incomplete."
        )

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # ------------------------------------------------------------
    # 2) Label count sanity check
    # ------------------------------------------------------------

    id2label = config.get("id2label", {})
    n_labels = len(id2label)

    if n_labels != 8:
        raise RuntimeError(
            f"Startup canary FAILED: expected 8 topic labels, but the "
            f"loaded classifier config has {n_labels}. This suggests "
            f"the wrong artefact is being served."
        )

    # ------------------------------------------------------------
    # 3) Pinned smoke-test input -> expected prediction
    # ------------------------------------------------------------

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_DIR)
    model.eval()

    smoke_text = "انقطاع المياه مستمر في حي النرجس منذ الصباح"
    expected_topic = "water"

    encoded = tokenizer(
        smoke_text, return_tensors="pt", truncation=True, max_length=256
    )
    with torch.inference_mode():
        logits = model(**encoded).logits
    pred_id = int(logits.argmax(dim=-1).item())
    predicted_topic = model.config.id2label[pred_id]

    if predicted_topic != expected_topic:
        raise RuntimeError(
            f"Startup canary FAILED: pinned smoke input expected topic "
            f"'{expected_topic}' but got '{predicted_topic}'. This "
            f"suggests a corrupted artefact or a version mismatch "
            f"between training and serving."
        )

    print("Startup canaries: all checks PASSED.")
    print(f"  Artefact: {CLASSIFIER_DIR}")
    print(f"  Labels: {n_labels}")
    print(f"  Smoke test: '{smoke_text}' -> '{predicted_topic}' (expected)")