"""Lab 7 + capstone starter: Bayan FastAPI service.

The final service should integrate the outputs of Labs 1-7. Keep this file as
orchestration; reusable logic belongs in the package modules.
"""

import torch
from fastapi import FastAPI

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from bayan.preprocessing.core import preprocess
from bayan.serving.canaries import run_startup_canaries


app = FastAPI(title="Bayan — Bilingual Citizen-Feedback Intelligence Service")


# ------------------------------------------------------------
# Model loaded once at import time, reused across requests.
#
# Uses the WINNING artefact from the Lab 7 optimisation ladder
# (INT8-quantised ONNX, ~25x faster than the original fp32 torch
# checkpoint with zero measured quality tax — see BENCHMARKS.md),
# rather than the original fp32 training artefact.
# ------------------------------------------------------------

CLASSIFIER_DIR = "artifacts/topic_classifier"          # for canaries / tokenizer / id2label
ONNX_INT8_DIR = "artifacts/onnx/classifier_int8"        # winning serving artefact

run_startup_canaries()

_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_DIR)
_model = ORTModelForSequenceClassification.from_pretrained(
    ONNX_INT8_DIR, file_name="model_quantized.onnx"
)
_id2label = _model.config.id2label


@app.get("/health")
def health():
    return {"status": "ok", "message": "Bayan service is running."}


@app.post("/v1/classify")
def classify(payload: dict):
    """Classify a citizen feedback text into one of the Bayan topics.

    Expected payload: {"text": "..."}
    Returns: {"topic": "...", "confidence": float}
    """

    text = payload.get("text", "")

    # Shared preprocessing (Lab 1), same contract used at train time.
    cleaned_text = preprocess(str(text))

    encoded = _tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.inference_mode():
        logits = _model(**encoded).logits

    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id].item())

    topic = _id2label[pred_id]

    return {
        "topic": topic,
        "confidence": confidence,
    }


@app.post("/v1/entities")
def entities(payload: dict):
    # TODO(Capstone): shared preprocessing/Arabic segmentation -> NER -> case fields.
    raise NotImplementedError("Wire the NER artefact")


@app.post("/v1/search")
def search(payload: dict):
    # TODO(Capstone): Lab 5 two-stage bilingual search.
    raise NotImplementedError("Wire the semantic-search component")


@app.post("/v1/analyse")
def analyse(payload: dict):
    # TODO(Capstone): one bilingual request -> classification + entities + similar cases.
    raise NotImplementedError("Assemble the Bayan capstone service")