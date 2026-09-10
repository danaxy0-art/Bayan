"""Lab 7 starter: ONNX export and dynamic INT8 quantisation."""

import time
import statistics
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

from bayan.models.data import build_topic_dataset


CHECKPOINT_DIR = "artifacts/topic_classifier"
DATA_PATH = "data/raw/bayan_feedback.csv"
SEED = 42

ONNX_FP32_DIR = "artifacts/onnx/classifier_fp32"
ONNX_INT8_DIR = "artifacts/onnx/classifier_int8"


def export_fp32_onnx():
    """Export the fp32 PyTorch classifier to ONNX format (this is our
    kept fp32 rollback artefact, per the course contract).
    """

    Path(ONNX_FP32_DIR).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

    print("Exporting fp32 model to ONNX...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        CHECKPOINT_DIR,
        export=True,
    )
    ort_model.save_pretrained(ONNX_FP32_DIR)
    tokenizer.save_pretrained(ONNX_FP32_DIR)

    print(f"Saved fp32 ONNX model to: {ONNX_FP32_DIR}")
    return tokenizer


def quantise_int8():
    """Dynamically quantise the exported fp32 ONNX model to INT8."""

    Path(ONNX_INT8_DIR).mkdir(parents=True, exist_ok=True)

    print("Quantising to INT8 (dynamic quantisation)...")

    quantizer = ORTQuantizer.from_pretrained(ONNX_FP32_DIR)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

    quantizer.quantize(
        save_dir=ONNX_INT8_DIR,
        quantization_config=qconfig,
    )

    print(f"Saved INT8 ONNX model to: {ONNX_INT8_DIR}")


def predict_topic(tokenizer, model, text, id2label):
    encoded = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256
    )
    with torch.inference_mode():
        logits = model(**encoded).logits
    pred_id = int(np.argmax(logits.numpy(), axis=-1)[0])
    return id2label[pred_id]


def evaluate_macro_f1(model_dir, test_df, is_onnx=False):
    """Load a model (torch or ONNX) from model_dir and compute macro-F1
    on the frozen test split, plus average per-example latency.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if is_onnx:
        # INT8-quantised models are saved with a non-standard file name
        # (model_quantized.onnx instead of model.onnx); fall back to it
        # explicitly so we never silently evaluate the wrong artefact.
        import os
        if os.path.exists(f"{model_dir}/model_quantized.onnx"):
            model = ORTModelForSequenceClassification.from_pretrained(
                model_dir, file_name="model_quantized.onnx"
            )
        else:
            model = ORTModelForSequenceClassification.from_pretrained(model_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()

    id2label = model.config.id2label

    preds = []
    latencies = []

    for text in test_df["text"]:
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )

        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(**encoded).logits
        latencies.append((time.perf_counter() - start) * 1000)

        pred_id = int(np.argmax(logits.numpy() if hasattr(logits, "numpy") else logits.detach().numpy(), axis=-1)[0])
        preds.append(id2label[pred_id])

    macro_f1 = f1_score(test_df["topic"], preds, average="macro")
    mean_latency = statistics.mean(latencies)
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

    return {
        "macro_f1": macro_f1,
        "mean_latency_ms": mean_latency,
        "p99_latency_ms": p99_latency,
    }


def main():
    print("Loading frozen test split (sampling 200 rows for speed)...")
    dataset = build_topic_dataset(DATA_PATH, seed=SEED)
    test_df = dataset["test"].to_pandas().sample(n=200, random_state=SEED)

    # ------------------------------------------------------------
    # 1) Export fp32 to ONNX (kept as the rollback artefact)
    # ------------------------------------------------------------

    export_fp32_onnx()

    # ------------------------------------------------------------
    # 2) Quantise to INT8
    # ------------------------------------------------------------

    quantise_int8()

    # ------------------------------------------------------------
    # 3) Paired accuracy + latency comparison: fp32 torch vs
    #    fp32 ONNX vs INT8 ONNX
    # ------------------------------------------------------------

    print("\nEvaluating fp32 torch (original)...")
    torch_result = evaluate_macro_f1(CHECKPOINT_DIR, test_df, is_onnx=False)

    print("Evaluating fp32 ONNX...")
    onnx_fp32_result = evaluate_macro_f1(ONNX_FP32_DIR, test_df, is_onnx=True)

    print("Evaluating INT8 ONNX...")
    onnx_int8_result = evaluate_macro_f1(ONNX_INT8_DIR, test_df, is_onnx=True)

    quality_tax = torch_result["macro_f1"] - onnx_int8_result["macro_f1"]

    print("\n" + "=" * 70)
    print("ONNX / INT8 EXPORT RESULTS")
    print("=" * 70)
    print(
        f"{'Configuration':25s} {'macro-F1':>10s} "
        f"{'mean(ms)':>10s} {'p99(ms)':>10s}"
    )
    for label, result in [
        ("fp32 torch (original)", torch_result),
        ("fp32 ONNX", onnx_fp32_result),
        ("INT8 ONNX", onnx_int8_result),
    ]:
        print(
            f"{label:25s} {result['macro_f1']:10.4f} "
            f"{result['mean_latency_ms']:10.2f} {result['p99_latency_ms']:10.2f}"
        )

    print(f"\nQuantisation quality tax (fp32 torch - INT8 macro-F1): {quality_tax:+.4f}")
    print(f"Target: quality tax <= 0.01 (1 macro-F1 point) -> "
          f"{'PASS' if quality_tax <= 0.01 else 'FAIL'}")

    speedup = torch_result["p99_latency_ms"] / onnx_int8_result["p99_latency_ms"]
    print(f"INT8 ONNX p99 speed-up vs fp32 torch: {speedup:.2f}x")

    # ------------------------------------------------------------
    # Append to BENCHMARKS.md
    # ------------------------------------------------------------

    with open("BENCHMARKS.md", "a", encoding="utf-8") as f:
        f.write("\n## Lab 7 - ONNX / INT8 Export\n\n")
        f.write("| Configuration | macro-F1 | mean (ms) | p99 (ms) |\n")
        f.write("|---|---:|---:|---:|\n")
        for label, result in [
            ("fp32 torch (original)", torch_result),
            ("fp32 ONNX", onnx_fp32_result),
            ("INT8 ONNX", onnx_int8_result),
        ]:
            f.write(
                f"| {label} | {result['macro_f1']:.4f} | "
                f"{result['mean_latency_ms']:.2f} | {result['p99_latency_ms']:.2f} |\n"
            )
        f.write(f"\nQuantisation quality tax: {quality_tax:+.4f} "
                 f"({'within' if quality_tax <= 0.01 else 'EXCEEDS'} 0.01 target)\n")
        f.write(f"\nINT8 ONNX p99 speed-up vs fp32 torch: {speedup:.2f}x\n")

    print("\nAppended results to BENCHMARKS.md")
    print(f"\nfp32 rollback artefact kept at: {ONNX_FP32_DIR}")


if __name__ == "__main__":
    main()