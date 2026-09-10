"""Lab 7 - Step 4: repeat ONNX export + INT8 quantisation for the NER model,
with an evidence-based quantisation decision (per-model, not assumed).
"""

import time
import statistics
from pathlib import Path

import numpy as np
import torch

from seqeval.metrics import f1_score as seqeval_f1
from transformers import AutoTokenizer, AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer


CHECKPOINT_DIR = "artifacts/ner"
CONLL_PATH = "data/models/bayan_ner.conll"
SEED = 42

ONNX_FP32_DIR = "artifacts/onnx/ner_fp32"
ONNX_INT8_DIR = "artifacts/onnx/ner_int8"


def read_conll(path):
    sentences = []
    tokens, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
                if len(parts) != 2:
                    continue
            word, tag = parts
            tokens.append(word)
            tags.append(tag)
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def get_test_split(sentences):
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(sentences))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    test_idx = indices[n_train + n_val:]
    return [sentences[i] for i in test_idx]


def export_fp32_onnx():
    Path(ONNX_FP32_DIR).mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

    print("Exporting NER fp32 model to ONNX...")
    ort_model = ORTModelForTokenClassification.from_pretrained(
        CHECKPOINT_DIR, export=True
    )
    ort_model.save_pretrained(ONNX_FP32_DIR)
    tokenizer.save_pretrained(ONNX_FP32_DIR)
    print(f"Saved fp32 ONNX NER model to: {ONNX_FP32_DIR}")


def quantise_int8():
    Path(ONNX_INT8_DIR).mkdir(parents=True, exist_ok=True)
    print("Quantising NER model to INT8...")

    quantizer = ORTQuantizer.from_pretrained(ONNX_FP32_DIR)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=ONNX_INT8_DIR, quantization_config=qconfig)

    print(f"Saved INT8 ONNX NER model to: {ONNX_INT8_DIR}")


def evaluate_model(model_dir, test_sentences, tokenizer, id2label, is_onnx, onnx_file=None):
    if is_onnx:
        if onnx_file:
            model = ORTModelForTokenClassification.from_pretrained(
                model_dir, file_name=onnx_file
            )
        else:
            model = ORTModelForTokenClassification.from_pretrained(model_dir)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        model.eval()

    all_true, all_pred = [], []
    latencies = []

    for tokens, true_tags in test_sentences:
        encoded = tokenizer(
            tokens, truncation=True, is_split_into_words=True, return_tensors="pt"
        )
        word_ids = encoded.word_ids(batch_index=0)

        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(**encoded).logits[0]
        latencies.append((time.perf_counter() - start) * 1000)

        pred_ids = logits.argmax(dim=-1).tolist() if hasattr(logits, "argmax") else np.argmax(logits, axis=-1).tolist()

        pred_tags = []
        prev_word_id = None
        for wid, pid in zip(word_ids, pred_ids):
            if wid is None:
                continue
            if wid != prev_word_id:
                pred_tags.append(id2label[pid])
            prev_word_id = wid

        pred_tags = pred_tags[:len(true_tags)]
        while len(pred_tags) < len(true_tags):
            pred_tags.append("O")

        all_true.append(true_tags)
        all_pred.append(pred_tags)

    entity_f1 = seqeval_f1(all_true, all_pred)
    mean_latency = statistics.mean(latencies)
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

    return {"entity_f1": entity_f1, "mean_latency_ms": mean_latency, "p99_latency_ms": p99_latency}


def main():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    torch_model_for_labels = AutoModelForTokenClassification.from_pretrained(CHECKPOINT_DIR)
    id2label = torch_model_for_labels.config.id2label
    del torch_model_for_labels

    print("Loading NER test split (same seed as training)...")
    sentences = read_conll(CONLL_PATH)
    test_sentences = get_test_split(sentences)
    # Sample for speed
    test_sentences = test_sentences[:150]
    print(f"Evaluating on {len(test_sentences)} test sentences")

    export_fp32_onnx()
    quantise_int8()

    print("\nEvaluating fp32 torch (original)...")
    torch_result = evaluate_model(
        CHECKPOINT_DIR, test_sentences, tokenizer, id2label, is_onnx=False
    )

    print("Evaluating fp32 ONNX...")
    onnx_fp32_result = evaluate_model(
        ONNX_FP32_DIR, test_sentences, tokenizer, id2label, is_onnx=True
    )

    print("Evaluating INT8 ONNX...")
    onnx_int8_result = evaluate_model(
        ONNX_INT8_DIR, test_sentences, tokenizer, id2label,
        is_onnx=True, onnx_file="model_quantized.onnx"
    )

    quality_tax = torch_result["entity_f1"] - onnx_int8_result["entity_f1"]
    speedup = torch_result["p99_latency_ms"] / onnx_int8_result["p99_latency_ms"]

    print("\n" + "=" * 70)
    print("NER ONNX / INT8 EXPORT RESULTS")
    print("=" * 70)
    print(f"{'Configuration':25s} {'entity-F1':>10s} {'mean(ms)':>10s} {'p99(ms)':>10s}")
    for label, result in [
        ("fp32 torch (original)", torch_result),
        ("fp32 ONNX", onnx_fp32_result),
        ("INT8 ONNX", onnx_int8_result),
    ]:
        print(
            f"{label:25s} {result['entity_f1']:10.4f} "
            f"{result['mean_latency_ms']:10.2f} {result['p99_latency_ms']:10.2f}"
        )

    print(f"\nQuantisation quality tax: {quality_tax:+.4f}")
    print(f"Target (<=0.01): {'PASS' if quality_tax <= 0.01 else 'FAIL'}")
    print(f"INT8 speed-up vs fp32 torch: {speedup:.2f}x")

    # ------------------------------------------------------------
    # Evidence-based quantisation decision
    # ------------------------------------------------------------

    decision = "quantise" if quality_tax <= 0.01 and speedup > 1.2 else "keep fp32"
    print(f"\nDecision: {decision} the NER model for serving")
    print(
        f"Rationale: quality tax={quality_tax:+.4f} (limit 0.01), "
        f"speed-up={speedup:.2f}x"
    )

    with open("BENCHMARKS.md", "a", encoding="utf-8") as f:
        f.write("\n## Lab 7 - NER ONNX / INT8 Export\n\n")
        f.write("| Configuration | entity-F1 | mean (ms) | p99 (ms) |\n")
        f.write("|---|---:|---:|---:|\n")
        for label, result in [
            ("fp32 torch (original)", torch_result),
            ("fp32 ONNX", onnx_fp32_result),
            ("INT8 ONNX", onnx_int8_result),
        ]:
            f.write(
                f"| {label} | {result['entity_f1']:.4f} | "
                f"{result['mean_latency_ms']:.2f} | {result['p99_latency_ms']:.2f} |\n"
            )
        f.write(f"\nQuantisation quality tax: {quality_tax:+.4f}\n")
        f.write(f"INT8 speed-up vs fp32 torch: {speedup:.2f}x\n")
        f.write(f"\n**NER quantisation decision:** {decision}\n")
        f.write(
            f"(quality tax={quality_tax:+.4f} vs 0.01 limit, "
            f"speed-up={speedup:.2f}x)\n"
        )

    print("\nAppended results to BENCHMARKS.md")


if __name__ == "__main__":
    main()