"""Lab 7 starter: honest p50/p99 benchmark harness over production length mix."""

import time
import statistics

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


CHECKPOINT_DIR = "artifacts/topic_classifier"

# A production-realistic mix of input lengths, drawn from the kinds of
# feedback text seen across the project (short, medium, long, bilingual).
PRODUCTION_MIX = [
    "انقطاع المياه",
    "حفرة في الطريق أمام حي العليا منذ 5 أيام",
    "There is a pothole near Dammam for 15 days, please fix it urgently",
    "دفعت الفاتورة لكن الحالة ما زالت غير مسددة والرقم المرجعي BYN-2025-090006",
    "The streetlight at Al Yasmin has been out for two weeks and it is a safety concern for pedestrians",
    "نحتاج حاوية إضافية",
    "Incorrect fees were charged on invoice BYN-2025-010548, please review and correct",
    "لووسمحت الممر في حديقة الرياض غير مناسب للكراسي المتحركة منذ فترة طويلة",
]


def load_model(checkpoint_dir):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()
    return tokenizer, model


def single_inference(tokenizer, model, text, max_length=None, padding="max_length" if False else "longest"):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=padding,
    )
    start = time.perf_counter()
    with torch.inference_mode():
        model(**encoded)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms


def benchmark(
    tokenizer,
    model,
    texts=None,
    n_warmup=10,
    n_runs=100,
    num_threads=1,
    padding="longest",
):
    """Run a warm-up phase, then n_runs timed inferences over a
    production-realistic text mix, pinning the torch thread count for
    reproducibility. Returns p50/p99 latency in milliseconds.
    """

    if texts is None:
        texts = PRODUCTION_MIX

    torch.set_num_threads(num_threads)

    # ------------------------------------------------------------
    # Warm-up: first few inferences are slower (lazy init, caches);
    # exclude them from the timed measurement.
    # ------------------------------------------------------------

    for i in range(n_warmup):
        text = texts[i % len(texts)]
        single_inference(tokenizer, model, text, padding=padding)

    # ------------------------------------------------------------
    # Timed runs, cycling through the production length mix
    # ------------------------------------------------------------

    latencies = []
    for i in range(n_runs):
        text = texts[i % len(texts)]
        elapsed_ms = single_inference(tokenizer, model, text, padding=padding)
        latencies.append(elapsed_ms)

    latencies.sort()
    p50 = statistics.median(latencies)
    p99_idx = int(len(latencies) * 0.99)
    p99 = latencies[min(p99_idx, len(latencies) - 1)]

    return {
        "n_runs": n_runs,
        "num_threads": num_threads,
        "padding": padding,
        "p50_ms": p50,
        "p99_ms": p99,
        "mean_ms": statistics.mean(latencies),
    }


def print_result_row(label, result):
    print(
        f"{label:30s} p50={result['p50_ms']:7.2f}ms  "
        f"p99={result['p99_ms']:7.2f}ms  "
        f"mean={result['mean_ms']:7.2f}ms  "
        f"(threads={result['num_threads']}, padding={result['padding']})"
    )


if __name__ == "__main__":
    print("Loading topic classifier...")
    tokenizer, model = load_model(CHECKPOINT_DIR)

    print("\n" + "=" * 70)
    print("LAB 7 - OPTIMISATION LADDER: fp32 torch baseline")
    print("=" * 70)

    # Row 1: fp32 torch, padded to a fixed max_length (worst case)
    result_padded = benchmark(
        tokenizer, model, num_threads=1, padding="max_length"
    )
    print_result_row("fp32 torch @512 padded", result_padded)

    # Row 2: fp32 torch, dynamic/"longest" padding within each batch
    # (free win: avoids wasting compute on padding tokens)
    result_dynamic = benchmark(
        tokenizer, model, num_threads=1, padding="longest"
    )
    print_result_row("fp32 torch dynamic padding", result_dynamic)

    speedup = result_padded["p99_ms"] / result_dynamic["p99_ms"]
    print(f"\nDynamic padding p99 speed-up vs fixed padding: {speedup:.2f}x")

    # ------------------------------------------------------------
    # Append to BENCHMARKS.md
    # ------------------------------------------------------------

    with open("BENCHMARKS.md", "a", encoding="utf-8") as f:
        f.write("\n## Lab 7 - Latency Optimisation Ladder\n\n")
        f.write("| Configuration | p50 (ms) | p99 (ms) | Threads |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(
            f"| fp32 torch @512 padded | {result_padded['p50_ms']:.2f} | "
            f"{result_padded['p99_ms']:.2f} | {result_padded['num_threads']} |\n"
        )
        f.write(
            f"| fp32 torch dynamic padding | {result_dynamic['p50_ms']:.2f} | "
            f"{result_dynamic['p99_ms']:.2f} | {result_dynamic['num_threads']} |\n"
        )

    print("\nAppended results to BENCHMARKS.md")