"""Lab 5 starter: labelled-query retrieval evaluation."""

import json
import time

from bayan.search.service import CaseSearch


QUERIES_PATH = "data/search/bayan_queries.jsonl"
INDEX_PREFIX = "artifacts/search/bayan_index"


def load_queries(path):
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def recall_at_10(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return None
    top10 = set(retrieved_ids[:10])
    relevant = set(relevant_ids)
    return 1.0 if top10 & relevant else 0.0


def reciprocal_rank(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return None
    relevant = set(relevant_ids)
    for rank, case_id in enumerate(retrieved_ids, start=1):
        if case_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_stage(search, answerable, rerank: bool):
    """Run all answerable queries through search() with the given
    rerank setting, and collect recall@10 / MRR@10 / per-query latency.
    """

    recalls = []
    rrs = []
    latencies = []

    per_lang_recalls = {}
    per_lang_rrs = {}

    for q in answerable:
        start_time = time.perf_counter()

        result = search.search(
            q["query"],
            k=10,
            candidates=50,
            min_score=0.0,
            rerank=rerank,
        )

        elapsed = time.perf_counter() - start_time
        latencies.append(elapsed)

        retrieved_ids = [r["case_id"] for r in result["results"]]

        r = recall_at_10(retrieved_ids, q["relevant_case_ids"])
        rr = reciprocal_rank(retrieved_ids, q["relevant_case_ids"])

        recalls.append(r)
        rrs.append(rr)

        lang = q["lang"]
        per_lang_recalls.setdefault(lang, []).append(r)
        per_lang_rrs.setdefault(lang, []).append(rr)

    return {
        "recall_at_10": sum(recalls) / len(recalls),
        "mrr_at_10": sum(rrs) / len(rrs),
        "avg_latency_ms": (sum(latencies) / len(latencies)) * 1000,
        "per_lang_recall": {
            lang: sum(vals) / len(vals)
            for lang, vals in per_lang_recalls.items()
        },
        "per_lang_mrr": {
            lang: sum(vals) / len(vals)
            for lang, vals in per_lang_rrs.items()
        },
    }


def main():
    queries = load_queries(QUERIES_PATH)
    print(f"Loaded {len(queries)} labelled queries")

    answerable = [q for q in queries if not q["no_answer"]]
    unanswerable = [q for q in queries if q["no_answer"]]

    print(f"  Answerable: {len(answerable)}")
    print(f"  No-answer:  {len(unanswerable)}")

    search = CaseSearch(INDEX_PREFIX, load_reranker=True)

    # ------------------------------------------------------------
    # Stage 1: bi-encoder only (no reranking)
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 1 (bi-encoder only)")
    print("=" * 60)

    stage1 = evaluate_stage(search, answerable, rerank=False)

    print(f"recall@10: {stage1['recall_at_10']:.4f}")
    print(f"MRR@10:    {stage1['mrr_at_10']:.4f}")
    print(f"Avg latency: {stage1['avg_latency_ms']:.1f} ms/query")

    print("\nBy query language:")
    for lang in sorted(stage1["per_lang_recall"]):
        print(
            f"  {lang:4s}: recall@10={stage1['per_lang_recall'][lang]:.4f}  "
            f"MRR@10={stage1['per_lang_mrr'][lang]:.4f}"
        )

    # ------------------------------------------------------------
    # Stage 2: bi-encoder + cross-encoder reranking
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 2 (bi-encoder + cross-encoder rerank)")
    print("=" * 60)

    stage2 = evaluate_stage(search, answerable, rerank=True)

    print(f"recall@10: {stage2['recall_at_10']:.4f}")
    print(f"MRR@10:    {stage2['mrr_at_10']:.4f}")
    print(f"Avg latency: {stage2['avg_latency_ms']:.1f} ms/query")

    print("\nBy query language:")
    for lang in sorted(stage2["per_lang_recall"]):
        print(
            f"  {lang:4s}: recall@10={stage2['per_lang_recall'][lang]:.4f}  "
            f"MRR@10={stage2['per_lang_mrr'][lang]:.4f}"
        )

    # ------------------------------------------------------------
    # MRR lift + latency cost from reranking
    # ------------------------------------------------------------

    mrr_lift = stage2["mrr_at_10"] - stage1["mrr_at_10"]
    latency_cost = stage2["avg_latency_ms"] - stage1["avg_latency_ms"]

    print("\n" + "=" * 60)
    print("RERANKING IMPACT")
    print("=" * 60)
    print(f"MRR@10 lift:      {mrr_lift:+.4f}")
    print(f"Latency cost:     {latency_cost:+.1f} ms/query")

    # ------------------------------------------------------------
    # Cross-lingual gap (using stage 2, our final configuration)
    # ------------------------------------------------------------

    lang_recalls = stage2["per_lang_recall"]
    if len(lang_recalls) >= 2:
        gap = max(lang_recalls.values()) - min(lang_recalls.values())
        print(f"\nCross-lingual recall@10 gap (stage 2): {gap:.4f}")

    # ------------------------------------------------------------
    # Honest no-answer threshold tuning (stage 1 score used for the
    # cutoff, since reranking only reorders candidates that already
    # passed the bi-encoder threshold)
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("No-answer threshold tuning")
    print("=" * 60)

    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        correct_empty = 0
        for q in unanswerable:
            result = search.search(
                q["query"],
                k=10,
                candidates=50,
                min_score=threshold,
                rerank=False,
            )
            if len(result["results"]) == 0:
                correct_empty += 1
        print(
            f"  min_score={threshold:.2f}: "
            f"{correct_empty}/{len(unanswerable)} correctly empty"
        )

    print(f"\nTotal no-answer queries: {len(unanswerable)}")


if __name__ == "__main__":
    main()