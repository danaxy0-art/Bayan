"""Lab 5 starter: labelled-query retrieval evaluation."""

import json

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
    """1 if any relevant case appears in the top-10 retrieved, else 0."""

    if not relevant_ids:
        return None

    top10 = set(retrieved_ids[:10])
    relevant = set(relevant_ids)

    return 1.0 if top10 & relevant else 0.0


def reciprocal_rank(retrieved_ids, relevant_ids):
    """1/rank of the first relevant case in the retrieved list, else 0."""

    if not relevant_ids:
        return None

    relevant = set(relevant_ids)

    for rank, case_id in enumerate(retrieved_ids, start=1):
        if case_id in relevant:
            return 1.0 / rank

    return 0.0


def main():
    queries = load_queries(QUERIES_PATH)
    print(f"Loaded {len(queries)} labelled queries")

    answerable = [q for q in queries if not q["no_answer"]]
    unanswerable = [q for q in queries if q["no_answer"]]

    print(f"  Answerable: {len(answerable)}")
    print(f"  No-answer:  {len(unanswerable)}")

    search = CaseSearch(INDEX_PREFIX)

    # ------------------------------------------------------------
    # 1) Evaluate recall@10 and MRR@10 on the answerable queries
    #    using a low min_score, so we measure raw bi-encoder ranking
    #    quality without the honest-empty-result cutoff interfering.
    # ------------------------------------------------------------

    recalls = []
    rrs = []

    per_lang_recalls = {}
    per_lang_rrs = {}

    for q in answerable:
        result = search.search(
            q["query"],
            k=10,
            candidates=50,
            min_score=0.0,
        )

        retrieved_ids = [r["case_id"] for r in result["results"]]

        r = recall_at_10(retrieved_ids, q["relevant_case_ids"])
        rr = reciprocal_rank(retrieved_ids, q["relevant_case_ids"])

        recalls.append(r)
        rrs.append(rr)

        lang = q["lang"]
        per_lang_recalls.setdefault(lang, []).append(r)
        per_lang_rrs.setdefault(lang, []).append(rr)

    mean_recall = sum(recalls) / len(recalls)
    mean_rr = sum(rrs) / len(rrs)

    print("\n" + "=" * 60)
    print("STAGE 1 (bi-encoder only) - Answerable queries")
    print("=" * 60)
    print(f"recall@10: {mean_recall:.4f}")
    print(f"MRR@10:    {mean_rr:.4f}")

    print("\nBy query language:")
    for lang in sorted(per_lang_recalls):
        lang_recall = sum(per_lang_recalls[lang]) / len(per_lang_recalls[lang])
        lang_rr = sum(per_lang_rrs[lang]) / len(per_lang_rrs[lang])
        n = len(per_lang_recalls[lang])
        print(
            f"  {lang:4s} (n={n:3d}): "
            f"recall@10={lang_recall:.4f}  MRR@10={lang_rr:.4f}"
        )

    # ------------------------------------------------------------
    # 2) Cross-lingual gap: difference between the best and worst
    #    performing language slices
    # ------------------------------------------------------------

    lang_recalls = {
        lang: sum(vals) / len(vals)
        for lang, vals in per_lang_recalls.items()
    }
    if len(lang_recalls) >= 2:
        gap = max(lang_recalls.values()) - min(lang_recalls.values())
        print(f"\nCross-lingual recall@10 gap: {gap:.4f}")

    # ------------------------------------------------------------
    # 3) Honest no-answer check: for queries explicitly marked
    #    no_answer=True, we check how many correctly return empty
    #    at a few candidate min_score thresholds.
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