"""Lab 5 - Step 5: verify the impact of a planted un-normalised-vector
failure, using metrics rather than eyeballing.

This script builds two small indexes from the same subset of cases:
  1. "normalised"   - L2-normalised embeddings (the correct pipeline)
  2. "unnormalised" - raw embeddings, NOT L2-normalised (the planted bug)

It then runs the same set of labelled queries against both, and
compares recall@10 to show, with numbers, how skipping L2
normalisation degrades retrieval quality when using an inner-product
(cosine-equivalent) FAISS index.
"""

import json

import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer


ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CASES_PATH = "data/search/bayan_cases.csv"
QUERIES_PATH = "data/search/bayan_queries.jsonl"


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
    return 1.0 if top10 & set(relevant_ids) else 0.0


def build_faiss_index(embeddings, normalise: bool):
    embeddings = embeddings.astype(np.float32).copy()

    if normalise:
        faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def evaluate(index, case_ids, encoder, queries, normalise_query: bool):
    recalls = []

    for q in queries:
        query_embedding = encoder.encode(
            [q["query"]],
            convert_to_numpy=True,
        ).astype(np.float32)

        if normalise_query:
            faiss.normalize_L2(query_embedding)

        scores, indices = index.search(query_embedding, 10)
        retrieved_ids = [case_ids[i] for i in indices[0] if i >= 0]

        r = recall_at_10(retrieved_ids, q["relevant_case_ids"])
        if r is not None:
            recalls.append(r)

    return sum(recalls) / len(recalls)


def main():
    # Use a manageable subset for this diagnostic (full 20k not needed
    # to demonstrate the effect).
    df = pd.read_csv(CASES_PATH).head(2000)
    case_ids = df["case_id"].tolist()
    texts = df["case_text"].astype(str).tolist()

    queries = load_queries(QUERIES_PATH)
    answerable = [
        q for q in queries
        if not q["no_answer"] and any(cid in case_ids for cid in q["relevant_case_ids"])
    ]
    print(f"Evaluating on {len(answerable)} answerable queries "
          f"whose relevant cases fall within the first 2000 rows")

    encoder = SentenceTransformer(ENCODER_NAME)

    print("\nEncoding case texts...")
    embeddings = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # ------------------------------------------------------------
    # Correct pipeline: L2-normalise both index and query vectors
    # ------------------------------------------------------------

    normalised_index = build_faiss_index(embeddings, normalise=True)
    recall_normalised = evaluate(
        normalised_index, case_ids, encoder, answerable, normalise_query=True
    )

    # ------------------------------------------------------------
    # Planted bug: skip L2 normalisation on the index vectors.
    # With an inner-product index, un-normalised vectors mean the
    # "similarity" score is dominated by vector magnitude rather than
    # direction/meaning, corrupting the ranking.
    # ------------------------------------------------------------

    unnormalised_index = build_faiss_index(embeddings, normalise=False)
    recall_unnormalised = evaluate(
        unnormalised_index, case_ids, encoder, answerable, normalise_query=False
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("PLANTED UNNORMALISED-VECTOR FAILURE CHECK")
    print("=" * 60)
    print(f"recall@10 WITH  L2 normalisation: {recall_normalised:.4f}")
    print(f"recall@10 WITHOUT L2 normalisation: {recall_unnormalised:.4f}")
    print(
        f"\nDelta: {recall_normalised - recall_unnormalised:+.4f} "
        "(positive = normalisation helps)"
    )

    if recall_normalised > recall_unnormalised:
        print(
            "\nVERIFIED: skipping L2 normalisation measurably degrades "
            "recall@10 on an inner-product FAISS index, confirming the "
            "index build pipeline's normalisation step is load-bearing, "
            "not cosmetic."
        )
    else:
        print(
            "\nWARNING: no measurable degradation detected on this "
            "subset/query sample; the planted-failure effect may need "
            "a larger sample or different metric to surface clearly."
        )


if __name__ == "__main__":
    main()