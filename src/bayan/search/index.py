"""Lab 5 starter: versioned FAISS index build."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer


# Multilingual bi-encoder: works across Arabic and English in the same
# embedding space, which is required for cross-lingual retrieval.
ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

PREPROCESSING_VERSION = "bayan_ar_v1"


def build_index(
    cases_path: str = "data/search/bayan_cases.csv",
    prefix: str = "artifacts/search/bayan_index",
    batch_size: int = 64,
    limit: int = None,
):
    """Build an L2-normalised FAISS index over case texts and persist it,
    along with row metadata and a manifest pinning the model/version used.

    Writes three files, all sharing `prefix`:
      {prefix}.faiss         - the FAISS index itself
      {prefix}.meta.jsonl    - one JSON line per case, in index order
      {prefix}_manifest.json - model/dim/preproc_version/n_vectors

    limit: if given, only the first `limit` rows of cases_path are
    indexed (useful for fast tests on a small subset).
    """

    output_path = Path(prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1) Load the cases
    # ------------------------------------------------------------

    df = pd.read_csv(cases_path)

    if limit is not None:
        df = df.head(limit)

    print(f"Loaded {len(df)} cases")

    texts = df["case_text"].astype(str).tolist()

    # ------------------------------------------------------------
    # 2) Encode all case texts with the multilingual bi-encoder
    # ------------------------------------------------------------

    encoder = SentenceTransformer(ENCODER_NAME)

    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype(np.float32)

    # ------------------------------------------------------------
    # 3) L2-normalise embeddings so inner product == cosine similarity
    # ------------------------------------------------------------

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    # ------------------------------------------------------------
    # 4) Build a flat inner-product FAISS index
    # ------------------------------------------------------------

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Index built: {index.ntotal} vectors, dim={dim}")

    # ------------------------------------------------------------
    # 5) Persist the index
    # ------------------------------------------------------------

    faiss.write_index(index, f"{prefix}.faiss")

    # ------------------------------------------------------------
    # 6) Persist metadata, one JSON line per case, in index order
    #    (row i in the index corresponds to line i in this file)
    # ------------------------------------------------------------

    with open(f"{prefix}.meta.jsonl", "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "case_id": row["case_id"],
                "lang": row["lang"],
                "topic": row["topic"],
                "case_text": row["case_text"],
                "resolution": row["resolution"],
                "status": row["status"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------
    # 7) Persist a manifest pinning the exact model/version used,
    #    so the search service can assert integrity on load.
    #    Key names match the Lab 5 reliability contract test.
    # ------------------------------------------------------------

    manifest = {
        "model": ENCODER_NAME,
        "dim": dim,
        "preproc_version": PREPROCESSING_VERSION,
        "n_vectors": index.ntotal,
        "metric": "inner_product_on_l2_normalised",
    }

    with open(f"{prefix}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved index + metadata + manifest to: {prefix}.*")

    return manifest


if __name__ == "__main__":
    build_index()