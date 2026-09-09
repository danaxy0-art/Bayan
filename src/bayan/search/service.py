"""Lab 5 starter: two-stage bilingual case search."""

import json

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


class CaseSearch:
    def __init__(self, prefix: str):
        """Load the FAISS index, its metadata, and manifest for a given
        index prefix (e.g. "artifacts/search/bayan_index"), and assert
        that the on-disk artefacts are internally consistent.
        """

        # ------------------------------------------------------------
        # 1) Load the manifest first, so we know what encoder to load
        #    and what to expect from the index/metadata.
        # ------------------------------------------------------------

        with open(f"{prefix}.manifest.json", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.encoder_name = self.manifest["encoder_name"]
        self.embedding_dim = self.manifest["embedding_dim"]
        self.num_cases = self.manifest["num_cases"]

        # ------------------------------------------------------------
        # 2) Load the FAISS index
        # ------------------------------------------------------------

        self.index = faiss.read_index(f"{prefix}.faiss")

        # ------------------------------------------------------------
        # 3) Load metadata, one record per line, in index order
        # ------------------------------------------------------------

        self.metadata = []
        with open(f"{prefix}.meta.jsonl", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        # ------------------------------------------------------------
        # 4) Assert manifest integrity on load
        # ------------------------------------------------------------

        assert self.index.ntotal == self.num_cases, (
            f"Manifest says {self.num_cases} cases, but the FAISS "
            f"index has {self.index.ntotal} vectors."
        )

        assert len(self.metadata) == self.num_cases, (
            f"Manifest says {self.num_cases} cases, but the metadata "
            f"file has {len(self.metadata)} records."
        )

        assert self.index.d == self.embedding_dim, (
            f"Manifest says embedding_dim={self.embedding_dim}, but "
            f"the FAISS index dimension is {self.index.d}."
        )

        # ------------------------------------------------------------
        # 5) Load the bi-encoder used to build this index, so queries
        #    are embedded in the exact same space as the case vectors.
        # ------------------------------------------------------------

        self.encoder = SentenceTransformer(self.encoder_name)

        print(
            f"Loaded index: {self.index.ntotal} cases, "
            f"dim={self.embedding_dim}, encoder={self.encoder_name}"
        )

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = 0.25,
    ):
        """Search for the top-k most relevant cases for a query.

        1. Normalise the query into the same space as the index.
        2. Retrieve `candidates` nearest neighbours via the bi-encoder
           (first-stage retrieval).
        3. Return the top-k, honestly reporting an empty result if the
           best score doesn't clear min_score.
        """

        # ------------------------------------------------------------
        # 1) Encode + L2-normalise the query
        # ------------------------------------------------------------

        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
        ).astype(np.float32)

        faiss.normalize_L2(query_embedding)

        # ------------------------------------------------------------
        # 2) Bi-encoder retrieval: get the top `candidates` nearest
        #    neighbours from the index (inner product == cosine
        #    similarity, since both sides are L2-normalised).
        # ------------------------------------------------------------

        n_candidates = min(candidates, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, n_candidates)

        scores = scores[0]
        indices = indices[0]

        # ------------------------------------------------------------
        # 3) Honest empty result: if even the best candidate doesn't
        #    clear min_score, we say so rather than returning noise.
        # ------------------------------------------------------------

        if len(scores) == 0 or scores[0] < min_score:
            return {
                "query": query,
                "results": [],
                "reason": "no_result_above_threshold" if len(scores) > 0 else "empty_index",
                "best_score": float(scores[0]) if len(scores) > 0 else None,
            }

        # ------------------------------------------------------------
        # 4) Take the top-k of the candidates that clear min_score
        # ------------------------------------------------------------

        results = []
        for score, idx in zip(scores, indices):
            if score < min_score:
                continue
            if idx < 0:
                continue

            case = self.metadata[idx]
            results.append({
                "case_id": case["case_id"],
                "lang": case["lang"],
                "topic": case["topic"],
                "case_text": case["case_text"],
                "resolution": case["resolution"],
                "score": float(score),
            })

            if len(results) >= k:
                break

        return {
            "query": query,
            "results": results,
            "reason": None,
            "best_score": float(scores[0]),
        }