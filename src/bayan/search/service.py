"""Lab 5 starter: two-stage bilingual case search."""

import json
import time

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder


# Multilingual cross-encoder for stage-2 reranking. Works across
# Arabic/English query-document pairs.
RERANKER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CaseSearch:
    def __init__(self, prefix: str, load_reranker: bool = True):
        """Load the FAISS index, its metadata, and manifest for a given
        index prefix (e.g. "artifacts/search/bayan_index"), and assert
        that the on-disk artefacts are internally consistent.
        """

        # ------------------------------------------------------------
        # 1) Load the manifest first, so we know what encoder to load
        #    and what to expect from the index/metadata.
        #    Manifest keys: model, dim, preproc_version, n_vectors
        #    (matches the Lab 5 reliability contract test).
        # ------------------------------------------------------------

        with open(f"{prefix}_manifest.json", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.encoder_name = self.manifest["model"]
        self.embedding_dim = self.manifest["dim"]
        self.num_cases = self.manifest["n_vectors"]
        self.preproc_version = self.manifest["preproc_version"]

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
            f"Manifest says {self.num_cases} vectors, but the FAISS "
            f"index has {self.index.ntotal} vectors."
        )

        assert len(self.metadata) == self.num_cases, (
            f"Manifest says {self.num_cases} cases, but the metadata "
            f"file has {len(self.metadata)} records."
        )

        assert self.index.d == self.embedding_dim, (
            f"Manifest says dim={self.embedding_dim}, but the FAISS "
            f"index dimension is {self.index.d}."
        )

        # ------------------------------------------------------------
        # 5) Load the bi-encoder used to build this index, so queries
        #    are embedded in the exact same space as the case vectors.
        # ------------------------------------------------------------

        self.encoder = SentenceTransformer(self.encoder_name)

        # ------------------------------------------------------------
        # 6) Optionally load the stage-2 cross-encoder reranker
        # ------------------------------------------------------------

        self.reranker = None
        if load_reranker:
            self.reranker = CrossEncoder(RERANKER_NAME)

        print(
            f"Loaded index: {self.index.ntotal} cases, "
            f"dim={self.embedding_dim}, encoder={self.encoder_name}, "
            f"reranker={'loaded' if self.reranker else 'disabled'}"
        )

    def _retrieve_candidates(self, query: str, candidates: int):
        """Stage 1: bi-encoder retrieval. Returns (scores, indices)."""

        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
        ).astype(np.float32)

        faiss.normalize_L2(query_embedding)

        n_candidates = min(candidates, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, n_candidates)

        return scores[0], indices[0]

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = 0.25,
        rerank: bool = True,
    ):
        """Search for the top-k most relevant cases for a query.

        1. Normalise the query into the same space as the index.
        2. Retrieve `candidates` nearest neighbours via the bi-encoder
           (stage 1). If the best bi-encoder score doesn't clear
           min_score, honestly return an empty result.
        3. If rerank=True, re-score the surviving candidates with a
           cross-encoder (stage 2) and re-sort by that score before
           taking the top-k. If rerank=False, take the top-k directly
           from the stage-1 (bi-encoder) ranking.
        """

        scores, indices = self._retrieve_candidates(query, candidates)

        if len(scores) == 0 or scores[0] < min_score:
            return {
                "query": query,
                "results": [],
                "reason": "no_result_above_threshold" if len(scores) > 0 else "empty_index",
                "best_score": float(scores[0]) if len(scores) > 0 else None,
            }

        surviving = [
            (score, idx)
            for score, idx in zip(scores, indices)
            if score >= min_score and idx >= 0
        ]

        if rerank and self.reranker is not None and surviving:
            pairs = [
                (query, self.metadata[idx]["case_text"])
                for _, idx in surviving
            ]
            rerank_scores = self.reranker.predict(pairs)

            surviving = [
                (float(rerank_scores[i]), surviving[i][1])
                for i in range(len(surviving))
            ]
            surviving.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in surviving[:k]:
            case = self.metadata[idx]
            results.append({
                "case_id": case["case_id"],
                "lang": case["lang"],
                "topic": case["topic"],
                "case_text": case["case_text"],
                "resolution": case["resolution"],
                "score": float(score),
            })

        return {
            "query": query,
            "results": results,
            "reason": None,
            "best_score": float(scores[0]),
        }