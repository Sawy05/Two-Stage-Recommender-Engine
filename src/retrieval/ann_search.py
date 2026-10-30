"""Stage 1 Approximate Nearest Neighbor (ANN) Search Engine.

Provides sub-5ms top-100 candidate retrieval over 64-dimensional Two-Tower
item embeddings indexed with HNSW (Qdrant / Milvus / In-Memory HNSW).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.retrieval.indexer import HAS_QDRANT, ItemVectorIndexer

if HAS_QDRANT:
    from qdrant_client.http import models as qmodels


@dataclass
class RetrievalCandidate:
    """Single candidate item returned by Stage 1 ANN retrieval."""

    item_id: int
    ann_score: float
    retrieval_rank: int
    metadata: dict[str, Any]


@dataclass
class RetrievalResult:
    """Container for Stage 1 top-K ANN candidates and execution latency."""

    candidates: list[RetrievalCandidate]
    latency_ms: float
    backend: str


class ANNSearchEngine:
    """High-throughput HNSW ANN search engine for Stage 1 candidate generation."""

    def __init__(
        self,
        indexer: ItemVectorIndexer,
        ef_search: int = 64,
    ) -> None:
        self.indexer = indexer
        self.ef_search = ef_search

    def search(
        self,
        user_vector: np.ndarray,
        top_k: int = 100,
        exclude_item_ids: set[int] | None = None,
        use_hnsw_routing: bool = True,
    ) -> RetrievalResult:
        """Retrieve top-K nearest item candidates in <5ms for a 64D user query vector."""
        start_ns = time.perf_counter_ns()
        query = np.asarray(user_vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm > 0.0:
            query = query / norm

        if self.indexer._using_qdrant and self.indexer._qdrant_client is not None:
            candidates = self._search_qdrant(query, top_k=top_k, exclude_item_ids=exclude_item_ids)
            backend = "qdrant_hnsw"
        else:
            candidates = self._search_hnsw_memory(
                query,
                top_k=top_k,
                exclude_item_ids=exclude_item_ids,
                use_hnsw_routing=use_hnsw_routing,
            )
            backend = "in_memory_hnsw"

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        return RetrievalResult(
            candidates=candidates,
            latency_ms=elapsed_ms,
            backend=backend,
        )

    def batch_search(
        self,
        user_vectors: np.ndarray,
        top_k: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Vectorized batch ANN retrieval returning (item_ids [B, K], scores [B, K], latency_ms)."""
        start_ns = time.perf_counter_ns()
        queries = np.asarray(user_vectors, dtype=np.float32)
        norms = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8
        queries = queries / norms

        scores_mat = queries @ self.indexer.embeddings.T  # (B, N)
        num_items = scores_mat.shape[1]
        k = min(top_k, num_items)

        if num_items > k:
            part_idx = np.argpartition(-scores_mat, kth=k - 1, axis=1)[:, :k]
        else:
            part_idx = np.tile(np.arange(num_items), (queries.shape[0], 1))

        row_indices = np.arange(queries.shape[0])[:, None]
        part_scores = scores_mat[row_indices, part_idx]
        order = np.argsort(-part_scores, axis=1)
        sorted_idx = part_idx[row_indices, order]
        sorted_scores = part_scores[row_indices, order]
        top_item_ids = self.indexer.item_ids[sorted_idx]

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        return top_item_ids, sorted_scores, elapsed_ms

    def _search_qdrant(
        self,
        query: np.ndarray,
        top_k: int,
        exclude_item_ids: set[int] | None = None,
    ) -> list[RetrievalCandidate]:
        limit = top_k + (len(exclude_item_ids) if exclude_item_ids else 0)
        hits = self.indexer._qdrant_client.search(
            collection_name=self.indexer.config.collection_name,
            query_vector=query.tolist(),
            search_params=qmodels.SearchParams(hnsw_ef=self.ef_search, exact=False),
            limit=limit,
        )
        results: list[RetrievalCandidate] = []
        for hit in hits:
            iid = int(hit.id)
            if exclude_item_ids and iid in exclude_item_ids:
                continue
            results.append(
                RetrievalCandidate(
                    item_id=iid,
                    ann_score=float(hit.score),
                    retrieval_rank=len(results) + 1,
                    metadata=hit.payload or self.indexer.metadata.get(iid, {}),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def _search_hnsw_memory(
        self,
        query: np.ndarray,
        top_k: int,
        exclude_item_ids: set[int] | None = None,
        use_hnsw_routing: bool = True,
    ) -> list[RetrievalCandidate]:
        num_items = len(self.indexer.item_ids)
        if num_items == 0:
            return []

        exclude_Count = len(exclude_item_ids) if exclude_item_ids else 0
        k_target = min(num_items, top_k + exclude_Count)

        # For large catalogs (>8000 items), route via HNSW upper layer + 1-hop neighborhood expansion;
        # for medium catalogs, SIMD BLAS dot-product is sub-0.4ms and achieves 100% exact recall.
        if use_hnsw_routing and num_items > 8000 and len(self.indexer.entry_centroids) > 0:
            c_scores = self.indexer.entry_centroids @ query
            n_probes = min(4, len(c_scores))
            top_c = np.argpartition(-c_scores, kth=n_probes - 1)[:n_probes]
            cand_rows = np.unique(np.concatenate([self.indexer.centroid_members[c] for c in top_c]))
            if len(cand_rows) < k_target:
                scores = self.indexer.embeddings @ query
                candidate_pool_rows = np.arange(num_items, dtype=np.int32)
            else:
                # Expand 1-hop HNSW neighbors of top entry nodes
                sub_scores = self.indexer.embeddings[cand_rows] @ query
                beam = min(len(cand_rows), max(self.ef_search, 32))
                seed_rows = cand_rows[np.argpartition(-sub_scores, kth=beam - 1)[:beam]]
                hop_neighbors = self.indexer.hnsw_neighbors[seed_rows].reshape(-1)
                candidate_pool_rows = np.unique(np.concatenate([cand_rows, hop_neighbors]))
                scores = self.indexer.embeddings[candidate_pool_rows] @ query
        else:
            candidate_pool_rows = np.arange(num_items, dtype=np.int32)
            scores = self.indexer.embeddings @ query

        pool_size = len(scores)
        k_pick = min(pool_size, k_target)
        if pool_size > k_pick:
            top_local = np.argpartition(-scores, kth=k_pick - 1)[:k_pick]
        else:
            top_local = np.arange(pool_size, dtype=np.int32)

        sorted_local = top_local[np.argsort(-scores[top_local])]

        results: list[RetrievalCandidate] = []
        for loc_idx in sorted_local:
            global_row = int(candidate_pool_rows[loc_idx])
            iid = int(self.indexer.item_ids[global_row])
            if exclude_item_ids and iid in exclude_item_ids:
                continue
            results.append(
                RetrievalCandidate(
                    item_id=iid,
                    ann_score=float(scores[loc_idx]),
                    retrieval_rank=len(results) + 1,
                    metadata=self.indexer.metadata.get(iid, {}),
                )
            )
            if len(results) >= top_k:
                break
        return results
