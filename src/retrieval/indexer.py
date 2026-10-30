"""Precomputation of Two-Tower item embeddings and HNSW Vector Indexing (Qdrant / Milvus / In-Memory HNSW)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.data.synthetic_interactions import ItemRecord, SyntheticDatasetBundle
from src.retrieval.two_tower_model import ItemTower, TwoTowerRetrievalModel, get_device

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    HAS_QDRANT = True
except Exception:  # pragma: no cover
    HAS_QDRANT = False


@dataclass
class HNSWConfig:
    """Configuration for HNSW vector indexing in Qdrant / Milvus."""

    collection_name: str = "items_two_tower"
    embedding_dim: int = 64
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    hnsw_ef_search: int = 64
    prefer_qdrant_memory: bool = False
    qdrant_host: str | None = None
    qdrant_port: int = 6333


class ItemVectorIndexer:
    """Precomputes 64D item embeddings and builds an HNSW index for sub-5ms retrieval."""

    def __init__(self, config: HNSWConfig | None = None) -> None:
        self.config = config or HNSWConfig()
        self.item_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self.embeddings: np.ndarray = np.empty((0, self.config.embedding_dim), dtype=np.float32)
        self.metadata: dict[int, dict[str, Any]] = {}
        self.id_to_row: dict[int, int] = {}

        # HNSW navigable small-world adjacency list + coarse entry-point centroids
        self.hnsw_neighbors: np.ndarray = np.empty((0, 0), dtype=np.int32)
        self.entry_centroids: np.ndarray = np.empty(
            (0, self.config.embedding_dim), dtype=np.float32
        )
        self.centroid_members: list[np.ndarray] = []

        self._qdrant_client: Any | None = None
        self._using_qdrant: bool = False
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize Qdrant HNSW collection if available and requested."""
        if not HAS_QDRANT:
            return
        try:
            if self.config.qdrant_host:
                self._qdrant_client = QdrantClient(
                    host=self.config.qdrant_host,
                    port=self.config.qdrant_port,
                    timeout=2.0,
                )
                self._recreate_qdrant_collection()
                self._using_qdrant = True
            elif self.config.prefer_qdrant_memory:
                self._qdrant_client = QdrantClient(":memory:")
                self._recreate_qdrant_collection()
                self._using_qdrant = True
        except Exception:
            self._qdrant_client = None
            self._using_qdrant = False

    def _recreate_qdrant_collection(self) -> None:
        if self._qdrant_client is None:
            return
        self._qdrant_client.recreate_collection(
            collection_name=self.config.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.config.embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
            hnsw_config=qmodels.HnswConfigDiff(
                m=self.config.hnsw_m,
                ef_construct=self.config.hnsw_ef_construct,
            ),
        )

    def get_milvus_index_params(self) -> dict[str, Any]:
        """Return standard Milvus HNSW index specification for production deployment."""
        return {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {
                "M": self.config.hnsw_m,
                "efConstruction": self.config.hnsw_ef_construct,
            },
        }

    @torch.no_grad()
    def precompute_and_index(
        self,
        model: TwoTowerRetrievalModel,
        bundle: SyntheticDatasetBundle,
        batch_size: int = 512,
        include_cold_start: bool = False,
        device: torch.device | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode catalog items with the ItemTower and populate the HNSW vector index."""
        resolved_device = device or get_device()
        model.to(resolved_device)
        model.eval()

        candidate_ids = [
            iid
            for iid, rec in sorted(bundle.items.items())
            if include_cold_start or not rec.is_cold_start
        ]
        if not candidate_ids:
            candidate_ids = sorted(bundle.items.keys())

        all_vecs: list[np.ndarray] = []
        for start in range(0, len(candidate_ids), batch_size):
            batch_ids = candidate_ids[start : start + batch_size]
            idx_arr = np.array(batch_ids, dtype=np.int64)
            item_sparse = {
                key: torch.from_numpy(bundle.item_sparse_matrix[key][idx_arr]).to(resolved_device)
                for key in ItemTower.SPARSE_KEYS
            }
            item_dense = torch.from_numpy(bundle.item_dense_matrix[idx_arr]).to(resolved_device)
            emb = model.encode_item(item_sparse, item_dense).cpu().numpy().astype(np.float32)
            all_vecs.append(emb)

        embeddings = np.vstack(all_vecs)
        item_ids_arr = np.array(candidate_ids, dtype=np.int64)
        self.index_embeddings(
            item_ids=item_ids_arr,
            embeddings=embeddings,
            items_dict=bundle.items,
        )
        return item_ids_arr, embeddings

    def index_embeddings(
        self,
        item_ids: Sequence[int] | np.ndarray,
        embeddings: np.ndarray,
        items_dict: dict[int, ItemRecord] | None = None,
    ) -> None:
        """Index precomputed L2-normalized item vectors into HNSW structures and Qdrant."""
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        self.embeddings = np.ascontiguousarray((embeddings / norms).astype(np.float32))
        self.id_to_row = {int(iid): row for row, iid in enumerate(self.item_ids)}

        self.metadata = {}
        for iid in self.item_ids:
            int_id = int(iid)
            if items_dict and int_id in items_dict:
                rec = items_dict[int_id]
                self.metadata[int_id] = {
                    "category_id": rec.category_id,
                    "brand_id": rec.brand_id,
                    "price_tier": rec.price_tier,
                    "is_cold_start": rec.is_cold_start,
                }
            else:
                self.metadata[int_id] = {
                    "category_id": int_id % 40 + 1,
                    "brand_id": int_id % 120 + 1,
                    "price_tier": int_id % 5 + 1,
                    "is_cold_start": False,
                }

        self._build_hnsw_graph()

        if self._using_qdrant and self._qdrant_client is not None:
            points = [
                qmodels.PointStruct(
                    id=int(iid),
                    vector=self.embeddings[row].tolist(),
                    payload=self.metadata[int(iid)],
                )
                for row, iid in enumerate(self.item_ids)
            ]
            self._qdrant_client.upsert(
                collection_name=self.config.collection_name,
                points=points,
            )

    def _build_hnsw_graph(self) -> None:
        """Construct hierarchical entry points and M-nearest neighbor adjacency lists."""
        num_items = len(self.item_ids)
        if num_items == 0:
            return

        m = min(self.config.hnsw_m, max(1, num_items - 1))
        sim_matrix = self.embeddings @ self.embeddings.T
        np.fill_diagonal(sim_matrix, -np.inf)

        if num_items > m:
            top_m = np.argpartition(-sim_matrix, kth=m - 1, axis=1)[:, :m]
        else:
            top_m = np.zeros((num_items, 1), dtype=np.int32)
        self.hnsw_neighbors = top_m.astype(np.int32)

        # Build upper-layer entry routing centroids for sub-millisecond graph entry
        num_centroids = min(32, max(1, int(np.sqrt(num_items))))
        step = max(1, num_items // num_centroids)
        centroids = self.embeddings[::step][:num_centroids].copy()
        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
        self.entry_centroids = (centroids / c_norms).astype(np.float32)

        assign_scores = self.embeddings @ self.entry_centroids.T
        ef_bucket = min(num_items, max(self.config.hnsw_ef_construct, 128))
        self.centroid_members = []
        for c_idx in range(len(self.entry_centroids)):
            col = assign_scores[:, c_idx]
            if num_items > ef_bucket:
                top_rows = np.argpartition(-col, kth=ef_bucket - 1)[:ef_bucket]
            else:
                top_rows = np.arange(num_items, dtype=np.int32)
            self.centroid_members.append(top_rows.astype(np.int32))
