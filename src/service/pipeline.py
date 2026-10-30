"""End-to-End Two-Stage Recommendation Pipeline with Redis User Vector Caching."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from src.bandits.contextual_bandit import LinUCBConfig, LinUCBContextualBandit
from src.data.synthetic_interactions import (
    SyntheticDatasetBundle,
    SyntheticEcommerceGenerator,
)
from src.ranking.dcn_v2 import DCNv2Config, DCNv2RankingModel
from src.ranking.export import ExportedDCNv2Scorer
from src.retrieval.ann_search import ANNSearchEngine, RetrievalCandidate
from src.retrieval.indexer import HNSWConfig, ItemVectorIndexer
from src.retrieval.two_tower_model import (
    TwoTowerConfig,
    TwoTowerRetrievalModel,
    UserTower,
)
from src.service.schemas import (
    LatencyBreakdown,
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
)

try:
    import redis

    HAS_REDIS = True
except Exception:  # pragma: no cover
    HAS_REDIS = False


class RedisUserVectorCache:
    """User 64D embedding cache backed by Redis with sub-millisecond in-memory fallback."""

    def __init__(
        self,
        host: str | None = None,
        port: int = 6379,
        ttl_seconds: int = 3600,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self._local_store: dict[int, tuple[np.ndarray, float]] = {}
        self._redis_client: Any | None = None
        self.backend_name: str = "in_memory_lru"

        if HAS_REDIS and host:
            try:
                client = redis.Redis(host=host, port=port, socket_connect_timeout=0.5)
                client.ping()
                self._redis_client = client
                self.backend_name = "redis"
            except Exception:
                self._redis_client = None
                self.backend_name = "in_memory_lru"

    def get(self, user_id: int) -> np.ndarray | None:
        """Retrieve cached 64D float32 user embedding if present and unexpired."""
        now = time.monotonic()
        if user_id in self._local_store:
            vec, expires_at = self._local_store[user_id]
            if now < expires_at:
                return vec
            del self._local_store[user_id]

        if self._redis_client is not None:
            try:
                raw = self._redis_client.get(f"user_vec:{user_id}")
                if raw is not None:
                    arr = np.frombuffer(raw, dtype=np.float32).copy()
                    self._local_store[user_id] = (arr, now + self.ttl_seconds)
                    return arr
            except Exception:
                pass
        return None

    def set(self, user_id: int, vector: np.ndarray) -> None:
        """Store 64D float32 user embedding in cache with configured TTL."""
        arr = np.ascontiguousarray(vector, dtype=np.float32)
        self._local_store[user_id] = (arr, time.monotonic() + self.ttl_seconds)
        if self._redis_client is not None:
            try:
                self._redis_client.setex(
                    f"user_vec:{user_id}",
                    self.ttl_seconds,
                    arr.tobytes(),
                )
            except Exception:
                pass


class TwoStageRecommendationPipeline:
    """Orchestrates Stage 1 (ANN) -> Stage 2 (DCNv2) -> Bandit Exploration -> Feed."""

    def __init__(
        self,
        bundle: SyntheticDatasetBundle,
        two_tower_model: TwoTowerRetrievalModel,
        dcn_model: DCNv2RankingModel,
        indexer: ItemVectorIndexer,
        ann_engine: ANNSearchEngine,
        bandit: LinUCBContextualBandit,
        user_cache: RedisUserVectorCache | None = None,
        torchscript_path: str | None = None,
        cvr_ranking_alpha: float = 2.5,
    ) -> None:
        self.bundle = bundle
        self.two_tower_model = two_tower_model.cpu().eval()
        self.dcn_model = dcn_model.cpu().eval()
        self.indexer = indexer
        self.ann_engine = ann_engine
        self.bandit = bandit
        self.user_cache = user_cache or RedisUserVectorCache()
        self.scorer = ExportedDCNv2Scorer(self.dcn_model, torchscript_path=torchscript_path)
        self.cvr_ranking_alpha = cvr_ranking_alpha

        # Precompute vectorized item catalog arrays for sub-millisecond Stage 2 feature assembly
        max_iid = max(bundle.items.keys())
        self._item_category = bundle.item_sparse_matrix["category_id"]
        self._item_brand = bundle.item_sparse_matrix["brand_id"]
        self._item_price = bundle.item_sparse_matrix["price_tier"]
        self._item_dense = bundle.item_dense_matrix
        self._item_hist_ctr = np.zeros(max_iid + 1, dtype=np.float32)
        self._item_hist_cvr = np.zeros(max_iid + 1, dtype=np.float32)
        for iid, item in bundle.items.items():
            self._item_hist_ctr[iid] = item.historical_ctr
            self._item_hist_cvr[iid] = item.historical_cvr

    @classmethod
    def build_default(
        cls,
        num_users: int = 500,
        num_items: int = 1500,
        num_interactions: int = 8000,
        seed: int = 42,
    ) -> TwoStageRecommendationPipeline:
        """Create and initialize a complete ready-to-serve two-stage pipeline."""
        torch.manual_seed(seed)
        gen = SyntheticEcommerceGenerator(
            num_users=num_users,
            num_items=num_items,
            num_interactions=num_interactions,
            seed=seed,
        )
        bundle = gen.generate()

        tt_cfg = TwoTowerConfig()
        two_tower = TwoTowerRetrievalModel(tt_cfg).eval()

        indexer = ItemVectorIndexer(HNSWConfig(embedding_dim=tt_cfg.embedding_dim))
        indexer.precompute_and_index(two_tower, bundle, include_cold_start=False)
        ann_engine = ANNSearchEngine(indexer=indexer)

        dcn_cfg = DCNv2Config()
        dcn = DCNv2RankingModel(dcn_cfg).eval()

        bandit = LinUCBContextualBandit(
            LinUCBConfig(context_dim=16, alpha=1.25, exploration_traffic_slice=0.05)
        )
        cat_map = {iid: bundle.items[iid].category_id for iid in bundle.cold_start_item_ids}
        prior_map = {iid: bundle.items[iid].latent_factor for iid in bundle.cold_start_item_ids}
        bandit.register_cold_start_items(
            bundle.cold_start_item_ids,
            item_categories=cat_map,
            prior_vectors=prior_map,
        )

        cache = RedisUserVectorCache()
        pipeline = cls(
            bundle=bundle,
            two_tower_model=two_tower,
            dcn_model=dcn,
            indexer=indexer,
            ann_engine=ann_engine,
            bandit=bandit,
            user_cache=cache,
        )
        pipeline.warm_up()
        return pipeline

    def warm_up(self, sample_user_id: int = 1) -> None:
        """Warm up PyTorch JIT/inference kernels and pre-populate user cache."""
        req = RecommendationRequest(user_id=sample_user_id, top_k_retrieval=100, top_k_final=20)
        self.recommend(req)

    def resolve_user_vector(self, user_id: int) -> tuple[np.ndarray, bool, float]:
        """Fetch 64D user vector from Redis cache or compute via UserTower on miss."""
        t0 = time.perf_counter_ns()
        cached = self.user_cache.get(user_id)
        if cached is not None:
            ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return cached, True, ms

        uid = min(max(1, user_id), len(self.bundle.users))
        with torch.inference_mode():
            user_sparse = {
                key: torch.tensor([int(self.bundle.user_sparse_matrix[key][uid])], dtype=torch.long)
                for key in UserTower.SPARSE_KEYS
            }
            user_dense = torch.from_numpy(self.bundle.user_dense_matrix[uid : uid + 1])
            vec = self.two_tower_model.encode_user(user_sparse, user_dense)[0].numpy()

        self.user_cache.set(user_id, vec)
        ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return vec, False, ms

    def _score_stage2_candidates(
        self,
        user_id: int,
        candidates: list[RetrievalCandidate],
        device_type: int,
        hour_of_day: int,
        day_of_week: int,
        active_category_id: int | None,
    ) -> tuple[list[dict[str, Any]], float]:
        """Vectorized Stage 2 DCNv2 batch scoring and diversity-aware re-ranking."""
        t0 = time.perf_counter_ns()
        if not candidates:
            return [], 0.0

        uid = min(max(1, user_id), len(self.bundle.users))
        user = self.bundle.users[uid]
        n_cand = len(candidates)

        cand_ids = np.array([c.item_id for c in candidates], dtype=np.int64)
        ann_scores = np.array([c.ann_score for c in candidates], dtype=np.float32)
        cat_ids = self._item_category[cand_ids]
        brand_ids = self._item_brand[cand_ids]
        price_tiers = self._item_price[cand_ids]

        # Build sparse_matrix of shape (N_cand, 7):
        # [user_id, item_id, category_id, brand_id, age_bucket, region, device_type]
        sparse_matrix = np.empty((n_cand, 7), dtype=np.int64)
        sparse_matrix[:, 0] = uid
        sparse_matrix[:, 1] = cand_ids
        sparse_matrix[:, 2] = cat_ids
        sparse_matrix[:, 3] = brand_ids
        sparse_matrix[:, 4] = user.age_bucket
        sparse_matrix[:, 5] = user.region
        sparse_matrix[:, 6] = device_type

        # Build dense_features of shape (N_cand, 26):
        # [user_dense(8), item_dense(8), session_dense(6), cross_features(4)]
        dense_features = np.empty((n_cand, 26), dtype=np.float32)
        dense_features[:, 0:8] = user.dense_features[None, :]
        item_dense_batch = self._item_dense[cand_ids]
        dense_features[:, 8:16] = item_dense_batch

        pref_c1, pref_c2 = user.preferred_categories
        if active_category_id is not None:
            cat_match = (
                (cat_ids == active_category_id) | (cat_ids == pref_c1) | (cat_ids == pref_c2)
            ).astype(np.float32)
        else:
            cat_match = ((cat_ids == pref_c1) | (cat_ids == pref_c2)).astype(np.float32)

        price_match = 1.0 - np.abs(float(user.device_tier) - price_tiers.astype(np.float32)) / 5.0
        dot_sim = item_dense_batch[:, :6] @ user.dense_features[:6]

        # Session context features (6D)
        dense_features[:, 16] = float(hour_of_day) / 24.0
        dense_features[:, 17] = float(day_of_week) / 7.0
        dense_features[:, 18] = cat_match
        dense_features[:, 19] = price_match
        dense_features[:, 20] = self._item_hist_ctr[cand_ids]
        dense_features[:, 21] = self._item_hist_cvr[cand_ids]

        # Explicit cross features (4D)
        dense_features[:, 22] = dot_sim
        dense_features[:, 23] = cat_match
        dense_features[:, 24] = price_match
        dense_features[:, 25] = dot_sim * cat_match

        p_ctr, p_cvr = self.scorer.score_batch(sparse_matrix, dense_features)
        combined_scores = (
            p_ctr + self.cvr_ranking_alpha * p_cvr + 0.15 * ann_scores + 0.05 * cat_match
        )

        # Sort by combined utility with category saturation penalty for feed diversity
        order = np.argsort(-combined_scores)
        category_counts: dict[int, int] = {}
        ranked: list[dict[str, Any]] = []

        for idx in order:
            iid = int(cand_ids[idx])
            cid = int(cat_ids[idx])
            seen = category_counts.get(cid, 0)
            diversity_penalty = 0.04 * seen
            adjusted_score = float(combined_scores[idx]) - diversity_penalty
            category_counts[cid] = seen + 1

            ranked.append(
                {
                    "item_id": iid,
                    "category_id": cid,
                    "p_ctr": float(p_ctr[idx]),
                    "p_cvr": float(p_cvr[idx]),
                    "final_score": adjusted_score,
                    "stage1_score": float(ann_scores[idx]),
                    "is_exploration": False,
                    "ucb_score": None,
                }
            )

        ranked.sort(key=lambda r: r["final_score"], reverse=True)
        for rank_idx, item_dict in enumerate(ranked, start=1):
            item_dict["rank"] = rank_idx

        ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return ranked, ms

    def build_bandit_context(self, user_id: int, user_vector: np.ndarray) -> np.ndarray:
        """Construct 16D context vector for LinUCB cold-start evaluation."""
        uid = min(max(1, user_id), len(self.bundle.users))
        user = self.bundle.users[uid]
        ctx = np.empty(16, dtype=np.float32)
        ctx[:8] = user_vector[:8]
        ctx[8:] = user.latent_factor[:8]
        return ctx

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """Execute full two-stage retrieval -> DCNv2 ranking -> LinUCB exploration funnel."""
        total_start = time.perf_counter_ns()

        # 1. Resolve User Embedding (Redis Cache / UserTower)
        user_vec, cache_hit, cache_ms = self.resolve_user_vector(request.user_id)

        # 2. Stage 1: HNSW ANN Candidate Retrieval (Top-100)
        exclude_set = (
            set(request.session_context.exclude_item_ids)
            if request.session_context.exclude_item_ids
            else None
        )
        ann_res = self.ann_engine.search(
            user_vector=user_vec,
            top_k=request.top_k_retrieval,
            exclude_item_ids=exclude_set,
        )
        stage1_ms = ann_res.latency_ms

        # 3. Stage 2: DCNv2 Multi-Task Scoring & Diversity Re-Ranking
        ranked_items, stage2_ms = self._score_stage2_candidates(
            user_id=request.user_id,
            candidates=ann_res.candidates,
            device_type=request.session_context.device_type,
            hour_of_day=request.session_context.hour_of_day,
            day_of_week=request.session_context.day_of_week,
            active_category_id=request.session_context.active_category_id,
        )

        # 4. Stage 3: LinUCB Contextual Bandit Cold-Start Exploration (5% traffic slice)
        t_bandit = time.perf_counter_ns()
        if request.enable_exploration:
            bandit_ctx = self.build_bandit_context(request.user_id, user_vec)
            final_dicts, explore_count = self.bandit.interleave_exploration(
                ranked_items=ranked_items,
                context_vector=bandit_ctx,
                top_k=request.top_k_final,
            )
        else:
            final_dicts = ranked_items[: request.top_k_final]
            explore_count = 0
        bandit_ms = (time.perf_counter_ns() - t_bandit) / 1_000_000.0

        total_ms = (time.perf_counter_ns() - total_start) / 1_000_000.0

        rec_models = [
            RecommendedItem(
                item_id=entry["item_id"],
                rank=entry["rank"],
                category_id=entry["category_id"],
                p_ctr=entry["p_ctr"],
                p_cvr=entry["p_cvr"],
                final_score=entry["final_score"],
                stage1_score=entry["stage1_score"],
                is_exploration=entry["is_exploration"],
                ucb_score=entry.get("ucb_score"),
            )
            for entry in final_dicts
        ]

        unique_cats = len({item.category_id for item in rec_models})
        diversity_ratio = unique_cats / max(1, len(rec_models))

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=rec_models,
            candidate_count_stage1=len(ann_res.candidates),
            exploration_count=explore_count,
            category_diversity_ratio=round(diversity_ratio, 4),
            cache_hit=cache_hit,
            latency=LatencyBreakdown(
                user_cache_ms=round(cache_ms, 4),
                stage1_ann_ms=round(stage1_ms, 4),
                stage2_ranking_ms=round(stage2_ms, 4),
                bandit_rerank_ms=round(bandit_ms, 4),
                total_ms=round(total_ms, 4),
            ),
        )
