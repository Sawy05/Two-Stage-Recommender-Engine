"""Offline Evaluation & End-to-End Latency Benchmark Suite for Two-Stage Recommender."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bandits.contextual_bandit import LinUCBConfig, LinUCBContextualBandit  # noqa: E402
from src.data.dataloader import (  # noqa: E402
    create_ranking_dataloader,
    create_two_tower_dataloader,
)
from src.data.synthetic_interactions import SyntheticEcommerceGenerator  # noqa: E402
from src.ranking.dcn_v2 import DCNv2Config, DCNv2RankingModel  # noqa: E402
from src.ranking.trainer import DCNv2Trainer  # noqa: E402
from src.retrieval.ann_search import ANNSearchEngine  # noqa: E402
from src.retrieval.indexer import HNSWConfig, ItemVectorIndexer  # noqa: E402
from src.retrieval.two_tower_model import (  # noqa: E402
    TwoTowerConfig,
    TwoTowerRetrievalModel,
    evaluate_recall_at_k,
    train_two_tower,
)
from src.service.pipeline import (  # noqa: E402
    RedisUserVectorCache,
    TwoStageRecommendationPipeline,
)
from src.service.schemas import RecommendationRequest, SessionContext  # noqa: E402


def run_benchmark(num_queries: int = 200) -> None:
    """Run offline evaluation (Recall@K, NDCG@10, AUC) and P50/P95/P99 latency benchmark."""
    torch.manual_seed(42)
    np.random.seed(42)

    print("1. Generating synthetic e-commerce interaction dataset...")
    generator = SyntheticEcommerceGenerator(
        num_users=500,
        num_items=1500,
        num_interactions=8000,
        seed=42,
    )
    bundle = generator.generate()

    print("2. Training Stage 1 Two-Tower Retrieval Model...")
    tt_loader = create_two_tower_dataloader(bundle, batch_size=128, shuffle=True)
    two_tower = TwoTowerRetrievalModel(TwoTowerConfig(embedding_dim=64))
    tt_losses = train_two_tower(two_tower, tt_loader, epochs=4, lr=3e-3)
    recalls = evaluate_recall_at_k(two_tower, bundle, k_values=(50, 100), max_eval_users=100)

    print("3. Indexing item vectors in HNSW index...")
    indexer = ItemVectorIndexer(HNSWConfig(embedding_dim=64, hnsw_m=16, hnsw_ef_construct=128))
    indexer.precompute_and_index(two_tower, bundle, include_cold_start=False)
    ann_engine = ANNSearchEngine(indexer=indexer, ef_search=64)

    print("4. Training Stage 2 Deep & Cross Network (DCNv2) Ranking Model...")
    rank_loader = create_ranking_dataloader(bundle, batch_size=128, shuffle=True)
    dcn_model = DCNv2RankingModel(DCNv2Config())
    trainer = DCNv2Trainer(dcn_model, lr=3e-3)
    dcn_losses = trainer.fit(rank_loader, epochs=4)
    rank_metrics = trainer.evaluate(rank_loader, k=10)

    bandit = LinUCBContextualBandit(
        LinUCBConfig(context_dim=16, alpha=1.25, exploration_traffic_slice=0.05)
    )
    cat_map = {iid: bundle.items[iid].category_id for iid in bundle.cold_start_item_ids}
    prior_map = {iid: bundle.items[iid].latent_factor for iid in bundle.cold_start_item_ids}
    bandit.register_cold_start_items(
        bundle.cold_start_item_ids, item_categories=cat_map, prior_vectors=prior_map
    )

    pipeline = TwoStageRecommendationPipeline(
        bundle=bundle,
        two_tower_model=two_tower,
        dcn_model=dcn_model,
        indexer=indexer,
        ann_engine=ann_engine,
        bandit=bandit,
        user_cache=RedisUserVectorCache(),
    )
    pipeline.warm_up()

    print(f"5. Running {num_queries} end-to-end recommendation queries...")
    cache_lat, s1_lat, s2_lat, bandit_lat, total_lat = [], [], [], [], []
    diversities = []

    for i in range(num_queries):
        uid = (i % 250) + 1
        req = RecommendationRequest(
            user_id=uid,
            top_k_retrieval=100,
            top_k_final=20,
            enable_exploration=True,
            session_context=SessionContext(
                device_type=(i % 4) + 1,
                hour_of_day=(i % 24),
                day_of_week=(i % 7),
            ),
        )
        resp = pipeline.recommend(req)
        cache_lat.append(resp.latency.user_cache_ms)
        s1_lat.append(resp.latency.stage1_ann_ms)
        s2_lat.append(resp.latency.stage2_ranking_ms)
        bandit_lat.append(resp.latency.bandit_rerank_ms)
        total_lat.append(resp.latency.total_ms)
        diversities.append(resp.category_diversity_ratio)

    print("\n================ OFFLINE EVALUATION METRICS ================")
    print(f"Two-Tower Final Loss : {tt_losses[-1]:.4f}")
    print(f"Retrieval Recall@50  : {recalls['Recall@50']:.4f}")
    print(f"Retrieval Recall@100 : {recalls['Recall@100']:.4f}")
    print(f"DCNv2 Final Loss     : {dcn_losses[-1]:.4f}")
    print(f"Ranking CTR ROC-AUC  : {rank_metrics.ctr_auc:.4f}")
    print(f"Ranking CVR ROC-AUC  : {rank_metrics.cvr_auc:.4f}")
    print(f"Ranking NDCG@10      : {rank_metrics.ndcg_at_10:.4f}")
    print(f"Mean Category Div.   : {float(np.mean(diversities)):.4f}")

    print("\n================ SYSTEM LATENCY BREAKDOWN (ms) ================")
    for label, arr in [
        ("User Vector Cache", cache_lat),
        ("Stage 1 HNSW ANN ", s1_lat),
        ("Stage 2 DCNv2    ", s2_lat),
        ("Bandit Reranking ", bandit_lat),
        ("Total End-to-End ", total_lat),
    ]:
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        print(f"{label} -> P50: {p50:6.3f} ms | P95: {p95:6.3f} ms | P99: {p99:6.3f} ms")


if __name__ == "__main__":
    run_benchmark()
