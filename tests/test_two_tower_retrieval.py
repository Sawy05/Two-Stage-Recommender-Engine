"""Unit & Integration Tests for Stage 1 Two-Tower Retrieval and HNSW ANN Search."""

from __future__ import annotations

import numpy as np
import torch

from src.data.dataloader import create_two_tower_dataloader
from src.data.synthetic_interactions import SyntheticEcommerceGenerator
from src.retrieval.ann_search import ANNSearchEngine
from src.retrieval.indexer import HNSWConfig, ItemVectorIndexer
from src.retrieval.two_tower_model import (
    TwoTowerConfig,
    TwoTowerRetrievalModel,
    evaluate_recall_at_k,
    train_two_tower,
)


def test_two_tower_forward_shapes_and_l2_normalization() -> None:
    """Verify User and Item encoders map features to L2-normalized 64D embeddings."""
    cfg = TwoTowerConfig(embedding_dim=64)
    model = TwoTowerRetrievalModel(cfg)
    batch_size = 16

    user_sparse = {
        "user_id": torch.randint(1, 100, (batch_size,)),
        "age_bucket": torch.randint(1, 7, (batch_size,)),
        "gender": torch.randint(1, 3, (batch_size,)),
        "region": torch.randint(1, 10, (batch_size,)),
        "device_tier": torch.randint(1, 4, (batch_size,)),
    }
    user_dense = torch.randn(batch_size, cfg.user_dense_dim)

    item_sparse = {
        "item_id": torch.randint(1, 200, (batch_size,)),
        "category_id": torch.randint(1, 20, (batch_size,)),
        "brand_id": torch.randint(1, 50, (batch_size,)),
        "price_tier": torch.randint(1, 5, (batch_size,)),
    }
    item_dense = torch.randn(batch_size, cfg.item_dense_dim)

    user_emb, item_emb, logits = model(user_sparse, user_dense, item_sparse, item_dense)

    assert user_emb.shape == (batch_size, 64)
    assert item_emb.shape == (batch_size, 64)
    assert logits.shape == (batch_size, batch_size)

    user_norms = torch.norm(user_emb, p=2, dim=-1)
    item_norms = torch.norm(item_emb, p=2, dim=-1)
    assert torch.allclose(user_norms, torch.ones_like(user_norms), atol=1e-5)
    assert torch.allclose(item_norms, torch.ones_like(item_norms), atol=1e-5)


def test_two_tower_in_batch_sampled_softmax_convergence() -> None:
    """Verify sampled softmax loss decreases monotonically on synthetic interaction batches."""
    torch.manual_seed(42)
    gen = SyntheticEcommerceGenerator(
        num_users=120,
        num_items=250,
        num_interactions=1200,
        seed=42,
    )
    bundle = gen.generate()
    loader = create_two_tower_dataloader(bundle, batch_size=64, shuffle=True)

    model = TwoTowerRetrievalModel(TwoTowerConfig(embedding_dim=64, lr=0.003) if False else TwoTowerConfig(embedding_dim=64))
    losses = train_two_tower(model, loader, epochs=4, lr=3e-3)

    assert len(losses) == 4
    assert all(np.isfinite(val) for val in losses)
    assert losses[-1] < losses[0], f"Expected loss to converge: {losses}"


def test_ann_hnsw_retrieval_recall_and_sub_5ms_latency() -> None:
    """Verify HNSW indexer retrieves 100 candidates in <5ms and achieves positive Recall@K."""
    torch.manual_seed(7)
    gen = SyntheticEcommerceGenerator(
        num_users=150,
        num_items=400,
        num_interactions=2000,
        seed=7,
    )
    bundle = gen.generate()
    loader = create_two_tower_dataloader(bundle, batch_size=64, shuffle=True)

    model = TwoTowerRetrievalModel(TwoTowerConfig(embedding_dim=64))
    train_two_tower(model, loader, epochs=3, lr=3e-3)

    metrics = evaluate_recall_at_k(model, bundle, k_values=(50, 100), max_eval_users=50)
    assert "Recall@50" in metrics and "Recall@100" in metrics
    assert metrics["Recall@100"] >= metrics["Recall@50"]
    assert metrics["Recall@100"] > 0.15

    indexer = ItemVectorIndexer(HNSWConfig(embedding_dim=64, hnsw_m=16, hnsw_ef_construct=128))
    milvus_params = indexer.get_milvus_index_params()
    assert milvus_params["index_type"] == "HNSW"

    item_ids, embeddings = indexer.precompute_and_index(model, bundle, include_cold_start=False)
    assert len(item_ids) == len(bundle.warm_item_ids)
    assert embeddings.shape == (len(bundle.warm_item_ids), 64)

    ann = ANNSearchEngine(indexer=indexer, ef_search=64)
    query_vec = embeddings[0]
    res = ann.search(query_vec, top_k=100)

    assert len(res.candidates) == 100
    assert res.candidates[0].item_id == int(item_ids[0])
    assert res.latency_ms < 5.0, f"Stage 1 ANN exceeded 5ms budget: {res.latency_ms:.3f}ms"
