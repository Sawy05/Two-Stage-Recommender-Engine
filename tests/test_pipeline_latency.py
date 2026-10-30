"""End-to-End Integration Tests for Pipeline Latency (<40ms), Diversity, Bandits, and FastAPI."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from src.bandits.contextual_bandit import LinUCBConfig, LinUCBContextualBandit
from src.service.main import create_app
from src.service.pipeline import TwoStageRecommendationPipeline
from src.service.schemas import RecommendationRequest, SessionContext


def test_linucb_contextual_bandit_5_percent_traffic_slice() -> None:
    """Verify LinUCB allocates 5% exploration slots to cold-start items and updates online."""
    bandit = LinUCBContextualBandit(
        LinUCBConfig(context_dim=16, alpha=1.25, exploration_traffic_slice=0.05)
    )
    bandit.register_cold_start_items(
        [9001, 9002, 9003], item_categories={9001: 10, 9002: 11, 9003: 12}
    )

    warm_ranked = [
        {
            "item_id": i,
            "rank": i,
            "category_id": (i % 5) + 1,
            "p_ctr": 0.25,
            "p_cvr": 0.08,
            "final_score": 1.0 - i * 0.01,
            "stage1_score": 0.8,
            "is_exploration": False,
            "ucb_score": None,
        }
        for i in range(1, 25)
    ]
    ctx = np.ones(16, dtype=np.float32)
    final_feed, explore_count = bandit.interleave_exploration(warm_ranked, ctx, top_k=20)

    assert len(final_feed) == 20
    # 5% of 20 items = 1 item
    assert explore_count == 1
    explore_items = [item for item in final_feed if item["is_exploration"]]
    assert len(explore_items) == 1
    chosen_id = explore_items[0]["item_id"]
    assert chosen_id in {9001, 9002, 9003}

    # Online Sherman-Morrison update with positive reward
    bandit.update(chosen_id, ctx, reward=1.0)
    assert bandit.arms[chosen_id].impressions == 1
    assert bandit.arms[chosen_id].cumulative_reward == 1.0


def test_end_to_end_pipeline_diversity_and_sub_40ms_latency() -> None:
    """Confirm end-to-end two-stage recommendation diversity and sub-40ms P99 latency."""
    pipeline = TwoStageRecommendationPipeline.build_default(
        num_users=300,
        num_items=1000,
        num_interactions=4000,
        seed=42,
    )

    latencies = []
    for uid in range(1, 26):
        req = RecommendationRequest(
            user_id=uid,
            top_k_retrieval=100,
            top_k_final=20,
            enable_exploration=True,
            session_context=SessionContext(device_type=2, hour_of_day=18, day_of_week=4),
        )
        resp = pipeline.recommend(req)
        assert len(resp.recommendations) == 20
        assert resp.candidate_count_stage1 == 100
        assert resp.exploration_count == 1
        assert resp.category_diversity_ratio >= 0.25
        assert resp.latency.stage1_ann_ms < 5.0
        latencies.append(resp.latency.total_ms)

    # Confirm second request for user 1 hits Redis/local user vector cache
    cached_resp = pipeline.recommend(RecommendationRequest(user_id=1))
    assert cached_resp.cache_hit is True

    p99_ms = float(np.percentile(latencies, 99))
    mean_ms = float(np.mean(latencies))
    assert mean_ms < 40.0, f"Mean latency {mean_ms:.2f}ms exceeded 40ms SLA"
    assert p99_ms < 40.0, f"P99 latency {p99_ms:.2f}ms exceeded 40ms SLA"


def test_fastapi_gateway_endpoints() -> None:
    """Verify FastAPI /health, /v1/recommend, and /v1/bandit/feedback endpoints."""
    pipeline = TwoStageRecommendationPipeline.build_default(
        num_users=200,
        num_items=600,
        num_interactions=2500,
        seed=42,
    )
    app = create_app(pipeline=pipeline)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    rec_resp = client.post(
        "/v1/recommend",
        json={
            "user_id": 5,
            "top_k_retrieval": 100,
            "top_k_final": 20,
            "enable_exploration": True,
            "session_context": {
                "device_type": 1,
                "hour_of_day": 12,
                "day_of_week": 2,
                "exclude_item_ids": [1, 2],
            },
        },
    )
    assert rec_resp.status_code == 200
    payload = rec_resp.json()
    assert payload["user_id"] == 5
    assert len(payload["recommendations"]) == 20
    assert payload["exploration_count"] == 1
    assert payload["latency"]["total_ms"] < 40.0

    explore_item = next(item for item in payload["recommendations"] if item["is_exploration"])
    fb_resp = client.post(
        "/v1/bandit/feedback",
        json={
            "user_id": 5,
            "item_id": explore_item["item_id"],
            "reward": 1.0,
        },
    )
    assert fb_resp.status_code == 200
    assert fb_resp.json()["updated_impressions"] >= 1
