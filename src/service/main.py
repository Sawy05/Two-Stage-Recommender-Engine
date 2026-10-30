"""High-Concurrency FastAPI Gateway for the Two-Stage Recommendation Engine."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI

from src.service.pipeline import TwoStageRecommendationPipeline
from src.service.schemas import (
    BanditFeedbackRequest,
    BanditFeedbackResponse,
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
)

_PIPELINE_INSTANCE: Optional[TwoStageRecommendationPipeline] = None


def get_pipeline() -> TwoStageRecommendationPipeline:
    """Return singleton TwoStageRecommendationPipeline, initializing on first request if needed."""
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = TwoStageRecommendationPipeline.build_default()
    return _PIPELINE_INSTANCE


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    """Initialize and warm up recommendation models and HNSW index on startup."""
    get_pipeline()
    yield


def create_app(
    pipeline: Optional[TwoStageRecommendationPipeline] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    global _PIPELINE_INSTANCE
    if pipeline is not None:
        _PIPELINE_INSTANCE = pipeline

    application = FastAPI(
        title="Enterprise Two-Stage Recommender API (Two-Tower + DCNv2 + LinUCB)",
        version="1.0.0",
        description=(
            "Sub-40ms two-stage recommendation funnel: Stage 1 Two-Tower HNSW ANN retrieval "
            "-> Stage 2 Deep & Cross Network (DCNv2) multi-task ranking -> LinUCB Cold-Start Bandit."
        ),
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health_check() -> HealthResponse:
        pipe = get_pipeline()
        return HealthResponse(
            status="healthy",
            indexed_items=len(pipe.indexer.item_ids),
            cold_start_items=len(pipe.bandit.get_cold_start_pool()),
            redis_backend=pipe.user_cache.backend_name,
            ann_backend="qdrant_hnsw" if pipe.indexer._using_qdrant else "in_memory_hnsw",
        )

    @application.post(
        "/v1/recommend",
        response_model=RecommendationResponse,
        tags=["recommendations"],
    )
    async def recommend(request: RecommendationRequest) -> RecommendationResponse:
        pipe = get_pipeline()
        return pipe.recommend(request)

    @application.post(
        "/v1/bandit/feedback",
        response_model=BanditFeedbackResponse,
        tags=["bandits"],
    )
    async def record_bandit_feedback(
        feedback: BanditFeedbackRequest,
    ) -> BanditFeedbackResponse:
        pipe = get_pipeline()
        user_vec, _, _ = pipe.resolve_user_vector(feedback.user_id)
        ctx = pipe.build_bandit_context(feedback.user_id, user_vec)
        pipe.bandit.update(
            item_id=feedback.item_id,
            context_vector=ctx,
            reward=feedback.reward,
        )
        arm = pipe.bandit.arms[feedback.item_id]
        return BanditFeedbackResponse(
            item_id=feedback.item_id,
            updated_impressions=arm.impressions,
            is_cold_start=arm.is_cold_start,
            status="updated",
        )

    return application


app = create_app()
