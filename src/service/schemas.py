"""Pydantic v2 Schemas for the Two-Stage Recommendation API Gateway."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SessionContext(BaseModel):
    """Real-time user session context passed at inference time."""

    device_type: int = Field(default=1, ge=1, le=4, description="User device type ID (1..4)")
    hour_of_day: int = Field(default=14, ge=0, le=23, description="Current hour of day (0..23)")
    day_of_week: int = Field(default=3, ge=0, le=6, description="Current day of week (0..6)")
    active_category_id: Optional[int] = Field(
        default=None, description="Optional category currently browsed in session"
    )
    exclude_item_ids: List[int] = Field(
        default_factory=list, description="Items to exclude (e.g., already purchased)"
    )


class RecommendationRequest(BaseModel):
    """Request payload for two-stage candidate retrieval + DCNv2 ranking + LinUCB exploration."""

    user_id: int = Field(..., ge=1, description="Target user ID")
    top_k_retrieval: int = Field(
        default=100, ge=10, le=500, description="Number of Stage 1 ANN candidates"
    )
    top_k_final: int = Field(
        default=20, ge=1, le=100, description="Number of final ranked items to return"
    )
    enable_exploration: bool = Field(
        default=True, description="Enable 5% LinUCB cold-start exploration slice"
    )
    session_context: SessionContext = Field(default_factory=SessionContext)


class RecommendedItem(BaseModel):
    """Single recommended item in the ranked response feed."""

    item_id: int
    rank: int
    category_id: int
    p_ctr: float = Field(..., ge=0.0, le=1.0)
    p_cvr: float = Field(..., ge=0.0, le=1.0)
    final_score: float
    stage1_score: float
    is_exploration: bool = False
    ucb_score: Optional[float] = None


class LatencyBreakdown(BaseModel):
    """Granular stage-by-stage execution latency in milliseconds."""

    user_cache_ms: float
    stage1_ann_ms: float
    stage2_ranking_ms: float
    bandit_rerank_ms: float
    total_ms: float


class RecommendationResponse(BaseModel):
    """Complete response returned by POST /v1/recommend."""

    user_id: int
    recommendations: List[RecommendedItem]
    candidate_count_stage1: int
    exploration_count: int
    category_diversity_ratio: float
    cache_hit: bool
    latency: LatencyBreakdown


class BanditFeedbackRequest(BaseModel):
    """Online user interaction feedback for updating LinUCB cold-start parameters."""

    user_id: int = Field(..., ge=1)
    item_id: int = Field(..., ge=1)
    reward: float = Field(..., ge=0.0, le=1.0, description="Observed reward (e.g. click=0.5, buy=1.0)")


class BanditFeedbackResponse(BaseModel):
    """Response confirming online LinUCB update."""

    item_id: int
    updated_impressions: int
    is_cold_start: bool
    status: str = "updated"


class HealthResponse(BaseModel):
    """Service readiness and component health check response."""

    status: str
    indexed_items: int
    cold_start_items: int
    redis_backend: str
    ann_backend: str
