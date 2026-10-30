"""Synthetic E-Commerce Interaction Data Generator.

Generates realistic multi-modal e-commerce user profiles, item catalog metadata,
and timestamped interaction events (view, click, buy) with planted latent factor
affinities so retrieval (Recall@K) and ranking (AUC / NDCG@K) models exhibit
realistic convergence out of the box without external dataset downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class UserRecord:
    """User entity with sparse categorical IDs and dense behavioral features."""

    user_id: int
    age_bucket: int
    gender: int
    region: int
    device_tier: int
    dense_features: np.ndarray  # shape: (user_dense_dim,)
    preferred_categories: Tuple[int, int]
    latent_factor: np.ndarray  # shape: (16,)


@dataclass
class ItemRecord:
    """Catalog item with sparse metadata, dense attributes, and cold-start flag."""

    item_id: int
    category_id: int
    brand_id: int
    price_tier: int
    dense_features: np.ndarray  # shape: (item_dense_dim,)
    is_cold_start: bool
    historical_impressions: int
    historical_ctr: float
    historical_cvr: float
    latent_factor: np.ndarray  # shape: (16,)


@dataclass
class InteractionEvent:
    """Single user-item interaction log with session context and multi-task labels."""

    user_id: int
    item_id: int
    timestamp: int
    event_type: str  # "view", "click", or "buy"
    clicked: int
    converted: int
    device_type: int
    session_dense: np.ndarray  # shape: (context_dense_dim,)
    cross_features: np.ndarray  # shape: (4,)


@dataclass
class SyntheticDatasetBundle:
    """Container holding generated users, items, interactions, and vectorized arrays."""

    users: Dict[int, UserRecord]
    items: Dict[int, ItemRecord]
    interactions: List[InteractionEvent]
    user_sparse_matrix: Dict[str, np.ndarray]
    user_dense_matrix: np.ndarray
    item_sparse_matrix: Dict[str, np.ndarray]
    item_dense_matrix: np.ndarray
    cold_start_item_ids: List[int]
    warm_item_ids: List[int]


class SyntheticEcommerceGenerator:
    """Deterministic generator for synthetic e-commerce users, items, and interaction logs."""

    def __init__(
        self,
        num_users: int = 1000,
        num_items: int = 2000,
        num_categories: int = 40,
        num_brands: int = 120,
        num_interactions: int = 20000,
        user_dense_dim: int = 8,
        item_dense_dim: int = 8,
        context_dense_dim: int = 6,
        cold_start_item_ratio: float = 0.08,
        seed: int = 42,
    ) -> None:
        self.num_users = num_users
        self.num_items = num_items
        self.num_categories = num_categories
        self.num_brands = num_brands
        self.num_interactions = num_interactions
        self.user_dense_dim = user_dense_dim
        self.item_dense_dim = item_dense_dim
        self.context_dense_dim = context_dense_dim
        self.cold_start_item_ratio = cold_start_item_ratio
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self) -> SyntheticDatasetBundle:
        """Generate complete user, item, and interaction dataset bundle."""
        category_centroids = self.rng.normal(0.0, 1.0, size=(self.num_categories + 1, 16)).astype(
            np.float32
        )
        norms = np.linalg.norm(category_centroids, axis=1, keepdims=True) + 1e-8
        category_centroids = category_centroids / norms

        users: Dict[int, UserRecord] = {}
        user_sparse_matrix = {
            "user_id": np.zeros(self.num_users + 1, dtype=np.int64),
            "age_bucket": np.zeros(self.num_users + 1, dtype=np.int64),
            "gender": np.zeros(self.num_users + 1, dtype=np.int64),
            "region": np.zeros(self.num_users + 1, dtype=np.int64),
            "device_tier": np.zeros(self.num_users + 1, dtype=np.int64),
        }
        user_dense_matrix = np.zeros((self.num_users + 1, self.user_dense_dim), dtype=np.float32)

        for uid in range(1, self.num_users + 1):
            age_bucket = int(self.rng.integers(1, 8))
            gender = int(self.rng.integers(1, 4))
            region = int(self.rng.integers(1, 12))
            device_tier = int(self.rng.integers(1, 4))

            pref_cat_1 = ((uid + age_bucket * 3) % self.num_categories) + 1
            pref_cat_2 = ((uid * 7 + region * 2) % self.num_categories) + 1
            latent = (
                0.65 * category_centroids[pref_cat_1]
                + 0.35 * category_centroids[pref_cat_2]
                + self.rng.normal(0.0, 0.12, size=16).astype(np.float32)
            )
            latent = latent / (np.linalg.norm(latent) + 1e-8)

            dense = np.zeros(self.user_dense_dim, dtype=np.float32)
            dense[: min(self.user_dense_dim, 6)] = latent[: min(self.user_dense_dim, 6)]
            if self.user_dense_dim > 6:
                dense[6:] = self.rng.uniform(0.0, 1.0, size=self.user_dense_dim - 6).astype(
                    np.float32
                )

            users[uid] = UserRecord(
                user_id=uid,
                age_bucket=age_bucket,
                gender=gender,
                region=region,
                device_tier=device_tier,
                dense_features=dense,
                preferred_categories=(pref_cat_1, pref_cat_2),
                latent_factor=latent,
            )
            user_sparse_matrix["user_id"][uid] = uid
            user_sparse_matrix["age_bucket"][uid] = age_bucket
            user_sparse_matrix["gender"][uid] = gender
            user_sparse_matrix["region"][uid] = region
            user_sparse_matrix["device_tier"][uid] = device_tier
            user_dense_matrix[uid] = dense

        num_cold_start = max(10, int(self.num_items * self.cold_start_item_ratio))
        cold_start_cutoff = self.num_items - num_cold_start

        items: Dict[int, ItemRecord] = {}
        item_sparse_matrix = {
            "item_id": np.zeros(self.num_items + 1, dtype=np.int64),
            "category_id": np.zeros(self.num_items + 1, dtype=np.int64),
            "brand_id": np.zeros(self.num_items + 1, dtype=np.int64),
            "price_tier": np.zeros(self.num_items + 1, dtype=np.int64),
        }
        item_dense_matrix = np.zeros((self.num_items + 1, self.item_dense_dim), dtype=np.float32)
        cold_start_item_ids: List[int] = []
        warm_item_ids: List[int] = []

        for iid in range(1, self.num_items + 1):
            category_id = ((iid - 1) % self.num_categories) + 1
            brand_id = ((iid * 3) % self.num_brands) + 1
            price_tier = int((iid % 5) + 1)
            is_cold = iid > cold_start_cutoff

            latent = category_centroids[category_id] + self.rng.normal(
                0.0, 0.10, size=16
            ).astype(np.float32)
            latent = latent / (np.linalg.norm(latent) + 1e-8)

            quality_score = float(np.clip(0.5 + 0.25 * latent[0], 0.05, 0.95))
            hist_impr = 0 if is_cold else int(self.rng.integers(50, 2000))
            hist_ctr = 0.0 if is_cold else float(np.clip(0.08 + 0.18 * quality_score, 0.01, 0.45))
            hist_cvr = 0.0 if is_cold else float(np.clip(0.02 + 0.12 * quality_score, 0.005, 0.25))

            dense = np.zeros(self.item_dense_dim, dtype=np.float32)
            dense[: min(self.item_dense_dim, 6)] = latent[: min(self.item_dense_dim, 6)]
            if self.item_dense_dim > 6:
                dense[6] = quality_score
            if self.item_dense_dim > 7:
                dense[7] = float(price_tier) / 5.0

            items[iid] = ItemRecord(
                item_id=iid,
                category_id=category_id,
                brand_id=brand_id,
                price_tier=price_tier,
                dense_features=dense,
                is_cold_start=is_cold,
                historical_impressions=hist_impr,
                historical_ctr=hist_ctr,
                historical_cvr=hist_cvr,
                latent_factor=latent,
            )
            item_sparse_matrix["item_id"][iid] = iid
            item_sparse_matrix["category_id"][iid] = category_id
            item_sparse_matrix["brand_id"][iid] = brand_id
            item_sparse_matrix["price_tier"][iid] = price_tier
            item_dense_matrix[iid] = dense

            if is_cold:
                cold_start_item_ids.append(iid)
            else:
                warm_item_ids.append(iid)

        # Index warm items by category for fast positive/negative sampling
        cat_to_warm_items: Dict[int, List[int]] = {c: [] for c in range(1, self.num_categories + 1)}
        for iid in warm_item_ids:
            cat_to_warm_items[items[iid].category_id].append(iid)

        interactions: List[InteractionEvent] = []
        base_ts = 1_700_000_000
        warm_items_arr = np.array(warm_item_ids, dtype=np.int64)

        for idx in range(self.num_interactions):
            uid = int(self.rng.integers(1, self.num_users + 1))
            user = users[uid]

            # 60% chance of interacting with preferred category, 40% random exploration
            if self.rng.random() < 0.60:
                target_cat = (
                    user.preferred_categories[0]
                    if self.rng.random() < 0.7
                    else user.preferred_categories[1]
                )
                candidates = cat_to_warm_items.get(target_cat, warm_item_ids)
                iid = int(candidates[self.rng.integers(0, len(candidates))])
            else:
                iid = int(warm_items_arr[self.rng.integers(0, len(warm_items_arr))])

            item = items[iid]
            affinity = float(np.dot(user.latent_factor, item.latent_factor))
            cat_match = 1.0 if item.category_id in user.preferred_categories else 0.0
            price_match = 1.0 - abs(user.device_tier - item.price_tier) / 5.0

            logit_ctr = -1.1 + 2.4 * affinity + 1.2 * cat_match + 0.5 * price_match
            prob_click = 1.0 / (1.0 + np.exp(-logit_ctr))
            clicked = int(self.rng.random() < prob_click)

            if clicked:
                logit_cvr = -1.6 + 2.2 * affinity + 1.0 * cat_match + 0.8 * price_match
                prob_buy = 1.0 / (1.0 + np.exp(-logit_cvr))
                converted = int(self.rng.random() < prob_buy)
            else:
                converted = 0

            event_type = "buy" if converted else ("click" if clicked else "view")
            device_type = int((uid + idx) % 4) + 1

            session_dense = np.array(
                [
                    float((idx % 24) / 24.0),  # normalized hour of day
                    float((idx % 7) / 7.0),  # day of week
                    float(cat_match),  # real-time session category affinity
                    float(price_match),  # price compatibility
                    float(item.historical_ctr),
                    float(item.historical_cvr),
                ],
                dtype=np.float32,
            )
            if self.context_dense_dim > 6:
                pad = np.zeros(self.context_dense_dim - 6, dtype=np.float32)
                session_dense = np.concatenate([session_dense, pad])
            else:
                session_dense = session_dense[: self.context_dense_dim]

            cross_features = np.array(
                [
                    affinity,
                    cat_match,
                    price_match,
                    affinity * cat_match,
                ],
                dtype=np.float32,
            )

            interactions.append(
                InteractionEvent(
                    user_id=uid,
                    item_id=iid,
                    timestamp=base_ts + idx * 15,
                    event_type=event_type,
                    clicked=clicked,
                    converted=converted,
                    device_type=device_type,
                    session_dense=session_dense,
                    cross_features=cross_features,
                )
            )

        return SyntheticDatasetBundle(
            users=users,
            items=items,
            interactions=interactions,
            user_sparse_matrix=user_sparse_matrix,
            user_dense_matrix=user_dense_matrix,
            item_sparse_matrix=item_sparse_matrix,
            item_dense_matrix=item_dense_matrix,
            cold_start_item_ids=cold_start_item_ids,
            warm_item_ids=warm_item_ids,
        )

    @staticmethod
    def build_ranking_dense_vector(
        user_dense: np.ndarray,
        item_dense: np.ndarray,
        session_dense: np.ndarray,
        cross_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Concatenate user, item, session, and cross features into a 26-dim dense vector."""
        if cross_features is None:
            dot_sim = float(
                np.dot(
                    user_dense[: min(len(user_dense), len(item_dense))],
                    item_dense[: min(len(user_dense), len(item_dense))],
                )
            )
            cat_match = float(session_dense[2]) if len(session_dense) > 2 else 0.5
            price_match = float(session_dense[3]) if len(session_dense) > 3 else 0.5
            cross_features = np.array(
                [dot_sim, cat_match, price_match, dot_sim * cat_match],
                dtype=np.float32,
            )
        return np.concatenate(
            [user_dense, item_dense, session_dense, cross_features], axis=-1
        ).astype(np.float32)
