"""Upper Confidence Bound (LinUCB) Contextual Bandit for Cold-Start Exploration.

Maintains per-arm ridge regression covariance matrices A_a and reward vectors b_a
using Sherman-Morrison rank-1 inverse updates for sub-millisecond UCB scoring:
    UCB_a(x) = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)
Allocates a configurable 5% exploration traffic slice to newly added cold-start
items with zero historical interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class LinUCBConfig:
    """Configuration for the Disjoint LinUCB Contextual Bandit."""

    context_dim: int = 16
    alpha: float = 1.25
    l2_reg: float = 1.0
    exploration_traffic_slice: float = 0.05
    cold_start_max_impressions: int = 5


@dataclass
class ArmState:
    """Ridge regression sufficient statistics for a single item arm."""

    item_id: int
    a_inv: np.ndarray  # shape: (d, d)
    b: np.ndarray  # shape: (d,)
    theta: np.ndarray  # shape: (d,)
    impressions: int = 0
    cumulative_reward: float = 0.0
    is_cold_start: bool = True
    category_id: int = 1


@dataclass
class BanditCandidateScore:
    """UCB breakdown for a candidate cold-start item."""

    item_id: int
    ucb_score: float
    expected_reward: float
    uncertainty_bonus: float
    impressions: int
    category_id: int


class LinUCBContextualBandit:
    """Disjoint LinUCB Contextual Bandit with Sherman-Morrison O(d^2) updates."""

    def __init__(self, config: Optional[LinUCBConfig] = None) -> None:
        self.config = config or LinUCBConfig()
        self.arms: Dict[int, ArmState] = {}
        self.total_requests: int = 0
        self.total_exploration_slots: int = 0
        self.total_impressions_served: int = 0

    def register_cold_start_items(
        self,
        item_ids: Sequence[int],
        item_categories: Optional[Dict[int, int]] = None,
        prior_vectors: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """Register newly added items with zero interaction history into the bandit pool."""
        d = self.config.context_dim
        inv_init = (1.0 / self.config.l2_reg) * np.eye(d, dtype=np.float32)

        for iid in item_ids:
            int_id = int(iid)
            if int_id in self.arms:
                continue
            b_init = np.zeros(d, dtype=np.float32)
            if prior_vectors and int_id in prior_vectors:
                vec = np.asarray(prior_vectors[int_id], dtype=np.float32).reshape(-1)
                b_init[: min(d, len(vec))] = 0.05 * vec[: min(d, len(vec))]
            theta_init = inv_init @ b_init
            cat_id = item_categories.get(int_id, (int_id % 40) + 1) if item_categories else (int_id % 40) + 1
            self.arms[int_id] = ArmState(
                item_id=int_id,
                a_inv=inv_init.copy(),
                b=b_init,
                theta=theta_init,
                impressions=0,
                cumulative_reward=0.0,
                is_cold_start=True,
                category_id=cat_id,
            )

    def get_cold_start_pool(self) -> List[int]:
        """Return item IDs that currently qualify as cold-start (zero/low impressions)."""
        return [
            iid
            for iid, state in self.arms.items()
            if state.is_cold_start and state.impressions <= self.config.cold_start_max_impressions
        ]

    def score_arms(
        self,
        context_vector: np.ndarray,
        candidate_item_ids: Optional[Sequence[int]] = None,
    ) -> List[BanditCandidateScore]:
        """Compute UCB_a(x) = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x) for cold-start items."""
        x = self._normalize_context(context_vector)
        pool = (
            list(candidate_item_ids)
            if candidate_item_ids is not None
            else self.get_cold_start_pool()
        )
        if not pool:
            return []

        scores: List[BanditCandidateScore] = []
        for iid in pool:
            int_id = int(iid)
            if int_id not in self.arms:
                self.register_cold_start_items([int_id])
            arm = self.arms[int_id]

            mean_reward = float(np.dot(arm.theta, x))
            quad_form = float(np.dot(x, arm.a_inv @ x))
            uncertainty = float(self.config.alpha * np.sqrt(max(0.0, quad_form)))
            # Zero-interaction bonus prioritizes strictly un-explored items (impressions == 0)
            zero_hist_bonus = 0.05 if arm.impressions == 0 else 0.0
            ucb = mean_reward + uncertainty + zero_hist_bonus

            scores.append(
                BanditCandidateScore(
                    item_id=int_id,
                    ucb_score=ucb,
                    expected_reward=mean_reward,
                    uncertainty_bonus=uncertainty,
                    impressions=arm.impressions,
                    category_id=arm.category_id,
                )
            )

        scores.sort(key=lambda s: s.ucb_score, reverse=True)
        return scores

    def update(
        self,
        item_id: int,
        context_vector: np.ndarray,
        reward: float,
    ) -> None:
        """Perform Sherman-Morrison O(d^2) rank-1 inverse update for arm `item_id`."""
        int_id = int(item_id)
        if int_id not in self.arms:
            self.register_cold_start_items([int_id])
        arm = self.arms[int_id]
        x = self._normalize_context(context_vector)

        # Sherman-Morrison update: A^{-1}_{new} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        a_inv_x = arm.a_inv @ x
        denom = 1.0 + float(np.dot(x, a_inv_x))
        arm.a_inv = arm.a_inv - np.outer(a_inv_x, a_inv_x) / max(denom, 1e-8)
        arm.b = arm.b + float(reward) * x
        arm.theta = arm.a_inv @ arm.b
        arm.impressions += 1
        arm.cumulative_reward += float(reward)

        if arm.impressions > self.config.cold_start_max_impressions:
            arm.is_cold_start = False

    def interleave_exploration(
        self,
        ranked_items: List[Dict[str, Any]],
        context_vector: np.ndarray,
        top_k: int = 20,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Allocate a 5% exploration traffic slice to cold-start items in the final feed.

        For a feed of size `top_k` (e.g. 20), 5% corresponds to 1 dedicated exploration
        slot (or proportional slot allocation) populated by the highest-UCB cold-start item.
        """
        self.total_requests += 1
        trimmed = list(ranked_items[:top_k])
        if not trimmed:
            return trimmed, 0

        # Calculate exact number of exploration slots for the 5% traffic slice
        raw_slots = top_k * self.config.exploration_traffic_slice
        num_explore_slots = int(np.floor(raw_slots))
        fractional = raw_slots - num_explore_slots
        # Deterministic fractional scheduling so exact empirical traffic slice converges to 5%
        if fractional > 0 and ((self.total_requests * int(round(fractional * 100))) % 100) < int(
            round(fractional * 100)
        ):
            num_explore_slots += 1
        if top_k >= 20 and self.config.exploration_traffic_slice >= 0.05:
            num_explore_slots = max(1, int(round(top_k * self.config.exploration_traffic_slice)))

        cold_scores = self.score_arms(context_vector)
        if not cold_scores or num_explore_slots <= 0:
            self.total_impressions_served += len(trimmed)
            return trimmed, 0

        existing_ids = {int(item["item_id"]) for item in trimmed}
        existing_top_cats = {int(item.get("category_id", -1)) for item in trimmed[:5]}

        # Prefer high-UCB cold-start items that also diversify categories
        selected_cold: List[BanditCandidateScore] = []
        for cand in cold_scores:
            if cand.item_id in existing_ids:
                continue
            selected_cold.append(cand)
            if len(selected_cold) >= num_explore_slots * 2:
                break

        # Re-order selected_cold slightly to favor unseen categories if UCB is close
        selected_cold.sort(
            key=lambda c: c.ucb_score + (0.02 if c.category_id not in existing_top_cats else 0.0),
            reverse=True,
        )
        chosen = selected_cold[:num_explore_slots]

        if not chosen:
            self.total_impressions_served += len(trimmed)
            return trimmed, 0

        # Inject cold-start exploration items into dedicated exploration positions
        output = list(trimmed[: top_k - len(chosen)])
        for idx, cold_cand in enumerate(chosen):
            insert_pos = min(len(output), max(2, len(output) - idx))
            exploration_entry = {
                "item_id": cold_cand.item_id,
                "category_id": cold_cand.category_id,
                "p_ctr": float(np.clip(0.12 + 0.05 * cold_cand.expected_reward, 0.01, 0.99)),
                "p_cvr": float(np.clip(0.04 + 0.02 * cold_cand.expected_reward, 0.005, 0.99)),
                "final_score": float(cold_cand.ucb_score),
                "is_exploration": True,
                "ucb_score": float(cold_cand.ucb_score),
                "uncertainty_bonus": float(cold_cand.uncertainty_bonus),
                "stage1_score": 0.0,
            }
            output.insert(insert_pos, exploration_entry)

        # Re-assign sequential rank 1..top_k
        final_feed = output[:top_k]
        for rank_idx, entry in enumerate(final_feed, start=1):
            entry["rank"] = rank_idx

        self.total_exploration_slots += len(chosen)
        self.total_impressions_served += len(final_feed)
        return final_feed, len(chosen)

    def _normalize_context(self, context_vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(context_vector, dtype=np.float32).reshape(-1)
        d = self.config.context_dim
        if len(vec) < d:
            padded = np.zeros(d, dtype=np.float32)
            padded[: len(vec)] = vec
            vec = padded
        elif len(vec) > d:
            vec = vec[:d]
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        return vec
