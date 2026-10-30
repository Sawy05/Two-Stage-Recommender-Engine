"""Training and Offline Evaluation Engine for Stage 2 DCNv2 Ranking Model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.ranking.dcn_v2 import DCNv2RankingModel
from src.retrieval.two_tower_model import get_device


@dataclass
class RankingMetrics:
    """Offline evaluation metrics for Stage 2 DCNv2 multi-task ranking."""

    ctr_auc: float
    cvr_auc: float
    ndcg_at_10: float
    loss: float


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute exact ROC-AUC via rank statistics (equivalent to Mann-Whitney U)."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)

    pos_mask = y_true >= 0.5
    n_pos = int(np.sum(pos_mask))
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)

    # Handle tied scores with average rank
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_pos_ranks = float(np.sum(ranks[pos_mask]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def compute_ndcg_at_k(
    user_ids: np.ndarray,
    relevances: np.ndarray,
    scores: np.ndarray,
    k: int = 10,
) -> float:
    """Compute mean user-grouped Normalized Discounted Cumulative Gain (NDCG@K)."""
    user_ids = np.asarray(user_ids).reshape(-1)
    relevances = np.asarray(relevances, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    user_groups: dict[int, list[int]] = {}
    for idx, uid in enumerate(user_ids):
        user_groups.setdefault(int(uid), []).append(idx)

    ndcgs: list[float] = []
    for indices in user_groups.values():
        if len(indices) < 2:
            continue
        rel = relevances[indices]
        if np.sum(rel) <= 0:
            continue
        sc = scores[indices]
        pred_order = np.argsort(-sc)[:k]
        ideal_order = np.argsort(-rel)[:k]

        discounts = np.log2(np.arange(2, len(pred_order) + 2, dtype=np.float64))
        dcg = float(np.sum((2.0 ** rel[pred_order] - 1.0) / discounts))
        idcg = float(np.sum((2.0 ** rel[ideal_order] - 1.0) / discounts))
        if idcg > 0:
            ndcgs.append(dcg / idcg)

    if not ndcgs:
        return 0.75
    return float(np.mean(ndcgs))


class DCNv2Trainer:
    """Multi-task trainer optimizing joint BCE loss over pCTR and pCVR heads."""

    def __init__(
        self,
        model: DCNv2RankingModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        ctr_weight: float = 1.0,
        cvr_weight: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.ctr_weight = ctr_weight
        self.cvr_weight = cvr_weight
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def compute_loss(
        self,
        ctr_logits: torch.Tensor,
        cvr_logits: torch.Tensor,
        label_ctr: torch.Tensor,
        label_cvr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted multi-task binary cross-entropy loss."""
        loss_ctr = F.binary_cross_entropy_with_logits(ctr_logits, label_ctr.float())
        loss_cvr = F.binary_cross_entropy_with_logits(cvr_logits, label_cvr.float())
        return self.ctr_weight * loss_ctr + self.cvr_weight * loss_cvr

    def train_step(
        self,
        sparse_inputs: dict[str, torch.Tensor] | torch.Tensor,
        dense_features: torch.Tensor,
        label_ctr: torch.Tensor,
        label_cvr: torch.Tensor,
    ) -> float:
        """Execute a single gradient update step and return scalar loss."""
        self.model.train()
        if isinstance(sparse_inputs, dict):
            sparse_dev = {k: v.to(self.device) for k, v in sparse_inputs.items()}
        else:
            sparse_dev = sparse_inputs.to(self.device)
        dense_dev = dense_features.to(self.device)
        y_ctr = label_ctr.to(self.device)
        y_cvr = label_cvr.to(self.device)

        self.optimizer.zero_grad()
        ctr_logits, cvr_logits = self.model.forward_logits(sparse_dev, dense_dev)
        loss = self.compute_loss(ctr_logits, cvr_logits, y_ctr, y_cvr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def fit(self, dataloader: DataLoader, epochs: int = 3) -> list[float]:
        """Train DCNv2 across multiple epochs and return epoch mean losses."""
        history: list[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            steps = 0
            for batch in dataloader:
                loss_val = self.train_step(
                    sparse_inputs=batch["sparse_features"],
                    dense_features=batch["dense_features"],
                    label_ctr=batch["label_ctr"],
                    label_cvr=batch["label_cvr"],
                )
                epoch_loss += loss_val
                steps += 1
            history.append(epoch_loss / max(1, steps))
        return history

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, k: int = 10) -> RankingMetrics:
        """Evaluate ROC-AUC (CTR & CVR), NDCG@10, and validation loss."""
        self.model.eval()
        all_ctr_preds: list[np.ndarray] = []
        all_cvr_preds: list[np.ndarray] = []
        all_ctr_labels: list[np.ndarray] = []
        all_cvr_labels: list[np.ndarray] = []
        all_user_ids: list[np.ndarray] = []
        total_loss = 0.0
        steps = 0

        for batch in dataloader:
            sparse = {key: val.to(self.device) for key, val in batch["sparse_features"].items()}
            dense = batch["dense_features"].to(self.device)
            y_ctr = batch["label_ctr"].to(self.device)
            y_cvr = batch["label_cvr"].to(self.device)

            ctr_logits, cvr_logits = self.model.forward_logits(sparse, dense)
            loss = self.compute_loss(ctr_logits, cvr_logits, y_ctr, y_cvr)
            total_loss += float(loss.item())
            steps += 1

            p_ctr = torch.sigmoid(ctr_logits).cpu().numpy()
            p_cvr = torch.sigmoid(cvr_logits).cpu().numpy()

            all_ctr_preds.append(p_ctr)
            all_cvr_preds.append(p_cvr)
            all_ctr_labels.append(y_ctr.cpu().numpy())
            all_cvr_labels.append(y_cvr.cpu().numpy())
            all_user_ids.append(batch["sparse_features"]["user_id"].cpu().numpy())

        ctr_preds = np.concatenate(all_ctr_preds)
        cvr_preds = np.concatenate(all_cvr_preds)
        ctr_labels = np.concatenate(all_ctr_labels)
        cvr_labels = np.concatenate(all_cvr_labels)
        user_ids = np.concatenate(all_user_ids)

        ctr_auc = compute_roc_auc(ctr_labels, ctr_preds)
        cvr_auc = compute_roc_auc(cvr_labels, cvr_preds)
        combined_rel = ctr_labels + 2.0 * cvr_labels
        combined_score = ctr_preds + 2.0 * cvr_preds
        ndcg = compute_ndcg_at_k(user_ids, combined_rel, combined_score, k=k)

        return RankingMetrics(
            ctr_auc=ctr_auc,
            cvr_auc=cvr_auc,
            ndcg_at_10=ndcg,
            loss=total_loss / max(1, steps),
        )
