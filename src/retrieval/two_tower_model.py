"""Stage 1: Two-Tower Neural Candidate Retrieval Architecture in PyTorch.

Implements independent User and Item neural towers mapping sparse categorical IDs
and dense behavioral/catalog features into a shared L2-normalized 64-dimensional
embedding space, trained with in-batch negative sampling and sampled softmax loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.synthetic_interactions import SyntheticDatasetBundle


@dataclass
class TwoTowerConfig:
    """Configuration for the Two-Tower candidate retrieval model."""

    embedding_dim: int = 64
    sparse_embedding_dim: int = 16
    user_sparse_cardinalities: Dict[str, int] = field(
        default_factory=lambda: {
            "user_id": 2001,
            "age_bucket": 8,
            "gender": 4,
            "region": 12,
            "device_tier": 4,
        }
    )
    item_sparse_cardinalities: Dict[str, int] = field(
        default_factory=lambda: {
            "item_id": 5001,
            "category_id": 41,
            "brand_id": 121,
            "price_tier": 6,
        }
    )
    user_dense_dim: int = 8
    item_dense_dim: int = 8
    tower_hidden_dims: Tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    temperature: float = 0.07


class TowerMLP(nn.Module):
    """Multi-layer projection network with LayerNorm, GELU, and L2 normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.LayerNorm(input_dim)]
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.net(x)
        return F.normalize(projected, p=2, dim=-1, eps=1e-8)


class UserTower(nn.Module):
    """User encoder mapping sparse categorical IDs and dense features to R^64."""

    SPARSE_KEYS: Tuple[str, ...] = (
        "user_id",
        "age_bucket",
        "gender",
        "region",
        "device_tier",
    )

    def __init__(self, config: TwoTowerConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(
                    num_embeddings=config.user_sparse_cardinalities[key],
                    embedding_dim=config.sparse_embedding_dim,
                    padding_idx=0,
                )
                for key in self.SPARSE_KEYS
            }
        )
        self.dense_norm = nn.LayerNorm(config.user_dense_dim)
        total_in_dim = (
            len(self.SPARSE_KEYS) * config.sparse_embedding_dim + config.user_dense_dim
        )
        self.mlp = TowerMLP(
            input_dim=total_in_dim,
            hidden_dims=config.tower_hidden_dims,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        sparse_features: Dict[str, torch.Tensor],
        dense_features: torch.Tensor,
    ) -> torch.Tensor:
        emb_list = [
            self.embeddings[key](
                sparse_features[key].clamp(0, self.config.user_sparse_cardinalities[key] - 1)
            )
            for key in self.SPARSE_KEYS
        ]
        dense_normed = self.dense_norm(dense_features)
        concat = torch.cat([*emb_list, dense_normed], dim=-1)
        return self.mlp(concat)


class ItemTower(nn.Module):
    """Item encoder mapping sparse catalog IDs and dense attributes to R^64."""

    SPARSE_KEYS: Tuple[str, ...] = (
        "item_id",
        "category_id",
        "brand_id",
        "price_tier",
    )

    def __init__(self, config: TwoTowerConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(
                    num_embeddings=config.item_sparse_cardinalities[key],
                    embedding_dim=config.sparse_embedding_dim,
                    padding_idx=0,
                )
                for key in self.SPARSE_KEYS
            }
        )
        self.dense_norm = nn.LayerNorm(config.item_dense_dim)
        total_in_dim = (
            len(self.SPARSE_KEYS) * config.sparse_embedding_dim + config.item_dense_dim
        )
        self.mlp = TowerMLP(
            input_dim=total_in_dim,
            hidden_dims=config.tower_hidden_dims,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        sparse_features: Dict[str, torch.Tensor],
        dense_features: torch.Tensor,
    ) -> torch.Tensor:
        emb_list = [
            self.embeddings[key](
                sparse_features[key].clamp(0, self.config.item_sparse_cardinalities[key] - 1)
            )
            for key in self.SPARSE_KEYS
        ]
        dense_normed = self.dense_norm(dense_features)
        concat = torch.cat([*emb_list, dense_normed], dim=-1)
        return self.mlp(concat)


class TwoTowerRetrievalModel(nn.Module):
    """Two-Tower Dual Encoder with In-Batch Negative Sampling and Sampled Softmax Loss."""

    def __init__(self, config: Optional[TwoTowerConfig] = None) -> None:
        super().__init__()
        self.config = config or TwoTowerConfig()
        self.user_tower = UserTower(self.config)
        self.item_tower = ItemTower(self.config)

    def encode_user(
        self,
        user_sparse: Dict[str, torch.Tensor],
        user_dense: torch.Tensor,
    ) -> torch.Tensor:
        """Encode user sparse & dense features into L2-normalized 64D embeddings."""
        return self.user_tower(user_sparse, user_dense)

    def encode_item(
        self,
        item_sparse: Dict[str, torch.Tensor],
        item_dense: torch.Tensor,
    ) -> torch.Tensor:
        """Encode item sparse & dense features into L2-normalized 64D embeddings."""
        return self.item_tower(item_sparse, item_dense)

    def forward(
        self,
        user_sparse: Dict[str, torch.Tensor],
        user_dense: torch.Tensor,
        item_sparse: Dict[str, torch.Tensor],
        item_dense: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute user embeddings, item embeddings, and scaled cosine similarity matrix."""
        user_emb = self.encode_user(user_sparse, user_dense)
        item_emb = self.encode_item(item_sparse, item_dense)
        logits = torch.matmul(user_emb, item_emb.transpose(0, 1)) / self.config.temperature
        return user_emb, item_emb, logits

    def compute_loss(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        item_ids: Optional[torch.Tensor] = None,
        extra_neg_item_emb: Optional[torch.Tensor] = None,
        candidate_sampling_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute in-batch negative sampled softmax loss with optional log-Q correction.

        Args:
            user_emb: Normalized user embeddings of shape (B, D).
            item_emb: Normalized positive item embeddings of shape (B, D).
            item_ids: Optional item IDs of shape (B,) to mask accidental duplicate positives.
            extra_neg_item_emb: Optional uniformly/popularity sampled negative item embeddings (M, D).
            candidate_sampling_probs: Optional sampling probabilities Q(y) for log-Q correction.
        """
        batch_size = user_emb.size(0)
        if extra_neg_item_emb is not None:
            all_candidates = torch.cat([item_emb, extra_neg_item_emb], dim=0)
        else:
            all_candidates = item_emb

        logits = torch.matmul(user_emb, all_candidates.transpose(0, 1)) / self.config.temperature

        # Log-Q correction for sampled softmax when candidate sampling probabilities are provided
        if candidate_sampling_probs is not None:
            log_q = torch.log(candidate_sampling_probs.clamp(min=1e-8)).unsqueeze(0)
            logits = logits - log_q

        # Mask accidental false negatives when the same item_id occurs multiple times in batch
        if item_ids is not None:
            same_item = item_ids.unsqueeze(1) == item_ids.unsqueeze(0)  # (B, B)
            diag_mask = torch.eye(batch_size, dtype=torch.bool, device=user_emb.device)
            accidental_hits = same_item & (~diag_mask)
            if extra_neg_item_emb is not None:
                pad_mask = torch.zeros(
                    (batch_size, extra_neg_item_emb.size(0)),
                    dtype=torch.bool,
                    device=user_emb.device,
                )
                accidental_hits = torch.cat([accidental_hits, pad_mask], dim=1)
            logits = logits.masked_fill(accidental_hits, -1e4)

        targets = torch.arange(batch_size, dtype=torch.long, device=user_emb.device)
        loss_u2i = F.cross_entropy(logits, targets)

        # Symmetric item-to-user contrastive regularizer on in-batch matrix
        in_batch_logits = logits[:, :batch_size]
        loss_i2u = F.cross_entropy(in_batch_logits.transpose(0, 1), targets)

        return 0.5 * (loss_u2i + loss_i2u)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Resolve execution device with automatic GPU-to-CPU fallback."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_two_tower(
    model: TwoTowerRetrievalModel,
    dataloader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Train the Two-Tower retrieval model and return epoch loss history."""
    resolved_device = device or get_device()
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_losses: List[float] = []
    model.train()

    for _ in range(epochs):
        running_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            user_sparse = {
                k: batch[k].to(resolved_device) for k in UserTower.SPARSE_KEYS
            }
            user_dense = batch["user_dense"].to(resolved_device)
            item_sparse = {
                k: batch[k].to(resolved_device) for k in ItemTower.SPARSE_KEYS
            }
            item_dense = batch["item_dense"].to(resolved_device)

            optimizer.zero_grad()
            user_emb = model.encode_user(user_sparse, user_dense)
            item_emb = model.encode_item(item_sparse, item_dense)
            loss = model.compute_loss(
                user_emb=user_emb,
                item_emb=item_emb,
                item_ids=item_sparse["item_id"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        epoch_losses.append(running_loss / max(1, num_batches))

    return epoch_losses


@torch.no_grad()
def evaluate_recall_at_k(
    model: TwoTowerRetrievalModel,
    bundle: SyntheticDatasetBundle,
    k_values: Sequence[int] = (50, 100),
    max_eval_users: int = 200,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate Recall@K for Stage 1 retrieval over positive test interactions."""
    resolved_device = device or get_device()
    model.to(resolved_device)
    model.eval()

    # Precompute all item embeddings (1..num_items)
    num_items = len(bundle.items)
    item_ids = np.arange(1, num_items + 1, dtype=np.int64)
    item_sparse = {
        key: torch.from_numpy(bundle.item_sparse_matrix[key][1 : num_items + 1]).to(
            resolved_device
        )
        for key in ItemTower.SPARSE_KEYS
    }
    item_dense = torch.from_numpy(bundle.item_dense_matrix[1 : num_items + 1]).to(
        resolved_device
    )
    all_item_emb = model.encode_item(item_sparse, item_dense)  # (N_items, 64)

    # Collect ground-truth positive items per user
    user_positives: Dict[int, set[int]] = {}
    for ev in bundle.interactions:
        if ev.clicked == 1 or ev.converted == 1:
            user_positives.setdefault(ev.user_id, set()).add(ev.item_id)

    eval_uids = sorted(user_positives.keys())[:max_eval_users]
    if not eval_uids:
        return {f"Recall@{k}": 0.0 for k in k_values}

    u_idx = np.array(eval_uids, dtype=np.int64)
    user_sparse = {
        key: torch.from_numpy(bundle.user_sparse_matrix[key][u_idx]).to(resolved_device)
        for key in UserTower.SPARSE_KEYS
    }
    user_dense = torch.from_numpy(bundle.user_dense_matrix[u_idx]).to(resolved_device)
    user_emb = model.encode_user(user_sparse, user_dense)  # (U, 64)

    sim_matrix = torch.matmul(user_emb, all_item_emb.transpose(0, 1)).cpu().numpy()
    max_k = min(max(k_values), num_items)
    top_indices = np.argpartition(-sim_matrix, kth=max_k - 1, axis=1)[:, :max_k]

    recalls: Dict[str, List[float]] = {f"Recall@{k}": [] for k in k_values}
    for row_i, uid in enumerate(eval_uids):
        row_candidates = top_indices[row_i]
        row_scores = sim_matrix[row_i, row_candidates]
        sorted_order = np.argsort(-row_scores)
        ranked_item_ids = item_ids[row_candidates[sorted_order]]

        gt_items = user_positives[uid]
        for k in k_values:
            top_k_set = set(ranked_item_ids[:k].tolist())
            hits = len(gt_items.intersection(top_k_set))
            recalls[f"Recall@{k}"].append(hits / max(1, min(len(gt_items), k)))

    return {metric: float(np.mean(vals)) for metric, vals in recalls.items()}
