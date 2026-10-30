"""Stage 2: Deep & Cross Network v2 (DCNv2) Multi-Task Ranking Model in PyTorch.

Implements low-rank cross layers x_{l+1} = x_0 * (U_l (V_l^T x_l) + b_l) + x_l
coupled in parallel with a deep MLP backbone and dual task towers outputting
predicted Click-Through Rate (pCTR) and Conversion Rate (pCVR).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class DCNv2Config:
    """Configuration for the Stage 2 DCNv2 multi-task ranking architecture."""

    sparse_cardinalities: Dict[str, int] = field(
        default_factory=lambda: {
            "user_id": 2001,
            "item_id": 5001,
            "category_id": 41,
            "brand_id": 121,
            "age_bucket": 8,
            "region": 12,
            "device_type": 5,
        }
    )
    sparse_embedding_dim: int = 16
    dense_feature_dim: int = 26
    num_cross_layers: int = 3
    low_rank_dim: int = 32
    deep_hidden_dims: Tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    ctr_loss_weight: float = 1.0
    cvr_loss_weight: float = 1.0


class CrossLayerV2(nn.Module):
    """Low-rank DCNv2 feature crossing layer: x_{l+1} = x_0 * (U (V^T x_l) + b) + x_l."""

    def __init__(self, input_dim: int, low_rank_dim: int = 32) -> None:
        super().__init__()
        self.v_proj = nn.Linear(input_dim, low_rank_dim, bias=False)
        self.u_proj = nn.Linear(low_rank_dim, input_dim, bias=True)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.u_proj.weight)
        nn.init.zeros_(self.u_proj.bias)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        low_rank = torch.tanh(self.v_proj(xl))
        crossed = self.u_proj(low_rank)
        return x0 * crossed + xl


class DCNv2RankingModel(nn.Module):
    """Deep & Cross Network v2 predicting multi-task pCTR and pCVR probabilities."""

    SPARSE_KEYS: Tuple[str, ...] = (
        "user_id",
        "item_id",
        "category_id",
        "brand_id",
        "age_bucket",
        "region",
        "device_type",
    )

    def __init__(self, config: Optional[DCNv2Config] = None) -> None:
        super().__init__()
        self.config = config or DCNv2Config()

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=self.config.sparse_cardinalities[key],
                    embedding_dim=self.config.sparse_embedding_dim,
                    padding_idx=0,
                )
                for key in self.SPARSE_KEYS
            ]
        )
        self.max_indices: List[int] = [
            self.config.sparse_cardinalities[key] - 1 for key in self.SPARSE_KEYS
        ]

        self.dense_norm = nn.LayerNorm(self.config.dense_feature_dim)
        self.input_dim = (
            len(self.SPARSE_KEYS) * self.config.sparse_embedding_dim
            + self.config.dense_feature_dim
        )

        # Explicit bounded-degree Cross Network
        self.cross_layers = nn.ModuleList(
            [
                CrossLayerV2(self.input_dim, self.config.low_rank_dim)
                for _ in range(self.config.num_cross_layers)
            ]
        )
        self.cross_norm = nn.LayerNorm(self.input_dim)

        # Parallel Deep MLP Network
        deep_modules: List[nn.Module] = []
        in_d = self.input_dim
        for h_dim in self.config.deep_hidden_dims:
            deep_modules.extend(
                [
                    nn.Linear(in_d, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.GELU(),
                    nn.Dropout(self.config.dropout),
                ]
            )
            in_d = h_dim
        self.deep_net = nn.Sequential(*deep_modules)

        combined_dim = self.input_dim + self.config.deep_hidden_dims[-1]

        # Task-specific towers for pCTR and pCVR
        self.ctr_tower = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.cvr_tower = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def pack_sparse_dict(self, sparse_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stack a dictionary of 1D sparse feature tensors into a 2D tensor of shape (B, 7)."""
        cols = [sparse_features[key] for key in self.SPARSE_KEYS]
        return torch.stack(cols, dim=-1)

    def _encode_inputs(
        self,
        sparse_matrix: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> torch.Tensor:
        emb_parts: List[torch.Tensor] = []
        for idx, emb_layer in enumerate(self.embeddings):
            col = sparse_matrix[:, idx].clamp(0, self.max_indices[idx])
            emb_parts.append(emb_layer(col))
        dense_normed = self.dense_norm(dense_features)
        return torch.cat([*emb_parts, dense_normed], dim=-1)

    def forward_logits(
        self,
        sparse_inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        dense_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute raw logits (ctr_logits, cvr_logits) of shape (B,)."""
        if isinstance(sparse_inputs, dict):
            sparse_matrix = self.pack_sparse_dict(sparse_inputs)
        else:
            sparse_matrix = sparse_inputs

        x0 = self._encode_inputs(sparse_matrix, dense_features)
        xl = x0
        for cross_layer in self.cross_layers:
            xl = cross_layer(x0, xl)
        cross_out = self.cross_norm(xl)
        deep_out = self.deep_net(x0)

        fused = torch.cat([cross_out, deep_out], dim=-1)
        ctr_logits = self.ctr_tower(fused).squeeze(-1)
        cvr_logits = self.cvr_tower(fused).squeeze(-1)
        return ctr_logits, cvr_logits

    def forward(
        self,
        sparse_inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        dense_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute calibrated probabilities (pCTR, pCVR) in (0, 1) of shape (B,)."""
        ctr_logits, cvr_logits = self.forward_logits(sparse_inputs, dense_features)
        p_ctr = torch.sigmoid(ctr_logits)
        p_cvr = torch.sigmoid(cvr_logits)
        return p_ctr, p_cvr
