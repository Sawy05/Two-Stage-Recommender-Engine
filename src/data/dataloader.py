"""PyTorch Dataset and DataLoader abstractions for Stage 1 (Two-Tower) and Stage 2 (DCNv2)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.synthetic_interactions import (
    InteractionEvent,
    SyntheticDatasetBundle,
    SyntheticEcommerceGenerator,
)


class TwoTowerDataset(Dataset):
    """Dataset of positive (user, item) pairs for Two-Tower contrastive/sampled-softmax training."""

    def __init__(
        self,
        bundle: SyntheticDatasetBundle,
        positive_only: bool = True,
    ) -> None:
        self.bundle = bundle
        if positive_only:
            events = [e for e in bundle.interactions if e.clicked == 1 or e.converted == 1]
            if not events:
                events = bundle.interactions
        else:
            events = bundle.interactions
        self.events: List[InteractionEvent] = events

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        event = self.events[idx]
        user = self.bundle.users[event.user_id]
        item = self.bundle.items[event.item_id]

        return {
            "user_id": torch.tensor(user.user_id, dtype=torch.long),
            "age_bucket": torch.tensor(user.age_bucket, dtype=torch.long),
            "gender": torch.tensor(user.gender, dtype=torch.long),
            "region": torch.tensor(user.region, dtype=torch.long),
            "device_tier": torch.tensor(user.device_tier, dtype=torch.long),
            "user_dense": torch.from_numpy(user.dense_features.astype(np.float32)),
            "item_id": torch.tensor(item.item_id, dtype=torch.long),
            "category_id": torch.tensor(item.category_id, dtype=torch.long),
            "brand_id": torch.tensor(item.brand_id, dtype=torch.long),
            "price_tier": torch.tensor(item.price_tier, dtype=torch.long),
            "item_dense": torch.from_numpy(item.dense_features.astype(np.float32)),
        }


class RankingDataset(Dataset):
    """Dataset for Stage 2 Deep & Cross Network (DCNv2) multi-task CTR + CVR ranking."""

    def __init__(self, bundle: SyntheticDatasetBundle) -> None:
        self.bundle = bundle
        self.events = bundle.interactions

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        event = self.events[idx]
        user = self.bundle.users[event.user_id]
        item = self.bundle.items[event.item_id]

        dense_vec = SyntheticEcommerceGenerator.build_ranking_dense_vector(
            user_dense=user.dense_features,
            item_dense=item.dense_features,
            session_dense=event.session_dense,
            cross_features=event.cross_features,
        )

        return {
            "user_id": torch.tensor(user.user_id, dtype=torch.long),
            "item_id": torch.tensor(item.item_id, dtype=torch.long),
            "category_id": torch.tensor(item.category_id, dtype=torch.long),
            "brand_id": torch.tensor(item.brand_id, dtype=torch.long),
            "age_bucket": torch.tensor(user.age_bucket, dtype=torch.long),
            "region": torch.tensor(user.region, dtype=torch.long),
            "device_type": torch.tensor(event.device_type, dtype=torch.long),
            "dense_features": torch.from_numpy(dense_vec),
            "label_ctr": torch.tensor(float(event.clicked), dtype=torch.float32),
            "label_cvr": torch.tensor(float(event.converted), dtype=torch.float32),
        }


class RankingCollator:
    """Collator that organizes batch dictionaries into structured sparse & dense inputs."""

    SPARSE_KEYS = (
        "user_id",
        "item_id",
        "category_id",
        "brand_id",
        "age_bucket",
        "region",
        "device_type",
    )

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        sparse_features: Dict[str, torch.Tensor] = {}
        for key in self.SPARSE_KEYS:
            sparse_features[key] = torch.stack([sample[key] for sample in batch], dim=0)

        dense_features = torch.stack([sample["dense_features"] for sample in batch], dim=0)
        label_ctr = torch.stack([sample["label_ctr"] for sample in batch], dim=0)
        label_cvr = torch.stack([sample["label_cvr"] for sample in batch], dim=0)

        return {
            "sparse_features": sparse_features,
            "dense_features": dense_features,
            "label_ctr": label_ctr,
            "label_cvr": label_cvr,
        }


def create_two_tower_dataloader(
    bundle: SyntheticDatasetBundle,
    batch_size: int = 128,
    shuffle: bool = True,
    positive_only: bool = True,
) -> DataLoader:
    """Build a DataLoader for Stage 1 Two-Tower retrieval training."""
    dataset = TwoTowerDataset(bundle=bundle, positive_only=positive_only)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def create_ranking_dataloader(
    bundle: SyntheticDatasetBundle,
    batch_size: int = 128,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader for Stage 2 DCNv2 ranking training."""
    dataset = RankingDataset(bundle=bundle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=RankingCollator(),
        drop_last=False,
    )
