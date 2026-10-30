"""Stage 2: Deep & Cross Network (DCNv2) Multi-Task Ranking Engine."""

from src.ranking.dcn_v2 import CrossLayerV2, DCNv2Config, DCNv2RankingModel
from src.ranking.export import (
    ExportedDCNv2Scorer,
    export_to_onnx,
    export_to_torchscript,
    verify_exported_parity,
)
from src.ranking.trainer import DCNv2Trainer, RankingMetrics, compute_ndcg_at_k, compute_roc_auc

__all__ = [
    "CrossLayerV2",
    "DCNv2Config",
    "DCNv2RankingModel",
    "DCNv2Trainer",
    "RankingMetrics",
    "compute_roc_auc",
    "compute_ndcg_at_k",
    "export_to_torchscript",
    "export_to_onnx",
    "verify_exported_parity",
    "ExportedDCNv2Scorer",
]
