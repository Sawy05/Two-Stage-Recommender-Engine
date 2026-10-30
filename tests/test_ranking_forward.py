"""Unit Tests for Stage 2 DCNv2 Ranking Model, AUC Metrics, and TorchScript/ONNX Export."""

from __future__ import annotations

from pathlib import Path

import torch

from src.data.dataloader import create_ranking_dataloader
from src.data.synthetic_interactions import SyntheticEcommerceGenerator
from src.ranking.dcn_v2 import DCNv2Config, DCNv2RankingModel
from src.ranking.export import (
    ExportedDCNv2Scorer,
    export_to_onnx,
    export_to_torchscript,
    verify_exported_parity,
)
from src.ranking.trainer import DCNv2Trainer


def test_dcn_v2_forward_shapes_and_probability_bounds() -> None:
    """Verify DCNv2 forward pass returns valid calibrated probabilities (pCTR, pCVR)."""
    cfg = DCNv2Config()
    model = DCNv2RankingModel(cfg)
    batch_size = 32

    sparse_dict = {
        "user_id": torch.randint(1, 100, (batch_size,)),
        "item_id": torch.randint(1, 200, (batch_size,)),
        "category_id": torch.randint(1, 20, (batch_size,)),
        "brand_id": torch.randint(1, 50, (batch_size,)),
        "age_bucket": torch.randint(1, 7, (batch_size,)),
        "region": torch.randint(1, 10, (batch_size,)),
        "device_type": torch.randint(1, 4, (batch_size,)),
    }
    dense_features = torch.randn(batch_size, cfg.dense_feature_dim)

    p_ctr, p_cvr = model(sparse_dict, dense_features)
    assert p_ctr.shape == (batch_size,)
    assert p_cvr.shape == (batch_size,)
    assert torch.all((p_ctr > 0.0) & (p_ctr < 1.0))
    assert torch.all((p_cvr > 0.0) & (p_cvr < 1.0))


def test_dcn_v2_loss_convergence_and_ranking_auc() -> None:
    """Verify DCNv2 multi-task BCE loss converges and achieves strong ROC-AUC & NDCG@10."""
    torch.manual_seed(42)
    gen = SyntheticEcommerceGenerator(
        num_users=150,
        num_items=300,
        num_interactions=1800,
        seed=42,
    )
    bundle = gen.generate()
    loader = create_ranking_dataloader(bundle, batch_size=128, shuffle=True)

    model = DCNv2RankingModel(DCNv2Config())
    trainer = DCNv2Trainer(model, lr=3e-3)

    losses = trainer.fit(loader, epochs=4)
    assert len(losses) == 4
    assert losses[-1] < losses[0], f"Expected DCNv2 loss to decrease: {losses}"

    metrics = trainer.evaluate(loader, k=10)
    assert metrics.ctr_auc > 0.65, f"Expected CTR AUC > 0.65, got {metrics.ctr_auc:.4f}"
    assert metrics.cvr_auc > 0.65, f"Expected CVR AUC > 0.65, got {metrics.cvr_auc:.4f}"
    assert metrics.ndcg_at_10 > 0.65, f"Expected NDCG@10 > 0.65, got {metrics.ndcg_at_10:.4f}"


def test_dcn_v2_torchscript_and_onnx_export(tmp_path: Path) -> None:
    """Verify TorchScript and ONNX exports succeed and preserve numerical parity."""
    torch.manual_seed(11)
    model = DCNv2RankingModel(DCNv2Config()).eval()

    ts_path = tmp_path / "dcn_v2_scripted.pt"
    onnx_path = tmp_path / "dcn_v2.onnx"

    exported_ts = export_to_torchscript(model, ts_path, batch_size=16)
    exported_onnx = export_to_onnx(model, onnx_path, batch_size=16)

    assert exported_ts.exists() and exported_ts.stat().st_size > 0
    assert exported_onnx.exists() and exported_onnx.stat().st_size > 0

    max_diff = verify_exported_parity(model, exported_ts, batch_size=16, atol=1e-5)
    assert max_diff < 1e-5

    scorer = ExportedDCNv2Scorer(model, torchscript_path=exported_ts)
    sp = torch.ones((100, 7), dtype=torch.long).numpy()
    dn = torch.randn((100, 26), dtype=torch.float32).numpy()
    p_ctr, p_cvr = scorer.score_batch(sp, dn)
    assert p_ctr.shape == (100,)
    assert p_cvr.shape == (100,)
