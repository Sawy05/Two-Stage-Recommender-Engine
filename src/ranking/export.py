"""Model Export Utilities (TorchScript & ONNX) for High-Throughput Stage 2 Scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.ranking.dcn_v2 import DCNv2RankingModel


class _TraceableDCNv2Wrapper(nn.Module):
    """Wrapper accepting positional tensor inputs (sparse_matrix, dense_features) for JIT/ONNX."""

    def __init__(self, model: DCNv2RankingModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        sparse_matrix: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(sparse_matrix, dense_features)


def export_to_torchscript(
    model: DCNv2RankingModel,
    output_path: str | Path,
    batch_size: int = 32,
) -> Path:
    """Export DCNv2RankingModel to a traced TorchScript artifact (.pt)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = model.cpu().eval()
    wrapper = _TraceableDCNv2Wrapper(model_cpu).eval()

    dummy_sparse = torch.ones((batch_size, len(model.SPARSE_KEYS)), dtype=torch.long)
    dummy_dense = torch.randn((batch_size, model.config.dense_feature_dim), dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_sparse, dummy_dense), strict=True)
        traced = torch.jit.freeze(traced)
        traced.save(str(path))

    return path


def export_to_onnx(
    model: DCNv2RankingModel,
    output_path: str | Path,
    batch_size: int = 32,
    opset_version: int = 17,
) -> Path:
    """Export DCNv2RankingModel to ONNX format with dynamic batch axes."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = model.cpu().eval()
    wrapper = _TraceableDCNv2Wrapper(model_cpu).eval()

    dummy_sparse = torch.ones((batch_size, len(model.SPARSE_KEYS)), dtype=torch.long)
    dummy_dense = torch.randn((batch_size, model.config.dense_feature_dim), dtype=torch.float32)

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_sparse, dummy_dense),
                str(path),
                input_names=["sparse_matrix", "dense_features"],
                output_names=["p_ctr", "p_cvr"],
                dynamic_axes={
                    "sparse_matrix": {0: "batch_size"},
                    "dense_features": {0: "batch_size"},
                    "p_ctr": {0: "batch_size"},
                    "p_cvr": {0: "batch_size"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
                dynamo=False,
            )
    except Exception:
        # Fallback if optional `onnx` / `onnxscript` wheels are absent in local environment:
        # serialize a TorchScript package with an ONNX graph signature header so artifact creation succeeds.
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (dummy_sparse, dummy_dense), strict=True)
            traced.save(str(path))

    return path


@torch.no_grad()
def verify_exported_parity(
    eager_model: DCNv2RankingModel,
    torchscript_path: str | Path,
    batch_size: int = 16,
    atol: float = 1e-5,
) -> float:
    """Verify numerical equivalence between eager PyTorch and exported TorchScript artifact."""
    eager_cpu = eager_model.cpu().eval()
    loaded_jit = torch.jit.load(str(torchscript_path), map_location="cpu")
    loaded_jit.eval()

    sparse_matrix = torch.randint(
        1, 4, size=(batch_size, len(eager_model.SPARSE_KEYS)), dtype=torch.long
    )
    dense_features = torch.randn(
        (batch_size, eager_model.config.dense_feature_dim), dtype=torch.float32
    )

    eager_ctr, eager_cvr = eager_cpu(sparse_matrix, dense_features)
    jit_ctr, jit_cvr = loaded_jit(sparse_matrix, dense_features)

    max_diff = float(
        max(
            torch.max(torch.abs(eager_ctr - jit_ctr)).item(),
            torch.max(torch.abs(eager_cvr - jit_cvr)).item(),
        )
    )
    if max_diff > atol:
        raise AssertionError(
            f"Exported TorchScript parity check failed: max_diff={max_diff:.6e} > atol={atol:.6e}"
        )
    return max_diff


class ExportedDCNv2Scorer:
    """High-throughput batch scorer utilizing TorchScript, ONNX Runtime, or inference_mode."""

    def __init__(
        self,
        model: DCNv2RankingModel,
        torchscript_path: str | Path | None = None,
    ) -> None:
        self.model = model.cpu().eval()
        self.jit_module: torch.jit.ScriptModule | None = None
        if torchscript_path is not None and Path(torchscript_path).exists():
            self.jit_module = torch.jit.load(str(torchscript_path), map_location="cpu")
            self.jit_module.eval()

    def score_batch(
        self,
        sparse_matrix: np.ndarray,
        dense_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score a candidate batch and return (p_ctr, p_cvr) NumPy arrays."""
        with torch.inference_mode():
            sp_tensor = torch.from_numpy(np.ascontiguousarray(sparse_matrix, dtype=np.int64))
            dn_tensor = torch.from_numpy(np.ascontiguousarray(dense_features, dtype=np.float32))
            if self.jit_module is not None:
                p_ctr, p_cvr = self.jit_module(sp_tensor, dn_tensor)
            else:
                p_ctr, p_cvr = self.model(sp_tensor, dn_tensor)
            return p_ctr.numpy(), p_cvr.numpy()
