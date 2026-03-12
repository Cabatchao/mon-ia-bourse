"""Support GNN (relations inter-actifs) avec fallback explicite."""
from __future__ import annotations


def gnn_support_status() -> str:
    try:
        import torch_geometric  # noqa: F401
        return "torch_geometric available"
    except Exception:
        return "torch_geometric not installed; GNN module disabled in this environment"
