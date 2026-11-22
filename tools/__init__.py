"""DebitMatch AI tool layer: deterministic calculator, RAG engine, data contracts.

The RAG engine is imported lazily so that lightweight consumers (e.g. the
calculator inside an MCP server) do not pay ChromaDB's import cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tools.calculator import IsrParameters, NetImpactResult, calculate_net_impact
from tools.schemas import (
    BankProduct,
    CatalogConsistencyError,
    CatalogIssue,
    MonthlyFee,
    Penalty,
    TransferFee,
    YieldTier,
    audit_catalog,
    load_catalog,
)

if TYPE_CHECKING:
    from tools.rag_engine import InstitutionalRAGEngine

__all__ = [
    "BankProduct",
    "CatalogConsistencyError",
    "CatalogIssue",
    "InstitutionalRAGEngine",
    "IsrParameters",
    "MonthlyFee",
    "NetImpactResult",
    "Penalty",
    "TransferFee",
    "YieldTier",
    "audit_catalog",
    "calculate_net_impact",
    "load_catalog",
]


def __getattr__(name: str) -> Any:
    if name == "InstitutionalRAGEngine":
        from tools.rag_engine import InstitutionalRAGEngine

        return InstitutionalRAGEngine
    raise AttributeError(f"module 'tools' has no attribute {name!r}")
