"""Shared runtime objects for the DebitMatch AI agent squad."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Optional

from tools.schemas import BankProduct

if TYPE_CHECKING:
    from agents.llm import StructuredLLM
    from tools.rag_engine import InstitutionalRAGEngine

DEFAULT_TIME_BUDGET_S = 10.0


@dataclass
class SquadResources:
    """Long-lived dependencies, built once and shared by every graph run.

    Expensive initialisation (catalog validation, embedding model load, Chroma
    client) happens here, outside the per-query 10-second execution budget.
    """

    catalog: List[BankProduct]
    rag_engine: Optional["InstitutionalRAGEngine"] = None
    llm: Optional["StructuredLLM"] = None
    reference_date: Optional[date] = None  # None -> today (pin it in tests)
    time_budget_s: float = DEFAULT_TIME_BUDGET_S
    enable_llm_critic: bool = True
    min_llm_call_budget_s: float = 2.0  # skip a short LLM call (profiling/critic) if less time remains
    advisory_llm_min_budget_s: float = 3.5  # narrative generation is the slowest call
    llm_safety_margin_s: float = 1.5  # time always reserved for downstream nodes
    extra: Dict[str, object] = field(default_factory=dict)

    def llm_timeout(self, remaining_s: float, cap_s: float) -> float:
        """Per-call timeout that always leaves ``llm_safety_margin_s`` for later nodes."""
        return max(0.0, min(remaining_s - self.llm_safety_margin_s, cap_s))

    @cached_property
    def products_by_key(self) -> Dict[str, BankProduct]:
        return {p.catalog_key: p for p in self.catalog}

    def effective_reference_date(self) -> date:
        return self.reference_date or date.today()


@dataclass(frozen=True)
class NodeContext:
    """Per-node execution context injected by the supervisor wrapper."""

    node_name: str
    deadline_epoch: float
    timed_out: bool = False

    def remaining_s(self) -> float:
        return max(0.0, self.deadline_epoch - time.time())

    def has_budget(self, seconds: float) -> bool:
        return not self.timed_out and self.remaining_s() >= seconds

    def log(self, message: str) -> str:
        return f"[{self.node_name}] {message}"
