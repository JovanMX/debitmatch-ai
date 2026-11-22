"""LangGraph shared state for the DebitMatch AI multi-agent squad and MCP server.

LangGraph nodes must treat the state as read-only and return *partial updates*
(a dict with only the keys they change). ``trajectory_log`` uses an additive
reducer, so a node returns only its new log lines and LangGraph appends them
instead of overwriting the history.
"""

from __future__ import annotations

import math
import operator
from datetime import date, datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, TypedDict

FRESHNESS_MAX_AGE_DAYS = 30
QueryComplexity = Literal["simple", "hybrid"]


def merge_dicts(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """LangGraph reducer: shallow-merge metadata updates instead of overwriting."""
    return {**(left or {}), **(right or {})}
PREFERENCE_KEYS = ("yield_importance", "fee_importance", "perks_importance")
DEFAULT_PREFERENCES: Dict[str, float] = {
    "yield_importance": 0.4,
    "fee_importance": 0.4,
    "perks_importance": 0.2,
}


class SquadState(TypedDict):
    # --- Input / user modelling ---
    user_query: str
    weighted_preference_vector: Dict[str, float]
    archetype_prior: Optional[str]

    # --- Evidence gathered by tools ---
    api_structured_data: List[Dict[str, Any]]
    rag_retrieved_chunks: List[Dict[str, Any]]

    # --- Tree-of-Thought reasoning & evaluation ---
    tot_candidate_branches: List[Dict[str, Any]]
    evaluator_score: float
    draft_recommendation: Optional[Dict[str, Any]]
    confidence_score: float

    # --- Guardrails / human-in-the-loop ---
    data_freshness_flag: bool  # False when any verification date is > 30 days old
    is_fallback_mode: bool  # True when live sources failed and cached CONDUSEF baseline data was used
    review_approved: bool
    requires_human_review: bool

    # --- Observability ---
    trajectory_log: Annotated[List[str], operator.add]
    execution_metadata: Annotated[Dict[str, Any], merge_dicts]

    # --- Orchestration extensions ---
    user_profile: Dict[str, Any]  # deposit, hard constraints, objectives, usage pattern
    query_complexity: Optional[QueryComplexity]  # supervisor routing decision
    revision_count: int  # critique loop counter (N_max = 2)
    review_feedback: List[str]  # issues found by the Evaluator in the last review
    data_conflicts: List[Dict[str, Any]]  # API vs RAG discrepancies
    guardrail_warnings: List[str]  # abstention / freshness / timeout warnings


def is_data_fresh(
    last_verified_date: date | str,
    reference_date: Optional[date] = None,
    max_age_days: int = FRESHNESS_MAX_AGE_DAYS,
) -> bool:
    """True when ``last_verified_date`` is at most ``max_age_days`` old.

    Unparseable or future dates are treated as NOT fresh (fail closed).
    """
    try:
        verified = last_verified_date if isinstance(last_verified_date, date) else date.fromisoformat(last_verified_date)
    except (TypeError, ValueError):
        return False
    today = reference_date or date.today()
    age_days = (today - verified).days
    return 0 <= age_days <= max_age_days


class FreshnessReport(TypedDict):
    data_freshness_flag: bool
    stale_products: List[str]
    stale_chunks: List[str]
    oldest_verified_date: Optional[str]


def compute_data_freshness(
    api_structured_data: List[Dict[str, Any]],
    rag_retrieved_chunks: List[Dict[str, Any]],
    reference_date: Optional[date] = None,
    max_age_days: int = FRESHNESS_MAX_AGE_DAYS,
) -> FreshnessReport:
    """Evaluate freshness over BOTH evidence sources.

    * Products: ``product["last_verified_date"]``
    * RAG chunks: ``chunk["metadata"]["last_verified_date"]``

    A missing or unparseable date counts as stale (fail closed). The flag is
    ``True`` only when every piece of evidence is within ``max_age_days``.
    """
    stale_products: List[str] = []
    stale_chunks: List[str] = []
    seen_dates: List[date] = []

    def check(raw_date: Any, label: str, bucket: List[str]) -> None:
        if not is_data_fresh(raw_date, reference_date, max_age_days):
            bucket.append(label)
        try:
            seen_dates.append(raw_date if isinstance(raw_date, date) else date.fromisoformat(raw_date))
        except (TypeError, ValueError):
            pass

    for product in api_structured_data:
        label = f"{product.get('institution_id', '?')}_{product.get('product_id', '?')}"
        check(product.get("last_verified_date"), label, stale_products)

    for chunk in rag_retrieved_chunks:
        metadata = chunk.get("metadata") or {}
        label = str(chunk.get("chunk_id") or metadata.get("source_file") or "?")
        check(metadata.get("last_verified_date"), label, stale_chunks)

    return FreshnessReport(
        data_freshness_flag=not stale_products and not stale_chunks,
        stale_products=stale_products,
        stale_chunks=stale_chunks,
        oldest_verified_date=min(seen_dates).isoformat() if seen_dates else None,
    )


def validate_preference_vector(vector: Mapping[str, float], tolerance: float = 1e-6) -> Dict[str, float]:
    """Validate weights: known keys, finite, non-negative, summing to 1.0."""
    unknown = set(vector) - set(PREFERENCE_KEYS)
    if unknown:
        raise ValueError(f"Unknown preference keys: {sorted(unknown)}")
    missing = set(PREFERENCE_KEYS) - set(vector)
    if missing:
        raise ValueError(f"Missing preference keys: {sorted(missing)}")

    cleaned: Dict[str, float] = {}
    for key in PREFERENCE_KEYS:
        value = vector[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Preference '{key}' must be a finite, non-negative number.")
        cleaned[key] = float(value)

    total = sum(cleaned.values())
    if not math.isclose(total, 1.0, abs_tol=tolerance):
        raise ValueError(f"Preference weights must sum to 1.0 (got {total:.6f}).")
    return cleaned


def create_initial_state(
    user_query: str,
    weighted_preference_vector: Optional[Mapping[str, float]] = None,
    archetype_prior: Optional[str] = None,
    execution_metadata: Optional[Dict[str, Any]] = None,
) -> SquadState:
    """Factory returning a fully-populated, type-correct initial state.

    Freshness starts as ``True`` and must be recomputed by the data-retrieval
    node once ``api_structured_data`` is populated.
    """
    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("user_query must be a non-empty string.")

    preferences = validate_preference_vector(weighted_preference_vector or DEFAULT_PREFERENCES)
    metadata: Dict[str, Any] = {"created_at_utc": datetime.now(timezone.utc).isoformat()}
    metadata.update(execution_metadata or {})

    return SquadState(
        user_query=user_query.strip(),
        weighted_preference_vector=preferences,
        archetype_prior=archetype_prior,
        api_structured_data=[],
        rag_retrieved_chunks=[],
        tot_candidate_branches=[],
        evaluator_score=0.0,
        draft_recommendation=None,
        confidence_score=0.0,
        data_freshness_flag=True,
        is_fallback_mode=False,
        review_approved=False,
        requires_human_review=False,
        trajectory_log=[],
        execution_metadata=metadata,
        user_profile={},
        query_complexity=None,
        revision_count=0,
        review_feedback=[],
        data_conflicts=[],
        guardrail_warnings=[],
    )
