#!/usr/bin/env python3
"""DebitMatch AI - Model Context Protocol (MCP) server.

Exposes the same deterministic tools the LangGraph squad uses, plus the full
squad as a single tool, to any MCP client (VS Code, Claude Code, Cursor...).

Primitives:
    tools      (model-controlled)       search_products, compute_net_impact, search_policy_clauses,
                                        recommend_account, start/send/get/end conversation
    resources  (application-controlled) catalog://products (static), product://{product_key} (template)
    prompts    (user-controlled)        compare_accounts(deposit_mxn, priorities)

Run (stdio transport):
    .venv\\Scripts\\python.exe mcp_server.py

Note: mcp>=2 renamed ``FastMCP`` to ``MCPServer`` (mcp.server.mcpserver).
Logs go to stderr; stdout is reserved for the MCP protocol.
"""

from __future__ import annotations

import logging
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ResourceNotFoundError  # noqa: E402

from agents.common import SquadResources  # noqa: E402
from tools.catalog_search import HARD_CONSTRAINTS, filter_products, product_net_impact  # noqa: E402

CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.json"

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger("debitmatch.mcp")

mcp = MCPServer(
    "debitmatch-ai",
    instructions=(
        "Tools for comparing Mexican debit and payroll accounts. All MXN figures are computed "
        "deterministically from the verified catalog; never estimate fees or yields yourself."
    ),
)


@lru_cache(maxsize=1)
def _resources() -> SquadResources:
    from graph import build_default_resources

    return build_default_resources()


@lru_cache(maxsize=1)
def _catalog() -> Dict[str, Any]:
    """Catalog only (``catalog_key -> BankProduct``): resources are cheap reads and must not start the squad
    (embedding model, ChromaDB, LLM client). Same validated file the squad loads."""
    from tools.schemas import load_catalog

    return {p.catalog_key: p for p in load_catalog(CATALOG_PATH, strict=True)}


@lru_cache(maxsize=1)
def _graph() -> Any:
    from graph import build_graph

    return build_graph(_resources())


def _summary(product: Any) -> Dict[str, Any]:
    return {
        "product_key": product.catalog_key,
        "institution_name": product.institution_name,
        "product_name": product.product_name,
        "monthly_maintenance_fee": product.monthly_maintenance_fee,
        "minimum_balance_for_fee_waiver": product.minimum_balance_for_fee_waiver,
        "is_digital_account": product.is_digital_account,
        "is_payroll_compatible": product.is_payroll_compatible,
        "for_minor_under_guardianship": product.for_minor_under_guardianship,
        "qualitative_perks_summary": product.qualitative_perks_summary,
        "last_verified_date": product.last_verified_date.isoformat(),
    }


@mcp.tool()
def search_products(
    zero_fee: bool = False,
    digital_opening: bool = False,
    payroll: bool = False,
    deposit_mxn: float = 0.0,
    free_insurance: bool = False,
    cashback: bool = False,
    no_minimum_balance: bool = False,
    minor_guardianship: bool = False,
) -> List[Dict[str, Any]]:
    """List catalog accounts that satisfy ALL selected hard constraints.

    zero_fee is evaluated at the given average balance (balance waivers count).
    minor_guardianship: account of a minor operated by a parent / tutor.
    """
    flags = {"zero_fee": zero_fee, "digital_opening": digital_opening, "payroll": payroll,
             "free_insurance": free_insurance, "cashback": cashback, "no_minimum_balance": no_minimum_balance,
             "minor_guardianship": minor_guardianship}
    constraints = [name for name in HARD_CONSTRAINTS if flags[name]]
    passing, _ = filter_products(_resources().catalog, constraints, deposit_mxn)
    return [_summary(p) for p in passing]


@mcp.tool()
def compute_net_impact(
    product_key: str,
    deposit_mxn: float,
    monthly_transfers: int = 2,
    preferred_channel: Optional[str] = None,
) -> Dict[str, Any]:
    """Exact annual net impact (MXN) of keeping deposit_mxn in a product.

    product_key format: '<institution_id>_<product_id>' (see search_products).
    preferred_channel examples: 'Mobile Banking App', 'Online Banking (Web)', 'Branch / Teller'.
    """
    product = _resources().products_by_key.get(product_key)
    if product is None:
        return {"error": f"Unknown product_key '{product_key}'."}
    return product_net_impact(product, deposit_mxn, monthly_transfers, preferred_channel)


@mcp.tool()
def search_policy_clauses(
    query: str,
    institution_id: Optional[str] = None,
    product_id: Optional[str] = None,
    k: int = 4,
) -> List[Dict[str, Any]]:
    """Retrieve RECA contract clauses with Maximal Marginal Relevance (default k=4)."""
    engine = _resources().rag_engine
    if engine is None:
        return []
    results = engine.query_mmr(query, institution_id=institution_id, product_id=product_id, k=k)
    return [
        {"text": r["text"], "relevance_score": r["relevance_score"], "metadata": r["metadata"]}
        for r in results
    ]


@mcp.tool()
def recommend_account(query: str) -> Dict[str, Any]:
    """Run the full 5-agent DebitMatch squad and return the reviewed recommendation."""
    from graph import run_squad

    state = run_squad(query, _resources(), _graph())
    draft = state.get("draft_recommendation") or {}
    return {
        "status": draft.get("status"),
        "base_option": draft.get("base_option"),
        "tradeoff_option": draft.get("tradeoff_option"),
        "narrative": draft.get("narrative"),
        "warnings": draft.get("warnings", []),
        "disclaimers": draft.get("disclaimers", []),
        "requires_human_review": state.get("requires_human_review"),
        "human_review_reasons": draft.get("human_review_reasons", []),
        "confidence_score": state.get("confidence_score"),
        "route": (state.get("execution_metadata") or {}).get("route"),
    }


# --------------------------------------------------------------- #
# Resources (application-controlled, read-only context addressed by URI) and
# prompts (user-controlled templates, e.g. a slash command in the host).
# --------------------------------------------------------------- #
@mcp.resource("catalog://products", title="DebitMatch product catalog", mime_type="application/json")
def catalog_index() -> List[Dict[str, Any]]:
    """Read-only index of every verified product (static resource)."""
    return [_summary(p) for p in _catalog().values()]


@mcp.resource("product://{product_key}", title="DebitMatch product record", mime_type="application/json")
def product_detail(product_key: str) -> Dict[str, Any]:
    """Full catalog record of one product: fees, yield tiers, SPEI fees, penalties (resource TEMPLATE).

    product_key format: '<institution_id>_<product_id>' (see catalog://products).
    """
    product = _catalog().get(product_key)
    if product is None:  # protocol-level error (resource not found), not a successful read
        raise ResourceNotFoundError(f"Unknown product_key '{product_key}'. List the keys with catalog://products.")
    return product.model_dump(mode="json")


@mcp.prompt(title="Compare debit accounts")
def compare_accounts(deposit_mxn: float, priorities: str) -> str:
    """Consultation template: compares accounts for an average balance and the user's priorities."""
    return (
        f"I keep about MXN {deposit_mxn:,.2f} on average in the account. My priorities: {priorities}.\n"
        f"Call recommend_account with my request, including the MXN {deposit_mxn:,.2f} average balance, to run the "
        "full DebitMatch analysis. To compare specific accounts, use search_products with "
        f"deposit_mxn={deposit_mxn:.2f} and compute_net_impact with that same deposit_mxn for each product_key. "
        "Never estimate fees, yields or taxes yourself: quote only the figures the tools return, and keep their "
        "warnings and disclaimers."
    )


# --------------------------------------------------------------- #
# Conversation state manager (ToT plan, section 2.4: "State Manager - MCP:
# shared state for Preference Vector and history"). Session-only, in memory.
# --------------------------------------------------------------- #
_SESSIONS: Dict[str, Any] = {}
_SESSIONS_LOCK = threading.Lock()
MAX_SESSIONS = 200


def _session(session_id: str) -> Any:
    with _SESSIONS_LOCK:
        memory = _SESSIONS.get(session_id)
    if memory is None:
        raise KeyError(f"Unknown session_id '{session_id}'. Call start_conversation first.")
    return memory


def _turn_result(memory: Any, plan: Any) -> Dict[str, Any]:
    from graph import run_squad
    from memory import assistant_summary

    if plan.action == "clarify":
        return {"session_id": memory.session_id, "action": "clarify", "question": plan.question,
                "profile_changes": plan.changes, "state": memory.snapshot()}
    state = run_squad(plan.query, _resources(), _graph(), weighted_preference_vector=(plan.overrides or {}).get("weights"),
                      run_metadata=plan.run_metadata)
    memory.record_assistant(assistant_summary(state), run_id=None, llm=_resources().llm)
    draft = state.get("draft_recommendation") or {}
    return {
        "session_id": memory.session_id, "action": "run", "profile_changes": plan.changes,
        "status": draft.get("status"), "answer": draft.get("answer"),
        "base_option": draft.get("base_option"), "tradeoff_option": draft.get("tradeoff_option"),
        "negotiation": draft.get("negotiation", []), "narrative": draft.get("narrative"),
        "warnings": draft.get("warnings", []), "disclaimers": draft.get("disclaimers", []),
        "requires_human_review": state.get("requires_human_review"),
        "weighted_preference_vector": state.get("weighted_preference_vector"),
        "state": memory.snapshot(),
    }


@mcp.tool()
def start_conversation(deposit_mxn: Optional[float] = None) -> Dict[str, Any]:
    """Open a multi-turn consultation session (short-term memory: typed profile + sliding window + summary)."""
    from memory import ConversationMemory

    memory = ConversationMemory(catalog=_resources().catalog)
    if deposit_mxn is not None:
        memory.profile.deposit_mxn, memory.profile.deposit_source = float(deposit_mxn), "client"
    with _SESSIONS_LOCK:
        if len(_SESSIONS) >= MAX_SESSIONS:
            _SESSIONS.pop(next(iter(_SESSIONS)))  # evict the oldest session
        _SESSIONS[memory.session_id] = memory
    return {"session_id": memory.session_id, "state": memory.snapshot()}


@mcp.tool()
def send_message(session_id: str, message: str) -> Dict[str, Any]:
    """Send one user turn: the profile is refined (deposit, constraints, excluded banks, weights) and the squad
    re-runs, or a clarification question is returned when essential data (the deposit) is missing."""
    memory = _session(session_id)
    return _turn_result(memory, memory.plan_turn(message, llm=_resources().llm))


@mcp.tool()
def get_conversation_state(session_id: str) -> Dict[str, Any]:
    """Return the session's typed profile (incl. preference vector), window and rolling summary."""
    return _session(session_id).snapshot()


@mcp.tool()
def end_conversation(session_id: str) -> Dict[str, Any]:
    """Delete the session (no personal data is kept beyond the session)."""
    with _SESSIONS_LOCK:
        removed = _SESSIONS.pop(session_id, None) is not None
    return {"session_id": session_id, "deleted": removed}


if __name__ == "__main__":
    mcp.run()
