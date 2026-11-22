"""DebitMatch AI - Streamlit interface for the multi-agent banking consultant.

Run with::

    .venv\\Scripts\\python.exe -m streamlit run app.py

Layout (5 sections + regulatory footer):
    1. Sidebar profile configuration (deposit, SPEI usage, preference vector w, hard constraints).
       Each group is locked by default: the Profiling Agent infers it from the chat and the widgets
       show the values the last run used, labelled with their origin (message / agent / default).
       A per-group switch makes the group manual (authoritative). A clarification question unlocks
       only the field it asks for (the deposit).
    2. Query input and squad execution
    3. Agentic reasoning inspector (trajectory log, ToT beam table, data freshness badge)
    4. Recommendation dashboard (Base vs Positive Trade-off, metrics, catalog chart, RAG fine print)
    5. Human-in-the-Loop review queue (Bounded Decision Card + advisor controls)

Every MXN figure shown is produced by ``tools.calculator.calculate_net_impact``
(via the graph or ``tools.catalog_search.product_net_impact``); the UI never
computes money itself.

Session-state contract (``st.session_state``):
    squad_state         - last SquadState returned by the LangGraph squad (None before the first run)
    is_evaluating       - True while the graph is executing
    execution_history   - list of run summaries for in-session comparison
    hitl_status         - {"pending", "approved_by_human", "advisor_notes", "decision",
                           "override_product_key", "case_id", ...}; cases persist via hitl_queue.ReviewQueue
    normalized_weights  - {"yield_importance", "fee_importance", "perks_importance"}, sum == 1.0

Set ``DEBITMATCH_UI_OFFLINE=1`` to start with the LLM disabled (deterministic mode).
"""

from __future__ import annotations

import copy
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents.evaluator import (BEAM_WIDTH, EARLY_STOP_SCORE, HIGHER_BALANCE_STAGE, MAX_DEPTH, compute_metrics,
                              composite_score, pays_yield_now)
from agents.advisory import build_option, supporting_clauses
from agents.profiling import PROFILE_OVERRIDES_KEY, classify_archetype, normalize_weights
from graph import build_default_resources, build_graph, run_squad
from guardrails import CONFIDENCE_THRESHOLD, MAX_REVISIONS
from hitl_queue import TERMINAL_DECISIONS, CaseClosedError, ReviewQueue, build_decision_card, hitl_reason_details
from memory import ConversationMemory, TurnPlan, assistant_summary
from prompts.profiling_prompts import ARCHETYPE_LABELS
from state import FRESHNESS_MAX_AGE_DAYS, PREFERENCE_KEYS, create_initial_state
from tools.catalog_search import (MEANINGFUL_YIELD_PERCENT, eligible_products, evaluate_hard_constraints,
                                  product_net_impact, yield_unlock)
from tools.schemas import CHANNEL_BRANCH, CHANNEL_MOBILE_APP, CHANNEL_ONLINE_WEB, CHANNEL_TELEPHONE

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
APP_TITLE = "DebitMatch AI"
APP_TAGLINE = "Agentic consultant for Mexican debit & payroll accounts"
FOOTER_DISCLAIMER = (
    "Informational financial comparison tool built for academic demonstration. "
    "Rates and terms subject to CONDUSEF / CASFIM regulatory updates."
)
SPINNER_TEXT = "Executing Multi-Agent Reasoning & ToT Beam Search..."
EXAMPLE_QUERY = "I have $50,000 MXN. I want maximum net yield, zero monthly fee, and free fraud insurance."
EXAMPLE_DEPOSIT_MXN = 50_000.0
OFFLINE_ENV_VAR = "DEBITMATCH_UI_OFFLINE"

DEFAULT_DEPOSIT_MXN = 20_000.0
MAX_DEPOSIT_MXN = 500_000.0
DEPOSIT_STEP_MXN = 1_000.0
DEFAULT_MONTHLY_TRANSFERS = 4
MAX_MONTHLY_TRANSFERS = 100
TOP_N_CHART = 5

DEFAULT_WEIGHTS: Dict[str, float] = {"yield_importance": 0.30, "fee_importance": 0.50, "perks_importance": 0.20}
WEIGHT_WIDGETS: Dict[str, Tuple[str, str, str]] = {  # preference key -> (widget key, label, help)
    "yield_importance": ("w_yield", "Yield Importance (w_yield)", "Weight of the annual net yield."),
    "fee_importance": ("w_fee", "Fee Sensitivity (w_fee)", "Weight of avoiding maintenance and transfer fees."),
    "perks_importance": ("w_perks", "Perks & Coverage Importance (w_perks)", "Weight of cashback and insurance."),
}
HARD_CONSTRAINT_OPTIONS: Tuple[Tuple[str, str, str], ...] = (  # (constraint, widget key, label)
    ("payroll", "hc_payroll", "Payroll Compatible Only"),
    ("zero_fee", "hc_zero_fee", "Zero Maintenance Fee Only"),
    ("digital_opening", "hc_digital_opening", "Digital Opening Required"),
    ("free_insurance", "hc_free_insurance", "Free Insurance Required"),
    ("cashback", "hc_cashback", "Cashback / Rewards Required"),
    ("no_minimum_balance", "hc_no_minimum_balance", "No Minimum Balance Required"),
    ("minor_guardianship", "hc_minor_guardianship", "Account for a Minor Under Guardianship (parent / tutor)"),
)
CONSTRAINT_LABELS = {name: label for name, _, label in HARD_CONSTRAINT_OPTIONS}
BANK_OPTIONS: Tuple[str, ...] = ("BaBien", "BANAMEX", "SANTANDER", "HSBC", "BBVA MEXICO")
REFINE_PLACEHOLDER = ("Refine your search, e.g. 'Actually I have 8,000 MXN', 'Forget the payroll requirement', "
                      "'Nothing from Santander' or 'Fees matter more than perks'.")
CHANNEL_AUTO = "Auto (inferred from query / cheapest listed)"
CHANNEL_OPTIONS: Tuple[str, ...] = (CHANNEL_AUTO, CHANNEL_MOBILE_APP, CHANNEL_ONLINE_WEB, CHANNEL_BRANCH,
                                    CHANNEL_TELEPHONE)

# Sidebar groups: locked by default (the Profiling Agent infers them from the chat and the
# widgets show what the last run used); the switch makes the group's values authoritative.
MANUAL_TOGGLES: Dict[str, Tuple[str, str]] = {  # group -> (toggle key, label)
    "financial": ("manual_financial", "Set financial inputs manually"),
    "weights": ("manual_weights", "Set preference weights manually"),
    "constraints": ("manual_constraints", "Set hard constraints manually"),
}
SOURCE_ICONS = {"message": "🗣️", "sidebar": "🎛️", "agent": "🤖", "default": "⚙️", "assumed": "⚙️", "pending": "⏳"}
SOURCE_LABELS = {
    "message": "from your message",
    "sidebar": "set manually",
    "agent": "inferred by the Profiling Agent",
    "default": "default assumption",
    "assumed": "assumed (deposit question skipped)",
    "pending": "not set yet",
}
_DEPOSIT_SOURCES = {"conversation": "message", "client": "message", "sidebar": "sidebar", "assumed": "assumed"}

DEFAULT_HITL_STATUS: Dict[str, Any] = {
    "pending": False,
    "approved_by_human": False,
    "advisor_notes": "",
    "decision": None,  # approved | modified | escalated
    "override_product_key": None,
    "case_id": None,  # persistent review-queue case
    "reviewed_at": None,
    "run_id": None,
}
DECISION_LABELS = {"approved": "Approved", "modified": "Approved with modification", "escalated": "Escalated"}
STATUS_LABELS = {
    "definitive": "Definitive recommendation",
    "abstained_provisional": "Provisional (abstention protocol active)",
    "pending_human_review": "Pending human review",
    "no_match": "No eligible product (criteria negotiation)",
    "direct_answer": "Direct answer from verified catalog facts",
}
AGENT_LABELS: Tuple[Tuple[str, str], ...] = (
    ("intent_router", "Supervisor Agent (intent router)"),
    ("direct_answer", "Research Agent (direct answer)"),
    ("profiling", "Profiling Agent"),
    ("research", "Research Agent"),
    ("supervisor", "Supervisor Agent"),
    ("review", "Evaluator Agent (critic)"),
    ("evaluator", "Evaluator Agent"),
    ("advisory", "Advisory Agent"),
)
# Chart colors encode the sign of the net annual impact (gain / cost / break-even). The sign is
# never color-alone: the label prints it ("+" / "−") and the bar grows right or left of zero.
# Constraint status is carried by the texture and the label suffix (✖ / 🔓), not by color.
CHART_GAIN_COLOR = "#2e9e5b"
CHART_COST_COLOR = "#d64545"
CHART_EVEN_COLOR = "#9aa3ad"
CHART_PATTERN_UNLOCK = "/"  # pays only with a higher balance
CHART_PATTERN_FAILS = "x"   # fails a hard constraint


# --------------------------------------------------------------------------- #
# Pure helpers (importable and unit-testable without a Streamlit runtime)
# --------------------------------------------------------------------------- #
def money(value: Optional[float]) -> str:
    return "n/a" if value is None else f"MXN {value:,.2f}"


def default_use_llm() -> bool:
    return os.environ.get(OFFLINE_ENV_VAR, "").strip().lower() not in {"1", "true", "yes"}


def rebalance_weights(changed_key: str, new_value: float, current: Mapping[str, float]) -> Dict[str, float]:
    """Keep the three sliders summing to 1.0 after one of them moves.

    The moved weight is kept; the remainder is split between the other two in
    proportion to their previous values (evenly if both were zero). Values are
    rounded to slider precision (0.01) and the rounding drift is absorbed by
    the largest other weight, so the result always sums to exactly 1.00.
    """
    if changed_key not in PREFERENCE_KEYS:
        raise ValueError(f"Unknown preference key '{changed_key}'.")
    value = min(1.0, max(0.0, float(new_value)))
    others = [k for k in PREFERENCE_KEYS if k != changed_key]
    remaining = 1.0 - value
    other_total = sum(max(0.0, float(current.get(k, 0.0))) for k in others)
    if other_total <= 0:
        shares = {k: remaining / len(others) for k in others}
    else:
        shares = {k: remaining * max(0.0, float(current.get(k, 0.0))) / other_total for k in others}

    rounded = {changed_key: round(value, 2), **{k: round(v, 2) for k, v in shares.items()}}
    drift = round(1.0 - sum(rounded.values()), 2)
    if drift:
        target = max(others, key=lambda k: rounded[k])
        rounded[target] = round(rounded[target] + drift, 2)
    return rounded


def selected_constraints(ss: Mapping[str, Any]) -> List[str]:
    return [name for name, key, _ in HARD_CONSTRAINT_OPTIONS if ss.get(key)]


def init_session_state(ss: MutableMapping[str, Any]) -> None:
    """Initialise the session-state contract and widget defaults (idempotent)."""
    defaults: Dict[str, Any] = {
        "squad_state": None,
        "is_evaluating": False,
        "execution_history": [],
        "hitl_status": dict(DEFAULT_HITL_STATUS),
        "normalized_weights": normalize_weights(DEFAULT_WEIGHTS),
        "pending_run": None,
        "run_errors": [],
        "progress_events": [],
        "preferred_channel_choice": CHANNEL_AUTO,
        "conversation": None,  # memory.ConversationMemory (short-term memory, session only)
        "deposit_touched": False,  # deposit field edited while the agent was asking for it
        "excluded_banks": [],
        "profile_provenance": [],  # rows of profile_provenance() for the last run (sidebar captions)
        "widget_sync": None,  # values of the last run to show in the locked widgets on the next rerun
        # widget defaults (widgets are created without ``value=`` to avoid double sources of truth);
        # None = empty field: nothing set yet, the agent infers it or asks for it
        "deposit_amount_mxn": None,
        "monthly_spei_transfers": None,
        "user_query_input": "",
        "advisor_notes_input": "",
        "use_llm": default_use_llm(),
    }
    for toggle_key, _ in MANUAL_TOGGLES.values():
        defaults[toggle_key] = False
    for key, (widget_key, _, _) in WEIGHT_WIDGETS.items():
        defaults[widget_key] = DEFAULT_WEIGHTS[key]
    for _, widget_key, _ in HARD_CONSTRAINT_OPTIONS:
        defaults[widget_key] = False
    for key, value in defaults.items():
        if key not in ss:
            ss[key] = copy.deepcopy(value)


def build_profile_overrides(
    deposit_mxn: float,
    monthly_transfers: int,
    constraints: Sequence[str],
    weights: Optional[Mapping[str, float]],
    preferred_channel: Optional[str] = None,
) -> Dict[str, Any]:
    """Structured profile consumed by the Profiling Agent (sidebar wins over free text)."""
    return {
        "deposit_mxn": float(deposit_mxn),
        "monthly_transfers": int(monthly_transfers),
        "hard_constraints": list(constraints),
        "weights": dict(weights) if weights else None,
        "preferred_channel": None if preferred_channel in (None, "", CHANNEL_AUTO) else preferred_channel,
    }


def profile_provenance(result: Mapping[str, Any], profile: Any) -> List[Dict[str, str]]:
    """Every profile value the run used, with its origin (makes all profiling assumptions explicit).

    ``profile`` is the conversation's ``ConversationProfile`` at run time (it records who set each
    field); values the user never gave are labelled as agent inferences or default assumptions.
    Returns rows ``{"group", "field", "value", "source"}``; empty for runs without a profile.
    """
    used = result.get("user_profile") or {}
    if "deposit_mxn" not in used:  # direct answers skip profiling
        return []
    sources = dict(getattr(profile, "sources", {}) or {})
    rows: List[Dict[str, str]] = []

    def add(group: str, name: str, value: str, source: str) -> None:
        rows.append({"group": group, "field": name, "value": value, "source": source})

    deposit_source = _DEPOSIT_SOURCES.get(getattr(profile, "deposit_source", None) or "", "default")
    add("financial", "Average balance", money(float(used.get("deposit_mxn") or 0.0)),
        "assumed" if not used.get("deposit_stated") and deposit_source != "sidebar" else deposit_source)
    add("financial", "SPEI transfers/month", str(used.get("monthly_transfers", "n/a")),
        sources.get("transfers", "default"))
    channel = used.get("preferred_channel")
    if channel and getattr(profile, "preferred_channel", None) == channel:
        add("financial", "SPEI channel", channel, sources.get("channel", "message"))
    elif channel:
        add("financial", "SPEI channel", channel, "agent")
    else:
        add("financial", "SPEI channel", "cheapest listed per product", "default")

    weights = result.get("weighted_preference_vector") or {}
    profiling = used.get("profiling_source") or (result.get("execution_metadata") or {}).get("profiling_source")
    w_text = " / ".join(f"{weights.get(k, 0.0):.2f}" for k in PREFERENCE_KEYS)
    if profiling == "user":
        add("weights", "Weights w (yield / fee / perks)", w_text, sources.get("weights", "sidebar"))
    else:
        add("weights", "Weights w (yield / fee / perks)", f"{w_text} ({profiling or 'rules'})", "agent")
    archetype = result.get("archetype_prior") or used.get("archetype")
    add("weights", "Archetype", ARCHETYPE_LABELS.get(archetype, str(archetype or "n/a")), "agent")

    constraints = list(used.get("hard_constraints") or [])
    add("constraints", "Hard constraints", ", ".join(CONSTRAINT_LABELS.get(c, c) for c in constraints) or "none",
        sources.get("constraints", "message" if constraints else "default"))
    if "minor_guardianship" in constraints:
        add("constraints", "Account holder", "minor under guardianship (parent / tutor)",
            sources.get("constraints", "message"))
    else:
        add("constraints", "Account holder", "adult (minor-only accounts excluded)", "default")
    excluded = list(used.get("excluded_institutions") or [])
    if excluded or "excluded" in sources:
        add("constraints", "Excluded banks", ", ".join(excluded) or "none", sources.get("excluded", "message"))
    return rows


_CAPTION_SOURCE_LABELS = {"message": "from your message", "sidebar": "set manually", "agent": "inferred",
                          "default": "default", "assumed": "assumed", "pending": "not set yet"}
# Usage assumptions that only price transfer fees; they never filter products.
_FEE_ONLY_FIELDS = frozenset({"SPEI transfers/month", "SPEI channel"})


def provenance_line(rows: Sequence[Mapping[str, str]]) -> str:
    """Compact chat caption: value + origin in words for every profile input of the run."""
    def origin(row: Mapping[str, str]) -> str:
        label = _CAPTION_SOURCE_LABELS.get(row["source"], row["source"])
        if row["source"] == "default" and row["field"] in _FEE_ONLY_FIELDS:
            label += ", fee calculation only"
        return label

    return " · ".join(f"{r['field']} {r['value']} ({origin(r)})" for r in rows)


def result_widget_values(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Widget values that mirror what a run used (shown in the locked sidebar groups)."""
    used = result.get("user_profile") or {}
    if "deposit_mxn" not in used:
        return {}
    channel = used.get("preferred_channel")
    values: Dict[str, Any] = {
        "deposit_amount_mxn": float(min(MAX_DEPOSIT_MXN, max(0.0, float(used.get("deposit_mxn") or 0.0)))),
        "monthly_spei_transfers": int(min(MAX_MONTHLY_TRANSFERS, max(0, int(used.get("monthly_transfers", 0))))),
        "preferred_channel_choice": channel if channel in CHANNEL_OPTIONS else CHANNEL_AUTO,
        "excluded_banks": [b for b in used.get("excluded_institutions") or [] if b in BANK_OPTIONS],
    }
    constraints = set(used.get("hard_constraints") or [])
    for name, widget, _ in HARD_CONSTRAINT_OPTIONS:
        values[widget] = name in constraints
    weights = result.get("weighted_preference_vector")
    if weights:
        rounded = {k: round(float(weights.get(k, 0.0)), 2) for k in PREFERENCE_KEYS}
        for key, (widget, _, _) in WEIGHT_WIDGETS.items():
            values[widget] = rounded[key]
        values["normalized_weights"] = normalize_weights(weights)
    return values


def _chroma_error_types() -> Tuple[type, ...]:
    try:
        from chromadb.errors import ChromaError

        return (ChromaError,)
    except Exception:  # noqa: BLE001 - chromadb layout differs between versions
        return ()


def execute_squad(
    query: str,
    overrides: Mapping[str, Any],
    resources: Any,
    app_graph: Any,
    run_id: str,
    on_progress: Optional[Any] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Invoke the LangGraph squad; never raises. Returns (final_state, error messages)."""
    try:
        result = run_squad(
            query,
            resources,
            compiled_graph=app_graph,
            weighted_preference_vector=overrides.get("weights") or None,
            run_metadata={**dict(extra_metadata or {}), PROFILE_OVERRIDES_KEY: dict(overrides), "ui_run_id": run_id,
                          "channel": "streamlit"},
            on_progress=on_progress,
        )
    except _chroma_error_types() as exc:
        return None, [f"ChromaDB error while retrieving RECA clauses: {type(exc).__name__}: {exc}"]
    except Exception as exc:  # noqa: BLE001 - surface any failure in the UI instead of crashing
        logger.exception("Squad execution failed")
        return None, [f"Squad execution failed: {type(exc).__name__}: {exc}"]
    return result, collect_run_errors(result)


def collect_run_errors(result: Mapping[str, Any]) -> List[str]:
    """Timeouts, ChromaDB failures and contained node errors recorded by the squad."""
    meta = result.get("execution_metadata") or {}
    errors: List[str] = []
    budget = float(meta.get("time_budget_s") or 10.0)
    if meta.get("timed_out") or meta.get("within_budget") is False:
        errors.append(
            f"Timeout: the squad exceeded its {budget:.0f}-second execution budget "
            f"(elapsed {meta.get('elapsed_s', 'n/a')} s). The answer below is provisional."
        )
    if meta.get("rag_error"):
        errors.append(f"ChromaDB / RAG retrieval error: {meta['rag_error']}. Continued without RECA evidence.")
    for key, value in sorted(meta.items()):
        if key.startswith("error."):
            errors.append(f"Node '{key.split('.', 1)[1]}' failed and was contained: {value}")
    return errors


def agent_label(node: str) -> str:
    for marker, label in AGENT_LABELS:
        if marker in node:
            return label
    return node


def trajectory_rows(trajectory: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for step, line in enumerate(trajectory, start=1):
        node, message = "", line
        if line.startswith("[") and "]" in line:
            node, message = line[1:].split("]", 1)
        rows.append({"Step": step, "Agent": agent_label(node), "Node": node, "Event": message.strip()})
    return rows


def final_branches(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    branches = [b for b in result.get("tot_candidate_branches") or [] if b.get("final")]
    return sorted(branches, key=lambda b: b.get("final_rank", 999))


def higher_balance_branches(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    branches = [b for b in result.get("tot_candidate_branches") or [] if b.get("stage") == HIGHER_BALANCE_STAGE]
    return sorted(branches, key=lambda b: b.get("higher_balance_rank", 999))


def display_branches(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Top-N order shown to the user.

    Yield objective: accounts that pay a meaningful yield at the stated balance (score order), then the
    accounts that would pay one with a higher balance (lowest extra balance first), then the rest.
    """
    finals = final_branches(result)
    upgrades = higher_balance_branches(result)
    if not upgrades:
        return finals
    keys = {b["product_key"] for b in upgrades}
    paying = [b for b in finals if pays_yield_now(b)]
    rest = [b for b in finals if not pays_yield_now(b) and b["product_key"] not in keys]
    return paying + upgrades + rest


def branch_note(branch: Mapping[str, Any]) -> str:
    strategy = branch.get("strategy") or {}
    if not strategy.get("requires_higher_balance"):
        return ""
    return (f"🔓 pays {strategy['unlock_rate_percent']:.2f}% from MXN {strategy['unlock_balance_mxn']:,.0f} "
            f"(you stated MXN {strategy['stated_deposit_mxn']:,.0f}); figures at that balance")


def branch_rows(branches: Sequence[Mapping[str, Any]], limit: int = BEAM_WIDTH) -> List[Dict[str, Any]]:
    rows = []
    for position, branch in enumerate(list(branches)[:limit], start=1):
        impact = branch.get("net_impact") or {}
        strategy = branch.get("strategy") or {}
        rows.append({
            "Rank": position,
            "Note": branch_note(branch),
            "Branch": branch.get("branch_id"),
            "Depth": branch.get("depth"),
            "Stage": branch.get("stage"),
            "Bank": branch.get("institution_name"),
            "Product": branch.get("product_name"),
            "Yield rate (%)": impact.get("applicable_yield_rate_percent"),
            "Balance evaluated (MXN)": impact.get("deposit_evaluated_mxn"),
            "SPEI channel": strategy.get("transfer_channel") or (impact.get("breakdown") or {}).get("transfer_channel_used"),
            "Gross interest (MXN)": impact.get("gross_annual_interest_mxn"),
            "Annual fees (MXN)": impact.get("total_annual_fees_mxn"),
            "Net annual (MXN)": impact.get("net_annual_impact_mxn"),
            "Composite score": branch.get("score"),
            "Hard constraints": "PASS" if branch.get("hard_constraints_met") else "FAIL",
        })
    return rows


def early_stop_explanation(result: Mapping[str, Any]) -> str:
    meta = result.get("execution_metadata") or {}
    depth = meta.get("tot_depth_reached")
    best = final_branches(result)[:1]
    if meta.get("early_stopped") and best:
        return (f"Early stopping triggered at depth {depth}/{MAX_DEPTH}: branch {best[0].get('branch_id')} "
                f"({best[0].get('institution_name')} {best[0].get('product_name')}) passed 100% of hard constraints "
                f"AND scored {best[0].get('score'):.2f} > {EARLY_STOP_SCORE:.0f}.")
    if not best:
        return "No branch survived the hard-constraint gate, so no early stop was possible."
    reason = ("the best branch failed a hard constraint" if not best[0].get("hard_constraints_met")
              else f"the best score {best[0].get('score'):.2f} did not exceed {EARLY_STOP_SCORE:.0f}")
    return f"No early stop: {reason}; the beam explored the full depth ({depth}/{MAX_DEPTH})."


def catalog_leaderboard(result: Mapping[str, Any], catalog: Sequence[Any]) -> List[Dict[str, Any]]:
    """Score every product the profile may hold (deterministic calculator only).

    Business and minor-only accounts are dropped like in the squad (``eligible_products``). With a yield
    objective, products that pay a meaningful yield at the stated balance come first, then products that
    would pay one with a higher balance (``unlock_*`` fields, evaluated at that balance), then the rest.
    """
    profile = result.get("user_profile") or {}
    weights = result.get("weighted_preference_vector") or normalize_weights(DEFAULT_WEIGHTS)
    deposit = float(profile.get("deposit_mxn") or 0.0)
    transfers = int(profile.get("monthly_transfers", DEFAULT_MONTHLY_TRANSFERS))
    channel = profile.get("preferred_channel")
    constraints = list(profile.get("hard_constraints") or [])
    yield_goal = "yield" in (profile.get("objectives") or [])
    rows = []
    for product in eligible_products(catalog, profile):
        impact = product_net_impact(product, deposit, transfers, channel)
        if impact.get("error"):
            continue
        metrics = compute_metrics(product, impact, deposit)
        checks = evaluate_hard_constraints(product, constraints, deposit)
        row = {
            "product_key": product.catalog_key,
            "label": f"{product.institution_name} {product.product_name}",
            "net_annual_impact_mxn": impact["net_annual_impact_mxn"],
            "gross_annual_interest_mxn": impact["gross_annual_interest_mxn"],
            "total_annual_fees_mxn": impact["total_annual_fees_mxn"],
            "yield_rate_percent": impact["applicable_yield_rate_percent"],
            "composite_score": composite_score(weights, metrics),
            "meets_constraints": all(checks.values()) if checks else True,
            "failed_constraints": [CONSTRAINT_LABELS.get(c, c) for c, ok in checks.items() if not ok],
            "unlock_balance_mxn": None,
            "_impact": impact, "_checks": checks,  # kept to build the same detail card as the recommendation
        }
        unlock = yield_unlock(product, deposit) if yield_goal else None
        if unlock:
            at_unlock = product_net_impact(product, unlock["unlock_balance_mxn"], transfers, channel)
            if not at_unlock.get("error") and at_unlock["applicable_yield_rate_percent"] >= MEANINGFUL_YIELD_PERCENT:
                row.update(unlock_balance_mxn=unlock["unlock_balance_mxn"],
                           unlock_rate_percent=at_unlock["applicable_yield_rate_percent"],
                           unlock_net_annual_impact_mxn=at_unlock["net_annual_impact_mxn"])
        rows.append(row)

    def order(r: Mapping[str, Any]) -> Tuple[Any, ...]:
        base = (-r["composite_score"], -r["net_annual_impact_mxn"], r["product_key"])
        fails = not r["meets_constraints"]  # products the user can actually choose come first
        if not yield_goal or r["yield_rate_percent"] >= MEANINGFUL_YIELD_PERCENT:
            return (fails, 0, False, 0.0, 0.0) + base
        if r["unlock_balance_mxn"] is not None:  # same order as the evaluator: positive net at that balance first
            return (fails, 1, r["unlock_net_annual_impact_mxn"] <= 0, r["unlock_balance_mxn"],
                    -r["unlock_rate_percent"]) + base
        return (fails, 2, False, 0.0, 0.0) + base

    rows.sort(key=order)
    return rows


def summarize_run(result: Mapping[str, Any], run_id: str, inputs: Mapping[str, Any]) -> Dict[str, Any]:
    draft = result.get("draft_recommendation") or {}
    meta = result.get("execution_metadata") or {}
    base = draft.get("base_option") or {}
    trade = draft.get("tradeoff_option") or {}
    weights = result.get("weighted_preference_vector") or {}
    return {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "query": result.get("user_query", ""),
        "deposit_mxn": (result.get("user_profile") or {}).get("deposit_mxn"),
        "weights": " / ".join(f"{weights.get(k, 0):.2f}" for k in PREFERENCE_KEYS),
        "archetype": result.get("archetype_prior"),
        "route": meta.get("route"),
        "status": draft.get("status"),
        "base_option": f"{base.get('institution_name', '')} {base.get('product_name', '')}".strip() or None,
        "base_net_mxn": base.get("net_annual_impact_mxn"),
        "tradeoff_option": f"{trade.get('institution_name', '')} {trade.get('product_name', '')}".strip() or None,
        "tradeoff_net_mxn": trade.get("net_annual_impact_mxn"),
        "evaluator_score": result.get("evaluator_score"),
        "confidence": result.get("confidence_score"),
        "requires_human_review": bool(result.get("requires_human_review")),
        "human_approved": False,
        "elapsed_s": meta.get("elapsed_s"),
        "tokens": (meta.get("token_usage") or {}).get("total_tokens"),
        "turn": meta.get("conversation_turn"),
        "llm": bool(inputs.get("use_llm")),
    }


# --------------------------------------------------------------------------- #
# Resources (loaded once per process, outside the 10 s query budget)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading catalog, ChromaDB index and embedding model...")
def load_squad(use_llm: bool) -> Tuple[Any, Any]:
    """Return (SquadResources, compiled LangGraph ``app_graph``)."""
    resources = build_default_resources(use_llm=use_llm)
    app_graph = build_graph(resources)
    return resources, app_graph


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def on_weight_change(changed_key: str) -> None:
    ss = st.session_state
    current = {key: float(ss[widget]) for key, (widget, _, _) in WEIGHT_WIDGETS.items()}
    balanced = rebalance_weights(changed_key, current[changed_key], current)
    for key, (widget, _, _) in WEIGHT_WIDGETS.items():
        ss[widget] = balanced[key]
    ss["normalized_weights"] = normalize_weights(balanced)


def load_example() -> None:
    st.session_state["user_query_input"] = EXAMPLE_QUERY  # the text states the deposit itself


def on_deposit_change() -> None:
    st.session_state["deposit_touched"] = True


def on_manual_toggle(group: str) -> None:
    """Switching a group to manual seeds empty widgets so their values are explicit before the next run."""
    ss = st.session_state
    toggle_key, _ = MANUAL_TOGGLES[group]
    if not ss[toggle_key] or group != "financial":
        return  # weights/constraints already show the last run's values; switching off re-enables inference
    memory = ss.get("conversation")
    if ss.get("deposit_amount_mxn") is None:
        known = memory.profile.deposit_mxn if memory is not None else None
        ss["deposit_amount_mxn"] = float(known if known is not None else DEFAULT_DEPOSIT_MXN)
    if ss.get("monthly_spei_transfers") is None:
        ss["monthly_spei_transfers"] = DEFAULT_MONTHLY_TRANSFERS


def manual(group: str, ss: Optional[Mapping[str, Any]] = None) -> bool:
    return bool((ss if ss is not None else st.session_state).get(MANUAL_TOGGLES[group][0]))


# --- conversation (short-term memory) -------------------------------------- #
def conversation() -> ConversationMemory:
    ss = st.session_state
    if ss.get("conversation") is None:
        ss["conversation"] = ConversationMemory()
    memory: ConversationMemory = ss["conversation"]
    if not memory.catalog:
        try:
            memory.catalog = list(load_squad(bool(ss["use_llm"]))[0].catalog)
        except Exception:  # noqa: BLE001 - intent detection degrades to "recommendation"
            pass
    return memory


def sync_sidebar_to_memory(memory: ConversationMemory) -> None:
    """Manual sidebar groups are authoritative inputs; locked groups are display-only (last change wins)."""
    ss = st.session_state
    p = memory.profile
    deposit = ss.get("deposit_amount_mxn")
    if manual("financial"):
        if deposit is not None:
            p.deposit_mxn, p.deposit_source = float(deposit), "sidebar"
        if ss.get("monthly_spei_transfers") is not None:
            p.monthly_transfers = int(ss["monthly_spei_transfers"])
            p.sources["transfers"] = "sidebar"
        channel = ss.get("preferred_channel_choice")
        p.preferred_channel = None if channel in (None, CHANNEL_AUTO) else channel
        p.sources["channel"] = "sidebar"
    elif memory.awaiting == "deposit_mxn" and ss.get("deposit_touched") and deposit is not None:
        # the deposit field is unlocked while the agent asks for it
        p.deposit_mxn, p.deposit_source = float(deposit), "sidebar"
    ss["deposit_touched"] = False
    if manual("constraints"):
        p.hard_constraints = selected_constraints(ss)
        p.excluded_institutions = list(ss.get("excluded_banks") or [])
        p.sources["constraints"] = p.sources["excluded"] = "sidebar"
    if manual("weights"):
        p.weights = normalize_weights(ss["normalized_weights"])
        p.sources["weights"] = "sidebar"
    else:
        p.weights = None  # inferred by the Profiling Agent
        p.sources.pop("weights", None)


def sync_memory_to_sidebar(memory: ConversationMemory) -> None:
    """Reflect conversation-driven changes in the sidebar widgets (only allowed inside callbacks)."""
    ss = st.session_state
    p = memory.profile
    if p.deposit_mxn is not None:
        ss["deposit_amount_mxn"] = float(min(MAX_DEPOSIT_MXN, max(0.0, p.deposit_mxn)))
    if p.monthly_transfers is not None:
        ss["monthly_spei_transfers"] = int(min(MAX_MONTHLY_TRANSFERS, max(0, p.monthly_transfers)))
    for name, widget, _ in HARD_CONSTRAINT_OPTIONS:
        ss[widget] = name in p.hard_constraints
    ss["excluded_banks"] = [b for b in p.excluded_institutions if b in BANK_OPTIONS]
    ss["preferred_channel_choice"] = p.preferred_channel if p.preferred_channel in CHANNEL_OPTIONS else CHANNEL_AUTO
    if p.weights and manual("weights"):
        for key, (widget, _, _) in WEIGHT_WIDGETS.items():
            ss[widget] = round(float(p.weights[key]), 2)
        ss["normalized_weights"] = normalize_weights(p.weights)


def apply_widget_sync(ss: MutableMapping[str, Any]) -> None:
    """Show the last run's values in the LOCKED groups (runs before any widget is created)."""
    values = ss.get("widget_sync")
    ss["widget_sync"] = None
    if not values:
        return
    groups = {
        "financial": ("deposit_amount_mxn", "monthly_spei_transfers", "preferred_channel_choice"),
        "weights": tuple(w for w, _, _ in WEIGHT_WIDGETS.values()) + ("normalized_weights",),
        "constraints": tuple(w for _, w, _ in HARD_CONSTRAINT_OPTIONS) + ("excluded_banks",),
    }
    for group, keys in groups.items():
        if manual(group, ss):
            continue
        for key in keys:
            if key in values:
                ss[key] = copy.deepcopy(values[key])


def queue_run(plan: TurnPlan) -> None:
    ss = st.session_state
    run_id = uuid.uuid4().hex[:8]
    weights = normalize_weights(ss["normalized_weights"])
    ss["squad_state"] = create_initial_state(plan.query, weighted_preference_vector=weights,
                                             execution_metadata={"ui_run_id": run_id})
    ss["pending_run"] = {"run_id": run_id, "query": plan.query, "message": plan.message,
                         "use_llm": bool(ss["use_llm"]), "overrides": dict(plan.overrides or {}),
                         "run_metadata": dict(plan.run_metadata or {}), "changes": list(plan.changes)}
    ss["run_errors"] = []
    ss["hitl_status"] = dict(DEFAULT_HITL_STATUS, run_id=run_id)
    ss["advisor_notes_input"] = ""
    ss["hitl_override_choice"] = None
    ss["is_evaluating"] = True


def turn_llm() -> Any:
    """LLM for the verified parameter extraction of each turn (None in deterministic mode / without a key)."""
    if not st.session_state.get("use_llm"):
        return None
    try:
        return getattr(load_squad(True)[0], "llm", None)
    except Exception:  # noqa: BLE001 - extraction degrades to the regex parser
        return None


def submit_turn(message: str) -> None:
    """One conversation turn: update the typed state, then clarify or run the squad."""
    ss = st.session_state
    memory = conversation()
    sync_sidebar_to_memory(memory)
    plan = memory.plan_turn(message, llm=turn_llm())
    sync_memory_to_sidebar(memory)
    ss["user_query_input"] = ""
    ss["run_errors"] = []
    if plan.action == "run":
        queue_run(plan)


def start_evaluation() -> None:
    """Button callback: send the composer text as a new conversation turn."""
    ss = st.session_state
    query = str(ss.get("user_query_input", "")).strip()
    if not query:
        ss["run_errors"] = ["Please describe your banking needs before running the analysis."]
        return
    submit_turn(query)


def use_sidebar_deposit() -> None:
    """Clarification quick reply: continue with the deposit currently shown in the sidebar."""
    ss = st.session_state
    if ss.get("deposit_amount_mxn") is None:
        ss["run_errors"] = ["Type your average balance in the sidebar field (or reply in the chat) first."]
        return
    memory = conversation()
    deposit = float(ss["deposit_amount_mxn"])
    sync_sidebar_to_memory(memory)
    plan = memory.resolve_clarification_with(deposit, source="sidebar")
    sync_memory_to_sidebar(memory)
    queue_run(plan)


def skip_deposit() -> None:
    submit_turn("skip")


def relax_constraints(constraints: Sequence[str]) -> None:
    """Criteria negotiation quick reply: drop the listed hard constraints and re-run."""
    labels = ", ".join(CONSTRAINT_LABELS.get(c, c).replace(" Only", "").replace(" Required", "") for c in constraints)
    submit_turn(f"Relax the {labels} requirement.")


def new_conversation() -> None:
    ss = st.session_state
    ss["conversation"] = None
    ss["squad_state"] = None
    ss["run_errors"] = []
    ss["hitl_status"] = dict(DEFAULT_HITL_STATUS)
    ss["profile_provenance"] = []
    # locked groups only mirrored the old conversation; manual groups keep the user's settings
    if not manual("financial"):
        ss["deposit_amount_mxn"] = ss["monthly_spei_transfers"] = None
        ss["preferred_channel_choice"] = CHANNEL_AUTO
    if not manual("constraints"):
        for _, widget, _ in HARD_CONSTRAINT_OPTIONS:
            ss[widget] = False
        ss["excluded_banks"] = []


def record_human_decision(decision: str) -> None:
    """Approve / modify / escalate callback; persisted to the review queue and Golden Dataset."""
    ss = st.session_state
    status = dict(ss["hitl_status"])
    notes = str(ss.get("advisor_notes_input", "")).strip()
    override = ss.get("hitl_override_choice") if decision == "modified" else None
    if decision == "modified" and not override:
        ss["run_errors"] = ["Select a replacement Base Option before submitting a modification."]
        return
    terminal = decision in {"approved", "modified"}
    status.update(pending=not terminal, approved_by_human=terminal, advisor_notes=notes, decision=decision,
                  override_product_key=override, reviewed_at=datetime.now().isoformat(timespec="seconds"))
    ss["hitl_status"] = status
    for entry in ss["execution_history"]:
        if entry["run_id"] == status.get("run_id"):
            entry["human_approved"] = terminal
            entry["human_decision"] = decision
            entry["advisor_notes"] = notes
    if status.get("case_id"):
        try:
            ReviewQueue().record_decision(status["case_id"], decision, notes, override, source="app")
        except CaseClosedError as exc:
            ss["run_errors"] = [f"{exc} Showing the reviewer's decision instead."]
            sync_hitl_from_queue()
        except Exception as exc:  # noqa: BLE001 - persistence failure must not lose the in-session decision
            ss["run_errors"] = [f"Decision kept in this session but could not be persisted: {type(exc).__name__}: {exc}"]


def sync_hitl_from_queue() -> None:
    """Adopt the latest decision recorded in the persistent queue for this run's case.

    The queue is the source of truth, so a decision taken in the Reviewer
    Console (any session) is reflected here on the next rerun.
    """
    ss = st.session_state
    status = dict(ss["hitl_status"])
    if not status.get("case_id"):
        return
    try:
        case = ReviewQueue().get(status["case_id"])
    except Exception:  # noqa: BLE001 - the queue is optional for rendering
        return
    if not case or not case["decisions"]:
        return
    last = case["decisions"][-1]
    if status.get("synced_decision_at") == last.get("at_utc"):
        return
    terminal = case["status"] in TERMINAL_DECISIONS
    status.update(pending=not terminal, approved_by_human=terminal, decision=last["decision"],
                  advisor_notes=last.get("advisor_notes", ""), override_product_key=last.get("override_product_key"),
                  reviewed_at=last.get("at_utc"), decided_via=last.get("source"),
                  synced_decision_at=last.get("at_utc"))
    ss["hitl_status"] = status
    for entry in ss["execution_history"]:
        if entry["run_id"] == status.get("run_id"):
            entry["human_approved"] = terminal
            entry["human_decision"] = last["decision"]


def approve_recommendation() -> None:
    record_human_decision("approved")


def modify_recommendation() -> None:
    record_human_decision("modified")


def escalate_recommendation() -> None:
    record_human_decision("escalated")


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def run_pending_evaluation() -> None:
    ss = st.session_state
    run = ss.get("pending_run")
    if not (ss.get("is_evaluating") and run):
        return
    errors: List[str] = []
    events: List[str] = []
    try:
        with st.status(SPINNER_TEXT, expanded=True) as status:
            def on_progress(node: str, update: Mapping[str, Any]) -> None:
                events.append(node)
                lines = list(update.get("trajectory_log") or [])
                detail = lines[0].split("] ", 1)[-1] if lines else ""
                status.write(f"✅ **{agent_label(node)}** · `{node}` — {detail[:140]}")
                status.update(label=f"{SPINNER_TEXT} · {agent_label(node)} finished")

            try:
                resources, app_graph = load_squad(run["use_llm"])
            except Exception as exc:  # noqa: BLE001 - ChromaDB / catalog initialisation failures
                result, errors = None, [f"Could not initialise the catalog / ChromaDB resources: "
                                        f"{type(exc).__name__}: {exc}"]
            else:
                result, errors = execute_squad(run["query"], run["overrides"], resources, app_graph, run["run_id"],
                                               on_progress=on_progress, extra_metadata=run.get("run_metadata"))
            elapsed = ((result or {}).get("execution_metadata") or {}).get("elapsed_s")
            status.update(label=f"Analysis finished in {elapsed} s" if result else "Analysis failed",
                          state="error" if errors else "complete", expanded=False)
        if result is not None:
            ss["squad_state"] = result
            ss["execution_history"].append(summarize_run(result, run["run_id"], run))
            draft = result.get("draft_recommendation") or {}
            memory = conversation()
            provenance = profile_provenance(result, memory.profile)
            if provenance:
                ss["profile_provenance"] = provenance
                ss["widget_sync"] = result_widget_values(result)  # applied before the sidebar renders
            memory.record_assistant(assistant_summary(result),
                                    kind="direct_answer" if draft.get("status") == "direct_answer"
                                    else "recommendation", run_id=run["run_id"],
                                    llm=getattr(resources, "llm", None),
                                    profile_used=provenance_line(provenance) or None)
            hitl = dict(DEFAULT_HITL_STATUS, run_id=run["run_id"], pending=bool(result.get("requires_human_review")))
            if hitl["pending"]:
                try:
                    hitl["case_id"] = ReviewQueue().enqueue(build_decision_card(result, run["run_id"]))
                except Exception as exc:  # noqa: BLE001 - disk issues must not hide the recommendation
                    errors.append(f"Could not persist the case to the review queue: {type(exc).__name__}: {exc}")
            ss["hitl_status"] = hitl
    finally:
        ss["progress_events"] = events
        ss["is_evaluating"] = False
        ss["pending_run"] = None
        ss["run_errors"] = errors
    st.rerun()


# --------------------------------------------------------------------------- #
# Section 1 - Sidebar
# --------------------------------------------------------------------------- #
def provenance_caption(group: str) -> None:
    """Origin of every value of a sidebar group in the last run (or that nothing was set yet)."""
    rows = [r for r in st.session_state.get("profile_provenance") or [] if r["group"] == group]
    if manual(group):
        st.caption("🎛️ Manual: these values are authoritative for the next run. Chat changes still update them.")
    elif not rows:
        st.caption(f"{SOURCE_ICONS['pending']} Locked: the Profiling Agent will infer these from your message "
                   "(and ask when something essential is missing).")
    else:
        st.caption("Last run used: " + " · ".join(
            f"{r['field']} **{r['value']}** {SOURCE_ICONS.get(r['source'], '')} {SOURCE_LABELS.get(r['source'], '')}"
            for r in rows))


def render_sidebar() -> None:
    ss = st.session_state
    apply_widget_sync(ss)  # must run before the widgets are created
    memory = ss.get("conversation")
    asking_deposit = memory is not None and memory.awaiting == "deposit_mxn"
    with st.sidebar:
        st.header("🎛️ Profile Configuration")
        st.caption("Every group is locked by default: the agent infers it from the chat and the fields show what "
                   "it used. Switch a group to manual to set it yourself.")

        st.subheader("Financial inputs")
        manual_financial = st.toggle(MANUAL_TOGGLES["financial"][1], key=MANUAL_TOGGLES["financial"][0],
                                     on_change=on_manual_toggle, args=("financial",))
        if asking_deposit and not manual_financial:
            st.warning("❓ The agent asked for your average balance: type it here and press **Use … (sidebar)** "
                       "in the chat, or just reply in the chat.", icon="✍️")
        st.number_input("Initial deposit / average monthly balance (MXN)", min_value=0.0, max_value=MAX_DEPOSIT_MXN,
                        step=DEPOSIT_STEP_MXN, format="%.2f", key="deposit_amount_mxn", on_change=on_deposit_change,
                        disabled=not (manual_financial or asking_deposit), placeholder="not set yet",
                        help="Used by calculate_net_impact for fee waivers, yield tiers and ISR withholding. "
                             "If your message has no amount, the assistant asks for it first.")
        st.number_input("Estimated SPEI transfers per month", min_value=0, max_value=MAX_MONTHLY_TRANSFERS, step=1,
                        key="monthly_spei_transfers", disabled=not manual_financial, placeholder="not set yet")
        st.selectbox("Preferred SPEI transfer channel", CHANNEL_OPTIONS, key="preferred_channel_choice",
                     disabled=not manual_financial,
                     help="Transfer fees differ by channel (e.g. branch vs. mobile app). 'Auto' uses the channel "
                          "implied by your query, or the cheapest listed channel per product.")
        provenance_caption("financial")

        st.subheader("Preference vector w")
        manual_weights = st.toggle(MANUAL_TOGGLES["weights"][1], key=MANUAL_TOGGLES["weights"][0],
                                   on_change=on_manual_toggle, args=("weights",),
                                   help="Off: the Profiling Agent (LLM or rules) builds w from your message and the "
                                        "sliders show the result. On: the sliders are used as they are.")
        for key, (widget, label, help_text) in WEIGHT_WIDGETS.items():
            st.slider(label, min_value=0.0, max_value=1.0, step=0.01, key=widget, help=help_text,
                      on_change=on_weight_change, args=(key,), disabled=not manual_weights)
        weights = ss["normalized_weights"]
        st.caption("Normalized w = [" + ", ".join(f"{weights[k]:.2f}" for k in PREFERENCE_KEYS)
                   + f"]  ·  Σ = {sum(weights.values()):.2f}")
        provenance_caption("weights")

        st.subheader("Hard constraints")
        manual_constraints = st.toggle(MANUAL_TOGGLES["constraints"][1], key=MANUAL_TOGGLES["constraints"][0],
                                       on_change=on_manual_toggle, args=("constraints",))
        for _, widget, label in HARD_CONSTRAINT_OPTIONS:
            st.checkbox(label, key=widget, disabled=not manual_constraints)
        st.multiselect("Exclude banks", BANK_OPTIONS, key="excluded_banks", disabled=not manual_constraints,
                       help="Banks you do not want recommended (also set by saying e.g. 'nothing from Santander').")
        provenance_caption("constraints")

        archetype = classify_archetype(weights, selected_constraints(ss))
        if manual_weights:
            st.info(f"Classified archetype: **{ARCHETYPE_LABELS.get(archetype, archetype)}** (`{archetype}`)")
        else:
            inferred = next((r["value"] for r in ss.get("profile_provenance") or [] if r["field"] == "Archetype"), None)
            st.info(f"Archetype inferred by the Profiling Agent: **{inferred}**" if inferred
                    else "Archetype will be inferred by the Profiling Agent from your query.")

        st.divider()
        st.toggle("Use LLMs (gpt-6-luna narrative & critic, gpt-4o-mini profiling)", key="use_llm",
                  help="Off = fully deterministic mode (template narrative, rule-based checks).")
        st.caption(f"Time budget: 10 s per query · Freshness window: {FRESHNESS_MAX_AGE_DAYS} days · "
                   "No personal data is stored beyond this browser session.")


# --------------------------------------------------------------------------- #
# Section 2 - Query & execution
# --------------------------------------------------------------------------- #
def render_conversation(memory: Optional[ConversationMemory]) -> None:
    """Chat transcript of the sliding window (+ rolling summary of older turns)."""
    if memory is None or not memory.turns:
        return
    if memory.summary:
        st.caption(f"🧠 Earlier conversation (summarized, {memory.folded_turns} turn(s)): {memory.summary}")
    for turn in memory.turns:
        with st.chat_message("user" if turn.role == "user" else "assistant"):
            st.markdown(turn.content)
            if turn.changes:
                st.caption("Profile updated: " + "; ".join(turn.changes))
            if turn.profile_used:
                st.caption("Profile used: " + turn.profile_used)
    if memory.awaiting == "deposit_mxn":
        deposit = st.session_state.get("deposit_amount_mxn")
        c1, c2, _ = st.columns([2.2, 1.6, 3])
        c1.button(f"Use {money(float(deposit))} (sidebar)" if deposit is not None else "Use the sidebar field (empty)",
                  key="clarify_use_sidebar", on_click=use_sidebar_deposit,
                  disabled=bool(st.session_state["is_evaluating"]) or deposit is None)
        c2.button("Skip (assume MXN 0)", key="clarify_skip", on_click=skip_deposit,
                  disabled=bool(st.session_state["is_evaluating"]))


def render_query_section() -> None:
    ss = st.session_state
    memory = ss.get("conversation")
    started = memory is not None and bool(memory.turns)
    st.subheader("💬 Refine your search" if started else "💬 Describe your banking needs")
    render_conversation(memory)
    st.text_area("Your message (English)" if started else "Natural-language request (English)",
                 key="user_query_input", height=90 if started else 110,
                 placeholder=REFINE_PLACEHOLDER if started else EXAMPLE_QUERY)
    col_run, col_example, col_new, _ = st.columns([2, 1.3, 1.6, 2])
    col_run.button("🚀 Analyze Banking Options", key="analyze_btn", type="primary", on_click=start_evaluation,
                   disabled=bool(ss["is_evaluating"]))
    col_example.button("Load example", key="example_btn", on_click=load_example, disabled=bool(ss["is_evaluating"]))
    col_new.button("🔄 New conversation", key="new_conversation_btn", on_click=new_conversation,
                   disabled=bool(ss["is_evaluating"]) or not started)

    run_pending_evaluation()

    for error in ss.get("run_errors") or []:
        st.error(error, icon="🚨")


# --------------------------------------------------------------------------- #
# Section 3 - Reasoning inspector
# --------------------------------------------------------------------------- #
def render_freshness_badge(result: Mapping[str, Any]) -> None:
    flag = result.get("data_freshness_flag")
    report = (result.get("execution_metadata") or {}).get("freshness_report") or {}
    oldest = report.get("oldest_verified_date") or "unknown"
    if flag is True:
        st.markdown(f":green-badge[🟢 Data Verified (<{FRESHNESS_MAX_AGE_DAYS} days)]  "
                    f"Oldest verification date: {oldest}")
    elif flag is False:
        stale = ", ".join(report.get("stale_products") or []) or "policy documents"
        st.markdown(f":orange-badge[🟡 Preliminary Data Warning (>{FRESHNESS_MAX_AGE_DAYS} days)]  "
                    f"Stale: {stale} · oldest verification date: {oldest}")
    else:
        st.markdown(":gray-badge[⚪ Freshness not evaluated]")


def render_reasoning_section(result: Mapping[str, Any]) -> None:
    meta = result.get("execution_metadata") or {}
    st.subheader("🧠 Agentic Reasoning Inspector")
    render_freshness_badge(result)
    if result.get("is_fallback_mode"):
        st.markdown(":red-badge[🟠 Fallback Mode — cached CONDUSEF baseline data]  "
                    + "; ".join((meta.get("fallback_reasons") or []) or ["live sources unavailable"]))
    tokens = meta.get("token_usage") or {}
    st.caption(f"Intent: `{meta.get('intent', 'recommendation')}` · tokens: prompt {tokens.get('prompt_tokens', 0):,}"
               f" · completion {tokens.get('completion_tokens', 0):,} · total {tokens.get('total_tokens', 0):,}"
               + (f" · conversation turn {meta.get('conversation_turn')}" if meta.get("conversation_turn") else ""))

    routes = {"evaluator_tot_beam": "ToT beam", "direct_answer": "Direct answer"}
    cols = st.columns(5)
    cols[0].metric("Route", routes.get(meta.get("route"), "Linear"))
    archetype = result.get("archetype_prior")
    cols[1].metric("Archetype", ARCHETYPE_LABELS.get(archetype, str(archetype or "n/a")).replace("The ", ""),
                   help=f"Agent Design Refinement archetype id: {archetype}")
    cols[2].metric("Confidence", f"{float(result.get('confidence_score') or 0):.2f}")
    cols[3].metric("Revisions", f"{int(result.get('revision_count') or 0)} / {MAX_REVISIONS}")
    cols[4].metric("Elapsed", f"{meta.get('elapsed_s', 0):.2f} s",
                   delta="within budget" if meta.get("within_budget") else "over budget",
                   delta_color="normal" if meta.get("within_budget") else "inverse")

    with st.expander("🔍 View Agent Execution Trajectory & ToT Reasoning Log"):
        st.caption(f"Routing: {meta.get('route_reason', 'n/a')} · profiling source: {meta.get('profiling_source', 'n/a')} "
                   f"· LLM: {meta.get('llm_model') or 'deterministic'}")
        st.dataframe(pd.DataFrame(trajectory_rows(result.get("trajectory_log") or [])), hide_index=True,
                     width="stretch", height=320)

        events = st.session_state.get("progress_events") or []
        if events:
            st.caption("Live execution order: " + " → ".join(events))
        timings = {k.split(".", 1)[1]: v for k, v in meta.items() if k.startswith("timing_ms.")}
        if timings:
            st.caption("Node latency (ms, last execution): "
                       + " · ".join(f"{agent_label(node)} [{node}] {ms:,.0f}" for node, ms in timings.items()))

        branches = display_branches(result)
        if higher_balance_branches(result):
            st.caption(f"Yield objective: accounts paying ≥ {MEANINGFUL_YIELD_PERCENT:.0f}% at your balance come first, "
                       "then 🔓 accounts that would pay with a higher balance (figures at that balance).")
        if meta.get("evaluation_mode") == "tot_beam" or meta.get("route") == "evaluator_tot_beam":
            st.markdown(f"**Tree-of-Thought Beam Search** (b = {BEAM_WIDTH}, D = {MAX_DEPTH}) · "
                        f"branches explored: {meta.get('tot_branches_explored', len(result.get('tot_candidate_branches') or []))}"
                        f" · depth reached: {meta.get('tot_depth_reached', 'n/a')}")
            st.dataframe(pd.DataFrame(branch_rows(branches)), hide_index=True, width="stretch")
            if meta.get("early_stopped"):
                st.success(early_stop_explanation(result), icon="⏹️")
            else:
                st.info(early_stop_explanation(result), icon="🌳")
        else:
            st.markdown("**Linear evaluation** (simple query; ToT not triggered). Top candidates:")
            st.dataframe(pd.DataFrame(branch_rows(branches)), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# Section 4 - Recommendation dashboard
# --------------------------------------------------------------------------- #
def render_option_card(option: Optional[Mapping[str, Any]], title: str, fallback_text: str) -> None:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        if not option:
            st.info(fallback_text)
            return
        st.markdown(f"**{option['institution_name']} — {option['product_name']}**")
        constraints_ok = option.get("hard_constraints_met")
        st.caption(("✅ Meets all hard constraints" if constraints_ok else "⚠️ Does not meet every hard constraint")
                   + f" · SPEI channel: {option.get('transfer_channel_used') or 'n/a'}"
                   + f" · verified {option.get('last_verified_date') or 'n/a'}")
        c1, c2 = st.columns(2)
        c1.metric("Gross Annual Yield (MXN)", money(option["gross_annual_interest_mxn"]),
                  help=f"Rate {option['applicable_yield_rate_percent']:.2f}% · ISR withheld "
                       f"{money(option.get('isr_withheld_mxn', 0.0))}")
        c2.metric("Effective Monthly Maintenance Fee (MXN)", money(option["effective_monthly_fee_mxn"]))
        if option.get("fee_waived_by_balance"):
            c2.caption(f"✅ Waived by your average balance (≥ {money(option.get('minimum_balance_for_fee_waiver_mxn'))})")
        net = float(option["net_annual_impact_mxn"])
        c3, c4 = st.columns(2)
        c3.metric("Net Annual Impact (MXN)", money(net), delta=f"{net:+,.2f} MXN" if net else "break-even",
                  delta_color="normal" if net else "off")
        c4.metric("Composite Match Score (0–100)", f"{float(option['composite_score']):.2f}")
        st.caption(f"Total annual fees: {money(option['total_annual_fees_mxn'])}")


def leaderboard_option(row: Mapping[str, Any], products_by_key: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Option-card fields of a Top-5 row, built exactly like the recommended options (stated balance)."""
    product = products_by_key.get(row["product_key"])
    if product is None or "_impact" not in row:
        return None
    branch = {"product_key": row["product_key"], "institution_name": product.institution_name,
              "product_name": product.product_name, "score": row["composite_score"],
              "hard_constraints_met": row["meets_constraints"], "hard_constraint_checks": row["_checks"],
              "net_impact": row["_impact"]}
    return build_option(branch, product.model_dump(mode="json"))


def recommendation_roles(draft: Mapping[str, Any]) -> Dict[str, str]:
    """product_key -> role label of the accounts the agent recommended."""
    roles: Dict[str, str] = {}
    if draft.get("tradeoff_option"):
        roles[draft["tradeoff_option"]["product_key"]] = "📈 Trade-off"
    if draft.get("base_option"):
        roles[draft["base_option"]["product_key"]] = "🛡️ Base Option"
    return roles


def render_catalog_chart(rows: Sequence[Mapping[str, Any]], roles: Optional[Mapping[str, str]] = None) -> None:
    roles = roles or {}
    top = list(rows)[:TOP_N_CHART]
    if not top:
        st.info("No catalog product could be scored.")
        return
    fig = go.Figure()

    def label(r: Mapping[str, Any]) -> str:
        suffix = "" if r["meets_constraints"] else " ✖"
        if r.get("unlock_balance_mxn") is not None:
            suffix += f" 🔓 from MXN {r['unlock_balance_mxn']:,.0f}"
        if r["product_key"] in roles:
            suffix += f" · {roles[r['product_key']]}"
        return r["label"] + suffix

    def signed_mxn(value: float) -> str:
        """Bar label that always shows the sign: '−240' (Unicode minus, easier to see), '+12', '0'."""
        rounded = round(value)
        if rounded == 0:
            return f"{value:+,.2f}".replace("-", "−") if value else "0"
        return f"{rounded:+,.0f}".replace("-", "−")

    def net(r: Mapping[str, Any]) -> float:
        return r["unlock_net_annual_impact_mxn"] if r.get("unlock_balance_mxn") is not None else r["net_annual_impact_mxn"]

    def pattern(r: Mapping[str, Any]) -> str:
        if r.get("unlock_balance_mxn") is not None:
            return CHART_PATTERN_UNLOCK
        return "" if r["meets_constraints"] else CHART_PATTERN_FAILS

    series = (  # color = sign of the net shown on the bar (rounded to the cent)
        ("Net gain", CHART_GAIN_COLOR, lambda r: round(net(r), 2) > 0),
        ("Net cost", CHART_COST_COLOR, lambda r: round(net(r), 2) < 0),
        ("Break-even", CHART_EVEN_COLOR, lambda r: round(net(r), 2) == 0),
    )
    for name, color, member in series:
        subset = [r for r in top if member(r)]
        if not subset:
            continue
        marker: Dict[str, Any] = {"color": color, "cornerradius": 4}
        if any(pattern(r) for r in subset):  # texture marks the constraint status without a new hue (CVD / print safe)
            marker["pattern"] = {"shape": [pattern(r) for r in subset], "fgcolor": "#ffffff", "solidity": 0.35}
        fig.add_trace(go.Bar(
            name=name,
            orientation="h",
            y=[label(r) for r in subset],
            x=[net(r) for r in subset],
            marker=marker,
            text=[signed_mxn(net(r)) for r in subset],
            textposition="outside",
            cliponaxis=False,  # a negative bar's label sits left of its end; never clip it (or its sign)
            customdata=[[r["composite_score"], r["gross_annual_interest_mxn"], r["total_annual_fees_mxn"],
                         ", ".join(r["failed_constraints"]) or "none",
                         r.get("unlock_rate_percent", r["yield_rate_percent"])] for r in subset],
            hovertemplate=("<b>%{y}</b><br>Net annual impact: MXN %{x:,.2f}<br>Yield rate: %{customdata[4]:.2f}%"
                           "<br>Composite score (stated balance): %{customdata[0]:.2f}"
                           "<br>Gross interest: MXN %{customdata[1]:,.2f}<br>Annual fees: MXN %{customdata[2]:,.2f}"
                           "<br>Failed constraints: %{customdata[3]}<extra></extra>"),
        ))
    order = [label(r) for r in top]
    values = [net(r) for r in top]
    lo, hi = min(0.0, *values), max(0.0, *values)
    pad = 0.22 * ((hi - lo) or 1.0)  # room for the outside labels on whichever side has bars
    x_range = [lo - pad if lo < 0 else 0.0, hi + pad if hi > 0 else 0.1 * pad]
    fig.update_layout(
        height=110 + 52 * len(top),
        bargap=0.35,
        margin={"l": 10, "r": 40, "t": 10, "b": 10},
        xaxis={"title": "Net annual impact (MXN, after fees and ISR)", "zeroline": True, "tickformat": ",.0f",
               "range": x_range},
        yaxis={"categoryorder": "array", "categoryarray": order[::-1], "title": None},
        legend={"orientation": "h", "y": -0.25},
        showlegend=len(fig.data) > 1,
    )
    st.plotly_chart(fig, width="stretch", theme="streamlit")
    markers = []
    if any(not r["meets_constraints"] and r.get("unlock_balance_mxn") is None for r in top):
        markers.append("✖ / crossed texture = fails a hard constraint")
    if any(r.get("unlock_balance_mxn") is not None for r in top):
        markers.append("🔓 / striped texture = pays only with a higher balance (net shown at that balance)")
    if any(r["product_key"] in roles for r in top):
        markers.append("🛡️ / 📈 = the account the agent recommends as Base Option / Positive Trade-off")
    st.caption(" · ".join(["Order follows the ranking in the title (composite match score), not the net impact; "
                           "green = net gain, red = net cost, gray = break-even"] + markers))
    with st.expander("Table view"):
        st.dataframe(pd.DataFrame([{
            "Product": r["label"], "Composite score": r["composite_score"],
            "Yield rate (%)": r["yield_rate_percent"],
            "Pays from (MXN)": r.get("unlock_balance_mxn"), "Rate there (%)": r.get("unlock_rate_percent"),
            "Net annual (MXN)": r["net_annual_impact_mxn"], "Gross interest (MXN)": r["gross_annual_interest_mxn"],
            "Annual fees (MXN)": r["total_annual_fees_mxn"], "Meets hard constraints": r["meets_constraints"],
            "Failed constraints": ", ".join(r["failed_constraints"]),
        } for r in rows]), hide_index=True, width="stretch")


def render_fine_print(draft: Mapping[str, Any], products_by_key: Mapping[str, Any]) -> None:
    st.markdown("##### 📜 Qualitative benefits & fine print (RECA context)")
    options = [o for o in (draft.get("base_option"), draft.get("tradeoff_option")) if o]
    if not options:
        st.caption("No option selected.")
        return
    for option in options:
        clauses = [c for c in draft.get("supporting_clauses") or [] if c.get("product_key") == option["product_key"]]
        render_option_fine_print(option, products_by_key.get(option["product_key"]), clauses)


def render_option_fine_print(option: Mapping[str, Any], product: Any, clauses: Sequence[Mapping[str, Any]]) -> None:
    """Benefits, penalties and RECA excerpts of one account (recommended option or any Top-5 account)."""
    lines: List[str] = []
    for perk in str(option.get("qualitative_perks_summary") or "").split(","):
        if perk.strip():
            lines.append(f"- {perk.strip().rstrip('.')}")
    if product is not None:
        lines.append(f"- Insurance coverage: {'complimentary life/fraud insurance (details vary)' if product.has_free_insurance else 'not included'}")
        if product.has_cashback_or_perks:
            lines.append("- Cashback / rewards: benefits available (details vary)")
        for penalty in product.penalties_detail:
            lines.append(f"- ⚠️ Penalty: {penalty.penalty_concept} — {money(penalty.cost_mxn)} ({penalty.billing_frequency})")
        if option.get("minimum_balance_for_fee_waiver_mxn"):
            lines.append(f"- ⚠️ Maintenance fee applies below an average balance of "
                         f"{money(option['minimum_balance_for_fee_waiver_mxn'])}")
    for clause in clauses:
        lines.append(f"- 📄 *{clause.get('document_type')}* (verified {clause.get('last_verified_date')}): "
                     f"“{clause.get('excerpt')}…”")
    st.markdown(f"**{option['institution_name']} — {option['product_name']}**\n" + "\n".join(lines))


def render_top_detail(result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], resources: Any,
                      roles: Mapping[str, str]) -> None:
    """Pick any Top-5 account and see the same card and fine print as the recommended options."""
    top = [r for r in list(rows)[:TOP_N_CHART] if r["product_key"] in resources.products_by_key]
    if not top:
        return
    names = {r["product_key"]: f"{i}. {r['label']}" + (f" · {roles[r['product_key']]}" if r["product_key"] in roles
                                                       else "") for i, r in enumerate(top, start=1)}
    if st.session_state.get("top5_detail_key") not in names:  # a new run changed the Top 5: reset the choice
        st.session_state.pop("top5_detail_key", None)
    with st.container(border=True):
        key = st.selectbox("🔎 See the details of a Top-5 account", list(names), format_func=names.get,
                           key="top5_detail_key")
        row = next(r for r in top if r["product_key"] == key)
        option = leaderboard_option(row, resources.products_by_key)
        if option is None:
            st.caption("Details unavailable for this account.")
            return
        explanation = ((result.get("draft_recommendation") or {}).get("top_score_explanation") or {})
        if explanation.get("product_key") == key:
            st.warning(explanation["text"], icon="ℹ️")
        elif key in roles:
            st.success(f"This is the account the agent recommends as {roles[key]}.", icon="✅")
        render_option_card(option, f"{row['label']}", "No data.")
        if row.get("unlock_balance_mxn") is not None:
            st.caption(f"Figures at your stated balance; the chart shows the net at MXN {row['unlock_balance_mxn']:,.0f}, "
                       "where this account starts paying its yield.")
        if not row["meets_constraints"]:
            st.caption(f"✖ Fails: {', '.join(row['failed_constraints'])}.")
        render_option_fine_print(option, resources.products_by_key.get(key), supporting_clauses(result, [key]))


def render_history() -> None:
    history = st.session_state["execution_history"]
    with st.expander(f"🕘 Session execution history ({len(history)} run(s)) — compare queries"):
        if not history:
            st.caption("Runs will appear here.")
            return
        st.dataframe(pd.DataFrame(history[::-1]), hide_index=True, width="stretch")


def render_direct_answer(draft: Mapping[str, Any]) -> None:
    """Routed factual / benefit question: deterministic catalog facts (no recommendation)."""
    with st.container(border=True):
        st.markdown("#### 📌 Direct answer")
        st.markdown(draft.get("answer") or "No catalog facts found.")
        rows = [{
            "Product": f"{f['institution_name']} {f['product_name']}",
            "Effective monthly fee (MXN)": f["effective_monthly_fee_mxn"],
            "Waived by balance": f["fee_waived_by_balance"],
            "Annual fees (MXN)": f["total_annual_fees_mxn"],
            "Yield rate (%)": f["annual_yield_rate_percent"],
            "Min. opening deposit (MXN)": f["minimum_opening_deposit_mxn"],
            "Verified": f["last_verified_date"],
        } for f in draft.get("facts") or []]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        for clause in draft.get("supporting_clauses") or []:
            st.caption(f"📄 {clause.get('document_type')} (verified {clause.get('last_verified_date')}): "
                       f"“{clause.get('excerpt')}…”")
        st.caption("Ask for a recommendation (e.g. 'Which account is best for me?') to run the full agent squad.")


def render_negotiation(options: Sequence[Mapping[str, Any]]) -> None:
    """Criteria negotiation: closest products and one-click constraint relaxation."""
    with st.container(border=True):
        st.markdown("#### 🤝 Criteria negotiation — closest alternatives")
        for index, option in enumerate(options):
            c1, c2 = st.columns([4, 2])
            c1.markdown(f"**{option['institution_name']} — {option['product_name']}**  \n"
                        f"Relax: {', '.join(option['relax_labels'])} · annual fees "
                        f"{money(option['total_annual_fees_mxn'])} · net {money(option['net_annual_impact_mxn'])}")
            c2.button(f"Relax {', '.join(option['relax_labels'])}", key=f"relax_btn_{index}",
                      on_click=relax_constraints, args=(tuple(option["relax_constraints"]),),
                      disabled=bool(st.session_state["is_evaluating"]))


def render_dashboard(result: Mapping[str, Any], resources: Any) -> None:
    draft = result.get("draft_recommendation") or {}
    st.subheader("🏆 Recommendation Dashboard")
    status = str(draft.get("status") or "n/a")
    narrative = draft.get("narrative") or {}
    if status in {"abstained_provisional", "pending_human_review"} or result.get("guardrail_warnings"):
        for warning in draft.get("warnings") or result.get("guardrail_warnings") or []:
            st.warning(warning, icon="⚠️")
    st.markdown(f"**Status:** {STATUS_LABELS.get(status, status)} · narrative source: "
                f"`{draft.get('narrative_source', 'n/a')}` · critic approved: "
                f"{'yes' if result.get('review_approved') else 'no'} · LLM critic: "
                f"`{(result.get('execution_metadata') or {}).get('llm_critic_status', 'n/a')}`")
    if status == "direct_answer":
        render_direct_answer(draft)
        render_history()
        return
    if narrative.get("summary"):
        st.markdown(narrative["summary"])
    if draft.get("negotiation"):
        render_negotiation(draft["negotiation"])

    left, right = st.columns(2)
    with left:
        render_option_card(draft.get("base_option"), "🛡️ Base Option — Risk-Free / $0 Fee",
                           narrative.get("base_option_rationale") or "No eligible product.")
        if draft.get("base_option") and narrative.get("base_option_rationale"):
            st.caption(narrative["base_option_rationale"])
    with right:
        render_option_card(draft.get("tradeoff_option"), "📈 Positive Trade-off — Highest Net Yield",
                           narrative.get("tradeoff_option_rationale") or "No trade-off option.")
        if draft.get("tradeoff_option") and narrative.get("tradeoff_option_rationale"):
            st.caption(narrative["tradeoff_option_rationale"])

    explanation = draft.get("top_score_explanation")
    if explanation:  # the #1 by match score is not recommended: say why, explicitly
        st.info("**Why the highest-scoring account is not the recommendation** — " + explanation["text"]
                + " Recommendations only include accounts with no fees (Base Option) or a higher positive net "
                  "return (Trade-off).", icon="ℹ️")

    if narrative.get("key_tradeoffs"):
        st.markdown("**Key trade-offs**\n" + "\n".join(f"- {t}" for t in narrative["key_tradeoffs"]))

    roles = recommendation_roles(draft)
    rows = catalog_leaderboard(result, resources.catalog) if resources is not None else []
    chart_col, print_col = st.columns([3, 2])
    with chart_col:
        yield_goal = "yield" in ((result.get("user_profile") or {}).get("objectives") or [])
        st.markdown(f"##### Net annual impact — top {TOP_N_CHART} of {len(rows)} eligible catalog products "
                    + ("(paying now first, then 🔓 paying with a higher balance)" if yield_goal else "by composite score"))
        if resources is not None:
            render_catalog_chart(rows, roles)
    with print_col:
        render_fine_print(draft, resources.products_by_key if resources is not None else {})
    if resources is not None:
        render_top_detail(result, rows, resources, roles)

    render_history()


# --------------------------------------------------------------------------- #
# Section 5 - HITL review queue
# --------------------------------------------------------------------------- #
def render_hitl_section(result: Mapping[str, Any]) -> None:
    ss = st.session_state
    draft = result.get("draft_recommendation") or {}
    st.subheader("🛡️ Human-in-the-Loop Review Queue")
    if not result.get("requires_human_review"):
        st.success("No escalation required: confidence, amount, data consistency and critique-loop checks "
                   "are all within thresholds.", icon="✅")
        render_review_queue()
        return

    reasons = draft.get("human_review_reasons") or []
    sync_hitl_from_queue()
    hitl = ss["hitl_status"]
    labels = {b["product_key"]: f"{b['institution_name']} {b['product_name']}" for b in final_branches(result)}
    if hitl.get("approved_by_human"):
        decision = DECISION_LABELS.get(hitl.get("decision"), "Approved")
        override = hitl.get("override_product_key")
        extra = f" Base Option replaced by {labels.get(override, override)}." if override else ""
        via = " (via Reviewer Console)" if hitl.get("decided_via") == "reviewer_console" else ""
        st.success(f"{decision} by human advisor{via} at {hitl.get('reviewed_at')}.{extra} "
                   f"Notes: {hitl.get('advisor_notes') or '—'}", icon="✅")
    elif hitl.get("decision") == "escalated":
        st.warning(f"Escalated to a senior reviewer at {hitl.get('reviewed_at')}; the case remains open. "
                   f"Notes: {hitl.get('advisor_notes') or '—'}", icon="🔺")
    else:
        st.error("This recommendation has been routed to the Human-in-the-Loop review queue "
                 "(Protocol B: asynchronous audit). It is shown to you, but it is not final until a "
                 "financial specialist approves it.", icon="🧑‍⚖️")

    with st.container(border=True):
        st.markdown("**Bounded Decision Card**")
        st.markdown("**Escalation cause(s)**")
        for reason in reasons:
            label, correction = hitl_reason_details(reason)
            st.markdown(f"- **{label}** — {reason}.  \n  *Suggested correction:* {correction}")
        profile = result.get("user_profile") or {}
        base = draft.get("base_option") or {}
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Deposit", money(profile.get("deposit_mxn")))
        k2.metric("Confidence", f"{float(result.get('confidence_score') or 0):.2f}",
                  help=f"Threshold {CONFIDENCE_THRESHOLD:.2f}")
        k3.metric("Best score", f"{float(result.get('evaluator_score') or 0):.2f}")
        k4.metric("Base net annual", money(base.get("net_annual_impact_mxn")))
        conflicts = result.get("data_conflicts") or []
        if conflicts:
            st.markdown("**Conflict vector (catalog API vs RECA clause)**")
            st.dataframe(pd.DataFrame([{
                "Product": c.get("product_key"), "Type": c.get("type"), "Catalog value": c.get("api_value"),
                "RECA value": c.get("document_value"), "Source": c.get("source_file"),
            } for c in conflicts]), hide_index=True, width="stretch")

        closed = bool(hitl.get("approved_by_human"))
        st.text_input("Add Human Advisor Notes", key="advisor_notes_input", disabled=closed)
        alternatives = [key for key in labels if key != base.get("product_key")]
        st.selectbox("Replacement Base Option (for Modify)", alternatives, index=None, key="hitl_override_choice",
                     format_func=lambda key: labels.get(key, key), placeholder="Choose another evaluated product",
                     disabled=closed or not alternatives)
        b1, b2, b3 = st.columns(3)
        b1.button("✅ Approve Recommendation", key="approve_btn", on_click=approve_recommendation, disabled=closed)
        b2.button("✏️ Modify Recommendation", key="modify_btn", on_click=modify_recommendation,
                  disabled=closed or not alternatives)
        b3.button("🔺 Escalate to Senior Reviewer", key="escalate_btn", on_click=escalate_recommendation,
                  disabled=closed)
        if hitl.get("case_id"):
            st.caption(f"Persistent review case `{hitl['case_id']}` · decisions are appended to the Golden Dataset.")

    render_review_queue()


def render_review_queue() -> None:
    try:
        queue = ReviewQueue()
        cases = queue.cases()
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Review queue unavailable: {type(exc).__name__}: {exc}")
        return
    pending = [c for c in cases if c["status"] in {"pending", "escalated"}]
    with st.expander(f"📥 Persistent review queue — {len(pending)} open / {len(cases)} total case(s)"):
        if not cases:
            st.caption("No escalated cases yet.")
            return
        st.dataframe(pd.DataFrame([{
            "Case": c["card"].get("case_id"),
            "Created (UTC)": c["card"].get("created_at_utc"),
            "Status": c["status"],
            "Deposit (MXN)": (c["card"].get("profile") or {}).get("deposit_mxn"),
            "Reasons": "; ".join(c["card"].get("escalation_reasons") or []),
            "Query (PII-redacted)": c["card"].get("query_redacted"),
        } for c in reversed(cases)]), hide_index=True, width="stretch")
        st.caption(f"Stored in {queue.directory} (queue + golden_dataset.jsonl). Free text is PII-redacted. "
                   "Reviewers can decide open cases from any session in the **Reviewer Console** page.")


# --------------------------------------------------------------------------- #
# Footer & page
# --------------------------------------------------------------------------- #
def render_bottom_new_conversation() -> None:
    """Second "New conversation" button after the results, so a new search does not require scrolling up."""
    st.divider()
    col_new, col_hint = st.columns([1.6, 5])
    col_new.button("🔄 New conversation", key="new_conversation_bottom_btn", on_click=new_conversation,
                   disabled=bool(st.session_state["is_evaluating"]))
    col_hint.caption("Done reviewing? Start a new search with a different need, or refine this one in the box above.")


def render_footer(result: Optional[Mapping[str, Any]]) -> None:
    st.divider()
    st.caption(f"⚖️ {FOOTER_DISCLAIMER}")
    draft = (result or {}).get("draft_recommendation") or {}
    for disclaimer in draft.get("disclaimers") or []:
        st.caption(disclaimer)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🏦", layout="wide")
    init_session_state(st.session_state)

    st.title(f"🏦 {APP_TITLE}")
    st.caption(f"{APP_TAGLINE} · 5-agent LangGraph squad · ChromaDB MMR (k=4) · ToT beam search (b={BEAM_WIDTH}, D={MAX_DEPTH})")

    render_sidebar()

    resources = None
    try:
        resources, _ = load_squad(bool(st.session_state["use_llm"]))
        if st.session_state["use_llm"] and resources.llm is None:
            st.sidebar.caption("⚠️ OpenAI key not configured; running in deterministic mode.")
    except Exception as exc:  # noqa: BLE001 - keep the page usable and explain the failure
        st.error(f"Could not initialise the catalog / ChromaDB resources: {type(exc).__name__}: {exc}", icon="🚨")

    render_query_section()

    result = st.session_state["squad_state"]
    if result and (result.get("execution_metadata") or {}).get("elapsed_s") is not None:
        st.divider()
        render_reasoning_section(result)
        st.divider()
        render_dashboard(result, resources)
        st.divider()
        render_hitl_section(result)
        render_bottom_new_conversation()
    else:
        st.info("Describe your needs and press **🚀 Analyze Banking Options**. The agent infers your profile from "
                "the message (and asks when something essential is missing); switch a sidebar group to manual to "
                "set it yourself.", icon="ℹ️")
        render_history()

    render_footer(result)


if __name__ == "__main__":
    main()
