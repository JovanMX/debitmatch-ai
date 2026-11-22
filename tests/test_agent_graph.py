"""Production verification for the DebitMatch AI agent graph.

Run with:
    python test_agent_graph.py              # auto: live gpt-4o-mini if OPENAI_API_KEY is set
    python test_agent_graph.py --offline    # deterministic fallback (no LLM calls)

Exit code 0 only when every test passes. Assertions are written to hold in both
modes: every number and routing decision is deterministic; only narrative text
comes from the LLM.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import time
import traceback
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # tests/ sits one level below the project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pydantic import BaseModel  # noqa: E402

from agents.common import SquadResources  # noqa: E402
from agents.evaluator import BEAM_WIDTH, EARLY_STOP_SCORE, MAX_DEPTH  # noqa: E402
from graph import build_default_resources, build_graph, run_squad  # noqa: E402
from guardrails import MAX_REVISIONS, check_human_intervention_needed  # noqa: E402
from prompts.advisory_prompts import AdvisoryNarrative, LEGAL_DISCLAIMER  # noqa: E402
from prompts.evaluator_prompts import CriticVerdict  # noqa: E402
from agents.extraction import ProfileExtraction  # noqa: E402
from prompts.profiling_prompts import ProfilingLLMOutput  # noqa: E402
from tools.catalog_search import product_net_impact  # noqa: E402
from tools.schemas import BankProduct  # noqa: E402

REFERENCE_DATE = date(2026, 11, 21)  # pinned (>= newest verification date: RECA PDFs 2026-11-21) so freshness results do not drift with the calendar
CORE_NODES = ("profiling_node", "research_node", "supervisor_node", "advisory_node", "review_node")

QUERY_SIMPLE = "I want a digital debit account with zero monthly fee."
QUERY_HYBRID = "I have $50,000 MXN. I want high yield, zero fees, and cashback perks."
QUERY_HIGH_AMOUNT = "I have $150,000 MXN to deposit and I want a digital account with zero monthly fees."
QUERY_CONFLICT = "I want a digital payroll account; I keep about 6,000 MXN and want to avoid the monthly fee."

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _assert_trajectory(state: Mapping[str, Any], extra_nodes: tuple[str, ...] = ()) -> None:
    log = "\n".join(state["trajectory_log"])
    for node in CORE_NODES + extra_nodes:
        assert f"[{node}]" in log, f"trajectory_log has no entry for {node}"
        assert f"[{node}] completed in" in log, f"{node} did not report completion"


def _with_catalog(resources: SquadResources, catalog: List[BankProduct]) -> SquadResources:
    return dataclasses.replace(resources, catalog=catalog)


def _without_early_stop_products(resources: SquadResources, query: str, max_rounds: int = 10) -> tuple:
    """Drop the products whose D1 thought clears EARLY_STOP_SCORE, so the beam must reach D=3.

    Keeps test (a) independent of how strong the real catalog is (it grows with every precatalog merge).
    """
    current = resources
    for _ in range(max_rounds):
        graph = build_graph(current)
        state = run_squad(query, current, graph)
        if not state["execution_metadata"].get("early_stopped"):
            return current, graph
        strong = {b["product_key"] for b in state["tot_candidate_branches"]
                  if b["depth"] == 1 and b["score"] > EARLY_STOP_SCORE}
        current = _with_catalog(current, [p for p in current.catalog if p.catalog_key not in strong])
    raise AssertionError("could not build a catalog without early-stop products")


def _patched_catalog(resources: SquadResources, key: str, **changes: Any) -> List[BankProduct]:
    patched = []
    for product in resources.catalog:
        if product.catalog_key == key:
            data = product.model_dump(mode="json")
            for field, value in changes.items():
                data[field] = value
            product = BankProduct.model_validate(data)
        patched.append(product)
    return patched


def _synthetic_high_yield_product() -> BankProduct:
    """Fictional product (not a real bank) used to exercise ToT early stopping."""
    return BankProduct.model_validate({
        "institution_id": "99999", "institution_name": "SYNTHETIC TEST BANK",
        "product_id": "YIELD12", "product_name": "Synthetic High-Yield Account",
        "minimum_opening_deposit": 0.0, "monthly_maintenance_fee": 0.0, "minimum_balance_for_fee_waiver": 0.0,
        "spei_transfer_fee": 0.0, "offers_yield": True, "annual_yield_rate": 12.0,
        "is_digital_account": True, "is_payroll_compatible": True, "has_no_maintenance_fee": True,
        "has_mobile_app": True, "has_virtual_card": True, "has_free_insurance": True, "has_cashback_or_perks": True,
        "qualitative_perks_summary": "Synthetic test fixture: 12% yield, cashback and insurance.",
        "last_verified_date": "2026-11-10",
        "yield_tiers": [{"tier_level": 1, "min_balance_mxn": 0.0, "max_balance_mxn": None,
                         "annual_yield_rate": 12.0, "payout_frequency": "daily"}],
        "monthly_fees_detail": [{"service_concept": "Account Management & Maintenance", "cost_mxn": 0.0,
                                 "has_minimum_balance_exemption": False, "waiver_minimum_average_balance_mxn": 0.0}],
        "transfer_fees_detail": [{"transfer_type": "Interbank SPEI Transfer", "channel": "Mobile Banking App",
                                  "cost_per_transaction_mxn": 0.0, "free_transactions_per_month": 0}],
        "penalties_detail": [],
    })


class HallucinatingLLM:
    """Fake LLM whose narrative always invents an amount -> must be rejected every time."""

    model_name = "fake-hallucinating-llm"

    def __init__(self) -> None:
        self.calls: Dict[str, int] = {}

    def generate(self, schema: Type[T], system_prompt: str, payload: Mapping[str, Any],
                 timeout_s: float, run_name: str) -> T:
        self.calls[run_name] = self.calls.get(run_name, 0) + 1
        if schema is ProfilingLLMOutput:
            return schema(archetype="Lifestyle_Maximizer", yield_importance=0.2, fee_importance=0.5,
                          perks_importance=0.3, rationale="Fake profile.")
        if schema is ProfileExtraction:  # hybrid extraction: also tries to invent a deposit -> must be discarded
            return schema(amounts=[{"amount_mxn": 999_999.0, "kind": "average_balance", "evidence": "a fortune"}],
                          constraints=[], monthly_transfers=None, monthly_transfers_evidence=None,
                          preferred_channel=None, channel_evidence=None, banks=[], objectives=["perks"],
                          business_account=False, archetype="Lifestyle_Maximizer", yield_importance=0.2,
                          fee_importance=0.5, perks_importance=0.3)
        if schema is AdvisoryNarrative:
            return schema(summary="This account pays a guaranteed bonus of MXN 9,999.99 every year.",
                          base_option_rationale="It also earns 15% annual interest.",
                          tradeoff_option_rationale="None.", key_tradeoffs=["Invented figures."])
        if schema is CriticVerdict:
            return schema(verdict="approve", issues=[], feature_claims=[])
        raise AssertionError(f"Unexpected schema {schema}")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_1_simple_linear_query(resources: SquadResources, graph: Any) -> Dict[str, Any]:
    state = run_squad(QUERY_SIMPLE, resources, graph)
    meta = state["execution_metadata"]

    assert state["query_complexity"] == "simple", state["query_complexity"]
    assert meta["route"] == "evaluator_linear"
    _assert_trajectory(state, ("evaluator_linear",))
    assert "[evaluator_tot_beam]" not in "\n".join(state["trajectory_log"]), "ToT must not run for a simple query"
    assert {b["depth"] for b in state["tot_candidate_branches"]} == {1}

    profile = state["user_profile"]
    assert set(profile["hard_constraints"]) == {"zero_fee", "digital_opening"}
    assert abs(sum(state["weighted_preference_vector"].values()) - 1.0) < 1e-9
    assert state["archetype_prior"] in {"Flow_Manager", "Strategic_Saver", "Basic_Protected_User", "Lifestyle_Maximizer"}

    draft = state["draft_recommendation"]
    # Catalog-agnostic: the base option is the best-scored root thought and truly satisfies both constraints.
    best_root = max(state["tot_candidate_branches"], key=lambda b: b["score"])
    assert draft["base_option"]["product_key"] == best_root["product_key"], draft["base_option"]
    chosen = resources.products_by_key[draft["base_option"]["product_key"]]
    assert chosen.is_digital_account and chosen.has_no_maintenance_fee, chosen.catalog_key
    assert draft["base_option"]["effective_monthly_fee_mxn"] == 0.0
    assert draft["base_option"]["hard_constraints_met"] is True
    assert draft["status"] in {"definitive", "pending_human_review"}
    assert LEGAL_DISCLAIMER in draft["disclaimers"]
    assert meta["within_budget"], f"exceeded 10 s budget: {meta['elapsed_s']}s"
    return state


def test_2_hybrid_tot_query(resources: SquadResources, graph: Any) -> Dict[str, Any]:
    # (a) Real catalog minus every product that clears 85 points at D1 -> full beam search to D=3.
    resources, graph = _without_early_stop_products(resources, QUERY_HYBRID)
    state = run_squad(QUERY_HYBRID, resources, graph)
    meta = state["execution_metadata"]
    assert state["query_complexity"] == "hybrid"
    assert meta["route"] == "evaluator_tot_beam"
    _assert_trajectory(state, ("evaluator_tot_beam",))
    assert state["user_profile"]["deposit_mxn"] == 50_000.0
    assert set(state["user_profile"]["objectives"]) == {"yield", "fee", "perks"}

    branches = state["tot_candidate_branches"]
    assert {b["depth"] for b in branches} == {1, 2, 3}, "ToT must explore D=3 levels"
    assert meta["tot_depth_reached"] == MAX_DEPTH and meta["early_stopped"] is False
    for depth in (1, 2, 3):
        in_beam = [b for b in branches if b["depth"] == depth and b["in_beam"]]
        assert 0 < len(in_beam) <= BEAM_WIDTH, f"beam width violated at depth {depth}: {len(in_beam)}"
    assert len([b for b in branches if b["depth"] == 1]) >= BEAM_WIDTH, "need >= b root thoughts"

    # Deterministic calculation: every scored thought matches calculate_net_impact exactly.
    profile = state["user_profile"]
    assert {b["stage"] for b in branches} == {"product_selection", "yield_assessment", "benefits_and_fees"}
    for branch in branches:
        product = resources.products_by_key[branch["product_key"]]
        # D1/D2 price the user's preferred channel; D3 (fee structure) prices each explored channel explicitly.
        channel = profile["preferred_channel"] if branch["depth"] in (1, 2) else branch["strategy"]["transfer_channel"]
        expected = product_net_impact(product, 50_000.0, profile["monthly_transfers"], channel)
        assert branch["net_impact"] == expected, f"{branch['branch_id']} diverges from calculate_net_impact"

    # (b) Add a fictional 12 % zero-fee product: it should trigger early stopping at depth 1.
    rich = _with_catalog(resources, [*resources.catalog, _synthetic_high_yield_product()])
    early = run_squad(QUERY_HYBRID, rich, build_graph(rich))
    early_meta = early["execution_metadata"]
    assert early_meta["route"] == "evaluator_tot_beam"
    assert early_meta["early_stopped"] is True and early_meta["tot_depth_reached"] == 1, early_meta
    assert early["evaluator_score"] > EARLY_STOP_SCORE
    assert any("EARLY STOP" in line for line in early["trajectory_log"])
    base = early["draft_recommendation"]["base_option"]
    assert base["product_key"] == "99999_YIELD12"
    assert base["gross_annual_interest_mxn"] == 6_000.0  # 50,000 x 12 %, ISR-exempt (< 5 UMA)
    assert base["net_annual_impact_mxn"] == 6_000.0
    assert meta["within_budget"] and early_meta["within_budget"]
    return state


def test_3_freshness_and_abstention(resources: SquadResources, graph: Any) -> Dict[str, Any]:
    # (a) Stale catalog record (> 30 days old).
    stale = _with_catalog(resources, _patched_catalog(resources, "40002_SwitchBanamex", last_verified_date="2026-09-01"))
    state = run_squad(QUERY_SIMPLE, stale, build_graph(stale))
    assert state["data_freshness_flag"] is False
    draft = state["draft_recommendation"]
    assert draft["status"] == "abstained_provisional", draft["status"]
    warnings = " ".join(state["guardrail_warnings"])
    assert "ABSTENTION NOTICE" in warnings and "older than 30 days" in warnings, warnings
    assert any("Abstention protocol ACTIVE" in line for line in state["trajectory_log"])
    assert any("2026-09-01" in d for d in draft["disclaimers"]), "freshness disclaimer must cite the stale date"
    assert state["execution_metadata"]["within_budget"], f"exceeded budget: {state['execution_metadata']['elapsed_s']}s"

    # (b) API vs RAG discrepancy: catalog says MXN 5,000 waiver, RECA clause says MXN 3,000.
    conflict_catalog = _patched_catalog(resources, "40014_060014", minimum_balance_for_fee_waiver=5000.0,
                                        monthly_fees_detail=[
                                            {"service_concept": "Below Minimum Average Balance Fee", "cost_mxn": 140.0,
                                             "has_minimum_balance_exemption": True,
                                             "waiver_minimum_average_balance_mxn": 5000.0}])
    conflicted = _with_catalog(resources, conflict_catalog)
    conflict_state = run_squad(QUERY_CONFLICT, conflicted, build_graph(conflicted))
    conflicts = conflict_state["data_conflicts"]
    assert any(c["product_key"] == "40014_060014" and c["type"] == "waiver_threshold_mismatch"
               and c["document_value"] == 3000.0 for c in conflicts), conflicts
    assert conflict_state["draft_recommendation"]["status"] == "abstained_provisional"
    assert conflict_state["requires_human_review"] is True
    assert any("regulatory conflict" in r for r in conflict_state["execution_metadata"]["hitl_reasons"])
    assert conflict_state["execution_metadata"]["within_budget"], conflict_state["execution_metadata"]["elapsed_s"]
    return state


def test_4_hitl_escalation(resources: SquadResources, graph: Any) -> Dict[str, Any]:
    state = run_squad(QUERY_HIGH_AMOUNT, resources, graph)
    assert state["user_profile"]["deposit_mxn"] == 150_000.0
    assert state["requires_human_review"] is True
    assert check_human_intervention_needed(state) is True
    reasons = state["execution_metadata"]["hitl_reasons"]
    assert any("high financial amount" in r for r in reasons), reasons
    draft = state["draft_recommendation"]
    assert draft["requires_human_review"] is True and "hitl_notice" in draft
    assert draft["status"] in {"pending_human_review", "abstained_provisional"}
    assert any("HITL escalation required" in line for line in state["trajectory_log"])
    assert state["execution_metadata"]["within_budget"], f"exceeded budget: {state['execution_metadata']['elapsed_s']}s"
    return state


def test_5_review_loop_is_bounded(resources: SquadResources, graph: Any) -> Dict[str, Any]:
    fake = HallucinatingLLM()
    looped = dataclasses.replace(resources, llm=fake)
    state = run_squad(QUERY_SIMPLE, looped, build_graph(looped))
    assert state["revision_count"] == MAX_REVISIONS, state["revision_count"]
    assert state["review_approved"] is False
    assert fake.calls.get("advisory_llm") == MAX_REVISIONS + 1, fake.calls  # initial draft + 2 revisions
    assert any("MXN amounts not produced by the calculator" in issue for issue in state["review_feedback"])
    assert state["requires_human_review"] is True
    assert any("critique loop exhausted" in r for r in state["execution_metadata"]["hitl_reasons"])
    log = "\n".join(state["trajectory_log"])
    assert log.count("[evaluator_node] Revision") == MAX_REVISIONS
    # the invented MXN 999,999 deposit (evidence not in the query) was discarded by the extraction verification
    assert state["user_profile"]["deposit_mxn"] != 999_999.0 and "Node profiling_node failed" not in log
    assert "LLM extraction item discarded: amount 999,999.00" in log
    return state


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
TESTS: List[Callable[[SquadResources, Any], Dict[str, Any]]] = [
    test_1_simple_linear_query,
    test_2_hybrid_tot_query,
    test_3_freshness_and_abstention,
    test_4_hitl_escalation,
    test_5_review_loop_is_bounded,
]


def _summary(state: Optional[Mapping[str, Any]]) -> str:
    if not state:
        return ""
    meta = state.get("execution_metadata") or {}
    draft = state.get("draft_recommendation") or {}
    base = (draft.get("base_option") or {}).get("product_key")
    return (f"route={meta.get('route')} status={draft.get('status')} base={base} "
            f"score={state.get('evaluator_score')} conf={state.get('confidence_score')} "
            f"HITL={state.get('requires_human_review')} narrative={draft.get('narrative_source')} "
            f"elapsed={meta.get('elapsed_s')}s")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DebitMatch AI agent graph verification.")
    parser.add_argument("--offline", action="store_true", help="Disable LLM calls (deterministic fallback).")
    parser.add_argument("--show-trajectory", action="store_true", help="Print trajectory_log for each test.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

    resources = build_default_resources(use_llm=not args.offline, reference_date=REFERENCE_DATE)
    graph = build_graph(resources)
    mode = f"live ({resources.llm.model_name})" if resources.llm else "offline (deterministic fallback)"
    print(f"DebitMatch AI agent graph tests - mode: {mode}, reference date {REFERENCE_DATE}\n")

    failures = 0
    for test in TESTS:
        started = time.perf_counter()
        state: Optional[Dict[str, Any]] = None
        try:
            state = test(resources, graph)
        except Exception:  # noqa: BLE001 - runner must report every failure
            failures += 1
            print(f"[FAIL] {test.__name__}")
            traceback.print_exc()
        else:
            print(f"[PASS] {test.__name__} ({time.perf_counter() - started:.2f}s) {_summary(state)}")
        if args.show_trajectory and state:
            print("       " + "\n       ".join(state["trajectory_log"]))
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
