"""Tests for the requirements recovered from the design specification review.

Covers (deterministic, no OpenAI calls):
    memory        - ConversationMemory deltas, removals, exclusions, weights, clarification, sliding window
    guardrails    - epistemic overconfidence markers, benefit-in-RECA-but-not-in-API conflicts
    fallback      - Fallback Mode when the structured source fails; hard-timeout baseline
    ToT           - stale-data x0.8 downweight, tie resolution within 5 points
    routing       - intent router (direct answers) and benchmark queries still routed to the squad
    negotiation   - closest alternatives when nothing satisfies all hard constraints
    RAG           - atomic benefit chunks with benefit_type metadata
    MCP           - conversation state-manager tools

Usage::

    .venv\\Scripts\\python.exe tests\\test_doc_alignment.py
"""

from __future__ import annotations

import dataclasses
import sys
import time
import traceback
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # tests/ sits one level below the project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

import agents.research as research_module  # noqa: E402
from agents.evaluator import STALE_DOWNWEIGHT_FACTOR, TIE_MARGIN, _rank_with_ties  # noqa: E402
from agents.router import classify_intent  # noqa: E402
from evaluation.dataset_generator import BENCHMARK_CASES, apply_scenario  # noqa: E402
from graph import _timeout_fallback, build_default_resources, build_graph, run_squad  # noqa: E402
from guardrails import (  # noqa: E402
    benefit_conflicts,
    detect_api_rag_conflicts,
    overconfidence_markers,
    review_draft,
    unresolved_conflicts,
)
from memory import DEPOSIT_QUESTION, ConversationMemory, assistant_summary, extract_delta  # noqa: E402
from prompts.advisory_prompts import FALLBACK_DISCLAIMER, LEGAL_DISCLAIMER  # noqa: E402
from state import create_initial_state  # noqa: E402
from tools.catalog_search import eligible_products, nearest_alternatives  # noqa: E402
from tools.rag_engine import split_benefit_entries  # noqa: E402
from tools.schemas import API_BENEFIT_DOC_TYPE  # noqa: E402

REFERENCE_DATE = date(2026, 11, 21)  # >= newest verification date (RECA PDFs 2026-11-21)
_R: Dict[str, Any] = {}


def _resources() -> Any:
    if "r" not in _R:
        _R["r"] = build_default_resources(use_llm=False, reference_date=REFERENCE_DATE)
        _R["g"] = build_graph(_R["r"])
    return _R["r"]


def _graph() -> Any:
    _resources()
    return _R["g"]


def _run(query: str, resources: Any = None, **overrides: Any) -> Dict[str, Any]:
    res = resources or _resources()
    graph = _graph() if resources is None else build_graph(res)
    return run_squad(query, res, graph, run_metadata={"profile_overrides": overrides} if overrides else None)


# --------------------------------------------------------------------------- #
# Memory (short-term: typed state + sliding window + summarizer)
# --------------------------------------------------------------------------- #
def test_memory_deltas() -> None:
    from agents.profiling import extract_signals

    # verb forms of the zero-fee requirement (benchmark arch_basic_protected: "never charges me a monthly fee")
    for text in ("I just want a basic account that never charges me a monthly fee.",
                 "an account that doesn't charge any fees", "I never want to pay a monthly fee",
                 "I don't want to pay fees", "without paying any commission"):
        assert "zero_fee" in extract_signals(text)["hard_constraints"], text
        assert "zero_fee" in extract_delta(text).add_constraints, text
    for text in ("the bank never charges interest", "I don't mind if it charges a fee",
                 "it charges me a monthly fee of 50"):
        assert "zero_fee" not in extract_signals(text)["hard_constraints"], text
    assert extract_delta("Actually I have 8,000 MXN.").deposit_mxn == 8_000.0
    assert extract_delta("Forget the payroll requirement.").remove_constraints == ["payroll"]
    assert extract_delta("I can't maintain that minimum balance.").add_constraints == ["no_minimum_balance"]
    assert extract_delta("I can't maintain that minimum balance of 3,000 MXN.").deposit_mxn is None  # rejection
    assert extract_delta("Nothing from Santander, please.").exclude_institutions == ["SANTANDER"]
    assert extract_delta("Fees matter more to me than perks.").weight_shifts == {"fee_importance": 0.15,
                                                                               "perks_importance": -0.15}
    assert extract_delta("I don't need to pay fees.").remove_constraints == []
    assert extract_delta("around 12k", awaiting_deposit=True).deposit_mxn == 12_000.0
    skip = extract_delta("skip", awaiting_deposit=True)
    assert skip.deposit_mxn == 0.0 and skip.deposit_assumed


def test_memory_clarification_and_window() -> None:
    memory = ConversationMemory(window_size=4, catalog=_resources().catalog)
    plan = memory.plan_turn("I want a digital account with no fees.")
    assert plan.action == "clarify" and plan.question == DEPOSIT_QUESTION
    plan = memory.plan_turn("about 9,000 pesos")
    assert plan.action == "run" and plan.message == "about 9,000 pesos"
    assert plan.overrides["deposit_mxn"] == 9_000.0 and plan.overrides["replace_hard_constraints"]
    assert plan.run_metadata["latest_user_message"] == "I want a digital account with no fees."  # resumed request
    memory.record_assistant("Base Option: X.")
    for message in ("Forget the digital requirement.", "Nothing from HSBC.", "Fees matter more than perks."):
        memory.plan_turn(message)
        memory.record_assistant("Base Option: Y.")
    assert len(memory.turns) == 4 and memory.folded_turns > 0 and memory.summary  # sliding window + summary
    assert memory.profile.excluded_institutions == ["HSBC"]
    assert "digital_opening" not in memory.profile.hard_constraints
    assert "Earlier:" in memory.compose_query("next")
    product_fact = ConversationMemory(catalog=_resources().catalog).plan_turn("What is the monthly fee of Cuenta Maestra?")
    assert product_fact.action == "run"  # factual question: no deposit clarification


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #
def test_overconfidence_markers_trigger_rewrite() -> None:
    assert overconfidence_markers("This is guaranteed and 100% safe; you will earn more.") == [
        "guaranteed", "100%", "you will earn"]
    assert overconfidence_markers("The Base Option (Risk-Free / $0 Fee) appears suitable.") == []
    state = _run("I want a digital debit account with zero monthly fee.", deposit_mxn=10_000.0)
    draft = dict(state["draft_recommendation"])
    draft["narrative"] = {**draft["narrative"], "summary": draft["narrative"]["summary"] + " It is guaranteed."}
    assert any("Epistemic overconfidence" in issue for issue in review_draft(draft, state))


def test_internal_identifiers_trigger_rewrite() -> None:
    """Field names or product keys leaking into the narrative send the draft back for revision."""
    from guardrails import internal_identifiers

    leaked = "No Positive Trade-off Option was identified among the accounts_compared candidates (40156_20_103)."
    assert internal_identifiers(leaked) == ["accounts_compared", "40156_20_103"]
    assert internal_identifiers("No Positive Trade-off Option was identified among the 12 accounts compared.") == []
    state = _run("I want a digital debit account with zero monthly fee.", deposit_mxn=10_000.0)
    draft = state["draft_recommendation"]
    assert not any("internal identifiers" in issue for issue in review_draft(draft, state))  # template is clean
    leaky = {**draft, "narrative": {**draft["narrative"], "tradeoff_option_rationale": leaked}}
    assert any("internal identifiers ['accounts_compared', '40156_20_103']" in issue
               for issue in review_draft(leaky, state))


def test_benefit_conflict_is_advisory() -> None:
    products = [{"institution_id": "40002", "product_id": "SwitchBanamex", "product_name": "Switch Banamex",
                 "has_free_insurance": False, "has_cashback_or_perks": True, "monthly_fees_detail": []}]
    chunk = {"chunk_id": "c1", "text": "Coverage 1. A complimentary fraud insurance benefit is included. "
                                       "Deposits are protected by IPAB up to 400,000 UDIs.",
             "metadata": {"institution_id": "40002", "product_id": "SwitchBanamex", "document_type": "poliza_seguros",
                          "last_verified_date": "2026-11-01"}}
    conflicts = detect_api_rag_conflicts(products, [chunk])
    assert [(c["type"], c["severity"], c["benefit"]) for c in conflicts] == [
        ("benefit_missing_in_api", "advisory", "complimentary insurance")]
    state = {"data_conflicts": conflicts}
    assert benefit_conflicts(state) and not unresolved_conflicts(state)  # no full abstention / HITL
    api_text = dict(chunk, metadata={**chunk["metadata"], "document_type": API_BENEFIT_DOC_TYPE})
    assert detect_api_rag_conflicts(products, [api_text]) == []  # same source as the API: never a conflict


# --------------------------------------------------------------------------- #
# Fallback Mode
# --------------------------------------------------------------------------- #
def test_fallback_mode_when_structured_source_fails() -> None:
    original = research_module._branch_a_structured

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise ConnectionError("Financial Product API unreachable")

    research_module._branch_a_structured = broken
    try:
        state = _run("I want a digital account with zero monthly fee.", deposit_mxn=10_000.0)
    finally:
        research_module._branch_a_structured = original
    draft = state["draft_recommendation"]
    assert state["is_fallback_mode"] is True and draft["is_fallback_mode"] is True
    assert draft["disclaimers"][0] == FALLBACK_DISCLAIMER and FALLBACK_DISCLAIMER in draft["warnings"]
    assert draft["base_option"]["total_annual_fees_mxn"] == 0.0  # cached zero-fee baseline


def test_hard_timeout_returns_cached_baseline() -> None:
    state = create_initial_state("I want an account with no fees.",
                                 execution_metadata={"profile_overrides": {"deposit_mxn": 5_000.0}})
    fallback = _timeout_fallback(state, 10.0, _resources())
    draft = fallback["draft_recommendation"]
    assert fallback["is_fallback_mode"] and draft["base_option"]["total_annual_fees_mxn"] == 0.0
    assert FALLBACK_DISCLAIMER in draft["disclaimers"] and LEGAL_DISCLAIMER in draft["disclaimers"]
    assert any("additional options may exist" in w for w in draft["warnings"])


# --------------------------------------------------------------------------- #
# ToT pruning rules
# --------------------------------------------------------------------------- #
def test_tot_stale_downweight() -> None:
    query = "I have $50,000 MXN. I want high yield, zero fees, and cashback perks."
    fresh = _run(query)
    stale_res = dataclasses.replace(_resources(), catalog=apply_scenario(
        _resources().catalog, {"type": "stale_products", "product_keys": ["40002_SwitchBanamex"],
                               "last_verified_date": "2026-09-01"}))
    stale = _run(query, stale_res)
    assert stale["execution_metadata"]["route"] == "evaluator_tot_beam"

    def d1(state: Dict[str, Any]) -> float:
        return next(b["score"] for b in state["tot_candidate_branches"]
                    if b["depth"] == 1 and b["product_key"] == "40002_SwitchBanamex")

    assert abs(d1(stale) - round(d1(fresh) * STALE_DOWNWEIGHT_FACTOR, 2)) < 0.011, (d1(fresh), d1(stale))
    assert any("downweight" in line for line in stale["trajectory_log"])


def test_tot_tie_resolution() -> None:
    class _Eval:
        flagged = {"B_stale"}
        preferred_channel = None
        evidence: Dict[str, int] = {"A_ok": 1}

        def adherence(self, branch: Dict[str, Any]) -> tuple:
            return (int(branch["hard_constraints_met"]), int(branch["product_key"] not in self.flagged),
                    1, int(bool(self.evidence.get(branch["product_key"]))))

    def branch(key: str, score: float) -> Dict[str, Any]:
        return {"branch_id": key, "product_key": key, "score": score, "hard_constraints_met": True,
                "net_impact": {"net_annual_impact_mxn": 0.0}, "strategy": {}}

    ranked = _rank_with_ties([branch("B_stale", 70.0), branch("A_ok", 70.0 - TIE_MARGIN + 1)], _Eval())
    assert [b["product_key"] for b in ranked] == ["A_ok", "B_stale"] and "tie_break" in ranked[0]
    far = _rank_with_ties([branch("B_stale", 70.0), branch("A_ok", 60.0)], _Eval())
    assert [b["product_key"] for b in far] == ["B_stale", "A_ok"]  # outside the 5-point margin: score wins


# --------------------------------------------------------------------------- #
# Routing, negotiation, RAG metadata
# --------------------------------------------------------------------------- #
def test_intent_router() -> None:
    catalog = _resources().catalog
    assert classify_intent("What is the monthly fee of Débito LikeU Pro?", catalog)[0] == "product_fact"
    assert classify_intent("Does Cuenta Maestra include insurance?", catalog)[0] == "benefit_lookup"
    assert classify_intent("Which is better for me, Switch Banamex or Cuenta Maestra?", catalog)[0] == "recommendation"
    intent, products = classify_intent("What does Cuenta Inmediata Bis charge for SPEI?", catalog)
    assert intent == "product_fact" and [p.catalog_key for p in products] == ["40014_72214"]
    # One-word short names ("NOW") only count when qualified: "from now on" is not a product mention.
    assert classify_intent("From now on, how much do my transfers cost from the app?", catalog)[1] == []
    intent, products = classify_intent("What is the yield of Cuenta Now?", catalog)
    assert intent == "product_fact" and [p.catalog_key for p in products] == ["40059_NW0001"]
    for case in BENCHMARK_CASES:  # the benchmark must keep exercising the full squad
        assert classify_intent(case["inputs"]["query"], catalog)[0] == "recommendation", case["inputs"]["case_id"]

    state = _run("Does Producto Básico General include insurance?")
    draft = state["draft_recommendation"]
    assert draft["status"] == "direct_answer" and draft["intent"] == "benefit_lookup"
    assert "complimentary insurance" in draft["answer"] and draft["supporting_clauses"]
    assert "profiling_node" not in " ".join(state["trajectory_log"])


def test_criteria_negotiation() -> None:
    catalog = _resources().catalog
    options = nearest_alternatives(catalog, ["cashback", "free_insurance", "no_minimum_balance"], 10_000.0)
    assert options and all(len(o["relax_constraints"]) >= 1 for o in options)
    # Fewest relaxed constraints first (several products only miss free insurance).
    assert options[0]["relax_constraints"] == ["free_insurance"]
    assert [len(o["relax_constraints"]) for o in options] == sorted(len(o["relax_constraints"]) for o in options)
    state = _run("I want an account with good benefits.", deposit_mxn=10_000.0,
                 hard_constraints=["cashback", "free_insurance", "no_minimum_balance"])
    draft = state["draft_recommendation"]
    assert draft["status"] == "no_match" and draft["negotiation"]
    assert "closest alternatives" in draft["narrative"]["summary"]


def test_atomic_benefit_chunks() -> None:
    engine = _resources().rag_engine
    assert engine.count_document_type(API_BENEFIT_DOC_TYPE) >= 13
    hits = engine.query_mmr("complimentary life insurance", institution_id="40021", product_id="CTAMAESTRA69",
                            document_types=[API_BENEFIT_DOC_TYPE], k=4)
    assert hits and all(h["metadata"]["document_type"] == API_BENEFIT_DOC_TYPE for h in hits)
    assert any(h["metadata"].get("benefit_type") == "insurance" for h in hits)
    # Thousands separators are not list separators ("$40,000 mxn" stays one benefit).
    assert split_benefit_entries("Digital opening & management, monthly fee waived with $40,000 mxn average "
                                 "balance, complimentary life/fraud insurance.") == [
        "Digital opening & management", "monthly fee waived with $40,000 mxn average balance",
        "complimentary life/fraud insurance"]
    efe = engine.collection.get(where={"$and": [{"document_type": API_BENEFIT_DOC_TYPE},
                                                {"institution_id": "40036"}, {"product_id": "1002"}]})
    assert any("$20,000 mxn average balance" in doc for doc in efe["documents"]), efe["documents"]  # Efe folleto
    assert not any(doc.rstrip(".").endswith(("(credit card", "insurance)")) for doc in efe["documents"]), efe["documents"]
    assert not any(doc.endswith(": 000 mxn average balance.") for doc in efe["documents"])


def test_business_accounts_excluded_by_default() -> None:
    catalog = _resources().catalog
    business = {p.catalog_key for p in catalog if p.is_business_account}
    assert "40113_BX+_Empresa_PFAE" in business and "40058_58135" in business
    assert "40113_Dinero_vista_PM_PF_PFAE" not in business  # also sold to individuals
    personal = eligible_products(catalog, {})
    assert not business & {p.catalog_key for p in personal}
    assert business <= {p.catalog_key for p in eligible_products(catalog, {"business_account": True})}

    state = _run("I want a checking account with low fees.", deposit_mxn=20_000.0)
    seen = {r["catalog_key"] for r in state["api_structured_data"]}
    assert not seen & business and any("business account(s)" in line for line in state["trajectory_log"])
    for_company = _run("I need an account for my business, PFAE, with low fees.", deposit_mxn=20_000.0)
    assert state["user_profile"]["business_account"] is False and for_company["user_profile"]["business_account"]
    assert business & {r["catalog_key"] for r in for_company["api_structured_data"]}


def test_minor_account_flags_profiling_and_filtering() -> None:
    from agents.profiling import minor_holder_constraint
    from tools.catalog_search import evaluate_hard_constraints

    catalog = _resources().catalog
    by_key = {p.catalog_key: p for p in catalog}
    guardianship = {k for k, p in by_key.items() if p.for_minor_under_guardianship}
    # 40127_01 GUARDADITO KIDS: corrected in the catalog build; the source file says false
    assert guardianship == {"40012_0201", "40030_CHAVOS", "40030_PEKES", "40036_1005", "40137_2500", "40127_01"}, guardianship
    assert "for_minor" not in type(catalog[0]).model_fields  # the file's forMinor flag is not used (unreliable)

    assert minor_holder_constraint("I want an account for my 10-year-old son") == "minor_guardianship"
    assert minor_holder_constraint("My kids need a savings account") == "minor_guardianship"
    assert minor_holder_constraint("Cuenta para mi hija") == "minor_guardianship"
    assert minor_holder_constraint("I'm 16 and I just got my first job") == "minor_guardianship"  # a minor's own account
    # ages written in words; deposit with "Mexican pesos"
    spelled_out = "I would like an account for my fifteen-year-old son, with an initial balance of 5,000 Mexican pesos."
    assert minor_holder_constraint(spelled_out) == "minor_guardianship"
    assert extract_delta(spelled_out).deposit_mxn == 5_000.0
    assert minor_holder_constraint("my seven-year-old son needs a savings account") == "minor_guardianship"
    assert minor_holder_constraint("I'm fifteen and want an account") == "minor_guardianship"
    for adult_text in ("account for my twenty-five-year-old son", "account for my eighteen-year-old daughter",
                       "I'm fifteen thousand pesos short"):
        assert minor_holder_constraint(adult_text) is None, adult_text
    assert extract_delta("I have 12,000 pesos mexicanos").deposit_mxn == 12_000.0
    assert extract_delta("I keep MX$8,000").deposit_mxn == 8_000.0
    # income / per-period amounts are not the average balance
    assert extract_delta("I earn 50000 mexican pesos a moth, I want a debit card with yields").deposit_mxn is None
    assert extract_delta("My salary is $30,000 and I keep 10,000 pesos in the account").deposit_mxn == 10_000.0
    assert extract_delta("I deposit 4,000 pesos a month").deposit_mxn is None
    assert extract_delta("I have 25,000 MXN saved").deposit_mxn == 25_000.0
    for adult in ("account for my 25-year-old daughter", "I'm 35 and want yield", "I'm 16k in debt",
                  "I have 16 transfers", "minor details do not matter", "I want a zero fee account"):
        assert minor_holder_constraint(adult) is None, adult
    assert "minor_guardianship" in extract_delta("Actually it is for my daughter.").add_constraints
    assert "minor_guardianship" in extract_delta("Forget the minor account, it is for me.").remove_constraints

    adult = {p.catalog_key for p in eligible_products(catalog, {})}
    assert not guardianship & adult and "40014_3120" in adult  # minor-only accounts never reach adult advice
    assert guardianship <= {p.catalog_key for p in eligible_products(catalog, {"hard_constraints": ["minor_guardianship"]})}
    assert evaluate_hard_constraints(by_key["40030_PEKES"], ["minor_guardianship"], 0.0) == {"minor_guardianship": True}
    assert evaluate_hard_constraints(by_key["40014_3120"], ["minor_guardianship"], 0.0) == {"minor_guardianship": False}

    state = _run("I want a savings account with yield for my 10-year-old daughter, I will keep 20,000 pesos")
    assert state["user_profile"]["hard_constraints"] == ["minor_guardianship"]
    assert {r["catalog_key"] for r in state["api_structured_data"]} == guardianship
    assert state["draft_recommendation"]["base_option"]["product_key"] == "40036_1005"  # Cuenta Educa pays 5.63 %
    adult_run = _run("I want a zero fee account.", deposit_mxn=20_000.0)
    assert not guardianship & {r["catalog_key"] for r in adult_run["api_structured_data"]}
    assert any("minor under guardianship excluded" in line for line in adult_run["trajectory_log"])


def test_yield_objective_lists_higher_balance_accounts() -> None:
    from agents.evaluator import HIGHER_BALANCE_STAGE, pays_yield_now
    from tools.catalog_search import yield_unlock

    by_key = {p.catalog_key: p for p in _resources().catalog}
    assert yield_unlock(by_key["40060_104"], 5_000.0) == {"unlock_balance_mxn": 10_000.0, "unlock_rate_percent": 3.0}
    assert yield_unlock(by_key["40059_NW0001"], 5_000.0) is None  # already pays 10 %
    assert yield_unlock(by_key["40060_104"], 10_000.0) is None
    assert yield_unlock(by_key["40042_PFADM"], 5_000.0) is None  # 0.01 % is nominal, no higher tier

    state = _run("I earn 50000 mexican pesos a moth, I want a debit card with yields over 10% a month. "
                 "I keep 5,000 pesos")
    branches = state["tot_candidate_branches"]
    upgrades = [b for b in branches if b["stage"] == HIGHER_BALANCE_STAGE]
    assert upgrades and not any(b.get("final") for b in upgrades)  # never a Base / Trade-off candidate
    order = [b["product_key"] for b in upgrades]
    # positive net at the unlock balance first (smallest extra balance first); Bansi's 3 % is eaten by its fees
    assert order[0] == "40036_1002" and order.index("40060_104") > order.index("40113_Cuenta_Verde"), order
    assert all(b["net_impact"]["deposit_evaluated_mxn"] == b["strategy"]["unlock_balance_mxn"] > 5_000 for b in upgrades)
    paying = [b["product_key"] for b in sorted((b for b in branches if b.get("final")), key=lambda b: b["final_rank"])
              if pays_yield_now(b)]
    # NOW pays 7.25 % (Cuenta Now folleto, Oct 2026), below BIM 9.47 % and Sabadell 7.87 %
    assert paying == ["40150_CuentaDigitalBIMTrad", "40156_20_103", "40059_NW0001", "40133_CARN4"], paying
    draft = state["draft_recommendation"]
    keys = {upgrade["product_key"] for upgrade in upgrades}
    assert draft["base_option"]["product_key"] not in keys
    assert not draft.get("tradeoff_option") or draft["tradeoff_option"]["product_key"] not in keys
    assert state["execution_metadata"]["higher_balance_options"] == len(upgrades)

    # the answer names the best higher-balance account with calculator figures, and passes the guardrails
    upgrade = draft["higher_balance_option"]
    assert upgrade["product_key"] == "40036_1002" and upgrade["net_annual_impact_mxn"] > 0
    assert upgrade == draft["verified_facts"]["higher_balance_option"]
    sentence = draft["narrative"]["key_tradeoffs"][-1]
    assert "MXN 40,000.00" in sentence and "2.46%" in sentence and "CUENTA EFE" in sentence, sentence
    assert f"MXN {upgrade['net_annual_impact_mxn']:,.2f}" in sentence and "MXN 35,000.00 more" in sentence
    assert upgrade["yield_rate_is_estimate"] and "estimated rate" in sentence  # 40 % of CETES (Efe folleto)
    from agents.evaluator import critic_narrative
    # the deterministic sentence is already verified: the LLM critic never sees it
    assert sentence not in critic_narrative(draft)["key_tradeoffs"]
    assert critic_narrative(draft)["summary"] == draft["narrative"]["summary"]
    assert review_draft(draft, state) == [] and state["review_approved"] is True
    from evaluation.custom_evaluators import deterministic_faithfulness
    # the benchmark accepts the higher-balance bank (its figures are verified); benchmark outputs name chunks rag_chunks
    score, detail = deterministic_faithfulness({**state, "rag_chunks": state["rag_retrieved_chunks"]})
    assert score == 1.0, detail
    assert "with MXN 40,000.00: INBURSA CUENTA EFE would pay 2.46%" in assistant_summary(state)
    no_upgrade = _run("I want a zero fee account.", deposit_mxn=5_000.0)
    assert no_upgrade["draft_recommendation"]["higher_balance_option"] is None

    no_yield = _run("I want a zero fee account.", deposit_mxn=5_000.0)  # no yield objective: nothing added
    assert not any(b["stage"] == HIGHER_BALANCE_STAGE for b in no_yield["tot_candidate_branches"])


def test_document_figures_and_reco_fee_detector() -> None:
    """RECA documents (folletos) are the ground truth; the detector reads the RECO 'Administración - renta' format."""
    by_key = {p.catalog_key: p for p in _resources().catalog}
    # figures taken from the institutions' own documents are in the live catalog
    flex, one, efe, educa = (by_key[k] for k in ("40021_FLEXSIM17", "40021_ADVANCE27", "40036_1002", "40036_1005"))
    assert flex.monthly_maintenance_fee == 73.5 and flex.minimum_balance_for_fee_waiver == 0.0
    assert one.minimum_balance_for_fee_waiver == 300000.0 and "$35,000 mxn" in one.qualitative_perks_summary
    assert efe.monthly_maintenance_fee == 119.83 and efe.minimum_balance_for_fee_waiver == 20000.0
    assert [t.annual_yield_rate for t in efe.yield_tiers] == [2.46, 3.7, 4.93] and all(t.estimated for t in efe.yield_tiers)
    assert [(t.min_balance_mxn, t.annual_yield_rate, t.estimated) for t in educa.yield_tiers] == [
        (15000.0, 3.08, False), (100000.0, 4.32, False)]
    assert by_key["40113_BX+_Beneficios"].minimum_opening_deposit == 0.0
    assert any(t.channel == "Branch / Teller" and t.cost_per_transaction_mxn == 110.0
               for t in by_key["40014_060014"].transfer_fees_detail)
    now = by_key["40059_NW0001"]
    assert now.annual_yield_rate == 7.25 and any(p.cost_mxn == 70.0 for p in now.penalties_detail)

    # detector: RECO concept + amount layouts, VAT-included figures, waiver phrasings and the published rate
    def conflicts(product: Any, text: str) -> List[str]:
        record = product.model_dump(mode="json")
        chunk = {"chunk_id": "c", "text": text,
                 "metadata": {"institution_id": product.institution_id, "product_id": product.product_id,
                              "document_type": "terminos_y_condiciones_producto"}}
        return sorted(f"{c['type']}={c['document_value']}" for c in detect_api_rag_conflicts([record], [chunk]))
    assert conflicts(flex, "La comisión por Administración Renta tendrá un pago fijo de $65.00 M.N. + IVA") == [
        "monthly_fee_mismatch=65.0"]
    assert conflicts(flex, "La comisión por Administración Renta tendrá un pago fijo de $73.50 M.N. + IVA") == []
    assert conflicts(efe, "Concepto Monto Periodicidad Administración - renta $139.00 Mensual") == []  # VAT included
    assert conflicts(efe, "Administración - renta $180.00 Mensual") == ["monthly_fee_mismatch=180.0"]
    verde = by_key["40113_Cuenta_Verde"]
    assert conflicts(verde, "Administración - renta 1 Mensual $ 265.00 Saldo mínimo") == []
    assert conflicts(verde, "Administración - renta; (banca por internet) 6 Mensual $ 150.00") == []  # other service
    assert conflicts(efe, "El Saldo Promedio Mensual (S.P.M.) mínimo requerido es de $40,000") == [
        "waiver_threshold_mismatch=40000.0"]
    assert conflicts(one, "a) Mantener un saldo promedio mínimo mensual en Cuentas a la Vista, Fondos de Inversión "
                          "e Inversiones a Plazo de $300,000.00 M.N.") == []
    assert conflicts(now, "Tasa de rendimiento 10.00% anual fija") == ["yield_rate_mismatch=10.0"]
    assert conflicts(now, "Tasa de rendimiento 7.25% anual fija") == []


def test_catalog_yield_tiers() -> None:
    """Yield tiers of the live catalog: official CASFIM rates, document rates and flagged estimates."""
    from tools.calculator import calculate_net_impact

    by_key = {p.catalog_key: p for p in _resources().catalog}
    bansi, patrimonial = by_key["40060_104"], by_key["40152_1500"]
    assert [t.annual_yield_rate for t in bansi.yield_tiers] == [3.0, 4.5, 5.0] and bansi.annual_yield_rate == 5.0
    assert len(patrimonial.yield_tiers) == 8 and "5.75%" in patrimonial.qualitative_perks_summary
    assert by_key["40150_CuentaDigitalBIMTrad"].yield_tiers[0].minimum_term_days == 28
    # Gap fills are explicit estimates: official tiers keep their rate at the shared boundary (Sabadell).
    sabadell = [t.model_dump() for t in by_key["40156_20_103"].yield_tiers]
    rate = lambda tiers, deposit: calculate_net_impact(deposit=deposit, yield_tiers=tiers)["applicable_yield_rate_percent"]
    assert (rate(sabadell, 100_000.0), rate(sabadell, 500_000.0), rate(sabadell, 1_000_000.0)) == (8.44, 9.85, 11.25)
    from agents.advisory import _rate_is_estimate
    record = by_key["40156_20_103"].model_dump(mode="json")
    assert _rate_is_estimate(record, 500_000.0) and not _rate_is_estimate(record, 50_000.0)
    # Cuenta Verde pays a share of CETES (Ve por Más fee table, Nov 2026): estimated from the reference CETES rate
    verde = by_key["40113_Cuenta_Verde"]
    assert [(t.min_balance_mxn, t.annual_yield_rate, t.estimated) for t in verde.yield_tiers] == [
        (200000.0, 3.08, True), (500000.0, 4.93, True), (1000000.0, 5.54, True), (2500000.0, 6.16, True)]
    assert verde.yield_tiers[0].payout_frequency == "monthly"
    assert all(t.annual_yield_rate > 0 for p in by_key.values() for t in p.yield_tiers)  # 0 % tiers are dropped

    # One product per near-duplicate group; Switch's branch SPEI follows its RECA fee annex.
    assert not {"40059_NWR022", "40059_NW0023", "40059_DIG042"} & set(by_key)
    # NOW account levels (NV/NVR 2-4) are ONE product, keyed by the CASFIM product id.
    now = by_key["40059_NW0001"]
    assert not {f"40059_{m}" for m in ("NW0021", "NW0031", "NW0041", "NWR021", "NWR031", "NWR041")} & set(by_key)
    # its tiers follow the Cuenta Now folleto (Oct 2026, valid to Dec 2026), not the CASFIM 10 %
    assert [(t.min_balance_mxn, t.max_balance_mxn, t.annual_yield_rate, t.estimated) for t in now.yield_tiers] == [
        (50.0, None, 7.25, False)]
    assert "level 4" in now.qualitative_perks_summary and "7.25%" in now.qualitative_perks_summary
    switch = by_key["40002_SwitchBanamex"]
    assert [t.cost_per_transaction_mxn for t in switch.transfer_fees_detail
            if t.transfer_type == "Interbank SPEI Transfer" and t.channel == "Branch / Teller"] == [108.0]


def test_task_model_routing_and_profiling_skip() -> None:
    """One model per task; temperature only for non-reasoning models; no profiling LLM on clear signals."""
    import os

    from agents.llm import TASK_BY_RUN_NAME
    from agents.profiling import extract_signals, rules_are_conclusive
    from config import TASK_MODEL_DEFAULTS, get_llm, is_reasoning_model

    assert TASK_MODEL_DEFAULTS["advisory"].model == "gpt-6-luna" and TASK_MODEL_DEFAULTS["advisory"].reasoning_effort == "none"
    assert TASK_MODEL_DEFAULTS["critic"].model == "gpt-6-luna" and TASK_MODEL_DEFAULTS["judge"].model == "gpt-6.1-terra"
    assert TASK_MODEL_DEFAULTS["profiling"].model == "gpt-4o-mini"
    assert TASK_BY_RUN_NAME["advisory_llm"] == "advisory" and TASK_BY_RUN_NAME["evaluator_critic_llm"] == "critic"
    assert is_reasoning_model("gpt-6.1-terra") and not is_reasoning_model("gpt-4o-mini")
    if os.getenv("OPENAI_API_KEY"):  # client construction only, no request is sent
        reasoning = get_llm(model="gpt-6-luna", reasoning_effort="none")
        assert reasoning.temperature is None and reasoning.reasoning_effort == "none"
        assert get_llm(model="gpt-4o-mini", reasoning_effort="none").reasoning_effort is None

    single = ["I just want a basic account that never charges me a monthly fee.",
              "I want the highest possible net yield and interest on my savings of 50,000 MXN.",
              "I need a payroll account for my salary."]
    mixed = ["I need a payroll account to receive my salary, with low fees and some cashback perks.",
             "Which account is best for me?"]
    assert all(rules_are_conclusive(extract_signals(q)) for q in single)
    assert not any(rules_are_conclusive(extract_signals(q)) for q in mixed)
    # hybrid extraction: one profiling-model call extracts parameters + archetype; offline only the regex parser
    assert TASK_BY_RUN_NAME["profiling_extraction_llm"] == "profiling"
    state = _run(single[0], deposit_mxn=5_000.0)
    assert state["user_profile"]["profiling_source"] == "rules"
    assert any("Parsed signals (regex parser)" in line for line in state["trajectory_log"])


def test_research_finalists_and_incomplete_evaluation() -> None:
    """RECA retrieval only for finalists; unscored candidates never produce a false 'no match'."""
    import time

    from agents.advisory import EVALUATION_INCOMPLETE_WARNING, advisory_node
    from agents.common import NodeContext
    from agents.research import MAX_RAG_FINALISTS
    from evaluation.custom_evaluators import deterministic_faithfulness

    state = _run("I want an account with no monthly fees.", deposit_mxn=20_000.0)
    line = next(l for l in state["trajectory_log"] if "RAG finalists:" in l)
    finalists = int(line.split("RAG finalists: ")[1].split(" of ")[0])
    candidates = len(state["api_structured_data"])
    audited = _resources().rag_engine.product_specific_keys() & {r["catalog_key"] for r in state["api_structured_data"]}
    assert finalists <= MAX_RAG_FINALISTS + len(audited) and finalists < candidates, line
    with_evidence = {k for c in state["rag_retrieved_chunks"] for k in c["matched_product_keys"]}
    assert len(with_evidence) <= finalists
    base_key = state["draft_recommendation"]["base_option"]["product_key"]
    assert base_key in with_evidence, "the winner must be a finalist with RECA evidence"
    assert state["draft_recommendation"]["verified_facts"]["accounts_compared"] == candidates

    # Evaluator skipped (budget): candidates exist, no scored branches -> provisional baseline, not 'no match'.
    unscored = {**state, "tot_candidate_branches": [], "draft_recommendation": None}
    ctx = NodeContext(node_name="advisory_node", deadline_epoch=time.time() + 10.0)
    draft = advisory_node(unscored, ctx, _resources())["draft_recommendation"]
    assert draft["status"] == "abstained_provisional" and draft["base_option"] is not None
    assert draft["negotiation"] == [] and EVALUATION_INCOMPLETE_WARNING in draft["warnings"]
    assert "satisfies all of your hard constraints" not in " ".join(map(str, draft["narrative"].values()))
    assert draft["verified_facts"]["comparison_completed"] is False
    score, detail = deterministic_faithfulness({"draft_recommendation": draft, **unscored})
    assert score == 1.0, detail


def test_memory_index_matches_chroma_and_embedding_cache() -> None:
    """The in-memory retrieval path returns the same MMR chunks as querying ChromaDB directly."""
    res = _resources()
    engine = res.rag_engine
    query = "I want a digital payroll account; I keep about 6,000 MXN and want to avoid the monthly fee."
    variants = research_module.rag_query_variants(query)
    assert len(variants) == 8 and variants[0] == query
    assert research_module._rag_query(query, ["perks", "fee"]) == research_module._rag_query(query, ["fee", "perks"])

    engine.prefetch_query_embeddings(variants)
    started = time.perf_counter()
    vector = engine.embed_query(variants[1])
    vector_again = engine.embed_query(variants[1])
    assert vector == vector_again and time.perf_counter() - started < 5.0

    doc_types = research_module._document_types(["fee", "perks"])
    sample = res.catalog[::6] + [res.products_by_key[k] for k in sorted(engine.product_specific_keys())
                                 if k in res.products_by_key]
    for product in sample:
        kwargs = dict(institution_id=product.institution_id, product_id=product.product_id, document_types=doc_types,
                      k=research_module.MMR_K, fetch_k=research_module.MMR_FETCH_K, query_embedding=vector)
        try:
            engine.use_memory_index = False
            expected = engine.query_mmr(query, **kwargs)
        finally:
            engine.use_memory_index = True
        actual = engine.query_mmr(query, **kwargs)
        assert [c["chunk_id"] for c in actual] == [c["chunk_id"] for c in expected], product.catalog_key
        for a, e in zip(actual, expected):
            assert abs(a["distance"] - e["distance"]) < 1e-4 and a["metadata"] == e["metadata"], product.catalog_key


class _FakeExtractionLLM:
    """Offline stand-in for the profiling model: returns a scripted ProfileExtraction per message."""

    model_name = "fake-extractor"

    def __init__(self, scripts: Dict[str, Dict[str, Any]], fail: bool = False) -> None:
        self.scripts, self.fail, self.calls = scripts, fail, []

    def generate(self, schema: Any, system_prompt: str, payload: Dict[str, Any], timeout_s: float, run_name: str) -> Any:
        from agents.extraction import EXTRACTION_RUN_NAME
        from agents.llm import LLMUnavailableError

        if run_name != EXTRACTION_RUN_NAME:  # narrative / critic / summarizer: use their deterministic fallbacks
            raise LLMUnavailableError(f"fake extractor does not serve {run_name}")
        self.calls.append((run_name, payload["user_message"]))
        if self.fail:
            raise LLMUnavailableError("simulated outage")
        base = {"amounts": [], "constraints": [], "monthly_transfers": None, "monthly_transfers_evidence": None,
                "preferred_channel": None, "channel_evidence": None, "banks": [], "objectives": [],
                "business_account": False, "archetype": "Basic_Protected_User", "yield_importance": 0.2,
                "fee_importance": 0.6, "perks_importance": 0.2}
        return schema.model_validate({**base, **self.scripts.get(payload["user_message"], {})})


def test_hybrid_llm_parameter_extraction() -> None:
    """LLM extraction verified against the message (no invented figures); regex fallback; one call per turn."""
    from agents.extraction import EXTRACTION_KEY, ProfileExtraction, numbers_in, verify_extraction

    son = "I would like an account for my fifteen-year-old son, with an initial balance of 5,000 Mexican pesos."
    earn = "I earn 50000 mexican pesos a moth, I want a debit card with yields over 10% a month"
    fifteen = {"amounts": [{"amount_mxn": 5000, "kind": "initial_deposit", "evidence": "5,000 Mexican pesos"}],
               "constraints": [{"constraint": "minor_guardianship", "action": "require",
                                "evidence": "for my fifteen-year-old son"}]}
    scripts = {
        son: fifteen,
        earn: {"amounts": [{"amount_mxn": 50000, "kind": "income", "evidence": "I earn 50000 mexican pesos"}],
               "objectives": ["yield"], "archetype": "Strategic_Saver", "yield_importance": 0.8,
               "fee_importance": 0.1, "perks_importance": 0.1},
        "12k": {"amounts": [{"amount_mxn": 12000, "kind": "average_balance", "evidence": "12k"}]},
        "I have $50,000 MXN. I want zero fees.": {  # live gpt-4o-mini mislabels this as "other" ~50% of calls
            "amounts": [{"amount_mxn": 50000, "kind": "other", "evidence": "$50,000 MXN"}]},
    }
    institutions = sorted({p.institution_name for p in _resources().catalog})

    # verification: every item must quote the message; numbers must be written in the quote; topic anchors
    def verify(message: str, **fields: Any) -> Any:
        base = _FakeExtractionLLM({}).generate(ProfileExtraction, "", {"user_message": message}, 1.0,
                                               "profiling_extraction_llm")
        return verify_extraction(base.model_copy(update=fields), message, institutions)
    from agents.extraction import AmountMention, BankMention, ConstraintMention
    invented = verify("I want a zero fee account.",
                      amounts=[AmountMention(amount_mxn=20000, kind="average_balance", evidence="20,000 pesos")])
    assert invented.deposit_mxn is None and "not in the message" in invented.rejected[0]
    wrong_number = verify("I keep 5,000 pesos", amounts=[AmountMention(amount_mxn=50000, kind="average_balance",
                                                                        evidence="5,000 pesos")])
    assert wrong_number.deposit_mxn is None and "not written" in wrong_number.rejected[0]
    off_topic = verify("nothing from Santander please", constraints=[ConstraintMention(
        constraint="zero_fee", action="require", evidence="nothing from Santander please")],
        banks=[BankMention(bank="Santander", action="exclude", evidence="nothing from Santander"),
               BankMention(bank="Fake Bank", action="exclude", evidence="nothing")])
    assert off_topic.require == [] and off_topic.exclude_banks == ["SANTANDER"]
    assert any("does not mention its topic" in r for r in off_topic.rejected)
    assert any("not a catalog institution" in r for r in off_topic.rejected)
    assert numbers_in("cinco mil pesos") == [5000.0] and numbers_in("twenty-five thousand") == [25000.0]

    # the institutions list is not sent to the model: names as the user writes them are resolved in code
    from agents.extraction import _bank_index, resolve_institution
    index = _bank_index(institutions)
    for written, canonical in {"Banco Azteca": "AZTECA", "BanCoppel": "BANCOPPEL", "Bajío": "BAJIO",
                               "BX+": "VE POR MAS", "Banco Ve por Más": "VE POR MAS", "BBVA México": "BBVA MEXICO",
                               "Citibanamex": "BANAMEX", "Banco del Bienestar": "BaBien", "Bineo": "BANCO BINEO",
                               "Banco Sabadell": "SABADELL", "Grupo Financiero Inbursa": "INBURSA"}.items():
        assert resolve_institution(written, index) == canonical, written
    assert resolve_institution("Nubank", index) is None and resolve_institution("my bank", index) is None
    # deterministic rationale: only the user's verified words
    assert '"nothing from Santander"' in off_topic.rationale  # the verified bank quote
    assert "nothing from Santander please" not in off_topic.rationale  # the rejected zero_fee quote is not cited

    # squad (benchmark / MCP path): ONE extraction call replaces the archetype call; income is not a deposit
    llm = _FakeExtractionLLM(scripts)
    resources = dataclasses.replace(_resources(), llm=llm)
    state = _run(son, resources)
    profile = state["user_profile"]
    assert profile["deposit_mxn"] == 5000.0 and profile["deposit_stated"] is True
    assert profile["hard_constraints"] == ["minor_guardianship"] and profile["profiling_source"] == "llm"
    assert any("LLM extraction verified against the message" in line for line in state["trajectory_log"])
    assert [c[0] for c in llm.calls] == ["profiling_extraction_llm"]  # one call, no separate archetype call
    income = _run(earn, resources)
    assert income["user_profile"]["deposit_stated"] is False and income["archetype_prior"] == "Strategic_Saver"
    other = _run("I have $50,000 MXN. I want zero fees.", resources)  # "other" is ambiguity: regex balance stands
    assert other["user_profile"]["deposit_mxn"] == 50_000.0 and other["user_profile"]["deposit_stated"] is True
    other_memory = ConversationMemory(catalog=_resources().catalog)
    other_memory.plan_turn("I have $50,000 MXN. I want zero fees.", llm=llm)
    assert other_memory.profile.deposit_mxn == 50_000.0

    # conversation memory (UI / MCP path): extraction before the graph, forwarded so the squad does not call again
    memory = ConversationMemory(catalog=_resources().catalog)
    first = memory.plan_turn(earn, llm=llm)
    assert first.action == "clarify" and memory.profile.deposit_mxn is None  # the MXN 50,000 is income
    second = memory.plan_turn("12k", llm=llm)
    assert second.action == "run" and memory.profile.deposit_mxn == 12000.0
    assert second.run_metadata[EXTRACTION_KEY]["archetype"] == "Strategic_Saver"  # the original request's profile
    calls = len(llm.calls)
    run = run_squad(second.query, resources, build_graph(resources), run_metadata=second.run_metadata)
    assert len(llm.calls) == calls, "the Profiling Agent must reuse the memory's extraction"
    assert run["user_profile"]["deposit_mxn"] == 12000.0 and run["archetype_prior"] == "Strategic_Saver"

    # failure -> regex fallback (same behaviour as the offline mode)
    down = dataclasses.replace(_resources(), llm=_FakeExtractionLLM(scripts, fail=True))
    fallback = _run(son, down)
    assert fallback["user_profile"]["profiling_source"] == "rules"
    assert fallback["user_profile"]["hard_constraints"] == ["minor_guardianship"]  # regex still understands it
    assert any("regex parser only" in line for line in fallback["trajectory_log"])


def test_mcp_resources_and_prompts() -> None:
    """MCP lab primitives over the protocol (in-memory client): static resource, resource template, prompt."""
    import asyncio
    import json

    import mcp_server
    from mcp import Client

    async def drive() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        async with Client(mcp_server.mcp) as client:
            out["tools"] = [t.name for t in (await client.list_tools()).tools]
            out["resources"] = [(str(r.uri), r.mime_type) for r in (await client.list_resources()).resources]
            out["templates"] = [(t.uri_template, t.mime_type)
                                for t in (await client.list_resource_templates()).resource_templates]
            started = time.perf_counter()
            catalog = await client.read_resource("catalog://products")
            out["catalog_s"] = time.perf_counter() - started
            out["catalog"] = json.loads(catalog.contents[0].text)
            out["now"] = json.loads((await client.read_resource("product://40059_NW0001")).contents[0].text)
            out["plus"] = json.loads((await client.read_resource("product://40113_BX+_Beneficios")).contents[0].text)
            try:
                await client.read_resource("product://nope_1")
                out["missing"] = "read succeeded"
            except Exception as exc:  # noqa: BLE001 - the protocol error surfaces as an exception
                out["missing"] = str(exc)
            out["prompts"] = [(p.name, p.title, [(a.name, a.required) for a in p.arguments or []])
                              for p in (await client.list_prompts()).prompts]
            prompt = await client.get_prompt("compare_accounts", {"deposit_mxn": "5000", "priorities": "high yield"})
            out["prompt"] = [(m.role, m.content.text) for m in prompt.messages]
        return out

    mcp_server._catalog.cache_clear()
    out = asyncio.run(drive())
    assert {"search_products", "compute_net_impact", "recommend_account", "send_message"} <= set(out["tools"])
    assert out["resources"] == [("catalog://products", "application/json")]
    assert out["templates"] == [("product://{product_key}", "application/json")]
    catalog_keys = {p.catalog_key for p in _resources().catalog}
    assert {p["product_key"] for p in out["catalog"]} == catalog_keys and len(out["catalog"]) == len(catalog_keys)
    assert out["catalog_s"] < 3.0, out["catalog_s"]  # catalog-only read: never starts the squad / ChromaDB
    assert out["now"]["product_name"] == "NOW" and out["now"]["annual_yield_rate"] == 7.25
    assert out["now"]["yield_tiers"] and "monthly_fees_detail" in out["now"]  # full record, not the summary
    assert out["plus"]["product_id"] == "BX+_Beneficios"  # keys with '+' resolve through the URI template
    assert "Unknown product_key 'nope_1'" in out["missing"], out["missing"]  # protocol error, not a payload
    assert out["prompts"] == [("compare_accounts", "Compare debit accounts", [("deposit_mxn", True), ("priorities", True)])]
    role, text = out["prompt"][0]
    assert role == "user" and "MXN 5,000.00" in text and "recommend_account" in text and "deposit_mxn=5000.00" in text
    assert "Never estimate fees" in text


def test_mcp_conversation_tools() -> None:
    import mcp_server

    mcp_server._resources.cache_clear()
    mcp_server._graph.cache_clear()
    mcp_server._resources = lambda: _resources()  # type: ignore[assignment]
    mcp_server._graph = lambda: _graph()  # type: ignore[assignment]
    session = mcp_server.start_conversation()["session_id"]
    first = mcp_server.send_message(session, "I want a digital account with no fees.")
    assert first["action"] == "clarify"
    second = mcp_server.send_message(session, "15,000 MXN")
    assert second["action"] == "run" and second["base_option"] is not None
    third = mcp_server.send_message(session, "Nothing from Banamex.")
    assert third["state"]["profile"]["excluded_institutions"] == ["BANAMEX"]
    assert (third["base_option"] or {}).get("institution_name") != "BANAMEX"
    assert mcp_server.get_conversation_state(session)["profile"]["deposit_mxn"] == 15_000.0
    assert mcp_server.end_conversation(session)["deleted"] is True


def test_confidence_separation_ignores_interchangeable_ties() -> None:
    """A near-tie between two safe accounts is not uncertainty; a tie with a flagged / failing runner-up is."""
    from types import SimpleNamespace

    from agents.evaluator import separation_term

    evaluation = SimpleNamespace(flagged={"STALE_1"})
    safe = lambda key, score, met=True: {"product_key": key, "score": score, "hard_constraints_met": met}  # noqa: E731
    assert separation_term([safe("A", 80.0)], evaluation) == 1.0  # no runner-up
    assert separation_term([safe("A", 80.0), safe("B", 79.0)], evaluation) == 1.0  # both safe -> interchangeable
    assert separation_term([safe("A", 80.0), safe("STALE_1", 79.0)], evaluation) == 0.1  # stale runner-up
    assert separation_term([safe("A", 80.0), safe("B", 76.0, met=False)], evaluation) == 0.4  # runner-up fails
    assert separation_term([safe("A", 95.0), safe("B", 70.0, met=False)], evaluation) == 1.0  # clear gap


def test_net_cost_penalty_by_archetype() -> None:
    """A net annual loss pushes the account down by archetype; fees the yield pays for are not penalised."""
    from agents.evaluator import NET_COST_MAX_PENALTY, composite_score, net_cost_penalty
    from agents.profiling import ARCHETYPE_PRIORS

    metrics = lambda cost: {"yield_metric": 0.0, "fee_metric": 0.9, "perks_metric": 0.7,  # noqa: E731
                            "net_cost_mxn": cost}
    for archetype, prior in ARCHETYPE_PRIORS.items():
        assert net_cost_penalty(prior, metrics(0.0)) == 0.0, archetype
        assert net_cost_penalty(prior, metrics(300.0)) == NET_COST_MAX_PENALTY[archetype] / 2, archetype
        assert net_cost_penalty(prior, metrics(5_000.0)) == NET_COST_MAX_PENALTY[archetype], archetype  # capped
    basic, lifestyle = ARCHETYPE_PRIORS["Basic_Protected_User"], ARCHETYPE_PRIORS["Lifestyle_Maximizer"]
    assert composite_score(basic, metrics(240.0)) == 62.5  # 72.5 - 25 * 240/600
    assert composite_score(lifestyle, metrics(240.0)) == round(100 * (0.30 * 0.9 + 0.55 * 0.7) - 2.0, 2)

    # The minor-account case: a zero-cost account now outranks one that costs MXN 240 a year.
    result = run_squad("I would like an account for my fifteen-year-old son, with an initial balance of 5,000 "
                       "Mexican pesos.", _resources(), compiled_graph=_R["g"])
    scores = {b["product_key"]: b["score"] for b in result["tot_candidate_branches"]}
    assert scores["40137_2500"] == 62.5 and scores["40012_0201"] == 76.0, scores  # BanCoppel Jóvenes vs BBVA Link Card
    assert scores["40036_1005"] > scores["40137_2500"], scores  # Inbursa Educa (MXN 0) above the MXN 240 account


def test_catalog_benefit_perks() -> None:
    """Translated benefit perks are in the catalog once, without restating what it already had."""
    catalog = {p.catalog_key: p.qualitative_perks_summary for p in _resources().catalog}
    assert "1% cashback on purchases with the digital card" in catalog["40002_SwitchBanamex"]
    assert "rebate of up to MXN 25 per withdrawal at other banks' ATMs" in catalog["40138_ABCCA04"]
    assert "Mi Monedero Chedraui" in catalog["40140_ConsucuentaN2"]
    assert catalog["40127_117"].endswith("rebates, annual yield up to 0.01%.")  # yield clause stays last
    link = catalog["40012_0201"]
    assert "choice of card design" in link and "no account management fee" not in link  # fee already stated
    assert "interest" not in catalog["40156_20"].lower()  # 'Pago de intereses' without a rate is not a perk
    assert "Sin Comentarios" not in " ".join(catalog.values())


def test_top_score_explanation() -> None:
    """The #1 by match score that is neither option gets an explicit, deterministic reason."""
    from agents.advisory import select_options, top_score_explanation

    def branch(key: str, rank: int, score: float, fees: float, net: float) -> Dict[str, Any]:
        return {"product_key": key, "institution_name": "BANK", "product_name": key, "score": score, "final": True,
                "final_rank": rank, "hard_constraints_met": True, "hard_constraint_checks": {},
                "net_impact": {"deposit_evaluated_mxn": 30_000.0, "applicable_yield_rate_percent": 0.0,
                               "gross_annual_interest_mxn": max(net, 0.0), "effective_monthly_fee_mxn": fees / 12,
                               "fee_waived_by_balance": False, "total_annual_fees_mxn": fees,
                               "net_annual_impact_mxn": net, "breakdown": {}}}

    def explain(*branches: Dict[str, Any]) -> Any:
        state = {"tot_candidate_branches": list(branches), "api_structured_data": []}
        return top_score_explanation(state, *select_options(state))

    # perks profile: the top account charges fees and loses money -> neither Base nor Trade-off
    lost = explain(branch("PERKS", 1, 95.0, 4_080.0, -4_080.0), branch("FREE", 2, 70.0, 0.0, 3.0),
                   branch("YIELD", 3, 15.0, 0.0, 2_841.0))
    assert lost["product_key"] == "PERKS" and "MXN 4,080.00 in fees a year" in lost["text"], lost
    assert "costs you money every year" in lost["text"] and "Base Option's MXN 3.00" in lost["text"], lost
    # top account has fees but earns: below the Trade-off's net -> says so
    below = explain(branch("FEE", 1, 90.0, 600.0, 900.0), branch("FREE", 2, 70.0, 0.0, 3.0),
                    branch("YIELD", 3, 60.0, 0.0, 2_841.0))
    assert "below the MXN 2,841.00 of the Positive Trade-off" in below["text"], below
    # the top account IS the Base Option -> nothing to explain
    assert explain(branch("FREE", 1, 79.0, 0.0, 0.5), branch("FEE", 2, 72.5, 240.0, -240.0)) is None


TESTS: List[Callable[[], None]] = [
    test_memory_deltas, test_memory_clarification_and_window, test_overconfidence_markers_trigger_rewrite,
    test_internal_identifiers_trigger_rewrite,
    test_benefit_conflict_is_advisory, test_fallback_mode_when_structured_source_fails,
    test_hard_timeout_returns_cached_baseline, test_tot_stale_downweight, test_tot_tie_resolution,
    test_intent_router, test_criteria_negotiation, test_atomic_benefit_chunks, test_hybrid_llm_parameter_extraction,
    test_mcp_resources_and_prompts,
    test_mcp_conversation_tools,
    test_business_accounts_excluded_by_default, test_catalog_yield_tiers,
    test_task_model_routing_and_profiling_skip, test_research_finalists_and_incomplete_evaluation,
    test_memory_index_matches_chroma_and_embedding_cache,
    test_minor_account_flags_profiling_and_filtering, test_yield_objective_lists_higher_balance_accounts,
    test_document_figures_and_reco_fee_detector, test_confidence_separation_ignores_interchangeable_ties,
    test_net_cost_penalty_by_archetype, test_catalog_benefit_perks, test_top_score_explanation,
]


def main() -> int:
    failures = 0
    for test in TESTS:
        started = time.perf_counter()
        try:
            test()
            print(f"PASS  {test.__name__}  ({time.perf_counter() - started:.1f}s)")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} doc-alignment tests passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
