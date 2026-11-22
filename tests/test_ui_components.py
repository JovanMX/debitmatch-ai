"""Synthetic UI tests for app.py using streamlit.testing.v1.AppTest.

Runs the real Streamlit script headlessly in deterministic mode (no LLM calls)
and checks that session state initialises, callbacks run and every section
renders without exceptions.

Usage::

    .venv\\Scripts\\python.exe tests\\test_ui_components.py
    .venv\\Scripts\\python.exe -m pytest tests\\test_ui_components.py   (if pytest is installed)
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # tests/ sits one level below the project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tempfile  # noqa: E402

os.environ["DEBITMATCH_UI_OFFLINE"] = "1"  # deterministic mode: never call OpenAI from UI tests
QUEUE_DIR = Path(tempfile.mkdtemp(prefix="debitmatch_queue_"))
os.environ["DEBITMATCH_REVIEW_QUEUE_DIR"] = str(QUEUE_DIR)  # never touch the real review queue

from streamlit.testing.v1 import AppTest  # noqa: E402

import app  # noqa: E402  (pure helpers only; the UI does not render on import)
from state import PREFERENCE_KEYS  # noqa: E402

APP_PATH = str(PROJECT_ROOT / "app.py")
CONSOLE_PATH = str(PROJECT_ROOT / "pages" / "1_Reviewer_Console.py")
APP_TIMEOUT_S = 240  # first run loads the embedding model and ChromaDB


def _new_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT_S)
    at.run()
    _assert_clean(at)
    return at


def _assert_clean(at: AppTest) -> None:
    assert not at.exception, f"Streamlit raised: {[e.value for e in at.exception]}"


def _texts(elements: Any) -> List[str]:
    return [str(e.value) for e in elements]


def _send(at: AppTest, message: str) -> None:
    at.text_area(key="user_query_input").input(message)
    at.button(key="analyze_btn").click().run()
    _assert_clean(at)


def _manual(at: AppTest, *groups: str) -> None:
    """Switch sidebar groups to manual (authoritative) input."""
    for group in groups:
        at.sidebar.toggle(key=app.MANUAL_TOGGLES[group][0]).set_value(True).run()
        _assert_clean(at)


def _run_query(at: AppTest, query: str, deposit: float = 20_000.0) -> Dict[str, Any]:
    _send(at, query)
    memory = at.session_state["conversation"]
    if memory is not None and memory.awaiting == "deposit_mxn":  # pre-generation clarification
        field = at.sidebar.number_input(key="deposit_amount_mxn")
        assert not field.disabled, "the clarification must unlock the deposit field"
        if field.value is None:
            field.set_value(deposit).run()
            _assert_clean(at)
        at.button(key="clarify_use_sidebar").click().run()
        _assert_clean(at)
    assert at.session_state["is_evaluating"] is False
    result = at.session_state["squad_state"]
    assert result is not None and result.get("draft_recommendation"), "squad_state was not populated"
    return result


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_rebalance_weights_keeps_sum_one() -> None:
    current = {"yield_importance": 0.30, "fee_importance": 0.50, "perks_importance": 0.20}
    out = app.rebalance_weights("yield_importance", 0.80, current)
    assert out["yield_importance"] == 0.80
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert out["fee_importance"] > out["perks_importance"]  # proportional split (0.5 : 0.2)

    edge = app.rebalance_weights("fee_importance", 1.0, current)
    assert edge == {"fee_importance": 1.0, "yield_importance": 0.0, "perks_importance": 0.0}

    zeros = app.rebalance_weights("perks_importance", 0.4, {"yield_importance": 0, "fee_importance": 0,
                                                            "perks_importance": 1})
    assert zeros["yield_importance"] == zeros["fee_importance"] == 0.3

    for value in (0.0, 0.01, 0.33, 0.67, 0.99):
        out = app.rebalance_weights("perks_importance", value, {"yield_importance": 0.33, "fee_importance": 0.34,
                                                               "perks_importance": 0.33})
        assert abs(sum(out.values()) - 1.0) < 1e-9 and min(out.values()) >= 0


def test_session_state_contract_on_plain_dict() -> None:
    ss: Dict[str, Any] = {}
    app.init_session_state(ss)
    for key in ("squad_state", "is_evaluating", "execution_history", "hitl_status", "normalized_weights"):
        assert key in ss, key
    assert ss["is_evaluating"] is False and ss["execution_history"] == []
    assert set(ss["hitl_status"]) >= {"pending", "approved_by_human", "advisor_notes"}
    assert set(ss["normalized_weights"]) == set(PREFERENCE_KEYS)
    assert abs(sum(ss["normalized_weights"].values()) - 1.0) < 1e-9
    ss["execution_history"].append({"run_id": "x"})
    app.init_session_state(ss)  # idempotent: does not reset existing values
    assert ss["execution_history"] == [{"run_id": "x"}]


def test_error_collection_and_hitl_labels() -> None:
    errors = app.collect_run_errors({"execution_metadata": {
        "timed_out": True, "within_budget": False, "elapsed_s": 12.1, "time_budget_s": 10,
        "rag_error": "InternalError: collection missing", "error.research_node": "ValueError: boom"}})
    assert any(e.startswith("Timeout") for e in errors)
    assert any("ChromaDB" in e for e in errors)
    assert any("research_node" in e for e in errors)
    assert app.collect_run_errors({"execution_metadata": {"within_budget": True}}) == []

    label, correction = app.hitl_reason_details("high financial amount (MXN 150,000.00 > MXN 100,000.00)")
    assert "100,000" in label and correction
    assert app.hitl_reason_details("low confidence (0.55 < 0.70)")[0].startswith("Low confidence")
    assert app.hitl_reason_details("unresolved regulatory conflict between ...")[0] == "API/RAG Policy Conflict"


def test_execute_squad_captures_chromadb_error() -> None:
    from chromadb.errors import ChromaError

    class _BrokenGraph:
        def invoke(self, state: Any, config: Any = None) -> Any:
            raise ChromaError("simulated ChromaDB outage")

    class _Resources:
        time_budget_s = 10.0
        llm = None

        @staticmethod
        def effective_reference_date():
            from datetime import date
            return date.today()

    result, errors = app.execute_squad("zero fee account", app.build_profile_overrides(1000, 2, [], None),
                                       _Resources(), _BrokenGraph(), "test")
    assert result is None
    assert errors and errors[0].startswith("ChromaDB error"), errors


# --------------------------------------------------------------------------- #
# AppTest - rendered application
# --------------------------------------------------------------------------- #
def test_initial_render_and_session_state() -> None:
    at = _new_app()
    ss = at.session_state
    assert ss["squad_state"] is None and ss["is_evaluating"] is False
    assert ss["execution_history"] == []
    assert ss["hitl_status"]["pending"] is False and ss["hitl_status"]["approved_by_human"] is False
    assert abs(sum(ss["normalized_weights"].values()) - 1.0) < 1e-9
    assert ss["deposit_amount_mxn"] is None and ss["monthly_spei_transfers"] is None  # nothing assumed silently
    assert any("DebitMatch AI" in t for t in _texts(at.title))
    assert any(app.FOOTER_DISCLAIMER in t for t in _texts(at.caption)), "footer disclaimer missing"
    assert len(at.sidebar.slider) == 3 and len(at.sidebar.checkbox) == len(app.HARD_CONSTRAINT_OPTIONS) == 7
    # every group starts locked (inferred by the agent)
    assert [t.value for t in at.sidebar.toggle][:3] == [False, False, False]
    assert all(s.disabled for s in at.sidebar.slider) and all(c.disabled for c in at.sidebar.checkbox)
    assert at.sidebar.number_input(key="deposit_amount_mxn").disabled
    assert at.sidebar.selectbox(key="preferred_channel_choice").disabled
    assert any("will infer these" in t for t in _texts(at.sidebar.caption))
    labels = [b.label for b in at.button]
    assert "🚀 Analyze Banking Options" in labels


def test_weight_slider_autonormalization_and_archetype() -> None:
    at = _new_app()
    _manual(at, "weights", "constraints")
    assert not at.sidebar.slider(key="w_yield").disabled
    at.sidebar.slider(key="w_yield").set_value(0.80).run()
    _assert_clean(at)
    sliders = {k: at.sidebar.slider(key=w).value for k, (w, _, _) in app.WEIGHT_WIDGETS.items()}
    assert abs(sum(sliders.values()) - 1.0) < 1e-9, sliders
    weights = at.session_state["normalized_weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9 and weights["yield_importance"] == 0.8
    assert any("Strategic Saver" in t for t in _texts(at.sidebar.info)), _texts(at.sidebar.info)

    at.sidebar.checkbox(key="hc_payroll").check().run()
    _assert_clean(at)
    assert any("Flow Manager" in t for t in _texts(at.sidebar.info))


def test_empty_query_is_rejected() -> None:
    at = _new_app()
    at.button(key="analyze_btn").click().run()
    _assert_clean(at)
    assert at.session_state["squad_state"] is None
    assert any("describe your banking needs" in t for t in _texts(at.error))


def test_simple_query_end_to_end() -> None:
    at = _new_app()
    _manual(at, "financial", "constraints")  # seeds the empty financial fields (20,000 MXN, 4 transfers)
    at.sidebar.checkbox(key="hc_zero_fee").check()
    at.sidebar.checkbox(key="hc_digital_opening").check()
    result = _run_query(at, "I want a digital debit account with zero monthly fee.")
    meta = result["execution_metadata"]
    assert meta["route"] == "evaluator_linear", meta.get("route")
    assert meta["within_budget"] is True
    profile = result["user_profile"]
    assert profile["deposit_mxn"] == 20_000.0 and profile["monthly_transfers"] == 4  # sidebar wins
    assert {"zero_fee", "digital_opening"} <= set(profile["hard_constraints"])
    assert result["draft_recommendation"]["base_option"]["hard_constraints_met"] is True

    history = at.session_state["execution_history"]
    assert len(history) == 1 and history[0]["route"] == "evaluator_linear"
    assert any("Data Verified" in t or "Preliminary Data" in t for t in _texts(at.markdown))
    expanders = [e.label for e in at.expander]
    assert "🔍 View Agent Execution Trajectory & ToT Reasoning Log" in expanders
    assert len(at.metric) >= 8  # inspector KPIs + option cards
    assert not at.error, _texts(at.error)


def test_hybrid_query_renders_tot_table() -> None:
    at = _new_app()
    _manual(at, "constraints")
    at.sidebar.checkbox(key="hc_free_insurance").check()
    result = _run_query(at, "I want net yield, cashback perks and low fees for my savings.")
    meta = result["execution_metadata"]
    assert meta["route"] == "evaluator_tot_beam", meta.get("route_reason")
    finals = [b for b in result["tot_candidate_branches"] if b.get("final")]
    assert 1 <= len(finals) <= 5
    assert all(b["hard_constraint_checks"].get("free_insurance") for b in finals)
    assert any("Tree-of-Thought Beam Search" in t for t in _texts(at.markdown))
    assert any("early stop" in t.lower() for t in _texts(at.info) + _texts(at.success))
    assert len(at.session_state["execution_history"]) == 1


def test_hitl_escalation_and_human_approval() -> None:
    at = _new_app()
    _manual(at, "financial")
    at.number_input(key="deposit_amount_mxn").set_value(150_000.0)
    result = _run_query(at, "I want a zero fee account for my savings.")
    assert result["requires_human_review"] is True
    reasons = result["draft_recommendation"]["human_review_reasons"]
    assert any("high financial amount" in r for r in reasons), reasons
    assert at.session_state["hitl_status"]["pending"] is True
    assert any("Human-in-the-Loop review queue" in t for t in _texts(at.error))
    assert any("exceeds $100,000 MXN" in t for t in _texts(at.markdown))

    case_id = at.session_state["hitl_status"]["case_id"]
    assert case_id, "escalated case was not persisted to the review queue"
    from hitl_queue import ReviewQueue
    assert ReviewQueue().get(case_id)["status"] == "pending"

    at.text_input(key="advisor_notes_input").input("Verified IPAB coverage with the client.")
    at.button(key="approve_btn").click().run()
    _assert_clean(at)
    hitl = at.session_state["hitl_status"]
    assert hitl["approved_by_human"] is True and hitl["pending"] is False and hitl["decision"] == "approved"
    assert hitl["advisor_notes"] == "Verified IPAB coverage with the client."
    assert at.session_state["execution_history"][-1]["human_approved"] is True
    assert any("Approved by human advisor" in t for t in _texts(at.success))
    assert ReviewQueue().get(case_id)["status"] == "approved"
    golden = (QUEUE_DIR / "golden_dataset.jsonl").read_text(encoding="utf-8")
    assert case_id in golden and '"human_decision": "approved"' in golden


def test_hitl_escalate_then_modify() -> None:
    from hitl_queue import ReviewQueue

    at = _new_app()
    _manual(at, "financial")
    at.number_input(key="deposit_amount_mxn").set_value(150_000.0)
    result = _run_query(at, "I want a zero fee account for my savings.")
    case_id = at.session_state["hitl_status"]["case_id"]

    at.button(key="escalate_btn").click().run()
    _assert_clean(at)
    hitl = at.session_state["hitl_status"]
    assert hitl["decision"] == "escalated" and hitl["pending"] is True and hitl["approved_by_human"] is False
    assert ReviewQueue().get(case_id)["status"] == "escalated"
    assert any("Escalated to a senior reviewer" in t for t in _texts(at.warning))

    base_key = result["draft_recommendation"]["base_option"]["product_key"]
    alternative = next(o for o in at.selectbox(key="hitl_override_choice").options)
    at.selectbox(key="hitl_override_choice").set_value(
        next(b["product_key"] for b in app.final_branches(result) if b["product_key"] != base_key))
    at.button(key="modify_btn").click().run()
    _assert_clean(at)
    hitl = at.session_state["hitl_status"]
    assert hitl["decision"] == "modified" and hitl["approved_by_human"] is True
    assert hitl["override_product_key"] and hitl["override_product_key"] != base_key
    assert alternative  # options rendered with human-readable labels
    assert ReviewQueue().get(case_id)["status"] == "modified"
    assert any("Base Option replaced by" in t for t in _texts(at.success))


def test_review_queue_redacts_pii() -> None:
    from hitl_queue import ReviewQueue, redact_pii

    text = ("Email me at ana.lopez@example.com or call 55 1234 5678; CURP GOLA850101MDFRPN09, "
            "card 4152 3134 5678 9012. I have $50,000 MXN.")
    redacted = redact_pii(text)
    for secret in ("ana.lopez@example.com", "1234 5678", "GOLA850101MDFRPN09", "4152 3134 5678 9012"):
        assert secret not in redacted, (secret, redacted)
    assert "$50,000 MXN" in redacted

    queue = ReviewQueue(Path(tempfile.mkdtemp(prefix="dm_q_")))
    card = {"case_id": "c1", "query_redacted": redacted, "machine_recommendation": {"base_option": {"product_key": "A"}}}
    queue.enqueue(card)
    assert [c["status"] for c in queue.pending()] == ["pending"]
    queue.record_decision("c1", "escalated", "call me at 55 9876 5432")
    assert queue.get("c1")["status"] == "escalated" and len(queue.pending()) == 1
    assert "9876" not in queue.get("c1")["decisions"][-1]["advisor_notes"]
    queue.record_decision("c1", "modified", "", override_product_key="B")
    assert queue.pending() == [] and "B" in queue.golden_path.read_text(encoding="utf-8")


def test_spei_channel_override_and_live_progress() -> None:
    at = _new_app()
    _manual(at, "financial")
    at.sidebar.selectbox(key="preferred_channel_choice").set_value(app.CHANNEL_OPTIONS[3])  # Branch / Teller
    result = _run_query(at, "I want a zero fee account.")
    assert result["user_profile"]["preferred_channel"] == app.CHANNEL_OPTIONS[3]  # sidebar channel applied
    result = _run_query(at, "From now on I do my transfers from the app.")  # later change wins
    assert result["user_profile"]["preferred_channel"] == app.CHANNEL_OPTIONS[1], result["user_profile"]
    assert at.sidebar.selectbox(key="preferred_channel_choice").value == app.CHANNEL_OPTIONS[1]  # sidebar synced
    events = at.session_state["progress_events"]
    assert events[:4] == ["intent_router_node", "profiling_node", "research_node", "supervisor_node"], events
    assert events[-1] == "review_node"
    assert any("Live execution order" in t for t in _texts(at.caption))


def test_llm_feature_claims_are_verified_deterministically() -> None:
    from guardrails import verify_feature_claims

    state = {"api_structured_data": [{"institution_id": "40002", "product_id": "SwitchBanamex",
                                      "has_free_insurance": False, "has_cashback_or_perks": True,
                                      "is_digital_account": True}]}
    option = {"product_key": "40002_SwitchBanamex", "institution_name": "BANAMEX", "product_name": "Switch Banamex"}
    draft = {"base_option": option, "tradeoff_option": None,
             "narrative": {"summary": "Switch Banamex protects you against fraud at no cost and opens online.",
                           "key_tradeoffs": []}}
    wrong = {"product_name": "Switch Banamex", "feature": "free_insurance", "asserted_available": True,
             "evidence_quote": "protects you against fraud at no cost"}
    right = {"product_name": "Base Option", "feature": "digital_opening", "asserted_available": True,
             "evidence_quote": "opens online"}
    ungrounded = dict(wrong, evidence_quote="text that is not in the narrative")
    issues = verify_feature_claims([wrong, right, ungrounded], draft, state)
    assert len(issues) == 1 and "free_insurance" in issues[0], issues


def _escalated_case(at: AppTest) -> str:
    _manual(at, "financial")
    at.number_input(key="deposit_amount_mxn").set_value(150_000.0)
    _run_query(at, "I want a zero fee account for my savings.")
    case_id = at.session_state["hitl_status"]["case_id"]
    assert case_id
    return case_id


def _console() -> AppTest:
    console = AppTest.from_file(CONSOLE_PATH, default_timeout=APP_TIMEOUT_S)
    console.run()
    _assert_clean(console)
    return console


def test_reviewer_console_decides_case_from_another_session() -> None:
    from hitl_queue import ReviewQueue

    user_session = _new_app()
    case_id = _escalated_case(user_session)

    console = _console()  # separate session (different browser / reviewer)
    assert any(case_id in label for label in console.selectbox(key="rc_case").options)  # formatted labels
    console.selectbox(key="rc_case").set_value(case_id).run()
    assert any("Bounded Decision Card" in t for t in _texts(console.markdown))
    assert any("exceeds $100,000 MXN" in t for t in _texts(console.markdown))
    console.text_input(key="rc_reviewer").input("reviewer-07")
    console.text_input(key="rc_notes").input("Suitability confirmed by phone 55 1234 5678.")
    console.button(key="rc_approve").click().run()
    _assert_clean(console)
    assert any("decision 'approved' recorded" in t for t in _texts(console.success))

    case = ReviewQueue().get(case_id)
    last = case["decisions"][-1]
    assert case["status"] == "approved" and last["source"] == "reviewer_console"
    assert last["reviewer"] == "reviewer-07" and "1234" not in last["advisor_notes"]  # PII redacted

    user_session.run()  # the user's session picks the decision up on its next rerun
    _assert_clean(user_session)
    hitl = user_session.session_state["hitl_status"]
    assert hitl["approved_by_human"] is True and hitl["decided_via"] == "reviewer_console"
    assert any("via Reviewer Console" in t for t in _texts(user_session.success))
    assert user_session.button(key="approve_btn").disabled


def test_reviewer_console_modify_and_closed_case_guard() -> None:
    from hitl_queue import CaseClosedError, ReviewQueue

    case_id = _escalated_case(_new_app())
    console = _console()
    console.selectbox(key="rc_case").set_value(case_id).run()
    console.button(key="rc_escalate").click().run()
    assert ReviewQueue().get(case_id)["status"] == "escalated"

    assert console.selectbox(key="rc_override").options, "no replacement candidates on the decision card"
    card = ReviewQueue().get(case_id)["card"]
    base_key = card["machine_recommendation"]["base_option"]["product_key"]
    replacement = next(o["product_key"] for o in card["candidate_options"] if o["product_key"] != base_key)
    console.selectbox(key="rc_override").set_value(replacement)
    console.button(key="rc_modify").click().run()
    _assert_clean(console)
    case = ReviewQueue().get(case_id)
    assert case["status"] == "modified" and case["decisions"][-1]["override_product_key"] == replacement
    golden = (QUEUE_DIR / "golden_dataset.jsonl").read_text(encoding="utf-8")
    assert f'"human_base_option": "{replacement}"' in golden

    try:  # a second reviewer cannot overwrite a closed case
        ReviewQueue().record_decision(case_id, "approved", source="reviewer_console")
        raise AssertionError("closed case accepted a second decision")
    except CaseClosedError:
        pass


def test_reviewer_console_empty_filter() -> None:
    console = _console()
    console.multiselect(key="rc_status_filter").set_value([]).run()
    _assert_clean(console)
    console.multiselect(key="rc_status_filter").set_value(["approved"]).run()
    _assert_clean(console)
    assert all("approved" in label for label in console.selectbox(key="rc_case").options) or \
        any("No cases match" in t for t in _texts(console.info))


def test_multi_turn_refinement_updates_profile_and_sidebar() -> None:
    at = _new_app()
    first = _run_query(at, "I want a digital payroll account with no fees.")
    assert {"payroll", "digital_opening", "zero_fee"} <= set(first["user_profile"]["hard_constraints"])
    assert at.sidebar.checkbox(key="hc_payroll").value is True  # conversation -> sidebar sync

    second = _run_query(at, "Forget the payroll requirement.")
    assert "payroll" not in second["user_profile"]["hard_constraints"], second["user_profile"]
    assert {"digital_opening", "zero_fee"} <= set(second["user_profile"]["hard_constraints"])
    assert at.sidebar.checkbox(key="hc_payroll").value is False

    third = _run_query(at, "Nothing from Banamex, please.")
    base = third["draft_recommendation"]["base_option"]
    assert third["user_profile"]["excluded_institutions"] == ["BANAMEX"]
    assert base is None or base["institution_name"] != "BANAMEX", base
    assert at.sidebar.multiselect(key="excluded_banks").value == ["BANAMEX"]

    memory = at.session_state["conversation"]
    assert [t.role for t in memory.turns].count("user") >= 3
    assert any("removed constraint" in c for t in memory.turns for c in t.changes)
    assert "[Conversation context]" in third["user_query"]  # earlier turns travel as compact context
    assert len(at.chat_message) >= 6
    assert [h["turn"] for h in at.session_state["execution_history"]][-1] >= 3

    # "New conversation" after the results resets like the one next to the query box
    at.button(key="new_conversation_bottom_btn").click().run()
    _assert_clean(at)
    assert at.session_state["conversation"] is None and at.session_state["squad_state"] is None
    assert not any(b.key == "new_conversation_bottom_btn" for b in at.button)  # no results, no bottom button
    assert at.session_state["execution_history"]  # the run history is kept


def test_clarification_asks_for_missing_deposit() -> None:
    at = _new_app()
    _send(at, "I want a basic account with no monthly fees.")
    memory = at.session_state["conversation"]
    assert memory.awaiting == "deposit_mxn" and at.session_state["squad_state"] is None  # no tool ran yet
    assert any("how much will you keep" in str(m.markdown[0].value) for m in at.chat_message)
    _send(at, "around 12k")
    result = at.session_state["squad_state"]
    assert result is not None and result["user_profile"]["deposit_mxn"] == 12_000.0
    assert "deposit_mxn" not in result["user_profile"]["missing_inputs"]
    assert at.session_state["deposit_amount_mxn"] == 12_000.0


def test_clarification_skip_keeps_missing_deposit_warning() -> None:
    from prompts.advisory_prompts import MISSING_DEPOSIT_WARNING

    at = _new_app()
    _send(at, "I want an account with zero fees.")
    at.button(key="clarify_skip").click().run()
    _assert_clean(at)
    result = at.session_state["squad_state"]
    assert result["user_profile"]["missing_inputs"] == ["deposit_mxn"]
    assert MISSING_DEPOSIT_WARNING in result["draft_recommendation"]["warnings"]


def test_criteria_negotiation_and_relax_button() -> None:
    at = _new_app()
    _manual(at, "constraints")
    for key in ("hc_cashback", "hc_free_insurance", "hc_no_minimum_balance"):
        at.sidebar.checkbox(key=key).check()
    result = _run_query(at, "I want an account with good benefits.")
    draft = result["draft_recommendation"]
    assert draft["status"] == "no_match" and draft["negotiation"], draft.get("negotiation")
    assert any("Criteria negotiation" in t for t in _texts(at.markdown))
    relax = draft["negotiation"][0]["relax_constraints"]
    at.button(key="relax_btn_0").click().run()
    _assert_clean(at)
    relaxed = at.session_state["squad_state"]
    assert not set(relax) & set(relaxed["user_profile"]["hard_constraints"]), relaxed["user_profile"]
    assert relaxed["draft_recommendation"]["base_option"] is not None


def test_direct_answer_route_for_product_fact() -> None:
    at = _new_app()
    _send(at, "What is the monthly fee of Débito LikeU Pro?")  # no deposit clarification for a factual question
    result = at.session_state["squad_state"]
    assert result["draft_recommendation"]["status"] == "direct_answer"
    assert result["execution_metadata"]["intent"] == "product_fact"
    assert "140.00" in result["draft_recommendation"]["answer"]
    assert any("Direct answer" in t for t in _texts(at.markdown))


def test_perk_hallucination_guardrail() -> None:
    """Regression: a live run claimed Switch Banamex includes free fraud insurance (it does not)."""
    from guardrails import unsupported_perk_claims

    state = {"api_structured_data": [{"institution_id": "40002", "product_id": "SwitchBanamex",
                                      "has_free_insurance": False, "has_cashback_or_perks": True}]}
    option = {"product_key": "40002_SwitchBanamex", "institution_name": "BANAMEX", "product_name": "Switch Banamex"}

    def draft(summary: str) -> Dict[str, Any]:
        return {"base_option": option, "tradeoff_option": None, "narrative": {"summary": summary, "key_tradeoffs": []}}

    bad = draft("The Base Option, Switch Banamex, offers a $0 fee and meets your requirement for zero monthly "
                "fees and free fraud insurance.")
    assert unsupported_perk_claims(bad, state), "hallucinated insurance claim was not flagged"
    assert not unsupported_perk_claims(draft("Switch Banamex does not include free insurance."), state)
    assert not unsupported_perk_claims(draft("Switch Banamex offers cashback on purchases."), state)


def test_example_query_enforces_free_insurance() -> None:
    at = _new_app()
    at.button(key="example_btn").click().run()
    _assert_clean(at)
    at.button(key="analyze_btn").click().run()
    _assert_clean(at)
    result = at.session_state["squad_state"]
    assert result["user_profile"]["deposit_mxn"] == app.EXAMPLE_DEPOSIT_MXN  # parsed from the example text
    assert at.session_state["deposit_amount_mxn"] == app.EXAMPLE_DEPOSIT_MXN  # locked field mirrors the run
    assert "free_insurance" in result["user_profile"]["hard_constraints"]
    base = result["draft_recommendation"]["base_option"]
    assert base is not None and base["includes_free_insurance"] is True, base


def test_history_accumulates_across_runs() -> None:
    at = _new_app()
    _manual(at, "financial")
    _run_query(at, "I want a zero fee account.")
    at.number_input(key="deposit_amount_mxn").set_value(5_000.0)
    _run_query(at, "I want a zero fee account with payroll.")
    history = at.session_state["execution_history"]
    assert len(history) == 2 and history[0]["run_id"] != history[1]["run_id"]
    assert [h["deposit_mxn"] for h in history] == [20_000.0, 5_000.0]


def test_profile_provenance_labels_every_input() -> None:
    from memory import ConversationProfile

    profile = ConversationProfile(deposit_mxn=5_000.0, deposit_source="sidebar", hard_constraints=["zero_fee"],
                                  sources={"constraints": "message"})
    result = {"user_profile": {"deposit_mxn": 5_000.0, "deposit_stated": True, "monthly_transfers": 2,
                               "preferred_channel": None, "hard_constraints": ["zero_fee"],
                               "excluded_institutions": [], "profiling_source": "llm"},
              "weighted_preference_vector": {"yield_importance": 0.7, "fee_importance": 0.2, "perks_importance": 0.1},
              "archetype_prior": "Strategic_Saver"}
    rows = {r["field"]: r for r in app.profile_provenance(result, profile)}
    assert rows["Average balance"]["source"] == "sidebar" and rows["Average balance"]["value"] == "MXN 5,000.00"
    assert rows["SPEI transfers/month"]["source"] == "default"
    assert rows["SPEI channel"]["source"] == "default"
    assert rows["Weights w (yield / fee / perks)"]["source"] == "agent" and "(llm)" in \
        rows["Weights w (yield / fee / perks)"]["value"]
    assert rows["Hard constraints"]["source"] == "message"
    assert "Excluded banks" not in rows
    line = app.provenance_line(list(rows.values()))
    assert "Average balance MXN 5,000.00 🎛️" in line and "⚙️" in line and "🤖" in line

    skipped = dict(result, user_profile=dict(result["user_profile"], deposit_stated=False, deposit_mxn=0.0))
    assumed = ConversationProfile(deposit_mxn=0.0, deposit_source="assumed")
    assert app.profile_provenance(skipped, assumed)[0]["source"] == "assumed"
    assert app.profile_provenance({"draft_recommendation": {"status": "direct_answer"}}, profile) == []

    widgets = app.result_widget_values(result)
    assert widgets["deposit_amount_mxn"] == 5_000.0 and widgets["monthly_spei_transfers"] == 2
    assert widgets["w_yield"] == 0.7 and widgets["hc_zero_fee"] is True and widgets["hc_payroll"] is False


def test_locked_sidebar_mirrors_inferred_profile() -> None:
    """Regression (Sabadell vs NOW): default slider weights silently overrode a yield-focused message."""
    at = _new_app()
    _send(at, "I earn 50000 mexican pesos a moth, I want a debit card with yields over 10% a month")
    memory = at.session_state["conversation"]
    assert memory.awaiting == "deposit_mxn"
    assert not at.sidebar.number_input(key="deposit_amount_mxn").disabled  # the question unlocks its field only
    assert at.sidebar.number_input(key="monthly_spei_transfers").disabled
    assert all(s.disabled for s in at.sidebar.slider)
    assert any("asked for your average balance" in t for t in _texts(at.sidebar.warning))
    assert at.button(key="clarify_use_sidebar").disabled  # empty field: nothing to use yet

    at.sidebar.number_input(key="deposit_amount_mxn").set_value(5_000.0).run()
    at.button(key="clarify_use_sidebar").click().run()
    _assert_clean(at)
    result = at.session_state["squad_state"]
    weights = result["weighted_preference_vector"]
    assert result["user_profile"]["profiling_source"] != "user", "sidebar weights must not be used while locked"
    assert weights["yield_importance"] > 0.5, weights
    # BIM pays 9.47 %; NOW 7.25 % since its Oct 2026 folleto
    assert result["draft_recommendation"]["base_option"]["product_key"] == "40150_CuentaDigitalBIMTrad"
    # yield objective: the top 5 lists the 4 accounts paying at MXN 5,000, then one paying with a higher balance
    rows = app.branch_rows(app.display_branches(result))
    assert [r["Note"] == "" for r in rows] == [True, True, True, True, False], rows
    assert rows[4]["Note"].startswith("🔓 pays 2.46% from MXN 40,000") and rows[4]["Bank"] == "INBURSA", rows[4]
    from tools.schemas import load_catalog
    chart = app.catalog_leaderboard(result, load_catalog(Path(APP_PATH).parent / "data" / "catalog.json", strict=False))
    assert chart[4]["unlock_balance_mxn"] == 40_000.0 and chart[3]["yield_rate_percent"] >= 1.0
    assert "INBURSA CUENTA EFE would pay 2.46%" in " ".join(result["draft_recommendation"]["narrative"]["key_tradeoffs"])
    assert not any(r["product_key"] == "40036_1005" for r in chart)  # guardianship account: not for an adult
    # Top-5 detail picker: same card fields as the recommended option, for any of the five accounts
    products = {p.catalog_key: p for p in load_catalog(Path(APP_PATH).parent / "data" / "catalog.json", strict=False)}
    base = result["draft_recommendation"]["base_option"]
    card = app.leaderboard_option(next(r for r in chart if r["product_key"] == base["product_key"]), products)
    for field in ("net_annual_impact_mxn", "total_annual_fees_mxn", "gross_annual_interest_mxn",
                  "effective_monthly_fee_mxn", "applicable_yield_rate_percent", "qualitative_perks_summary"):
        assert card[field] == base[field], (field, card[field], base[field])
    picker = at.selectbox(key="top5_detail_key")
    assert len(picker.options) == app.TOP_N_CHART and "🛡️ Base Option" in picker.options[0], picker.options
    picker.set_value(chart[2]["product_key"]).run()
    _assert_clean(at)
    assert any(chart[2]["label"] in t for t in _texts(at.markdown)), "selected Top-5 account card missing"

    assert at.sidebar.number_input(key="deposit_amount_mxn").disabled  # locked again after the answer
    assert at.sidebar.slider(key="w_yield").value == round(weights["yield_importance"], 2)  # shows inferred w
    assert at.sidebar.number_input(key="monthly_spei_transfers").value == result["user_profile"]["monthly_transfers"]
    rows = {r["field"]: r["source"] for r in at.session_state["profile_provenance"]}
    assert rows["Average balance"] == "sidebar" and rows["SPEI transfers/month"] == "default"
    assert rows["Weights w (yield / fee / perks)"] == "agent"
    assert any("Last run used" in t and "default assumption" in t for t in _texts(at.sidebar.caption))
    assert any(t.startswith("Profile used:") for t in _texts(at.caption)), "chat line with the profile used missing"

    _manual(at, "weights")  # switching to manual keeps the inferred values as the starting point
    assert not at.sidebar.slider(key="w_yield").disabled
    assert at.sidebar.slider(key="w_yield").value == round(weights["yield_importance"], 2)


TESTS: List[Callable[[], None]] = [
    test_rebalance_weights_keeps_sum_one,
    test_session_state_contract_on_plain_dict,
    test_error_collection_and_hitl_labels,
    test_execute_squad_captures_chromadb_error,
    test_initial_render_and_session_state,
    test_weight_slider_autonormalization_and_archetype,
    test_empty_query_is_rejected,
    test_simple_query_end_to_end,
    test_hybrid_query_renders_tot_table,
    test_hitl_escalation_and_human_approval,
    test_hitl_escalate_then_modify,
    test_review_queue_redacts_pii,
    test_spei_channel_override_and_live_progress,
    test_llm_feature_claims_are_verified_deterministically,
    test_multi_turn_refinement_updates_profile_and_sidebar,
    test_clarification_asks_for_missing_deposit,
    test_clarification_skip_keeps_missing_deposit_warning,
    test_criteria_negotiation_and_relax_button,
    test_direct_answer_route_for_product_fact,
    test_reviewer_console_decides_case_from_another_session,
    test_reviewer_console_modify_and_closed_case_guard,
    test_reviewer_console_empty_filter,
    test_perk_hallucination_guardrail,
    test_example_query_enforces_free_insurance,
    test_history_accumulates_across_runs,
    test_profile_provenance_labels_every_input,
    test_locked_sidebar_mirrors_inferred_profile,
]


def main() -> int:
    failures = 0
    for test in TESTS:
        started = time.perf_counter()
        try:
            test()
            print(f"PASS  {test.__name__}  ({time.perf_counter() - started:.1f}s)")
        except Exception:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} UI tests passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
