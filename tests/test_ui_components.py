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


TESTS: List[Callable[[], None]] = [
    test_initial_render_and_session_state,
    test_hitl_escalation_and_human_approval,
    test_review_queue_redacts_pii,
    test_llm_feature_claims_are_verified_deterministically,
    test_multi_turn_refinement_updates_profile_and_sidebar,
    test_clarification_asks_for_missing_deposit,
    test_reviewer_console_decides_case_from_another_session,
    test_reviewer_console_modify_and_closed_case_guard,
    test_perk_hallucination_guardrail,
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
