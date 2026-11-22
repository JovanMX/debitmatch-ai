"""DebitMatch AI - Human Reviewer Console (Protocol B: asynchronous audit).

A separate Streamlit page (auto-discovered from ``pages/``) that reads the
persistent review queue, so a financial specialist can decide escalated cases
created by ANY user session. Decisions are appended to the queue and, when
terminal (approve / modify), to the Golden Dataset. The originating session
picks the decision up on its next rerun (``app.sync_hitl_from_queue``).

Each case is shown as a Bounded Decision Card: escalation reasons with a
suggested correction, key financial values, the conflict vector and the
machine recommendation - nothing else, to keep reviewer cognitive load low.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root (pages/ is one level down)

from hitl_queue import TERMINAL_DECISIONS, CaseClosedError, ReviewQueue, hitl_reason_details  # noqa: E402

PAGE_TITLE = "Reviewer Console"
OPEN_STATUSES = ("pending", "escalated")
ALL_STATUSES = ("pending", "escalated", "approved", "modified")
STATUS_ICONS = {"pending": "🕒", "escalated": "🔺", "approved": "✅", "modified": "✏️"}
FOOTER = ("Informational financial comparison tool built for academic demonstration. "
          "Rates and terms subject to CONDUSEF / CASFIM regulatory updates.")


def money(value: Optional[float]) -> str:
    return "n/a" if value is None else f"MXN {float(value):,.2f}"


def option_label(option: Optional[Mapping[str, Any]]) -> str:
    if not option:
        return "—"
    return f"{option.get('institution_name')} {option.get('product_name')}"


def case_label(case: Mapping[str, Any]) -> str:
    card = case["card"]
    deposit = (card.get("profile") or {}).get("deposit_mxn")
    return (f"{STATUS_ICONS.get(case['status'], '')} {card.get('case_id')} · {case['status']} · "
            f"{money(deposit)} · {card.get('created_at_utc')}")


def replacement_candidates(card: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Evaluated alternatives a reviewer may promote to Base Option."""
    machine = card.get("machine_recommendation") or {}
    base_key = (machine.get("base_option") or {}).get("product_key")
    pool = list(card.get("candidate_options") or [])
    if machine.get("tradeoff_option"):
        pool.append(machine["tradeoff_option"])  # older cards have no candidate list
    seen, out = {base_key}, []
    for option in pool:
        if option and option.get("product_key") not in seen:
            seen.add(option["product_key"])
            out.append(option)
    return out


# --------------------------------------------------------------------------- #
# Callback
# --------------------------------------------------------------------------- #
def submit_decision(case_id: str, decision: str) -> None:
    ss = st.session_state
    override = ss.get("rc_override") if decision == "modified" else None
    if decision == "modified" and not override:
        ss["rc_message"] = ("error", "Select a replacement Base Option before submitting a modification.")
        return
    try:
        ReviewQueue().record_decision(case_id, decision, str(ss.get("rc_notes", "")), override,
                                      reviewer=str(ss.get("rc_reviewer", "")), source="reviewer_console")
    except CaseClosedError as exc:
        ss["rc_message"] = ("warning", str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - show persistence failures to the reviewer
        ss["rc_message"] = ("error", f"Could not record the decision: {type(exc).__name__}: {exc}")
        return
    ss["rc_message"] = ("success", f"Case {case_id}: decision '{decision}' recorded.")
    ss["rc_notes"] = ""
    ss["rc_override"] = None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_card(case: Mapping[str, Any]) -> None:
    card = case["card"]
    profile = card.get("profile") or {}
    values = card.get("key_values") or {}
    machine = card.get("machine_recommendation") or {}
    base = machine.get("base_option") or {}

    with st.container(border=True):
        st.markdown(f"#### Bounded Decision Card · `{card.get('case_id')}` · status **{case['status']}**")
        st.caption(f"Run {card.get('run_id')} · created {card.get('created_at_utc')} · route {values.get('route')} · "
                   f"LLM {values.get('llm_model') or 'deterministic'}")
        st.markdown(f"**Query (PII-redacted):** {card.get('query_redacted') or '—'}")

        st.markdown("**Escalation cause(s)**")
        for reason in card.get("escalation_reasons") or []:
            label, correction = hitl_reason_details(reason)
            st.markdown(f"- **{label}** — {reason}.  \n  *Suggested correction:* {correction}")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Deposit", money(profile.get("deposit_mxn")))
        k2.metric("Confidence", f"{float(values.get('confidence_score') or 0):.2f}")
        k3.metric("Best score", f"{float(values.get('evaluator_score') or 0):.2f}")
        k4.metric("Base net annual", money(base.get("net_annual_impact_mxn")))
        st.caption(f"Archetype {profile.get('archetype')} · hard constraints {profile.get('hard_constraints') or []} · "
                   f"transfers/month {profile.get('monthly_transfers')} · channel {profile.get('preferred_channel') or 'auto'}"
                   f" · data fresh: {values.get('data_freshness_flag')} · revisions {values.get('revision_count')}")

        conflicts = card.get("conflict_vector") or []
        if conflicts:
            st.markdown("**Conflict vector (catalog API vs RECA clause)**")
            st.dataframe(pd.DataFrame(conflicts), hide_index=True, width="stretch")

        st.markdown("**Machine recommendation**")
        rows = [dict(role="Base Option", **(machine.get("base_option") or {}))]
        if machine.get("tradeoff_option"):
            rows.append(dict(role="Positive Trade-off", **machine["tradeoff_option"]))
        rows += [dict(role="Alternative", **o) for o in replacement_candidates(card)
                 if o.get("product_key") != (machine.get("tradeoff_option") or {}).get("product_key")]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        if case["decisions"]:
            st.markdown("**Decision history**")
            st.dataframe(pd.DataFrame([{
                "At (UTC)": d.get("at_utc"), "Decision": d.get("decision"), "Reviewer": d.get("reviewer") or "—",
                "Via": d.get("source", "app"), "Override": d.get("override_product_key") or "—",
                "Notes": d.get("advisor_notes") or "—",
            } for d in case["decisions"]]), hide_index=True, width="stretch")


def render_controls(case: Mapping[str, Any]) -> None:
    card = case["card"]
    case_id = str(card.get("case_id"))
    if case["status"] in TERMINAL_DECISIONS:
        st.info(f"This case is closed ({case['status']}); decisions are final and already in the Golden Dataset.")
        return
    candidates = replacement_candidates(card)
    labels = {o["product_key"]: f"{option_label(o)} · net {money(o.get('net_annual_impact_mxn'))} · "
                                f"score {float(o.get('composite_score') or 0):.2f}" for o in candidates}
    st.text_input("Reviewer ID (optional)", key="rc_reviewer")
    st.text_input("Add Human Advisor Notes", key="rc_notes")
    st.selectbox("Replacement Base Option (for Modify)", list(labels), index=None, key="rc_override",
                 format_func=lambda key: labels.get(key, key), placeholder="Choose another evaluated product",
                 disabled=not candidates)
    c1, c2, c3 = st.columns(3)
    c1.button("✅ Approve", key="rc_approve", on_click=submit_decision, args=(case_id, "approved"), type="primary")
    c2.button("✏️ Modify", key="rc_modify", on_click=submit_decision, args=(case_id, "modified"),
              disabled=not candidates)
    c3.button("🔺 Escalate", key="rc_escalate", on_click=submit_decision, args=(case_id, "escalated"),
              disabled=case["status"] == "escalated")


def main() -> None:
    st.set_page_config(page_title=f"DebitMatch AI · {PAGE_TITLE}", page_icon="🧑‍⚖️", layout="wide")
    ss = st.session_state
    ss.setdefault("rc_message", None)
    ss.setdefault("rc_status_filter", list(OPEN_STATUSES))

    st.title("🧑‍⚖️ Human Reviewer Console")
    st.caption("Protocol B · asynchronous audit of escalated DebitMatch AI recommendations (all sessions).")

    queue = ReviewQueue()
    try:
        cases = queue.cases()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the review queue at {queue.directory}: {type(exc).__name__}: {exc}")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pending", sum(c["status"] == "pending" for c in cases))
    m2.metric("Escalated", sum(c["status"] == "escalated" for c in cases))
    m3.metric("Decided", sum(c["status"] in TERMINAL_DECISIONS for c in cases))
    m4.metric("Golden Dataset records", queue.golden_count())

    message = ss.get("rc_message")
    if message:
        getattr(st, message[0])(message[1])
        ss["rc_message"] = None

    st.multiselect("Show statuses", ALL_STATUSES, key="rc_status_filter")
    visible = [c for c in reversed(cases) if c["status"] in (ss["rc_status_filter"] or ALL_STATUSES)]
    visible.sort(key=lambda c: (c["status"] not in OPEN_STATUSES, c["status"] != "escalated"))
    if not visible:
        st.info("No cases match the selected statuses.", icon="📭")
        st.caption(f"Queue location: {queue.directory}")
        return

    by_id = {str(c["card"].get("case_id")): c for c in visible}
    if ss.get("rc_case") not in by_id:
        ss["rc_case"] = next(iter(by_id))
    st.selectbox("Case", list(by_id), key="rc_case", format_func=lambda cid: case_label(by_id[cid]))
    case = by_id[ss["rc_case"]]

    render_card(case)
    render_controls(case)

    st.divider()
    st.caption(f"Queue location: {queue.directory} · free text is PII-redacted before storage.")
    st.caption(f"⚖️ {FOOTER}")


if __name__ == "__main__":
    main()
