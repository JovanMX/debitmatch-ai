"""Persistent Human-in-the-Loop review queue and Golden Dataset (Protocol B).

Escalated cases are written as Bounded Decision Cards to an append-only JSONL
event log; every human decision (approve / modify / escalate) is appended to
the same log and, for terminal decisions, to a Golden Dataset JSONL file used
to refine routing thresholds offline.

Privacy: free-text queries are stored only after PII redaction (emails, phone
numbers, CURP, RFC, card / account numbers); structured inputs (deposit,
constraints, weights) are kept because they are not personally identifying.

Location: ``data/review_queue/`` (override with ``DEBITMATCH_REVIEW_QUEUE_DIR``).
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from guardrails import CONFIDENCE_THRESHOLD, HIGH_AMOUNT_THRESHOLD_MXN, MAX_REVISIONS

QUEUE_DIR_ENV_VAR = "DEBITMATCH_REVIEW_QUEUE_DIR"
MAX_CANDIDATES_ON_CARD = 5

# (substring of guardrails.human_review_reasons(), short label, suggested correction)
HITL_REASON_RULES: Tuple[Tuple[str, str, str], ...] = (
    ("low confidence", f"Low confidence score (<{CONFIDENCE_THRESHOLD:.2f})",
     "Collect missing profile variables (average balance, usage pattern) and re-run the analysis."),
    ("high financial amount", f"Initial deposit exceeds ${HIGH_AMOUNT_THRESHOLD_MXN:,.0f} MXN threshold",
     "Confirm IPAB deposit-insurance coverage and suitability with a certified advisor before opening."),
    ("regulatory conflict", "API/RAG Policy Conflict",
     "Re-verify the conflicting term against the bank's current RECA contract and correct the catalog."),
    ("critique loop exhausted", f"Critique loop exhausted (N_max = {MAX_REVISIONS}) without approval",
     "Review the narrative against the verified figures and edit or reject it."),
    ("time budget", "Execution time budget exceeded",
     "Re-run the analysis; if it persists, inspect node latencies in the trajectory log."),
)


def hitl_reason_details(reason: str) -> Tuple[str, str]:
    """(short label, suggested correction) for one escalation reason."""
    lowered = reason.lower()
    for marker, label, correction in HITL_REASON_RULES:
        if marker in lowered:
            return label, correction
    return reason, "Review the case manually."


class CaseClosedError(ValueError):
    """Raised when a decision targets a case that already has a terminal decision."""
DEFAULT_QUEUE_DIR = Path(__file__).resolve().parent / "data" / "review_queue"
QUEUE_FILE = "hitl_queue.jsonl"
GOLDEN_FILE = "golden_dataset.jsonl"

Decision = Literal["approved", "modified", "escalated"]
DECISIONS: tuple[str, ...] = ("approved", "modified", "escalated")
TERMINAL_DECISIONS = frozenset({"approved", "modified"})

_LOCK = threading.Lock()  # Streamlit serves several sessions from one process

# --------------------------------------------------------------------------- #
# PII redaction
# --------------------------------------------------------------------------- #
_PII_PATTERNS = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b[A-Z][AEIOUX][A-Z]{2}\d{6}[HM][A-Z]{5}[A-Z\d]\d\b", re.IGNORECASE), "[CURP]"),
    (re.compile(r"\b[A-ZÑ&]{3,4}\d{6}[A-Z\d]{3}\b", re.IGNORECASE), "[RFC]"),
    (re.compile(r"\b(?:\d[ -]?){15,18}\d\b"), "[CARD_OR_CLABE]"),
    (re.compile(r"(?:\+?52[ -]?)?\(?\d{2,3}\)?[ -]?\d{3,4}[ -]?\d{4}\b"), "[PHONE]"),
)


def redact_pii(text: str) -> str:
    """Mask personal identifiers; money amounts such as $50,000 are preserved."""
    redacted = str(text or "")
    for pattern, token in _PII_PATTERNS:
        redacted = pattern.sub(token, redacted)
    return redacted


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _option_card(option: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not option:
        return None
    keys = ("product_key", "institution_name", "product_name", "composite_score", "net_annual_impact_mxn",
            "total_annual_fees_mxn", "gross_annual_interest_mxn", "hard_constraints_met")
    return {k: option.get(k) for k in keys}


def build_decision_card(result: Mapping[str, Any], run_id: str) -> Dict[str, Any]:
    """Bounded Decision Card: conflict vector, key financial values, machine recommendation."""
    draft = result.get("draft_recommendation") or {}
    profile = result.get("user_profile") or {}
    meta = result.get("execution_metadata") or {}
    return {
        "case_id": uuid.uuid4().hex[:12],
        "run_id": run_id,
        "created_at_utc": _now(),
        "query_redacted": redact_pii(result.get("user_query", "")),
        "profile": {
            "deposit_mxn": profile.get("deposit_mxn"),
            "monthly_transfers": profile.get("monthly_transfers"),
            "hard_constraints": list(profile.get("hard_constraints") or []),
            "preferred_channel": profile.get("preferred_channel"),
            "archetype": result.get("archetype_prior"),
            "weights": dict(result.get("weighted_preference_vector") or {}),
        },
        "escalation_reasons": list(draft.get("human_review_reasons") or []),
        "conflict_vector": [
            {k: c.get(k) for k in ("product_key", "type", "api_value", "document_value", "source_file")}
            for c in result.get("data_conflicts") or []
        ],
        "key_values": {
            "confidence_score": result.get("confidence_score"),
            "evaluator_score": result.get("evaluator_score"),
            "revision_count": result.get("revision_count"),
            "data_freshness_flag": result.get("data_freshness_flag"),
            "route": meta.get("route"),
            "llm_model": meta.get("llm_model"),
        },
        "machine_recommendation": {
            "status": draft.get("status"),
            "base_option": _option_card(draft.get("base_option")),
            "tradeoff_option": _option_card(draft.get("tradeoff_option")),
        },
        # evaluated alternatives a reviewer may promote to Base Option (Modify)
        "candidate_options": [_option_card(_branch_option(b)) for b in _final_branches(result)[:MAX_CANDIDATES_ON_CARD]],
    }


def _final_branches(result: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    branches = [b for b in result.get("tot_candidate_branches") or [] if b.get("final")]
    return sorted(branches, key=lambda b: b.get("final_rank", 999))


def _branch_option(branch: Mapping[str, Any]) -> Dict[str, Any]:
    impact = branch.get("net_impact") or {}
    return {
        "product_key": branch.get("product_key"),
        "institution_name": branch.get("institution_name"),
        "product_name": branch.get("product_name"),
        "composite_score": branch.get("score"),
        "net_annual_impact_mxn": impact.get("net_annual_impact_mxn"),
        "total_annual_fees_mxn": impact.get("total_annual_fees_mxn"),
        "gross_annual_interest_mxn": impact.get("gross_annual_interest_mxn"),
        "hard_constraints_met": branch.get("hard_constraints_met"),
    }


class ReviewQueue:
    """Append-only JSONL review queue folded into case views on read."""

    def __init__(self, directory: Optional[Path | str] = None) -> None:
        root = directory or os.environ.get(QUEUE_DIR_ENV_VAR) or DEFAULT_QUEUE_DIR
        self.directory = Path(root)
        self.queue_path = self.directory / QUEUE_FILE
        self.golden_path = self.directory / GOLDEN_FILE

    # -- low level ---------------------------------------------------------- #
    def _append(self, path: Path, record: Mapping[str, Any]) -> None:
        with _LOCK:
            self.directory.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _events(self) -> List[Dict[str, Any]]:
        if not self.queue_path.is_file():
            return []
        events: List[Dict[str, Any]] = []
        with _LOCK, self.queue_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # tolerate a partially written last line
        return events

    # -- public API --------------------------------------------------------- #
    def enqueue(self, card: Mapping[str, Any]) -> str:
        self._append(self.queue_path, {"event": "enqueued", "at_utc": _now(), "card": dict(card)})
        return str(card["case_id"])

    def record_decision(
        self,
        case_id: str,
        decision: str,
        advisor_notes: str = "",
        override_product_key: Optional[str] = None,
        reviewer: str = "",
        source: str = "app",
    ) -> Dict[str, Any]:
        if decision not in DECISIONS:
            raise ValueError(f"Unknown decision '{decision}'. Allowed: {DECISIONS}")
        case = self.get(case_id)
        if case is None:
            raise KeyError(f"Unknown review case '{case_id}'.")
        if case["status"] in TERMINAL_DECISIONS:
            last = case["decisions"][-1]
            raise CaseClosedError(f"Case '{case_id}' was already {last['decision']} at {last['at_utc']}"
                                  f" ({last.get('source', 'app')}).")
        record = {"event": "decision", "at_utc": _now(), "case_id": case_id, "decision": decision,
                  "advisor_notes": redact_pii(advisor_notes), "override_product_key": override_product_key,
                  "reviewer": redact_pii(reviewer.strip()), "source": source}
        self._append(self.queue_path, record)
        if decision in TERMINAL_DECISIONS:
            machine_base = ((case["card"].get("machine_recommendation") or {}).get("base_option") or {})
            self._append(self.golden_path, {
                "case_id": case_id,
                "recorded_at_utc": record["at_utc"],
                "input": {"query_redacted": case["card"].get("query_redacted"), "profile": case["card"].get("profile")},
                "escalation_reasons": case["card"].get("escalation_reasons"),
                "conflict_vector": case["card"].get("conflict_vector"),
                "key_values": case["card"].get("key_values"),
                "machine_base_option": machine_base.get("product_key"),
                "human_decision": decision,
                "human_base_option": override_product_key or machine_base.get("product_key"),
                "correction_applied": decision == "modified",
                "advisor_notes": record["advisor_notes"],
                "reviewer": record["reviewer"],
                "source": source,
            })
        return record

    def cases(self) -> List[Dict[str, Any]]:
        """Fold events into case views: card + decision history + current status."""
        by_id: Dict[str, Dict[str, Any]] = {}
        for event in self._events():
            if event.get("event") == "enqueued":
                card = event.get("card") or {}
                by_id[card.get("case_id")] = {"card": card, "decisions": [], "status": "pending"}
            elif event.get("event") == "decision" and event.get("case_id") in by_id:
                case = by_id[event["case_id"]]
                case["decisions"].append(event)
                case["status"] = event["decision"]
        return list(by_id.values())

    def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        return next((c for c in self.cases() if c["card"].get("case_id") == case_id), None)

    def pending(self) -> List[Dict[str, Any]]:
        """Cases still awaiting a terminal decision (pending or escalated)."""
        return [c for c in self.cases() if c["status"] not in TERMINAL_DECISIONS]

    def golden_count(self) -> int:
        if not self.golden_path.is_file():
            return 0
        with _LOCK, self.golden_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
