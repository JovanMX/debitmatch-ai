"""Short-term conversation memory for multi-turn refinement (typed state + sliding window + summarizer).

Design (Agent Design Refinement V1.3, "Memory requirements"):
    * Typed state  - ``ConversationProfile``: deposit, transfers, hard constraints, excluded banks,
                     channel and preference weights. Every user message is converted into a
                     deterministic *profile delta*; numbers never come from an LLM.
    * Sliding window - the last ``window_size`` turns are kept verbatim.
    * Summarizer   - turns leaving the window are folded into a rolling summary (LLM when
                     available, deterministic fallback).

Each turn re-runs the full squad on the consolidated profile (structured overrides are
authoritative), so every turn keeps the 10 s budget and all guardrails. Pre-generation
completeness checks (Safety Guardrails plan) ask for the deposit before any tool runs.

The module is UI-agnostic: used by the Streamlit app and by the MCP server (state manager).
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from agents.profiling import (
    BUSINESS_ACCOUNT_PATTERN,
    OBJECTIVE_BOOST,
    PROFILE_OVERRIDES_KEY,
    extract_signals,
    normalize_weights,
    parse_deposit,
)
from agents.router import DIRECT_INTENTS, classify_intent
from prompts.memory_prompts import MEMORY_SUMMARIZER_SYSTEM_PROMPT, ConversationSummary
from state import PREFERENCE_KEYS
from tools.catalog_search import CONSTRAINT_DESCRIPTIONS, HARD_CONSTRAINTS
from tools.schemas import CHANNEL_BRANCH, CHANNEL_MOBILE_APP, CHANNEL_ONLINE_WEB, CHANNEL_TELEPHONE

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SIZE = 6  # turns kept verbatim (user + assistant)
SUMMARY_LLM_TIMEOUT_S = 4.0
DEPOSIT_QUESTION = (
    "Before I compare accounts: roughly how much will you keep in the account on average (MXN)? Fee waivers and "
    "interest depend on it. You can also reply 'skip' and I will assume MXN 0."
)

# ------------------------------------------------------------------------ #
# Deterministic delta extraction
# ------------------------------------------------------------------------ #
_REMOVAL_VERB = (r"(?:forget|remove|drop|ignore|skip|relax|don't need|do not need|no longer need|doesn't need to be|"
                 r"does not need to be|doesn't have to be|does not have to be|don't care about|do not care about|"
                 r"not necessary|not required|i can live without)")
_PAY_INTENT = re.compile(r"\b(?:need|want|have) to pay\b", re.IGNORECASE)  # "I don't need to pay fees" = keep zero_fee
_CONSTRAINT_KEYWORDS: Dict[str, str] = {
    "payroll": r"payroll|n[oó]mina|salary",
    "zero_fee": r"(?:zero|no|monthly|maintenance)[- ]?fees?|fee requirement|fees|commissions?",
    "digital_opening": r"digital(?: opening)?|online opening|open(?:ing)? (?:it )?online|app",
    "free_insurance": r"(?:free |fraud |life )?insurance",
    "cashback": r"cash ?back|rewards?|perks?",
    "no_minimum_balance": r"(?:no )?minimum balance",
    "minor_guardianship": (r"guardianship|minor(?:'s)?(?: account)?|teen(?:ager)?(?: account)?|under[- ]18"
                           r"|(?:son|daughter|kids?|child(?:ren)?)(?:'s)? account|for my (?:son|daughter|kids?|child)"),
}
_REMOVALS = {c: re.compile(rf"{_REMOVAL_VERB}\b[^.?!]{{0,40}}?\b(?:{kw})\b", re.IGNORECASE)
             for c, kw in _CONSTRAINT_KEYWORDS.items()}
_REMOVALS_SUFFIX = {c: re.compile(rf"\b(?:{kw})\b[^.?!]{{0,25}}\b(?:is|are)\s+(?:not (?:necessary|required|important)|"
                                  rf"optional|fine either way)", re.IGNORECASE) for c, kw in _CONSTRAINT_KEYWORDS.items()}
_EXCLUDE_VERB = r"(?:not|no|exclude|except|avoid|don't want|do not want|without|other than|anything but|nothing from)"
_INCLUDE_VERB = r"(?:include|consider|add|allow|bring back)(?: back| again)?"
_BANK_ALIASES: Dict[str, Sequence[str]] = {
    "BANAMEX": ("banamex", "citibanamex"), "SANTANDER": ("santander",), "HSBC": ("hsbc",),
    "BaBien": ("babien", "banco del bienestar"), "BBVA MEXICO": ("bbva",),
}
_BARE_AMOUNT = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s?(k|mil)?\b", re.IGNORECASE)
_SKIP = re.compile(r"^\s*(?:skip|i don't know|i do not know|not sure|prefer not|n/?a|no idea)\b", re.IGNORECASE)
_EXPLICIT_CHANNEL = {
    CHANNEL_BRANCH: re.compile(r"\b(?:branch|teller|sucursal|ventanilla)\b", re.IGNORECASE),
    CHANNEL_MOBILE_APP: re.compile(r"\b(?:mobile app|phone app|from the app|via the app|by app)\b", re.IGNORECASE),
    CHANNEL_ONLINE_WEB: re.compile(r"\b(?:online banking|web banking|website)\b", re.IGNORECASE),
    CHANNEL_TELEPHONE: re.compile(r"\b(?:telephone banking|by phone call|call center)\b", re.IGNORECASE),
}
_WEIGHT_KEYWORDS = {"yield_importance": r"yield|interest|returns?|earn", "fee_importance": r"fees?|commissions?|costs?",
                    "perks_importance": r"perks?|cash ?back|rewards?|insurance|benefits?"}
_MORE_AFTER = r"(?:matters? (?:more|most)|(?:is|are) (?:more|most) important|(?:is|are) my priority)"
_MORE_BEFORE = r"(?:prioriti[sz]e|focus on|care (?:most|more) about|more weight (?:to|on))"
_LESS_AFTER = r"(?:matters? less|(?:is|are) less important|(?:is|are) not (?:that )?important)"
_LESS_BEFORE = r"(?:than|deprioriti[sz]e|don't care much about|less weight (?:to|on))"


def _bare_amount(text: str) -> Optional[float]:
    amounts = []
    for raw, mult in _BARE_AMOUNT.findall(text):
        value = float(raw.replace(",", ""))
        amounts.append(value * 1000 if mult else value)
    plausible = [a for a in amounts if a >= 100 or a == 0]
    return max(plausible) if plausible else None


@dataclass
class ProfileDelta:
    deposit_mxn: Optional[float] = None
    deposit_assumed: bool = False
    monthly_transfers: Optional[int] = None
    add_constraints: List[str] = field(default_factory=list)
    remove_constraints: List[str] = field(default_factory=list)
    exclude_institutions: List[str] = field(default_factory=list)
    include_institutions: List[str] = field(default_factory=list)
    preferred_channel: Optional[str] = None
    weight_shifts: Dict[str, float] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any(asdict(self).values())


def extract_delta(message: str, awaiting_deposit: bool = False) -> ProfileDelta:
    """Convert one user message into structured profile changes (deterministic)."""
    delta = ProfileDelta()
    signals = extract_signals(message)

    removed = [c for c in HARD_CONSTRAINTS
               if (_REMOVALS[c].search(message) or _REMOVALS_SUFFIX[c].search(message))
               and not (c == "zero_fee" and _PAY_INTENT.search(message))]
    delta.remove_constraints = removed
    delta.add_constraints = [c for c in signals["hard_constraints"] if c not in removed]

    can_not_keep = "no_minimum_balance" in delta.add_constraints
    if not can_not_keep:  # "I can't keep 3,000 MXN" is a rejection, not the user's balance
        amount = parse_deposit(message)
        if amount is None and awaiting_deposit:
            amount = _bare_amount(message)
        if amount is not None:
            delta.deposit_mxn = float(amount)
    if delta.deposit_mxn is None and awaiting_deposit and _SKIP.search(message):
        delta.deposit_mxn, delta.deposit_assumed = 0.0, True

    if re.search(r"\btransfers?\b", message, re.IGNORECASE) and re.search(r"\d", message):
        delta.monthly_transfers = int(signals["monthly_transfers"])
    for channel, pattern in _EXPLICIT_CHANNEL.items():
        if pattern.search(message):
            delta.preferred_channel = channel
            break

    lowered = message.lower()
    for bank, aliases in _BANK_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{_EXCLUDE_VERB}\b[^.?!]{{0,25}}\b{re.escape(alias)}\b", lowered):
                delta.exclude_institutions.append(bank)
                break
            if re.search(rf"\b{_INCLUDE_VERB}\b[^.?!]{{0,15}}\b{re.escape(alias)}\b", lowered):
                delta.include_institutions.append(bank)
                break

    for key, keywords in _WEIGHT_KEYWORDS.items():
        # "fees matter more (than perks)" -> fee up, perks down; "prioritize yield" -> yield up
        if re.search(rf"\b(?:{_LESS_BEFORE})\s+(?:the\s+|my\s+)?(?:{keywords})\b", lowered) or \
                re.search(rf"\b(?:{keywords})\b[^.?!]{{0,15}}\b{_LESS_AFTER}", lowered):
            delta.weight_shifts[key] = -OBJECTIVE_BOOST
        elif re.search(rf"\b(?:{keywords})\b[^.?!]{{0,15}}\b{_MORE_AFTER}", lowered) or \
                re.search(rf"\b{_MORE_BEFORE}\s+(?:the\s+|my\s+)?(?:{keywords})\b", lowered):
            delta.weight_shifts[key] = OBJECTIVE_BOOST
    return delta


def delta_from_extraction(regex: ProfileDelta, verified: Any) -> ProfileDelta:
    """Profile delta from a verified LLM extraction; the regex delta fills in what the model did not mention.

    * deposit: the model's verified balance amount; else the regex value, unless the model identified that
      same amount as income or a periodic deposit (an amount it only labelled "other" is ambiguity, not
      counter-evidence, so the regex balance stands - as it does when the model saw no amount at all).
    * constraints / banks: the model's verified requirements and removals (each quotes the message).
    * transfers / channel: the model's verified value, else the regex value.
    * weight shifts ('fees matter more'): regex (they only adjust user-set weights).
    """
    from agents.extraction import NOT_BALANCE_KINDS

    delta = ProfileDelta(weight_shifts=dict(regex.weight_shifts))
    not_balance = [a["amount_mxn"] for a in verified.amounts if a["kind"] in NOT_BALANCE_KINDS]
    if verified.deposit_mxn is not None:
        delta.deposit_mxn = float(verified.deposit_mxn)
    elif regex.deposit_mxn is None or not any(abs(regex.deposit_mxn - a) <= 0.5 for a in not_balance):
        delta.deposit_mxn, delta.deposit_assumed = regex.deposit_mxn, regex.deposit_assumed
    delta.monthly_transfers = (verified.monthly_transfers if verified.monthly_transfers is not None
                               else regex.monthly_transfers)
    delta.add_constraints = list(verified.require)
    delta.remove_constraints = list(verified.remove)
    delta.exclude_institutions = list(verified.exclude_banks)
    delta.include_institutions = list(verified.include_banks)
    delta.preferred_channel = verified.preferred_channel or regex.preferred_channel
    return delta


# ------------------------------------------------------------------------ #
# Typed state
# ------------------------------------------------------------------------ #
@dataclass
class ConversationProfile:
    deposit_mxn: Optional[float] = None
    deposit_source: Optional[str] = None  # conversation | sidebar | assumed
    monthly_transfers: Optional[int] = None
    hard_constraints: List[str] = field(default_factory=list)
    excluded_institutions: List[str] = field(default_factory=list)
    preferred_channel: Optional[str] = None
    weights: Optional[Dict[str, float]] = None  # None -> inferred by the Profiling Agent
    business_account: bool = False  # sticky once the user asks for business banking
    # who set each field: "message" (parsed from the chat) | "sidebar" (manual UI input); absent = not set
    # keys: transfers, channel, constraints, excluded, weights (the deposit uses ``deposit_source``)
    sources: Dict[str, str] = field(default_factory=dict)


@dataclass
class Turn:
    role: str  # user | assistant
    content: str
    kind: str = "message"  # message | clarification | recommendation | direct_answer | summary
    changes: List[str] = field(default_factory=list)
    run_id: Optional[str] = None
    profile_used: Optional[str] = None  # assistant turns: the profile values (and their origin) the run used
    at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


@dataclass
class TurnPlan:
    action: str  # "clarify" | "run"
    message: str
    changes: List[str]
    question: Optional[str] = None
    query: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None
    run_metadata: Optional[Dict[str, Any]] = None


class ConversationMemory:
    """Typed profile + sliding window of turns + rolling summary for one session."""

    def __init__(self, session_id: Optional[str] = None, window_size: int = DEFAULT_WINDOW_SIZE,
                 catalog: Optional[Sequence[Any]] = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.window_size = max(2, window_size)
        self.catalog = list(catalog or [])
        self.profile = ConversationProfile()
        self.turns: List[Turn] = []  # sliding window (verbatim)
        self.summary = ""
        self.folded_turns = 0
        self.awaiting: Optional[str] = None  # field asked in a clarification question
        self.pending_message: Optional[str] = None  # request waiting for the clarification answer
        self.clarifications_asked: List[str] = []
        # request text -> verified LLM extraction (agents.extraction); forwarded to the squad so the Profiling
        # Agent reuses it instead of calling the model again
        self.extractions: Dict[str, Dict[str, Any]] = {}
        self.last_extraction_notes: List[str] = []

    # ----------------------------------------------------------------- profile
    def apply_delta(self, delta: ProfileDelta) -> List[str]:
        """Apply a delta to the typed state; returns human-readable change notes."""
        notes: List[str] = []
        p = self.profile
        if delta.deposit_mxn is not None and delta.deposit_mxn != p.deposit_mxn:
            notes.append(f"deposit {self._money(p.deposit_mxn)} -> {self._money(delta.deposit_mxn)}"
                         + (" (assumed)" if delta.deposit_assumed else ""))
            p.deposit_mxn = delta.deposit_mxn
            p.deposit_source = "assumed" if delta.deposit_assumed else "conversation"
        if delta.monthly_transfers is not None and delta.monthly_transfers != p.monthly_transfers:
            notes.append(f"SPEI transfers/month {p.monthly_transfers} -> {delta.monthly_transfers}")
            p.monthly_transfers = delta.monthly_transfers
            p.sources["transfers"] = "message"
        for constraint in delta.remove_constraints:
            if constraint in p.hard_constraints:
                p.hard_constraints.remove(constraint)
                notes.append(f"removed constraint '{CONSTRAINT_DESCRIPTIONS[constraint]}'")
                p.sources["constraints"] = "message"
        for constraint in delta.add_constraints:
            if constraint not in p.hard_constraints:
                p.hard_constraints.append(constraint)
                notes.append(f"added constraint '{CONSTRAINT_DESCRIPTIONS[constraint]}'")
                p.sources["constraints"] = "message"
        for bank in delta.include_institutions:
            if bank in p.excluded_institutions:
                p.excluded_institutions.remove(bank)
                notes.append(f"{bank} considered again")
                p.sources["excluded"] = "message"
        for bank in delta.exclude_institutions:
            if bank not in p.excluded_institutions:
                p.excluded_institutions.append(bank)
                notes.append(f"excluded {bank}")
                p.sources["excluded"] = "message"
        if delta.preferred_channel and delta.preferred_channel != p.preferred_channel:
            notes.append(f"SPEI channel -> {delta.preferred_channel}")
            p.preferred_channel = delta.preferred_channel
            p.sources["channel"] = "message"
        if delta.weight_shifts and p.weights:
            shifted = {k: p.weights.get(k, 0.0) + delta.weight_shifts.get(k, 0.0) for k in PREFERENCE_KEYS}
            p.weights = normalize_weights(shifted)
            notes.append("preference weights -> [" + ", ".join(f"{p.weights[k]:.2f}" for k in PREFERENCE_KEYS) + "]")
            p.sources["weights"] = "message"
        return notes

    def missing_information(self) -> List[str]:
        return [] if self.profile.deposit_mxn is not None else ["deposit_mxn"]

    # ----------------------------------------------------------------- turns
    def _llm_extract(self, llm: Any, message: str, awaiting_deposit: bool) -> Optional[Any]:
        """Verified LLM extraction of one message, or None (no LLM / failure: the regex delta is used)."""
        from agents.extraction import extract_with_llm

        p = self.profile
        current = {"deposit_mxn": p.deposit_mxn, "hard_constraints": list(p.hard_constraints),
                   "excluded_institutions": list(p.excluded_institutions), "preferred_channel": p.preferred_channel,
                   "monthly_transfers": p.monthly_transfers}
        try:
            return extract_with_llm(llm, message, sorted({c.institution_name for c in self.catalog}),
                                    current_profile=current if self.turns else None,
                                    pending_question=DEPOSIT_QUESTION if awaiting_deposit else None)
        except Exception as exc:  # noqa: BLE001 - any extraction failure degrades to the regex parser
            logger.info("LLM parameter extraction unavailable (%s: %s); using the regex parser.", type(exc).__name__, exc)
            return None

    def plan_turn(self, message: str, llm: Any = None) -> TurnPlan:
        """Record the user message, update the typed state and decide: clarify or run the squad.

        With ``llm`` the message is parsed by the verified LLM extraction (agents.extraction) and the regex
        parser fills in what the model did not mention; without it (offline) only the regex parser is used.
        """
        message = message.strip()
        if not message:
            raise ValueError("message must be a non-empty string.")
        awaiting_deposit = self.awaiting == "deposit_mxn"
        delta = extract_delta(message, awaiting_deposit=awaiting_deposit)
        verified = self._llm_extract(llm, message, awaiting_deposit) if llm is not None else None
        self.last_extraction_notes = []
        if verified is not None:
            from agents.extraction import compare_with_regex

            regex_view = {"deposit_mxn": delta.deposit_mxn or 0.0, "deposit_stated": delta.deposit_mxn is not None,
                          "hard_constraints": delta.add_constraints}
            self.last_extraction_notes = compare_with_regex(verified, regex_view) + verified.rejected
            delta = delta_from_extraction(delta, verified)
        changes = self.apply_delta(delta)
        business = verified.business_account if verified is not None else bool(BUSINESS_ACCOUNT_PATTERN.search(message))
        if not self.profile.business_account and business:
            self.profile.business_account = True
            changes.append("business accounts (PFAE / persona moral) included")
        self._add(Turn("user", message, changes=changes))
        if verified is not None and not awaiting_deposit:
            self.extractions[message] = verified.as_dict()
            while len(self.extractions) > self.window_size:
                self.extractions.pop(next(iter(self.extractions)))

        request = message
        if awaiting_deposit:
            self.awaiting = None
            request = self.pending_message or message  # answer to a clarification: resume the original request
            self.pending_message = None

        intent = classify_intent(request, self.catalog)[0] if self.catalog else "recommendation"
        missing = [] if intent in DIRECT_INTENTS else self.missing_information()
        if missing and "deposit_mxn" not in self.clarifications_asked:
            self.awaiting, self.pending_message = "deposit_mxn", request
            self.clarifications_asked.append("deposit_mxn")
            self._add(Turn("assistant", DEPOSIT_QUESTION, kind="clarification"))
            return TurnPlan("clarify", message, changes, question=DEPOSIT_QUESTION)

        return TurnPlan("run", message, changes, query=self.compose_query(request),
                        overrides=self.overrides(), run_metadata=self.run_metadata(request))

    def record_assistant(self, content: str, kind: str = "recommendation", run_id: Optional[str] = None,
                         llm: Any = None, profile_used: Optional[str] = None) -> None:
        self._add(Turn("assistant", content, kind=kind, run_id=run_id, profile_used=profile_used), llm=llm)

    def resolve_clarification_with(self, deposit_mxn: float, source: str = "sidebar") -> TurnPlan:
        """Quick reply: continue the pending request with a deposit supplied by the UI."""
        request = self.pending_message or (self.user_messages()[-1] if self.user_messages() else "")
        self.awaiting, self.pending_message = None, None
        before = self.profile.deposit_mxn
        self.profile.deposit_mxn, self.profile.deposit_source = float(deposit_mxn), source
        change = f"deposit {self._money(before)} -> {self._money(deposit_mxn)} ({source})"
        self._add(Turn("user", f"Use {self._money(deposit_mxn)} as my average balance.", changes=[change]))
        return TurnPlan("run", request, [change], query=self.compose_query(request), overrides=self.overrides(),
                        run_metadata=self.run_metadata(request))

    # ----------------------------------------------------------------- squad inputs
    def overrides(self) -> Dict[str, Any]:
        p = self.profile
        return {
            "deposit_mxn": p.deposit_mxn if p.deposit_mxn is not None else 0.0,
            "deposit_assumed": p.deposit_source == "assumed" or p.deposit_mxn is None,
            "monthly_transfers": p.monthly_transfers,
            "hard_constraints": list(p.hard_constraints),
            "replace_hard_constraints": True,  # the typed state is authoritative (supports removals)
            "excluded_institutions": list(p.excluded_institutions),
            "preferred_channel": p.preferred_channel,
            "weights": dict(p.weights) if p.weights else None,
            "business_account": p.business_account,
        }

    def run_metadata(self, request: str) -> Dict[str, Any]:
        from agents.extraction import EXTRACTION_KEY

        metadata = {PROFILE_OVERRIDES_KEY: self.overrides(), "conversation_id": self.session_id,
                    "conversation_turn": len(self.user_messages()) + self.folded_turns,
                    "latest_user_message": request}
        if request in self.extractions:  # the Profiling Agent reuses it (no second model call)
            metadata[EXTRACTION_KEY] = self.extractions[request]
        return metadata

    def compose_query(self, request: str) -> str:
        """Latest request plus compact conversation context (summary + recent user turns)."""
        context = self.context_text(exclude_last_user=True)
        return f"{request}\n\n[Conversation context] {context}" if context else request

    def context_text(self, exclude_last_user: bool = False) -> str:
        users = self.user_messages()
        if exclude_last_user and users:
            users = users[:-1]
        parts = []
        if self.summary:
            parts.append(f"Earlier: {self.summary}")
        if users:
            parts.append("Recent requests: " + " | ".join(u[:160] for u in users[-3:]))
        last = next((t for t in reversed(self.turns) if t.role == "assistant" and t.kind == "recommendation"), None)
        if last:
            parts.append(f"Last recommendation: {last.content[:200]}")
        return " ".join(parts)

    def user_messages(self) -> List[str]:
        return [t.content for t in self.turns if t.role == "user"]

    # ----------------------------------------------------------------- window + summarizer
    def _add(self, turn: Turn, llm: Any = None) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.window_size:
            overflow = self.turns[: len(self.turns) - self.window_size]
            self.turns = self.turns[len(overflow):]
            self.summary = self.summarize(overflow, llm)
            self.folded_turns += len(overflow)

    def summarize(self, overflow: Sequence[Turn], llm: Any = None) -> str:
        if llm is not None:
            try:
                out = llm.generate(ConversationSummary, MEMORY_SUMMARIZER_SYSTEM_PROMPT,
                                   {"previous_summary": self.summary,
                                    "turns_to_fold": [{"role": t.role, "content": t.content[:400]} for t in overflow]},
                                   timeout_s=SUMMARY_LLM_TIMEOUT_S, run_name="memory_summarizer_llm")
                return out.summary.strip()[:600]
            except Exception as exc:  # noqa: BLE001 - deterministic fallback below
                logger.info("Memory summarizer LLM unavailable (%s); using deterministic summary.", exc)
        folded = [t.content[:120] for t in overflow if t.role == "user"]
        changes = [c for t in overflow for c in t.changes]
        pieces = ([self.summary] if self.summary else []) + (["User asked: " + " | ".join(folded)] if folded else [])
        if changes:
            pieces.append("Changes: " + "; ".join(changes))
        return " ".join(pieces)[-600:]

    # ----------------------------------------------------------------- serialization (MCP / UI)
    def snapshot(self) -> Dict[str, Any]:
        return {"session_id": self.session_id, "profile": asdict(self.profile), "summary": self.summary,
                "window": [asdict(t) for t in self.turns], "folded_turns": self.folded_turns,
                "awaiting": self.awaiting, "pending_message": self.pending_message}

    @staticmethod
    def _money(value: Optional[float]) -> str:
        return "unknown" if value is None else f"MXN {value:,.2f}"


def assistant_summary(result: Mapping[str, Any]) -> str:
    """One-line assistant turn describing a squad result (stored in memory and shown in the chat)."""
    draft = result.get("draft_recommendation") or {}
    status = draft.get("status")
    if status == "direct_answer":
        return str(draft.get("answer") or "")[:600]
    base, trade = draft.get("base_option"), draft.get("tradeoff_option")
    if not base:
        alternatives = draft.get("negotiation") or []
        if alternatives:
            first = alternatives[0]
            return (f"No product meets every constraint. Closest: {first['institution_name']} {first['product_name']} "
                    f"(relax {', '.join(first['relax_labels'])}).")
        return "No product meets every constraint."
    text = (f"Base Option: {base['institution_name']} {base['product_name']} (annual fees "
            f"MXN {base['total_annual_fees_mxn']:,.2f}, net MXN {base['net_annual_impact_mxn']:,.2f})")
    if trade:
        text += (f"; Trade-off: {trade['institution_name']} {trade['product_name']} "
                 f"(net MXN {trade['net_annual_impact_mxn']:,.2f})")
    upgrade = draft.get("higher_balance_option")
    if upgrade:
        text += (f"; with MXN {upgrade['unlock_balance_mxn']:,.2f}: {upgrade['institution_name']} "
                 f"{upgrade['product_name']} would pay {upgrade['applicable_yield_rate_percent']:.2f}%")
    if status != "definitive":
        text += f" [{status}]"
    if result.get("requires_human_review"):
        text += " - escalated to human review"
    return text + "."
