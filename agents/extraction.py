"""Hybrid parameter extraction: the LLM reads the user's message, deterministic code verifies it.

Why hybrid (Safety Guardrails plan: auditable figures, no silent assumptions):

* The LLM (profiling model) maps free text to typed parameters - deposit / balance, hard constraints and
  their removal, excluded banks, SPEI channel, transfers, objectives - plus the archetype and preference
  weights, in ONE call. It understands paraphrases the regex parser misses ("fifteen-year-old son",
  "never charges me", "5,000 Mexican pesos", "I earn 50,000 a month" = income, not a balance).
* Every extracted item must quote its ``evidence`` verbatim from the message, and every number must be
  written in that quote. :func:`verify_extraction` drops anything that fails, so no figure, constraint or
  bank can be invented: numbers still never come from the model, only from the user's own words.
* The regex parser (``agents.profiling.extract_signals``) stays as the offline / failure fallback and as a
  cross-check: disagreements are reported in the trajectory log.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from prompts.profiling_prompts import Archetype
from tools.schemas import CHANNEL_BRANCH, CHANNEL_MOBILE_APP, CHANNEL_ONLINE_WEB, CHANNEL_TELEPHONE

EXTRACTION_RUN_NAME = "profiling_extraction_llm"
EXTRACTION_KEY = "llm_extraction"  # execution_metadata key: extraction precomputed by the conversation memory
EXTRACTION_TIMEOUT_S = 4.0
BALANCE_KINDS = ("average_balance", "initial_deposit")  # amounts that are the account balance
# Amounts the model positively identified as NOT a balance. "other" is only ambiguity: it never overrides a
# balance the regex parser read from the same words (gpt-4o-mini labels "I have $50,000 MXN" as "other" in
# about half of the calls when its quote omits the verb).
NOT_BALANCE_KINDS = ("income", "periodic_deposit")

ConstraintName = Literal["zero_fee", "digital_opening", "payroll", "free_insurance", "cashback",
                         "no_minimum_balance", "minor_guardianship"]
ChannelName = Literal["Mobile Banking App", "Online Banking (Web)", "Branch / Teller", "Telephone Banking"]
assert {CHANNEL_MOBILE_APP, CHANNEL_ONLINE_WEB, CHANNEL_BRANCH, CHANNEL_TELEPHONE} == set(ChannelName.__args__)


class AmountMention(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount_mxn: float = Field(description="The amount in MXN exactly as the user wrote it (5k -> 5000).")
    kind: Literal["average_balance", "initial_deposit", "income", "periodic_deposit", "other"] = Field(
        description="average_balance / initial_deposit = money kept in the account; income = salary or earnings; "
                    "periodic_deposit = an amount deposited every month/week; other = anything else.")
    evidence: str = Field(description="Exact substring of the message that contains the amount.")


class ConstraintMention(BaseModel):
    model_config = ConfigDict(extra="forbid")
    constraint: ConstraintName
    action: Literal["require", "remove"] = Field(description="require = the user needs it; remove = drop it.")
    evidence: str = Field(description="Exact substring of the message that states the requirement or its removal.")


class BankMention(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bank: str = Field(description="Bank name as the user wrote it (mapped to the catalog by the system).")
    action: Literal["exclude", "include"]
    evidence: str = Field(description="Exact substring of the message naming the bank.")


class ProfileExtraction(BaseModel):
    """Structured output of the extraction call (validated, then verified against the message)."""

    model_config = ConfigDict(extra="forbid")
    amounts: List[AmountMention] = Field(description="Every money amount the user states (may be empty).")
    constraints: List[ConstraintMention] = Field(description="Hard requirements stated or dropped (may be empty).")
    monthly_transfers: Optional[int] = Field(description="SPEI transfers per month if stated, else null.")
    monthly_transfers_evidence: Optional[str] = Field(description="Exact substring stating the transfers, or null.")
    preferred_channel: Optional[ChannelName] = Field(description="Channel the user says they will use, or null.")
    channel_evidence: Optional[str] = Field(description="Exact substring stating the channel, or null.")
    banks: List[BankMention] = Field(description="Banks the user rejects or wants back (may be empty).")
    objectives: List[Literal["yield", "fee", "perks"]] = Field(
        description="yield = interest/returns; fee = avoiding fees/costs; perks = cashback, rewards, insurance.")
    business_account: bool = Field(description="True only if the account is for a business / PFAE / company.")
    archetype: Archetype = Field(description="Best-matching user archetype.")
    yield_importance: float = Field(description="Raw weight for interest/yield, between 0 and 1.")
    fee_importance: float = Field(description="Raw weight for avoiding fees, between 0 and 1.")
    perks_importance: float = Field(description="Raw weight for cashback, insurance and digital perks, 0 to 1.")
    # No free-text rationale: it was ~20% of the output tokens (latency) and is rebuilt deterministically from
    # the verified evidence quotes (see build_rationale), so it can only cite the user's own words.


# --------------------------------------------------------------------------- #
# Deterministic verification
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).replace("’", "'").replace("‘", "'")
    return " ".join(text.lower().split()).strip(" .,;:!?\"'")


def evidence_in_message(evidence: Optional[str], message: str) -> bool:
    quote = _norm(evidence or "")
    return len(quote) >= 2 and quote in _norm(message)


_NUMBER = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(k\b|mil\b|thousand\b|million\b|millones\b)?", re.I)
_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                       "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_UNITS.update({w: i for i, w in enumerate("cero un dos tres cuatro cinco seis siete ocho nueve diez once doce trece "
                                            "catorce quince dieciseis diecisiete dieciocho diecinueve".split())})
_UNITS.update({"uno": 1, "una": 1, "veintiun": 21, "veintiuno": 21, "veintidos": 22, "veintitres": 23,
               "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28,
               "veintinueve": 29})
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_TENS.update({w: 10 * i for i, w in enumerate("_ _ veinte treinta cuarenta cincuenta sesenta setenta ochenta "
                                                "noventa".split()) if w != "_"})
_HUNDREDS = {"cien": 100, "ciento": 100, "doscientos": 200, "trescientos": 300, "cuatrocientos": 400,
             "quinientos": 500, "seiscientos": 600, "setecientos": 700, "ochocientos": 800, "novecientos": 900}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "mil": 1_000, "millon": 1_000_000,
           "millones": 1_000_000}


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).lower()


def _word_numbers(text: str) -> List[float]:
    """Number words in ``text``: English ("twenty-five thousand", "fifteen hundred") and Spanish ("cinco mil",
    "veinte mil", "doscientos mil", "mil quinientos")."""
    values, total, current, active = [], 0, 0, False
    for token in re.findall(r"[a-z]+", _fold(text).replace("-", " ")) + ["<end>"]:
        if token in _UNITS or token in _TENS or token in _HUNDREDS:
            current += _UNITS.get(token, 0) + _TENS.get(token, 0) + _HUNDREDS.get(token, 0)
            active = True
        elif token in _SCALES and (active or token == "mil"):
            if token == "hundred":
                current *= 100
            else:
                total += (current or 1) * _SCALES[token]  # "mil" alone = 1,000
                current = 0
            active = True
        elif token in {"and", "y"} and active:
            continue
        elif active:
            values.append(float(total + current))
            total, current, active = 0, 0, False
    return values


def numbers_in(text: str) -> List[float]:
    """Every amount written in ``text`` (digits with k / mil / thousand / million, or English words)."""
    values = []
    for whole, decimals, scale in _NUMBER.findall(text):
        value = float(whole.replace(",", "") + (f".{decimals}" if decimals else ""))
        multiplier = {"k": 1_000, "mil": 1_000, "thousand": 1_000, "million": 1_000_000,
                      "millones": 1_000_000}.get(scale.lower(), 1) if scale else 1
        values.append(value * multiplier)
    return values + _word_numbers(text)


def number_in_evidence(value: float, evidence: str) -> bool:
    return any(abs(value - candidate) <= 0.5 for candidate in numbers_in(evidence))


# Topic anchors: a constraint's quote must talk about its topic (a real quote about something else - e.g.
# "nothing from Santander" offered as evidence for zero_fee - is rejected). English and Spanish stems.
CONSTRAINT_ANCHORS: Dict[str, "re.Pattern[str]"] = {
    "zero_fee": re.compile(r"fee|charg|commission|comisi|cost|cobr|pay|pag|free|grat", re.I),
    "digital_opening": re.compile(r"digital|online|on-line|app|phone|celular|movil|internet|web|en linea|remot", re.I),
    "payroll": re.compile(r"payroll|salary|paycheck|paid|wage|nomina|sueldo|pension|jubila", re.I),
    "free_insurance": re.compile(r"insur|seguro|cover|protecc", re.I),
    "cashback": re.compile(r"cash ?back|reward|point|recompens|punto|bonifica|devoluc", re.I),
    "no_minimum_balance": re.compile(r"minim|balance|saldo|keep|mantener|maintain", re.I),
    "minor_guardianship": re.compile(r"son\b|daughter|child|kid|teen|minor|year|boy|girl|nephew|niece|grand|"
                                     r"hij|menor|nin|años|anos|adolesc|sobrin|niet|tutel|i'?m \d|i am \d", re.I),
}


def constraint_anchor_ok(constraint: str, evidence: str) -> bool:
    anchor = CONSTRAINT_ANCHORS.get(constraint)
    return anchor is None or bool(anchor.search(_fold(evidence)))


@dataclass
class VerifiedExtraction:
    """What survived verification, plus an audit trail for the trajectory log."""

    deposit_mxn: Optional[float] = None  # largest verified balance amount
    amounts: List[Dict[str, Any]] = field(default_factory=list)
    require: List[str] = field(default_factory=list)
    remove: List[str] = field(default_factory=list)
    exclude_banks: List[str] = field(default_factory=list)
    include_banks: List[str] = field(default_factory=list)
    monthly_transfers: Optional[int] = None
    preferred_channel: Optional[str] = None
    objectives: List[str] = field(default_factory=list)
    business_account: bool = False
    archetype: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    rationale: str = ""
    rejected: List[str] = field(default_factory=list)
    evidence_quotes: List[str] = field(default_factory=list)  # verified non-amount quotes (rationale basis)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# Common written forms of catalog institutions that a plain prefix strip does not reach.
_EXTRA_BANK_ALIASES: Dict[str, Sequence[str]] = {
    "VE POR MAS": ("bx+", "bx mas", "banco ve por mas", "ve por mas"),
    "BBASE": ("banco base", "base"),
    "BANCO BINEO": ("bineo",),
    "BANCOPPEL": ("coppel", "bancoppel"),
    "INMOBILIARIO": ("banco inmobiliario mexicano", "bim", "inmobiliario mexicano"),
    "BaBien": ("bienestar", "banco del bienestar", "babien"),
    "BANAMEX": ("citi", "citibanamex"),
    "DONDE": ("banco donde", "donde"),
}
_BANK_NOISE = re.compile(r"^(?:el |the )?(?:grupo financiero |banco |bank |banca )|(?:,? (?:mexico|s\.?a\.?))+$")


def _bank_key(name: str) -> str:
    key = _fold(_norm(name))
    previous = None
    while previous != key:  # "Grupo Financiero Banco Azteca" -> "azteca"; "BBVA México" -> "bbva"
        previous, key = key, _BANK_NOISE.sub("", key).strip()
    return key


def _bank_index(institutions: Iterable[str]) -> Dict[str, str]:
    from memory import _BANK_ALIASES  # canonical UI names + aliases (Citibanamex, Banco del Bienestar, ...)

    names = list(institutions)
    index: Dict[str, str] = {}
    for name in names:
        index.setdefault(_fold(_norm(name)), name)
        index.setdefault(_bank_key(name), name)
    for table in (_BANK_ALIASES, _EXTRA_BANK_ALIASES):
        for canonical, aliases in table.items():
            if names and canonical not in names:
                continue
            for alias in (canonical, *aliases):
                index.setdefault(_fold(_norm(alias)), canonical)
                index.setdefault(_bank_key(alias), canonical)
    return index


def resolve_institution(name: str, index: Mapping[str, str]) -> Optional[str]:
    """Catalog institution for a bank name as the user wrote it ('Banco Azteca' -> AZTECA), or None."""
    for key in (_fold(_norm(name)), _bank_key(name)):
        if key in index:
            return index[key]
    key = _bank_key(name)
    if len(key) < 3:
        return None
    # A unique catalog name contained as whole words in the mention ("banco sabadell mexico" -> SABADELL).
    hits = {canonical for alias, canonical in index.items()
            if len(alias) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", key)}
    return hits.pop() if len(hits) == 1 else None


def build_rationale(verified: "VerifiedExtraction") -> str:
    """Deterministic profile rationale: the archetype plus the user's own verified words."""
    quotes = [a["evidence"] for a in verified.amounts] + list(verified.evidence_quotes)
    unique = list(dict.fromkeys(q.strip() for q in quotes if q and q.strip()))[:3]
    basis = "; ".join(f'"{q}"' for q in unique) if unique else "the message's stated objectives"
    objectives = ", ".join(verified.objectives) or "none stated"
    return f"{verified.archetype or 'Profile'} inferred from {basis} (objectives: {objectives})."


def verify_extraction(output: ProfileExtraction, message: str, institutions: Sequence[str] = ()) -> VerifiedExtraction:
    """Keep only items whose evidence is a verbatim quote of the message (numbers written in it)."""
    from agents.profiling import normalize_weights

    verified = VerifiedExtraction()
    for amount in output.amounts:
        if not evidence_in_message(amount.evidence, message):
            verified.rejected.append(f"amount {amount.amount_mxn:,.2f}: evidence {amount.evidence!r} is not in the message")
        elif not number_in_evidence(amount.amount_mxn, amount.evidence):
            verified.rejected.append(f"amount {amount.amount_mxn:,.2f}: not written in {amount.evidence!r}")
        elif amount.amount_mxn < 0:
            verified.rejected.append(f"amount {amount.amount_mxn:,.2f}: negative")
        else:
            verified.amounts.append({"amount_mxn": float(amount.amount_mxn), "kind": amount.kind,
                                     "evidence": amount.evidence})
    balances = [a["amount_mxn"] for a in verified.amounts if a["kind"] in BALANCE_KINDS]
    verified.deposit_mxn = max(balances) if balances else None

    for item in output.constraints:
        if not evidence_in_message(item.evidence, message):
            verified.rejected.append(f"constraint {item.constraint} ({item.action}): evidence {item.evidence!r} not in "
                                     "the message")
            continue
        if not constraint_anchor_ok(item.constraint, item.evidence):
            verified.rejected.append(f"constraint {item.constraint} ({item.action}): evidence {item.evidence!r} does "
                                     "not mention its topic")
            continue
        target = verified.require if item.action == "require" else verified.remove
        if item.constraint not in target:
            target.append(item.constraint)
        verified.evidence_quotes.append(item.evidence)
    verified.require = [c for c in verified.require if c not in verified.remove]

    banks = _bank_index(institutions)
    for item in output.banks:
        name = resolve_institution(item.bank, banks)
        if name is None:
            verified.rejected.append(f"bank {item.bank!r}: not a catalog institution")
        elif not evidence_in_message(item.evidence, message):
            verified.rejected.append(f"bank {item.bank!r}: evidence {item.evidence!r} not in the message")
        else:
            target = verified.exclude_banks if item.action == "exclude" else verified.include_banks
            if name not in target:
                target.append(name)
            verified.evidence_quotes.append(item.evidence)

    if output.monthly_transfers is not None:
        if (evidence_in_message(output.monthly_transfers_evidence, message)
                and number_in_evidence(float(output.monthly_transfers), output.monthly_transfers_evidence or "")
                and 0 <= output.monthly_transfers <= 1000):
            verified.monthly_transfers = int(output.monthly_transfers)
        else:
            verified.rejected.append(f"monthly_transfers {output.monthly_transfers}: not verifiable in the message")
    if output.preferred_channel is not None:
        if evidence_in_message(output.channel_evidence, message):
            verified.preferred_channel = output.preferred_channel
        else:
            verified.rejected.append(f"channel {output.preferred_channel}: evidence not in the message")

    verified.objectives = [o for o in ("yield", "fee", "perks") if o in output.objectives]
    verified.business_account = bool(output.business_account)
    verified.archetype = output.archetype
    verified.weights = normalize_weights({"yield_importance": output.yield_importance,
                                          "fee_importance": output.fee_importance,
                                          "perks_importance": output.perks_importance})
    verified.rationale = build_rationale(verified)
    return verified


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #
def extract_with_llm(llm: Any, message: str, institutions: Sequence[str] = (),
                     current_profile: Optional[Mapping[str, Any]] = None, pending_question: Optional[str] = None,
                     timeout_s: float = EXTRACTION_TIMEOUT_S) -> VerifiedExtraction:
    """One structured call + deterministic verification. Raises LLMUnavailableError / ValueError on failure.

    ``institutions`` is used only to VERIFY bank mentions (resolve_institution); it is not sent to the model
    (the ~25-name list cost input tokens and latency, and the catalog mapping is deterministic anyway).
    """
    from prompts.profiling_prompts import PROFILE_EXTRACTION_SYSTEM_PROMPT

    payload: Dict[str, Any] = {"user_message": message}
    if current_profile:
        payload["current_profile"] = dict(current_profile)
    if pending_question:
        payload["assistant_question_being_answered"] = pending_question
    output = llm.generate(ProfileExtraction, PROFILE_EXTRACTION_SYSTEM_PROMPT, payload, timeout_s=timeout_s,
                          run_name=EXTRACTION_RUN_NAME)
    return verify_extraction(output, message, institutions)


def compare_with_regex(verified: VerifiedExtraction, signals: Mapping[str, Any]) -> List[str]:
    """Human-readable disagreements between the verified LLM extraction and the regex parser."""
    notes = []
    regex_deposit = signals.get("deposit_mxn") if signals.get("deposit_stated") else None
    if (regex_deposit or None) != (verified.deposit_mxn or None):
        notes.append(f"deposit: regex={regex_deposit} vs llm={verified.deposit_mxn}")
    regex_constraints, llm_constraints = set(signals.get("hard_constraints") or []), set(verified.require)
    if regex_constraints != llm_constraints:
        notes.append(f"constraints: regex-only={sorted(regex_constraints - llm_constraints)}, "
                     f"llm-only={sorted(llm_constraints - regex_constraints)}")
    return notes
