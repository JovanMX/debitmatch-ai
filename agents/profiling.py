"""Profiling & Dialogue Agent: intent categorisation and preference vector.

Split of responsibilities:
* Deterministic parser -> deposit amount, hard constraints, objectives and
  usage pattern (auditable; numbers never come from the LLM).
* LLM (optional)       -> archetype + relative weights + rationale.
* Rules fallback       -> archetype priors adjusted by explicit objectives.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from agents.common import NodeContext, SquadResources
from agents.llm import LLMUnavailableError
from prompts.profiling_prompts import ARCHETYPES, PROFILING_SYSTEM_PROMPT, Archetype, ProfilingLLMOutput
from state import PREFERENCE_KEYS, SquadState
from tools.schemas import CHANNEL_BRANCH, CHANNEL_MOBILE_APP

# Reference priors for the four archetypes of "Agent Design Refinement V1.3".
ARCHETYPE_PRIORS: Dict[str, Dict[str, float]] = {
    "Flow_Manager": {"yield_importance": 0.30, "fee_importance": 0.40, "perks_importance": 0.30},
    "Strategic_Saver": {"yield_importance": 0.65, "fee_importance": 0.25, "perks_importance": 0.10},
    "Basic_Protected_User": {"yield_importance": 0.15, "fee_importance": 0.65, "perks_importance": 0.20},
    "Lifestyle_Maximizer": {"yield_importance": 0.15, "fee_importance": 0.30, "perks_importance": 0.55},
}
_PERK_CONSTRAINTS_FOR_ARCHETYPE = frozenset({"free_insurance", "cashback"})
OBJECTIVE_BOOST = 0.15
_OBJECTIVE_TO_WEIGHT = {"yield": "yield_importance", "fee": "fee_importance", "perks": "perks_importance"}

# Structured profile supplied by a UI (e.g. the Streamlit sidebar) through
# ``execution_metadata[PROFILE_OVERRIDES_KEY]``. Keys (all optional):
#   deposit_mxn, monthly_transfers, hard_constraints, weights.
PROFILE_OVERRIDES_KEY = "profile_overrides"
OBJECTIVE_WEIGHT_THRESHOLD = 0.35  # a user-set weight >= this counts as an explicit objective
_PERK_CONSTRAINTS = frozenset({"free_insurance", "cashback"})

# --------------------------------------------------------------------------- #
# Deterministic signal extraction
# --------------------------------------------------------------------------- #
_NUM = r"\d{1,3}(?:,\d{3})+|\d+"
_MONEY_PREFIXED = re.compile(rf"(?:\bmx\$|\$|\bmxn\s?)\s?({_NUM})(?:\.\d+)?\s?(k\b|mil\b)?", re.IGNORECASE)
# "5,000 pesos", "5,000 MXN", "5,000 Mexican pesos", "5 mil pesos mexicanos"
_MONEY_SUFFIXED = re.compile(rf"\b({_NUM})(?:\.\d+)?\s?(k|mil)?\s?(?:mexican\s+)?(?:mxn|pesos)\b", re.IGNORECASE)

_ZERO_FEE = re.compile(
    r"\b(?:zero|no|0|without|free of)\s+(?:monthly\s+|maintenance\s+|account\s+)*(?:fees?|commissions?|charges?)\b"
    r"|\$0\s+(?:monthly\s+)?fees?|\bfee[- ]free\b|\bzero[- ]fee\b|\bsin comisi"
    r"|\bavoid (?:the |any |paying )?(?:monthly )?(?:fees?|commissions?)"
    # verb forms: "never charges me a monthly fee", "doesn't charge any fees", "won't charge commissions"
    r"|\b(?:never|doesn't|does not|don't|do not|won't|will not)\s+charges?\s+(?:me\s+|us\s+)?(?:a\s+|any\s+)?"
    r"(?:monthly\s+|maintenance\s+|account\s+)*(?:fees?|commissions?)\b"
    # "I never want to pay a monthly fee", "I don't want to pay fees", "without paying any fee"
    r"|\b(?:never|don't want to|do not want to|refuse to|without)\s+(?:want to\s+)?pay(?:ing)?\s+(?:a\s+|any\s+)?"
    r"(?:monthly\s+|maintenance\s+|account\s+)*(?:fees?|commissions?)\b",
    re.IGNORECASE,
)
_DIGITAL = re.compile(r"\bdigital\b|\bonline\b|\bmobile\b|\bapp\b|\bfrom my phone\b|\bvirtual card\b", re.IGNORECASE)
_PAYROLL = re.compile(r"\bpayroll\b|\bn[oó]mina\b|\bsalary\b|\bpaycheck\b|\bget paid\b|\bsueldo\b", re.IGNORECASE)
_PENSION = re.compile(r"\bpensions?\b|\bpensi[oó]n\b|\bretirement income\b|\bjubilaci[oó]n\b", re.IGNORECASE)
_YIELD = re.compile(r"\byield\b|\binterest\b|\breturns?\b|\brendimiento\b|\bearn\b|\bgat\b", re.IGNORECASE)
_PERKS = re.compile(
    r"\bcashback\b|\bperks?\b|\brewards?\b|\bdiscounts?\b|\binsurance\b|\bbenefits?\b|\bbeneficios?\b|\bseguro\b",
    re.IGNORECASE,
)
_FEE = re.compile(r"\bfees?\b|\bcommissions?\b|\bcomisi[oó]n|\bcharges?\b|\bcheap\b|\blow[- ]cost\b", re.IGNORECASE)
_FREE_INSURANCE = re.compile(
    r"\b(?:free|complimentary|included|no[- ]cost)\s+(?:\w+\s+){0,2}insurance\b"
    r"|\b(?:must|need to|has to)\s+(?:have|include|offer)\s+(?:\w+\s+){0,3}insurance\b"
    r"|\binsurance\s+(?:is\s+)?(?:required|mandatory|a must)\b|\bseguro\s+(?:gratis|incluido)\b",
    re.IGNORECASE,
)
_CASHBACK_REQUIRED = re.compile(
    r"\b(?:must|need to|has to)\s+(?:have|include|offer)\s+(?:\w+\s+){0,3}cash ?back\b"
    r"|\bcash ?back\s+(?:is\s+)?(?:required|mandatory|a must)\b",
    re.IGNORECASE,
)
_NO_MINIMUM_BALANCE = re.compile(
    r"\bno minimum (?:average )?balance\b|\bwithout (?:a |any )?minimum (?:average )?balance\b"
    r"|\b(?:can't|cannot|can not|unable to|won't be able to|don't want to)\s+(?:keep|maintain|hold)\b[^.]{0,40}"
    r"\b(?:minimum|average|that)\b[^.]{0,20}\bbalance\b|\bsin saldo m[ií]nimo\b",
    re.IGNORECASE,
)
# Business banking request: unlocks business accounts (PFAE / persona moral), excluded by default.
BUSINESS_ACCOUNT_PATTERN = re.compile(
    r"\bbusiness (?:account|banking)\b|\bfor my (?:business|company|shop|store)\b|\bmy (?:business|company)\b"
    r"|\bself[- ]employed\b|\bpfae\b|\bpersona (?:f[ií]sica con actividad empresarial|moral)\b"
    r"|\bactividad empresarial\b|\bcuenta (?:empresarial|de negocios?)\b|\bmi (?:negocio|empresa)\b",
    re.IGNORECASE,
)
# Account holder is a minor. Guardianship: a parent / tutor opens and operates the child's account.
_CHILD = r"(?:son|daughter|kids?|child(?:ren)?|boy|girl|nephew|niece|grand(?:son|daughter|child(?:ren)?)|teen(?:ager)?s?)"
# Ages written in words ("fifteen-year-old"); longest alternatives first so "seventeen" is not read as "seven".
_MINOR_AGE_WORD = (r"(?:thirteen|fourteen|fifteen|sixteen|seventeen|eleven|twelve|three|seven|eight|four|five|nine|"
                   r"ten|one|two|six)")
_TENS_WORD = r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
_ADULT_AGE_WORD = rf"(?:eighteen|nineteen|{_TENS_WORD}(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?)"
_AGE = rf"(?:\d{{1,2}}|{_ADULT_AGE_WORD}|{_MINOR_AGE_WORD})"
_AGED = rf"(?:{_AGE}[- ]years?[- ]old |little |teenage |young )?"
MINOR_GUARDIANSHIP_PATTERN = re.compile(
    rf"\bfor (?:my|our) {_AGED}{_CHILD}\b"
    rf"|\b(?:my|our) {_AGED}{_CHILD}(?:'s)? (?:first )?(?:account|card)\b"
    rf"|\b(?:my|our) {_AGED}{_CHILD} (?:needs?|wants?|would like|should have)"
    r" (?:an? |their |his |her )?(?:first |own )?(?:\w+ ){0,2}(?:account|card)\b"
    r"|\b(?:kids?|child(?:ren)?)(?:'s)? (?:debit |savings |bank )?(?:account|card)\b|\bguardianship\b"
    r"|\blegal guardian\b|\bas (?:a|his|her|their) (?:parent|guardian|tutor)\b|\bunder (?:my|our) (?:care|custody)\b"
    r"|\bpara mis? (?:hij[oa]s?|nietos?|sobrin[oa]s?)\b|\bmenor (?:tutelado|bajo tutela)\b|\bcuenta infantil\b",
    re.IGNORECASE,
)
MINOR_HOLDER_PATTERN = re.compile(
    r"\bi(?:'m| am) (?:a |still a |only )?(?:minor|teen(?:ager)?|high[- ]school student)\b"
    r"|\bi(?:'m| am) (?:only )?(?:1[0-7]|[5-9])(?: years? old| yo)?\b(?!\s*(?:k\b|mil\b|%|,\d|\.\d|pesos|mxn|\d))"
    rf"|\bi(?:'m| am) (?:only )?{_MINOR_AGE_WORD}(?:[- ]years?[- ]old)?\b(?![- ](?:thousand|hundred|k\b|mil\b))"
    r"|\bunder[- ]18\b|\bunder(?:age| the age of 18)\b|\b(?:for|is|as) a minor\b|\bfor minors\b"
    r"|\bminor(?:'s)? (?:account|card)\b|\bteen(?:ager)?(?:'s)? (?:account|card)\b"
    r"|\bsoy menor de edad\b|\bmenor(?:es)? de edad\b|\btengo (?:1[0-7]|[5-9]) años\b",
    re.IGNORECASE,
)
_ADULT_AGE = re.compile(rf"\b(?:1[89]|[2-9]\d|{_ADULT_AGE_WORD})[- ]years?[- ]old\b"
                        rf"|\bi(?:'m| am) (?:1[89]|[2-9]\d|{_ADULT_AGE_WORD})\b", re.IGNORECASE)
_MINOR_AGE = re.compile(rf"\b(?:1[0-7]|[1-9]|{_MINOR_AGE_WORD})[- ]years?[- ]old\b", re.IGNORECASE)


def minor_holder_constraint(query: str) -> Optional[str]:
    """``minor_guardianship`` when the account is for a minor, otherwise None (adult holder).

    Covers both a parent / tutor opening a child's account and a minor writing for themselves: a minor's
    account is always operated under guardianship.
    """
    # "my 20-year-old son" / "my twenty-five-year-old son" is an adult account; adult ages are removed first so
    # the "five-year-old" inside "twenty-five-year-old" is not read as a minor's age
    if _ADULT_AGE.search(query) and not _MINOR_AGE.search(_ADULT_AGE.sub(" ", query)):
        return None
    if MINOR_GUARDIANSHIP_PATTERN.search(query) or MINOR_HOLDER_PATTERN.search(query):
        return "minor_guardianship"
    return None


_BRANCH = re.compile(r"\bbranch\b|\bsucursal\b|\bteller\b|\bventanilla\b", re.IGNORECASE)
_TRANSFERS = re.compile(r"\b(\d{1,3})\s+(?:spei\s+|bank\s+|interbank\s+|monthly\s+)?transfers?\b", re.IGNORECASE)

DEFAULT_MONTHLY_TRANSFERS = 2


def _amount(raw: str, multiplier: Optional[str]) -> float:
    value = float(raw.replace(",", ""))
    return value * 1000 if multiplier and multiplier.lower() in {"k", "mil"} else value


# Income / flow amounts are not the account balance: "I earn 50000 pesos a month", "my salary is $30,000".
_INCOME_BEFORE = re.compile(r"\b(?:earn(?:s|ing)?|make|making|salary|income|wage|paid|paycheck|sueldo|gano|ingreso)\b"
                            r"[^.,;$\d]{0,25}$", re.IGNORECASE)
_PER_PERIOD_AFTER = re.compile(r"^\s*(?:mexican\s+)?(?:pesos|mxn)?\s*(?:a|per|each|every|/)\s*"
                               r"(?:month|week|year|fortnight|biweekly)\b|^\s*(?:mexican\s+)?(?:pesos|mxn)?\s*"
                               r"(?:monthly|weekly|yearly|annually|al mes|mensuales)\b", re.IGNORECASE)


def parse_deposit(query: str) -> Optional[float]:
    """Largest money amount stated in the query (MXN) that is a balance, or None.

    Income or per-period amounts ("I earn 50,000 pesos a month") are not the average balance and are ignored.
    """
    amounts = []
    for pattern in (_MONEY_PREFIXED, _MONEY_SUFFIXED):
        for match in pattern.finditer(query):
            before, after = query[max(0, match.start() - 40):match.start()], query[match.end():match.end() + 25]
            if _INCOME_BEFORE.search(before) or _PER_PERIOD_AFTER.search(after):
                continue
            amounts.append(_amount(match.group(1), match.group(2)))
    positive = [a for a in amounts if a > 0]
    return max(positive) if positive else None


def extract_signals(query: str) -> Dict[str, Any]:
    """Deterministically parse hard constraints, objectives and usage pattern."""
    deposit = parse_deposit(query)
    zero_fee = bool(_ZERO_FEE.search(query))
    digital = bool(_DIGITAL.search(query))
    payroll = bool(_PAYROLL.search(query))

    hard_constraints: List[str] = []
    if zero_fee:
        hard_constraints.append("zero_fee")
    if digital:
        hard_constraints.append("digital_opening")
    if payroll:
        hard_constraints.append("payroll")
    if _FREE_INSURANCE.search(query):
        hard_constraints.append("free_insurance")
    if _CASHBACK_REQUIRED.search(query):
        hard_constraints.append("cashback")
    if _NO_MINIMUM_BALANCE.search(query):
        hard_constraints.append("no_minimum_balance")
    minor = minor_holder_constraint(query)
    if minor:
        hard_constraints.append(minor)

    objectives: List[str] = []
    if _YIELD.search(query):
        objectives.append("yield")
    if zero_fee or _FEE.search(query):
        objectives.append("fee")
    if _PERKS.search(query):
        objectives.append("perks")

    transfers_match = _TRANSFERS.search(query)
    if _BRANCH.search(query):
        channel: Optional[str] = CHANNEL_BRANCH
    elif digital:
        channel = CHANNEL_MOBILE_APP
    else:
        channel = None  # evaluator will use the cheapest listed channel

    return {
        "deposit_mxn": deposit or 0.0,
        "deposit_stated": deposit is not None,
        "hard_constraints": hard_constraints,
        "objectives": objectives,
        "monthly_transfers": int(transfers_match.group(1)) if transfers_match else DEFAULT_MONTHLY_TRANSFERS,
        "preferred_channel": channel,
        "business_account": bool(BUSINESS_ACCOUNT_PATTERN.search(query)),
        "signals": {"digital": digital, "payroll": payroll, "zero_fee": zero_fee,
                    "pension": bool(_PENSION.search(query))},
    }


def rule_based_archetype(signals: Mapping[str, Any]) -> Archetype:
    """Payroll/pension -> Flow Manager; yield -> Strategic Saver; perks -> Lifestyle Maximizer;
    otherwise the zero-cost Basic/Protected User."""
    objectives = set(signals["objectives"])
    flags = signals["signals"]
    if flags["payroll"] or flags.get("pension"):
        return "Flow_Manager"
    if "yield" in objectives:
        return "Strategic_Saver"
    if "perks" in objectives or set(signals.get("hard_constraints") or []) & _PERK_CONSTRAINTS_FOR_ARCHETYPE:
        return "Lifestyle_Maximizer"
    return "Basic_Protected_User"


def archetype_cues(signals: Mapping[str, Any]) -> List[Archetype]:
    """Archetypes the parsed signals point to (Flow / Saver / Lifestyle; Basic only as the sole fee cue)."""
    flags = signals["signals"]
    objectives = set(signals["objectives"])
    cues: List[Archetype] = []
    if flags["payroll"] or flags.get("pension"):
        cues.append("Flow_Manager")
    if "yield" in objectives:
        cues.append("Strategic_Saver")
    if "perks" in objectives or set(signals.get("hard_constraints") or []) & _PERK_CONSTRAINTS_FOR_ARCHETYPE:
        cues.append("Lifestyle_Maximizer")
    if not cues and "fee" in objectives:
        cues.append("Basic_Protected_User")
    return cues


def rules_are_conclusive(signals: Mapping[str, Any]) -> bool:
    """True when exactly one archetype cue fired: the rules already decide the profile, so no LLM call.

    Mixed cues (e.g. payroll + cashback, yield + perks) or no cue at all (a vague query) still go to
    the LLM, which weighs the competing objectives.
    """
    return len(archetype_cues(signals)) == 1


def normalize_weights(raw: Mapping[str, float]) -> Dict[str, float]:
    """Clip to >= 0 and normalise so the weights sum to exactly 1.0."""
    clipped = {k: max(0.0, float(raw.get(k, 0.0))) for k in PREFERENCE_KEYS}
    total = sum(clipped.values())
    if total <= 0:
        clipped, total = dict(ARCHETYPE_PRIORS["Flow_Manager"]), 1.0
    normalized = {k: round(v / total, 4) for k, v in clipped.items()}
    last = PREFERENCE_KEYS[-1]
    normalized[last] = round(1.0 - sum(v for k, v in normalized.items() if k != last), 4)
    return normalized


def rule_based_weights(archetype: str, objectives: List[str]) -> Dict[str, float]:
    weights = dict(ARCHETYPE_PRIORS[archetype])
    for objective in objectives:
        weights[_OBJECTIVE_TO_WEIGHT[objective]] += OBJECTIVE_BOOST
    return normalize_weights(weights)


def classify_archetype(weights: Mapping[str, float], hard_constraints: Iterable[str] = ()) -> Archetype:
    """Deterministic archetype for an explicit preference vector.

    Payroll requirement -> Flow Manager; otherwise the nearest archetype prior
    (squared Euclidean distance) of Agent Design Refinement V1.3.
    """
    constraints = set(hard_constraints)
    w = normalize_weights(weights)
    if "payroll" in constraints:
        return "Flow_Manager"

    def distance(prior: Mapping[str, float]) -> float:
        return sum((w[k] - prior[k]) ** 2 for k in PREFERENCE_KEYS)

    return min(ARCHETYPES, key=lambda name: distance(ARCHETYPE_PRIORS[name]))


def apply_profile_overrides(signals: Dict[str, Any], overrides: Mapping[str, Any]) -> List[str]:
    """Merge structured UI inputs into the parsed signals (in place).

    Structured inputs win over free-text parsing; hard constraints are the
    union of both sources. Returns human-readable notes for the trajectory log.
    """
    notes: List[str] = []
    deposit = overrides.get("deposit_mxn")
    if deposit is not None:
        deposit = float(deposit)
        if signals["deposit_stated"] and abs(signals["deposit_mxn"] - deposit) > 0.005:
            notes.append(f"Query mentions MXN {signals['deposit_mxn']:,.2f} but the structured input is "
                         f"MXN {deposit:,.2f}; using the structured input.")
        # an assumed deposit (user skipped the clarification) is still a missing input
        signals["deposit_mxn"], signals["deposit_stated"] = deposit, not overrides.get("deposit_assumed", False)

    transfers = overrides.get("monthly_transfers")
    if transfers is not None:
        signals["monthly_transfers"] = max(0, int(transfers))

    if overrides.get("replace_hard_constraints"):
        # conversation memory owns the typed state: its constraint set is authoritative
        # (lets a user REMOVE a constraint that the text or an earlier turn introduced)
        dropped = [c for c in signals["hard_constraints"] if c not in (overrides.get("hard_constraints") or [])]
        if dropped:
            notes.append(f"Constraints {dropped} found in the text are not active in the conversation profile.")
        signals["hard_constraints"] = []
    extra = [c for c in overrides.get("hard_constraints") or [] if c not in signals["hard_constraints"]]
    signals["hard_constraints"] = signals["hard_constraints"] + extra
    excluded = [str(i) for i in overrides.get("excluded_institutions") or [] if str(i).strip()]
    if excluded:
        signals["excluded_institutions"] = excluded
    if overrides.get("business_account"):
        signals["business_account"] = True
    constraints = set(signals["hard_constraints"])
    authoritative = bool(overrides.get("replace_hard_constraints"))
    signals["signals"].update({
        "zero_fee": "zero_fee" in constraints,
        "digital": ("digital_opening" in constraints) if authoritative
        else (signals["signals"]["digital"] or "digital_opening" in constraints),
        "payroll": ("payroll" in constraints) if authoritative
        else (signals["signals"]["payroll"] or "payroll" in constraints),
    })
    channel = overrides.get("preferred_channel")
    if channel:
        if signals["preferred_channel"] and signals["preferred_channel"] != channel:
            notes.append(f"Query implies channel '{signals['preferred_channel']}' but the structured input is "
                         f"'{channel}'; using the structured input.")
        signals["preferred_channel"] = str(channel)
    elif "digital_opening" in constraints and signals["preferred_channel"] is None:
        signals["preferred_channel"] = CHANNEL_MOBILE_APP

    objectives = list(signals["objectives"])
    if constraints & _PERK_CONSTRAINTS:
        objectives.append("perks")
    if "zero_fee" in constraints:
        objectives.append("fee")
    weights = overrides.get("weights")
    if weights:
        normalized = normalize_weights(weights)
        objectives += [obj for obj, key in _OBJECTIVE_TO_WEIGHT.items()
                       if normalized[key] >= OBJECTIVE_WEIGHT_THRESHOLD]
    signals["objectives"] = [o for o in ("yield", "fee", "perks") if o in objectives]
    if extra or overrides:
        notes.append(f"Structured inputs applied: {sorted(k for k, v in overrides.items() if v is not None)}.")
    return notes


def apply_extraction(signals: Dict[str, Any], verified: Any) -> List[str]:
    """Merge a verified LLM extraction into the regex signals (in place); returns log notes.

    The verified extraction wins where it speaks (every item quotes the message, every number is written in its
    quote); the regex parser fills in what the model did not mention. Structured UI overrides still win after.
    """
    from agents.extraction import NOT_BALANCE_KINDS, compare_with_regex

    notes = [f"LLM extraction (verified) disagrees with the regex parser on {d}" for d in compare_with_regex(verified,
                                                                                                           signals)]
    notes += [f"LLM extraction item discarded: {r}" for r in verified.rejected]
    non_balance = {a["amount_mxn"] for a in verified.amounts if a["kind"] in NOT_BALANCE_KINDS}
    if verified.deposit_mxn is not None:
        signals["deposit_mxn"], signals["deposit_stated"] = float(verified.deposit_mxn), True
    elif signals["deposit_stated"] and any(abs(signals["deposit_mxn"] - a) <= 0.5 for a in non_balance):
        notes.append(f"MXN {signals['deposit_mxn']:,.2f} is not a balance (income / periodic amount): not used as the "
                     "deposit.")
        signals["deposit_mxn"], signals["deposit_stated"] = 0.0, False
    signals["hard_constraints"] = list(verified.require)
    constraints = set(verified.require)
    signals["signals"].update({"zero_fee": "zero_fee" in constraints, "digital": "digital_opening" in constraints,
                               "payroll": "payroll" in constraints})
    if verified.objectives:
        signals["objectives"] = list(verified.objectives)
    if verified.monthly_transfers is not None:
        signals["monthly_transfers"] = int(verified.monthly_transfers)
    if verified.preferred_channel:
        signals["preferred_channel"] = verified.preferred_channel
    elif signals["preferred_channel"] == CHANNEL_MOBILE_APP and "digital_opening" not in constraints:
        signals["preferred_channel"] = None  # the regex inferred the app from a word the model did not read as a need
    if verified.exclude_banks:
        signals["excluded_institutions"] = list(dict.fromkeys(verified.exclude_banks))
    signals["business_account"] = bool(verified.business_account)
    signals["extraction_source"] = "llm"
    return notes


def _enforce_objective_floor(weights: Dict[str, float], objectives: List[str], floor: float = 0.2) -> Dict[str, float]:
    """Guarantee every explicitly stated objective keeps a meaningful weight."""
    adjusted = dict(weights)
    for objective in objectives:
        key = _OBJECTIVE_TO_WEIGHT[objective]
        adjusted[key] = max(adjusted[key], floor)
    return normalize_weights(adjusted)


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #
def _verified_extraction(state: SquadState, ctx: NodeContext, resources: SquadResources,
                         log: List[str]) -> Optional[Any]:
    """The conversation memory's extraction of this request, or one LLM extraction call (None = regex only)."""
    from agents.extraction import EXTRACTION_KEY, VerifiedExtraction, extract_with_llm

    precomputed = (state.get("execution_metadata") or {}).get(EXTRACTION_KEY)
    if precomputed:
        log.append(ctx.log("Parameters extracted by the LLM in the conversation memory (verified); reused, no new call."))
        return VerifiedExtraction(**precomputed)
    if resources.llm is None or not ctx.has_budget(resources.min_llm_call_budget_s):
        return None
    institutions = sorted({p.institution_name for p in resources.catalog})
    try:
        return extract_with_llm(resources.llm, state["user_query"], institutions,
                                timeout_s=resources.llm_timeout(ctx.remaining_s(), 4.0))
    except Exception as exc:  # noqa: BLE001 - any extraction failure degrades to the regex parser, never the node
        log.append(ctx.log(f"LLM parameter extraction unavailable ({type(exc).__name__}: {exc}); using the regex "
                           "parser only."))
        return None


def profiling_node(state: SquadState, ctx: NodeContext, resources: SquadResources) -> Dict[str, Any]:
    query = state["user_query"]
    signals = extract_signals(query)  # deterministic baseline / fallback
    overrides: Mapping[str, Any] = (state.get("execution_metadata") or {}).get(PROFILE_OVERRIDES_KEY) or {}
    log: List[str] = []
    verified = _verified_extraction(state, ctx, resources, log)
    if verified is not None:
        log += [ctx.log(note) for note in apply_extraction(signals, verified)]
    override_notes = apply_profile_overrides(signals, overrides) if overrides else []
    log.append(
        ctx.log(
            f"Parsed signals ({'LLM extraction verified against the message' if verified else 'regex parser'}): "
            f"deposit=MXN {signals['deposit_mxn']:,.2f} (stated={signals['deposit_stated']}), "
            f"hard_constraints={signals['hard_constraints']}, objectives={signals['objectives']}, "
            f"monthly_transfers={signals['monthly_transfers']}, channel={signals['preferred_channel']}"
        )
    )
    log += [ctx.log(note) for note in override_notes]

    archetype: Archetype
    weights: Dict[str, float]
    rationale: str
    source = "rules"
    llm_result: Optional[Tuple[Archetype, Dict[str, float], str]] = None
    user_weights = overrides.get("weights")
    if verified is not None and verified.archetype in ARCHETYPES and verified.weights:
        llm_result = (verified.archetype, dict(verified.weights), verified.rationale)  # same call as the extraction

    if user_weights:
        # Explicit user weights are authoritative: no objective floor.
        weights = normalize_weights(user_weights)
        archetype = classify_archetype(weights, signals["hard_constraints"])
        rationale = "User-defined preference weights; archetype classified by nearest archetype prior."
        source = "user"
    elif llm_result is not None:
        archetype, weights, rationale = llm_result
        source = "llm"
        if not signals["objectives"]:
            # No stated yield / fee / perks preference: the model's free weights have nothing to rest on, so the
            # archetype it chose speaks through its reference prior (same vector as the rule path for that archetype).
            prior = rule_based_weights(archetype, [])
            if prior != normalize_weights(weights):
                log.append(ctx.log(f"No objective stated: LLM weights {normalize_weights(weights)} replaced by the "
                                   f"'{archetype}' prior {prior}."))
            weights = prior
    else:
        archetype = rule_based_archetype(signals)
        weights = rule_based_weights(archetype, signals["objectives"])
        rationale = f"Rule-based: archetype prior '{archetype}' boosted for explicit objectives {signals['objectives']}."

    if source != "user":
        weights = _enforce_objective_floor(weights, signals["objectives"])
    assert archetype in ARCHETYPES  # guaranteed by the Literal type / rules

    log.append(
        ctx.log(
            f"Archetype={archetype} (source={source}); preference vector w=[yield={weights['yield_importance']:.4f}, "
            f"fee={weights['fee_importance']:.4f}, perks={weights['perks_importance']:.4f}], sum={sum(weights.values()):.4f}"
        )
    )

    profile = {key: value for key, value in signals.items() if key != "signals"}
    profile.setdefault("excluded_institutions", [])
    # Pre-generation completeness check (Safety Guardrails plan): never assume silently.
    profile["missing_inputs"] = [] if signals["deposit_stated"] else ["deposit_mxn"]
    if profile["missing_inputs"]:
        log.append(ctx.log("Completeness check: no deposit / average balance stated; figures will assume MXN 0.00 "
                           "and the user is asked to clarify."))
    profile.update({"archetype": archetype, "profiling_rationale": rationale, "profiling_source": source})
    return {
        "weighted_preference_vector": weights,
        "archetype_prior": archetype,
        "user_profile": profile,
        "trajectory_log": log,
        "execution_metadata": {"profiling_source": source},
    }
