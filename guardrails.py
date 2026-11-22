"""Safety guardrails for DebitMatch AI.

1. Freshness & Abstention Protocol  -> :func:`assess_abstention`
2. Review Feedback Loop (critic)    -> :func:`review_draft` (+ ``MAX_REVISIONS``)
3. Human-in-the-Loop escalation     -> :func:`check_human_intervention_needed`

Also hosts :func:`detect_api_rag_conflicts`, which cross-checks catalog (API)
figures against figures stated in the retrieved RECA clauses.

Everything here is deterministic so that safety decisions are reproducible
and auditable; LLM output is only ever *checked*, never trusted.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from prompts.advisory_prompts import ABSTENTION_WARNING, LEGAL_DISCLAIMER
from state import FRESHNESS_MAX_AGE_DAYS, SquadState

# --------------------------------------------------------- #
# Policy constants
# --------------------------------------------------------- #
MAX_REVISIONS = 2  # N_max for the critique loop
CONFIDENCE_THRESHOLD = 0.70
HIGH_AMOUNT_THRESHOLD_MXN = 100_000.0
ISR_RETENTION_RATE_PERCENT = 0.90  # mirrors tools.calculator.DEFAULT_ISR_PARAMETERS

_MONEY_TOLERANCE = 0.01
_PERCENT_TOLERANCE = 0.05

# --------------------------------------------------------- #
# Numeric claim extraction
# --------------------------------------------------------- #
_NUMBER = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?"
_MONEY_PREFIX = re.compile(rf"(?:MXN|\$)\s?({_NUMBER})", re.IGNORECASE)
_MONEY_SUFFIX = re.compile(rf"({_NUMBER})\s?(?:MXN|pesos)\b", re.IGNORECASE)
_PERCENT = re.compile(rf"({_NUMBER})\s?%")


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def extract_numeric_claims(text: str) -> Dict[str, Set[float]]:
    """Return the money amounts and percentages stated in ``text``."""
    money = {abs(_to_float(m)) for m in _MONEY_PREFIX.findall(text)}
    money |= {abs(_to_float(m)) for m in _MONEY_SUFFIX.findall(text)}
    percents = {abs(_to_float(p)) for p in _PERCENT.findall(text)}
    return {"money": money, "percent": percents}


def _collect_numbers(value: Any, sink: Set[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)) and math.isfinite(value):
        sink.add(abs(float(value)))
    elif isinstance(value, Mapping):
        for item in value.values():
            _collect_numbers(item, sink)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_numbers(item, sink)


def _matches(value: float, allowed: Iterable[float], tolerance: float) -> bool:
    return any(abs(value - candidate) <= tolerance for candidate in allowed)


# --------------------------------------------------------- #
# API vs RAG conflict detection
# --------------------------------------------------------- #
_AMOUNT = r"(?:MXN|\$)\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_WAIVER_PATTERNS = (
    re.compile(r"minimum (?:monthly )?average balance of " + _AMOUNT, re.IGNORECASE),
    re.compile(r"saldo promedio (?:mensual )?m[ií]nimo de " + _AMOUNT, re.IGNORECASE),
    # RECO / folleto phrasings: "Saldo promedio mínimo mensual requerido $3,000.00", "El Saldo Promedio Mensual
    # (S.P.M.) mínimo requerido es de $20,000", "saldo promedio mínimo mensual en Cuentas a la Vista ... de $300,000"
    re.compile(r"saldo promedio m[ií]nimo mensual(?: requerido)?(?: es)?(?: de)?\s*" + _AMOUNT, re.IGNORECASE),
    re.compile(r"saldo promedio mensual \(s\.p\.m\.\) m[ií]nimo requerido es de\s*" + _AMOUNT, re.IGNORECASE),
    re.compile(r"mantener un saldo promedio m[ií]nimo mensual en [^$.]{0,90}?de\s*" + _AMOUNT, re.IGNORECASE),
)
_FEE_PATTERNS = (
    re.compile(
        r"(?:below[- ]minimum[- ]balance|maintenance|management|account management) fee of " + _AMOUNT,
        re.IGNORECASE,
    ),
    re.compile(r"comisi[oó]n (?:mensual )?(?:por (?:manejo|administraci[oó]n) de cuenta )?de " + _AMOUNT, re.IGNORECASE),
    # Official RECO concept "Administración - renta" (account maintenance): "Administración - renta $139.00 Mensual",
    # "Administración - renta 1 Mensual $ 265.00", "La comisión por Administración Renta tendrá un pago fijo de $73.50".
    # Not "Administración - renta de banca por internet" / "; (banca por internet)": a separate optional service.
    re.compile(r"administraci[oó]n\s*[-–]?\s*renta(?!\s*(?:de banca|;?\s*\(banca))\s*(?:\(\d\)\s*)?(?:\d\s+)?"
               r"(?:mensual\s*)?(?:tendr[aá] un pago fijo de\s*)?" + _AMOUNT, re.IGNORECASE),
)
# "Tasa de rendimiento 7.25% anual" / "Tasa de interés anual fija ... 0.01%": the account's published rate.
_YIELD_RATE_PATTERNS = (
    re.compile(r"tasa de (?:rendimiento|inter[eé]s)\s*(?:anual\s*)?(?:fija\s*)?(\d{1,2}(?:\.\d{1,2})?)\s?%\s*anual",
               re.IGNORECASE),
)
VAT_RATE = 0.16  # catalog fees exclude VAT; documents may publish them VAT-included
_VAT_ROUNDING_TOLERANCE = 0.25  # VAT-included MXN 139.00 = 119.83 + VAT; a catalog 120.00 is the same fee
_RATE_TOLERANCE = 0.01


def _fee_matches(stated: float, catalog_costs: Iterable[float]) -> bool:
    """A document figure matches with or without VAT (catalog amounts exclude VAT)."""
    costs = list(catalog_costs)
    return (_matches(stated, costs, _MONEY_TOLERANCE)
            or _matches(stated / (1 + VAT_RATE), costs, _VAT_ROUNDING_TOLERANCE))
_NO_FEE_PATTERNS = (
    re.compile(r"do(?:es)? not charge (?:a )?monthly maintenance fee", re.IGNORECASE),
    re.compile(r"no cobra(?:n)? comisi[oó]n (?:mensual )?por (?:manejo|administraci[oó]n) de cuenta", re.IGNORECASE),
)


from tools.schemas import API_BENEFIT_DOC_TYPE  # noqa: E402  catalog perk texts in ChromaDB (same source as the API)

# Benefit statements in RECA documents (deposit insurance by IPAB is a legal protection, not a perk).
_BENEFIT_CLAIMS: Tuple[Tuple[str, str, "re.Pattern[str]"], ...] = (
    ("has_free_insurance", "complimentary insurance",
     re.compile(r"\bcomplimentary\b[^.]{0,40}\binsurance\b|\binsurance coverage\b|\bseguro (?:de vida|contra fraude) "
                r"(?:gratuito|sin costo)\b", re.IGNORECASE)),
    ("has_cashback_or_perks", "cashback / rewards",
     re.compile(r"\bcash ?back\b|\brewards? (?:and discounts )?program\b|\brewards? and discounts\b", re.IGNORECASE)),
)
_DEPOSIT_INSURANCE = re.compile(r"\bIPAB\b|\bdeposit insurance\b|\bUDIs\b", re.IGNORECASE)


def _product_specific_chunks(product: Mapping[str, Any], chunks: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    selected = []
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        if meta.get("document_type") == API_BENEFIT_DOC_TYPE:
            continue  # same source as the catalog: cannot contradict it
        if meta.get("institution_id") == product.get("institution_id") and meta.get("product_id") == product.get(
            "product_id"
        ):
            selected.append(chunk)
    return selected


def detect_api_rag_conflicts(
    products: Iterable[Mapping[str, Any]],
    chunks: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Compare catalog fee figures with figures stated in product-specific clauses.

    Only product-specific chunks are compared (institution-wide "ALL" clauses
    describe ranges of products and would produce false positives).
    Detected conflicts are *unresolved* by definition: the system never picks
    a side automatically; it abstains and escalates instead.
    """
    chunk_list = list(chunks)
    conflicts: List[Dict[str, Any]] = []

    for product in products:
        key = f"{product.get('institution_id')}_{product.get('product_id')}"
        fees = product.get("monthly_fees_detail") or []
        fee_costs = {float(f.get("cost_mxn", 0.0)) for f in fees}
        waiver_thresholds = {
            float(f.get("waiver_minimum_average_balance_mxn", 0.0))
            for f in fees
            if f.get("has_minimum_balance_exemption") and float(f.get("waiver_minimum_average_balance_mxn", 0.0)) > 0
        }
        total_monthly_fees = sum(fee_costs)

        def add(conflict_type: str, api_value: Any, doc_value: Any, chunk: Mapping[str, Any],
                severity: str = "critical", benefit: Optional[str] = None) -> None:
            meta = chunk.get("metadata") or {}
            conflicts.append(
                {
                    "product_key": key,
                    "product_name": product.get("product_name"),
                    "type": conflict_type,
                    "api_value": api_value,
                    "document_value": doc_value,
                    "benefit": benefit,
                    "chunk_id": chunk.get("chunk_id"),
                    "source_file": meta.get("source_file"),
                    "last_verified_date": meta.get("last_verified_date"),
                    "severity": severity,
                    "resolved": False,
                }
            )

        for chunk in _product_specific_chunks(product, chunk_list):
            text = str(chunk.get("text", ""))
            for pattern in _WAIVER_PATTERNS:
                for raw in pattern.findall(text):
                    stated = _to_float(raw)
                    if not waiver_thresholds:
                        add("waiver_missing_in_api", None, stated, chunk)
                    elif not _matches(stated, waiver_thresholds, _MONEY_TOLERANCE):
                        add("waiver_threshold_mismatch", sorted(waiver_thresholds), stated, chunk)
            for pattern in _FEE_PATTERNS:
                for raw in pattern.findall(text):
                    stated = _to_float(raw)
                    if not _fee_matches(stated, fee_costs):
                        add("monthly_fee_mismatch", sorted(fee_costs), stated, chunk)
            tier_rates = {float(t.get("annual_yield_rate", 0.0)) for t in product.get("yield_tiers") or []}
            for pattern in _YIELD_RATE_PATTERNS:
                for raw in pattern.findall(text):
                    stated = _to_float(raw)
                    if not _matches(stated, tier_rates or {0.0}, _RATE_TOLERANCE):
                        add("yield_rate_mismatch", sorted(tier_rates), stated, chunk)
            if total_monthly_fees > 0 and any(p.search(text) for p in _NO_FEE_PATTERNS):
                add("no_fee_claim_vs_api_fee", total_monthly_fees, 0.0, chunk)
            # Benefit present in the RECA document but missing from the live catalog (RAG design, section 5).
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if _DEPOSIT_INSURANCE.search(sentence):
                    continue
                for flag, benefit, pattern in _BENEFIT_CLAIMS:
                    if pattern.search(sentence) and not product.get(flag, False):
                        add("benefit_missing_in_api", False, True, chunk, severity="advisory", benefit=benefit)

    # Deduplicate (the same clause can be retrieved for several queries).
    unique: Dict[tuple, Dict[str, Any]] = {}
    for conflict in conflicts:
        unique.setdefault((conflict["product_key"], conflict["type"], str(conflict["document_value"]),
                           conflict.get("benefit")), conflict)
    return list(unique.values())


def unresolved_conflicts(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Critical (figure-level) conflicts: trigger abstention and HITL."""
    return [c for c in state.get("data_conflicts") or []
            if not c.get("resolved", False) and c.get("severity", "critical") == "critical"]


def benefit_conflicts(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Advisory conflicts: a benefit stated in a RECA document is absent from the catalog."""
    return [c for c in state.get("data_conflicts") or []
            if not c.get("resolved", False) and c.get("type") == "benefit_missing_in_api"]


# --------------------------------------------------------- #
# 1. Freshness & Abstention Protocol
# --------------------------------------------------------- #
@dataclass(frozen=True)
class AbstentionDecision:
    abstain: bool
    reasons: List[str] = field(default_factory=list)

    @property
    def warning(self) -> Optional[str]:
        return ABSTENTION_WARNING.format(reasons="; ".join(self.reasons)) if self.abstain else None


def assess_abstention(state: Mapping[str, Any]) -> AbstentionDecision:
    """Abstain from definitive advice on stale data, API/RAG conflicts or timeouts."""
    reasons: List[str] = []
    if not state.get("data_freshness_flag", True):
        report = (state.get("execution_metadata") or {}).get("freshness_report") or {}
        stale = (report.get("stale_products") or []) + (report.get("stale_chunks") or [])
        detail = f" ({', '.join(stale[:5])})" if stale else ""
        reasons.append(f"some product or policy data is older than {FRESHNESS_MAX_AGE_DAYS} days{detail}")
    conflicts = unresolved_conflicts(state)
    if conflicts:
        keys = sorted({c["product_key"] for c in conflicts})
        reasons.append(f"the catalog and the RECA policy clauses disagree for {', '.join(keys)}")
    if (state.get("execution_metadata") or {}).get("timed_out"):
        reasons.append("the analysis exceeded its execution time budget")
    elif (state.get("execution_metadata") or {}).get("rag_timed_out"):
        reasons.append("the RECA policy documents could not be checked within the execution time budget")
    return AbstentionDecision(abstain=bool(reasons), reasons=reasons)


# --------------------------------------------------------- #
# 2. Review Feedback Loop (deterministic critic)
# --------------------------------------------------------- #
def verified_numbers(draft: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, Set[float]]:
    """Every number the narrative is allowed to mention."""
    money: Set[float] = {0.0}
    percent: Set[float] = {0.0, ISR_RETENTION_RATE_PERCENT, float(FRESHNESS_MAX_AGE_DAYS)}

    options = [draft.get("base_option"), draft.get("tradeoff_option")]
    option_keys = {o["product_key"] for o in options if o}
    for option in options + [draft.get("higher_balance_option")]:  # higher-balance figures: calculator output too
        if option:
            _collect_numbers(option, money)
            percent.add(abs(float(option.get("applicable_yield_rate_percent", 0.0))))

    for product in state.get("api_structured_data") or []:
        if f"{product.get('institution_id')}_{product.get('product_id')}" in option_keys:
            for array in ("monthly_fees_detail", "transfer_fees_detail", "penalties_detail", "yield_tiers"):
                _collect_numbers(product.get(array) or [], money)
            for tier in product.get("yield_tiers") or []:
                percent.add(float(tier.get("annual_yield_rate", 0.0)))
            money.add(float(product.get("minimum_opening_deposit", 0.0)))

    profile = state.get("user_profile") or {}
    money.add(abs(float(profile.get("deposit_mxn") or 0.0)))
    for weight in (state.get("weighted_preference_vector") or {}).values():
        percent.update({round(weight * 100, 1), float(round(weight * 100))})
    return {"money": money, "percent": percent}


_PERK_CLAIMS: Tuple[Tuple[str, str, "re.Pattern[str]"], ...] = (
    ("has_free_insurance", "insurance coverage", re.compile(r"\binsurance\b|\bseguros?\b", re.IGNORECASE)),
    ("has_cashback_or_perks", "cashback / rewards", re.compile(r"\bcash ?back\b|\brewards?\b", re.IGNORECASE)),
)
_NEGATION = re.compile(
    r"\b(?:no|not|without|lacks?|lacking|doesn't|does not|isn't|is not|missing|neither|nor|none|absent|unavailable)\b",
    re.IGNORECASE,
)


def unsupported_perk_claims(draft: Mapping[str, Any], state: Mapping[str, Any]) -> List[str]:
    """Flag affirmative perk statements (insurance, cashback) not backed by catalog flags.

    A sentence that mentions a perk without negation must refer to at least one
    recommended option whose catalog record carries that perk flag. If the
    sentence names specific options, only those are considered.
    """
    records = {f"{p.get('institution_id')}_{p.get('product_id')}": p for p in state.get("api_structured_data") or []}
    options = [o for o in (draft.get("base_option"), draft.get("tradeoff_option")) if o]
    narrative = draft.get("narrative") or {}
    text = " ".join([str(v) for k, v in narrative.items() if k != "key_tradeoffs"]
                    + [str(t) for t in narrative.get("key_tradeoffs", [])])
    issues: List[str] = []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n", text):
        if _NEGATION.search(sentence):
            continue
        lowered = sentence.lower()
        named = [o for o in options if str(o.get("product_name", "")).lower() in lowered
                 or str(o.get("institution_name", "")).lower() in lowered]
        referenced = named or options
        for flag, perk, pattern in _PERK_CLAIMS:
            if pattern.search(sentence) and not any(records.get(o["product_key"], {}).get(flag) for o in referenced):
                issues.append(f"Narrative claims {perk} that the recommended option(s) do not offer "
                              f"according to the catalog: \"{sentence.strip()[:160]}\"")
    return issues


# Epistemic tone check (Safety Guardrails plan, "During-Generation"): absolute markers of
# overconfidence. "Risk-Free" is excluded on purpose: it is the official name of the Base Option.
_OVERCONFIDENCE = re.compile(
    r"\bguarantee[sd]?\b|\bguaranteeing\b|\b100\s?%|\bone hundred percent\b|\bdefinitely\b|\bcertainly\b"
    r"|\bwithout (?:any )?risk\b|\bno risk\b|\bzero risk\b|\byou will (?:definitely )?(?:earn|save|get)\b"
    r"|\babsolutely\b|\bundoubtedly\b|\bcannot lose\b|\bbest possible\b",
    re.IGNORECASE,
)


def overconfidence_markers(text: str) -> List[str]:
    """Distinct absolute markers found in ``text`` (lower-cased, in order of appearance)."""
    found: List[str] = []
    for match in _OVERCONFIDENCE.finditer(text or ""):
        marker = " ".join(match.group(0).lower().split())
        if marker not in found:
            found.append(marker)
    return found


# Field names (accounts_compared) and product keys (40156_20_103); no official bank or product name has "_".
_INTERNAL_IDENTIFIER = re.compile(r"\b\w+_\w+\b")


def internal_identifiers(text: str) -> List[str]:
    """Distinct identifiers with underscores found in ``text``, in order of appearance."""
    return list(dict.fromkeys(_INTERNAL_IDENTIFIER.findall(text or "")))


FEATURE_FLAGS: Dict[str, str] = {
    "free_insurance": "has_free_insurance",
    "cashback_or_perks": "has_cashback_or_perks",
    "virtual_card": "has_virtual_card",
    "mobile_app": "has_mobile_app",
    "digital_opening": "is_digital_account",
    "payroll_compatible": "is_payroll_compatible",
}


def _resolve_option(name: str, draft: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    base, trade = draft.get("base_option"), draft.get("tradeoff_option")
    lowered = name.strip().lower()
    if "base option" in lowered or lowered == "base":
        return base
    if "trade-off" in lowered or "tradeoff" in lowered or "trade off" in lowered:
        return trade
    for option in (o for o in (base, trade) if o):
        product, bank = str(option.get("product_name", "")).lower(), str(option.get("institution_name", "")).lower()
        if (product and (product in lowered or lowered in product)) or (bank and bank in lowered):
            return option
    return None


def verify_feature_claims(
    claims: Iterable[Mapping[str, Any]], draft: Mapping[str, Any], state: Mapping[str, Any]
) -> List[str]:
    """Deterministically verify LLM-extracted feature claims against catalog flags.

    Only claims whose evidence_quote appears verbatim in the narrative and that
    resolve to a recommended option are checked (grounded extraction).
    """
    records = {f"{p.get('institution_id')}_{p.get('product_id')}": p for p in state.get("api_structured_data") or []}
    narrative = draft.get("narrative") or {}
    text = " ".join([str(v) for k, v in narrative.items() if k != "key_tradeoffs"]
                    + [str(t) for t in narrative.get("key_tradeoffs", [])]).lower()
    issues: List[str] = []
    for claim in claims:
        quote = str(claim.get("evidence_quote", "")).strip()
        flag = FEATURE_FLAGS.get(str(claim.get("feature")))
        if not quote or quote.lower() not in text or flag is None:
            continue
        if claim.get("option") in ("base_option", "tradeoff_option"):
            option = draft.get(str(claim["option"]))
        else:  # free-text product reference
            option = _resolve_option(str(claim.get("product_name", "")), draft)
        record = records.get(option.get("product_key")) if option else None
        if record is None:
            continue
        actual = bool(record.get(flag))
        if actual != bool(claim.get("asserted_available")):
            issues.append(
                f"Feature claim contradicts the catalog: narrative says {option.get('product_name')} "
                f"{'has' if claim.get('asserted_available') else 'lacks'} {claim.get('feature')}, catalog says "
                f"{'it does' if actual else 'it does not'} [quote: '{quote}']"
            )
    return issues


def review_draft(draft: Optional[Mapping[str, Any]], state: Mapping[str, Any]) -> List[str]:
    """Deterministic critique: math hallucinations and hard-constraint violations.

    Returns a list of human-readable issues; an empty list means approved.
    """
    if not draft:
        return ["No draft recommendation was produced."]

    issues: List[str] = []
    catalog_keys = {
        f"{p.get('institution_id')}_{p.get('product_id')}" for p in state.get("api_structured_data") or []
    }

    for label in ("base_option", "tradeoff_option"):
        option = draft.get(label)
        if not option:
            continue
        if option.get("product_key") not in catalog_keys:
            issues.append(f"{label} '{option.get('product_key')}' is not among the verified candidate products.")
        if not option.get("hard_constraints_met", False):
            issues.append(f"{label} '{option.get('product_name')}' violates at least one hard constraint.")

    narrative = draft.get("narrative") or {}
    narrative_text = " ".join(
        [str(v) for k, v in narrative.items() if k != "key_tradeoffs"] + [str(t) for t in narrative.get("key_tradeoffs", [])]
    )
    claims = extract_numeric_claims(narrative_text)
    allowed = verified_numbers(draft, state)
    bad_money = sorted(v for v in claims["money"] if not _matches(v, allowed["money"], _MONEY_TOLERANCE))
    bad_percent = sorted(v for v in claims["percent"] if not _matches(v, allowed["percent"], _PERCENT_TOLERANCE))
    if bad_money:
        issues.append(f"Narrative states MXN amounts not produced by the calculator: {bad_money}.")
    if bad_percent:
        issues.append(f"Narrative states percentages not present in verified data: {bad_percent}.")
    issues.extend(unsupported_perk_claims(draft, state))
    markers = overconfidence_markers(narrative_text)
    if markers:
        issues.append(f"Epistemic overconfidence: absolute markers {markers} must be rewritten with calibrated wording.")
    identifiers = internal_identifiers(narrative_text)
    if identifiers:
        issues.append(f"Narrative exposes internal identifiers {identifiers}; describe them in plain words and use "
                      "the official bank and product names.")

    if LEGAL_DISCLAIMER not in (draft.get("disclaimers") or []):
        issues.append("Mandatory legal disclaimer is missing.")
    if assess_abstention(state).abstain and draft.get("status") == "definitive":
        issues.append("Draft is marked definitive although the abstention protocol is active.")
    return issues


# --------------------------------------------------------- #
# 3. Human-in-the-Loop escalation
# --------------------------------------------------------- #
def human_review_reasons(state: Mapping[str, Any]) -> List[str]:
    """Explain every HITL criterion that is currently met."""
    reasons: List[str] = []
    confidence = float(state.get("confidence_score", 0.0) or 0.0)
    if confidence < CONFIDENCE_THRESHOLD:
        reasons.append(f"low confidence ({confidence:.2f} < {CONFIDENCE_THRESHOLD:.2f})")

    deposit = float((state.get("user_profile") or {}).get("deposit_mxn") or 0.0)
    if deposit > HIGH_AMOUNT_THRESHOLD_MXN:
        reasons.append(f"high financial amount (MXN {deposit:,.2f} > MXN {HIGH_AMOUNT_THRESHOLD_MXN:,.2f})")

    if unresolved_conflicts(state):
        reasons.append("unresolved regulatory conflict between catalog (API) data and RECA policy clauses")

    if int(state.get("revision_count", 0) or 0) >= MAX_REVISIONS and not state.get("review_approved", False):
        reasons.append(f"critique loop exhausted ({MAX_REVISIONS} revisions) without approval")

    if (state.get("execution_metadata") or {}).get("timed_out"):
        reasons.append("execution time budget exceeded")
    elif (state.get("execution_metadata") or {}).get("rag_timed_out"):
        reasons.append("RECA document retrieval exceeded its time budget (fine print not verified)")
    return reasons


def check_human_intervention_needed(state: SquadState) -> bool:
    """True when at least one HITL escalation criterion is met."""
    return bool(human_review_reasons(state))
