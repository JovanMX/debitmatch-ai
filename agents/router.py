"""Intent routing layer (RAG and Retrieval Design Integration V1, section 2).

    * Structured numerical question about a named product ("What is the monthly fee of LikeU Pro?")
      -> answered directly from the catalog with the deterministic calculator (exact, low latency).
    * Semantic benefit question about a named product ("Does Cuenta Maestra include insurance?")
      -> catalog facts + MMR retrieval of that product's RECA / benefit chunks.
    * Anything else (recommendations, comparisons, hybrid needs) -> full agent squad.

The router is deterministic and conservative: it only short-circuits when a
specific product is named AND the user asks a factual question without asking
for a recommendation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Literal, Mapping, Sequence

from agents.common import NodeContext, SquadResources
from agents.profiling import extract_signals
from guardrails import API_BENEFIT_DOC_TYPE, assess_abstention
from prompts.advisory_prompts import (
    DETERMINISTIC_FIGURES_NOTE,
    FRESHNESS_DISCLAIMER_TEMPLATE,
    LEGAL_DISCLAIMER,
    MISSING_DEPOSIT_WARNING,
)
from state import FRESHNESS_MAX_AGE_DAYS, SquadState, compute_data_freshness
from tools.catalog_search import product_net_impact
from tools.schemas import SPEI_TRANSFER, BankProduct

Intent = Literal["product_fact", "benefit_lookup", "recommendation"]
DIRECT_INTENTS = frozenset({"product_fact", "benefit_lookup"})

_GENERIC_TOKENS = frozenset({
    "cuenta", "de", "para", "el", "la", "debito", "producto", "basico", "basica", "general", "nomina",
    "super", "hsbc", "banamex", "santander", "babien", "one", "bis",
    # Words that also appear in ordinary English / Spanish requests ("a digital account", "plus").
    "digital", "account", "card", "tarjeta", "plus", "total", "con", "sin", "y", "del", "en", "mi", "nivel",
    "publico", "ahorro", "inversion", "full", "link", "smart", "trade", "exchange", "multiple", "flexible",
    "lite", "deposito", "cheques", "clasica", "valor", "verde", "relacion", "clic", "premier",
})
# A token shared by this many product names is not distinctive (e.g. "efectiva", "now", "naranja").
_SHARED_TOKEN_MIN_PRODUCTS = 3
_QUESTION = re.compile(r"\?|\b(?:what|how much|how many|does|do|is|are|tell me|explain)\b", re.IGNORECASE)
_RECOMMEND = re.compile(
    r"\b(?:recommend|best|better|which account|what account|suggest|i want|i need|looking for|compare|versus|vs\.?"
    r"|should i|alternatives?|options?)\b",
    re.IGNORECASE,
)
_BENEFIT_TOPIC = re.compile(r"\b(?:benefits?|insurance|cash ?back|rewards?|perks?|discounts?|coverage|seguros?)\b",
                            re.IGNORECASE)
_FACT_TOPIC = re.compile(
    r"\b(?:fees?|commissions?|costs?|charges?|minimum|balance|rate|yield|interest|gat|spei|transfers?|opening)\b",
    re.IGNORECASE,
)
_BENEFIT_DOC_TYPES = ("terminos_y_condiciones_producto", "bases_promociones_cashback", "poliza_seguros",
                      API_BENEFIT_DOC_TYPE)
_MAX_CLAUSES = 4


def _normalize(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", stripped.lower()).split())


def _catalog_generic_tokens(catalog: Sequence[BankProduct]) -> frozenset:
    """Static generic words + institution-name tokens + tokens shared by many product names."""
    institution_tokens = {t for p in catalog for t in _normalize(p.institution_name).split()}
    frequency: Dict[str, int] = {}
    for product in catalog:
        for token in set(_normalize(product.product_name).split()):
            frequency[token] = frequency.get(token, 0) + 1
    shared = {t for t, n in frequency.items() if n >= _SHARED_TOKEN_MIN_PRODUCTS}
    return _GENERIC_TOKENS | institution_tokens | shared


_SHORT_NAME_MAX_CHARS = 4  # one-word names this short are everyday words ("NOW": "from now on ...")


def _aliases(product: BankProduct, generic: frozenset = _GENERIC_TOKENS) -> List[str]:
    full = _normalize(product.product_name)
    if " " not in full and len(full) <= _SHORT_NAME_MAX_CHARS:  # only count it when qualified
        institution = _normalize(product.institution_name)
        return sorted({f"{institution} {full}", f"{full} {institution}", f"cuenta {full}", f"tarjeta {full}",
                       f"{full} account", f"{full} card", f"{full} debit"}, key=len, reverse=True)
    aliases = {full}
    distinctive = [t for t in full.split() if t not in generic]
    if distinctive and len(" ".join(distinctive)) >= 4:
        aliases.add(" ".join(distinctive))
    return sorted(aliases, key=len, reverse=True)


def mentioned_products(query: str, catalog: Sequence[BankProduct]) -> List[BankProduct]:
    """Products named in the query (longest alias wins; 'Cuenta Inmediata Bis' beats 'Cuenta Inmediata')."""
    text = f" {_normalize(query)} "
    generic = _catalog_generic_tokens(catalog)
    hits: List[tuple[str, BankProduct]] = []
    for product in catalog:
        for alias in _aliases(product, generic):
            if f" {alias} " in text:
                hits.append((alias, product))
                break
    selected = []
    for alias, product in sorted(hits, key=lambda h: -len(h[0])):
        if not any(alias != other and alias in other for other, _ in hits):
            selected.append(product)
    return selected


def classify_intent(query: str, catalog: Sequence[BankProduct]) -> tuple[Intent, List[BankProduct]]:
    products = mentioned_products(query, catalog)
    if not products or _RECOMMEND.search(query) or not _QUESTION.search(query):
        return "recommendation", products
    if _BENEFIT_TOPIC.search(query):
        return "benefit_lookup", products
    if _FACT_TOPIC.search(query):
        return "product_fact", products
    return "recommendation", products


def _money(value: float) -> str:
    return f"MXN {value:,.2f}"


def _product_facts(product: BankProduct, deposit: float, transfers: int, channel: Any) -> Dict[str, Any]:
    impact = product_net_impact(product, deposit, transfers, channel)
    spei = [{"channel": t.channel, "cost_per_transaction_mxn": t.cost_per_transaction_mxn,
             "free_transactions_per_month": t.free_transactions_per_month}
            for t in product.transfer_fees_detail if t.transfer_type == SPEI_TRANSFER]
    fees = [{"service_concept": f.service_concept, "cost_mxn": f.cost_mxn,
             "waiver_minimum_average_balance_mxn": f.waiver_minimum_average_balance_mxn if f.has_minimum_balance_exemption
             else None} for f in product.monthly_fees_detail]
    return {
        "product_key": product.catalog_key,
        "institution_name": product.institution_name,
        "product_name": product.product_name,
        "deposit_evaluated_mxn": impact["deposit_evaluated_mxn"],
        "monthly_transfers_evaluated": transfers,
        "minimum_opening_deposit_mxn": product.minimum_opening_deposit,
        "monthly_fees_detail": fees,
        "spei_fees_by_channel": spei,
        "annual_yield_rate_percent": product.annual_yield_rate,
        "effective_monthly_fee_mxn": impact["effective_monthly_fee_mxn"],
        "fee_waived_by_balance": impact["fee_waived_by_balance"],
        "total_annual_fees_mxn": impact["total_annual_fees_mxn"],
        "gross_annual_interest_mxn": impact["gross_annual_interest_mxn"],
        "net_annual_impact_mxn": impact["net_annual_impact_mxn"],
        "transfer_channel_used": (impact.get("breakdown") or {}).get("transfer_channel_used"),
        "perks": {"free_insurance": product.has_free_insurance, "cashback_or_perks": product.has_cashback_or_perks,
                  "virtual_card": product.has_virtual_card, "mobile_app": product.has_mobile_app,
                  "digital_opening": product.is_digital_account, "payroll_compatible": product.is_payroll_compatible,
                  "minor_under_guardianship": product.for_minor_under_guardianship},
        "qualitative_perks_summary": product.qualitative_perks_summary,
        "last_verified_date": product.last_verified_date.isoformat(),
    }


def _fact_sentence(facts: Mapping[str, Any]) -> str:
    name = f"{facts['product_name']} ({facts['institution_name']})"
    fee_parts = []
    for fee in facts["monthly_fees_detail"]:
        if fee["cost_mxn"] > 0:
            waiver = (f", waived at an average balance of at least {_money(fee['waiver_minimum_average_balance_mxn'])}"
                      if fee["waiver_minimum_average_balance_mxn"] else "")
            fee_parts.append(f"{fee['service_concept']} {_money(fee['cost_mxn'])}/month{waiver}")
    fees = "; ".join(fee_parts) if fee_parts else "no monthly maintenance fees"
    spei = ", ".join(f"{s['channel']} {_money(s['cost_per_transaction_mxn'])}" for s in facts["spei_fees_by_channel"])
    return (f"{name}: {fees}. At an average balance of {_money(facts['deposit_evaluated_mxn'])} the effective monthly "
            f"fee is {_money(facts['effective_monthly_fee_mxn'])}"
            + (" (waived by balance)" if facts["fee_waived_by_balance"] else "")
            + f". SPEI transfer fees: {spei or 'not listed'}. Estimated total annual fees for "
            f"{facts['monthly_transfers_evaluated']} transfer(s)/month: {_money(facts['total_annual_fees_mxn'])}; "
            f"annual yield rate {facts['annual_yield_rate_percent']:.2f}%."
            + (" Account of a minor operated by a parent or legal guardian."
               if facts["perks"].get("minor_under_guardianship") else ""))


def _benefit_sentence(facts: Mapping[str, Any]) -> str:
    perks = facts["perks"]
    listed = [label for key, label in (("free_insurance", "complimentary insurance"),
                                       ("cashback_or_perks", "cashback / rewards"),
                                       ("virtual_card", "virtual card"), ("mobile_app", "mobile app"),
                                       ("digital_opening", "digital opening"),
                                       ("payroll_compatible", "payroll compatibility")) if perks.get(key)]
    return (f"{facts['product_name']} ({facts['institution_name']}) lists: "
            f"{', '.join(listed) if listed else 'no additional benefits in the catalog'} "
            f"(benefits available, details vary). Catalog summary: {facts['qualitative_perks_summary']}")


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def intent_router_node(state: SquadState, ctx: NodeContext, resources: SquadResources) -> Dict[str, Any]:
    # classify the latest user message only (conversation context must not trigger a direct answer)
    latest = (state.get("execution_metadata") or {}).get("latest_user_message") or state["user_query"]
    intent, products = classify_intent(latest, resources.catalog)
    keys = [p.catalog_key for p in products]
    route = "direct_answer_node" if intent in DIRECT_INTENTS else "profiling_node"
    return {
        "trajectory_log": [ctx.log(f"Intent={intent}; products named={keys or 'none'} -> {route}.")],
        "execution_metadata": {"intent": intent, "intent_products": keys, "intent_route": route},
    }


def intent_edge(state: SquadState) -> str:
    return (state.get("execution_metadata") or {}).get("intent_route") or "profiling_node"


def direct_answer_node(state: SquadState, ctx: NodeContext, resources: SquadResources) -> Dict[str, Any]:
    meta = state.get("execution_metadata") or {}
    intent = meta.get("intent", "product_fact")
    overrides = meta.get("profile_overrides") or {}
    signals = extract_signals(meta.get("latest_user_message") or state["user_query"])
    deposit = float(overrides.get("deposit_mxn") if overrides.get("deposit_mxn") is not None else signals["deposit_mxn"])
    deposit_stated = signals["deposit_stated"] or overrides.get("deposit_mxn") is not None
    transfers = int(overrides.get("monthly_transfers") if overrides.get("monthly_transfers") is not None
                    else signals["monthly_transfers"])
    channel = overrides.get("preferred_channel") or signals["preferred_channel"]
    products = [resources.products_by_key[k] for k in meta.get("intent_products") or [] if k in resources.products_by_key]

    facts = [_product_facts(p, deposit, transfers, channel) for p in products]
    chunks: List[Dict[str, Any]] = []
    if intent == "benefit_lookup" and resources.rag_engine is not None and ctx.has_budget(0.5):
        for product in products:
            for chunk in resources.rag_engine.query_mmr(state["user_query"], institution_id=product.institution_id,
                                                        product_id=product.product_id,
                                                        document_types=list(_BENEFIT_DOC_TYPES), k=_MAX_CLAUSES):
                chunks.append({**chunk, "matched_product_keys": [product.catalog_key]})

    records = [{**p.model_dump(mode="json"), "catalog_key": p.catalog_key} for p in products]
    freshness = compute_data_freshness(records, chunks, reference_date=resources.effective_reference_date())
    probe = {"data_freshness_flag": freshness["data_freshness_flag"],
             "execution_metadata": {**meta, "freshness_report": dict(freshness)}}
    decision = assess_abstention(probe)
    warnings = [decision.warning] if decision.warning else []
    if not deposit_stated and intent == "product_fact":
        warnings.append(MISSING_DEPOSIT_WARNING)

    sentences = [(_benefit_sentence if intent == "benefit_lookup" else _fact_sentence)(f) for f in facts]
    answer = " ".join(sentences)
    clauses = [{"product_key": c["matched_product_keys"][0], "chunk_id": c["chunk_id"],
                "document_type": (c.get("metadata") or {}).get("document_type"),
                "last_verified_date": (c.get("metadata") or {}).get("last_verified_date"),
                "excerpt": " ".join(str(c.get("text", "")).split())[:240]} for c in chunks]
    draft = {
        "status": "direct_answer",
        "intent": intent,
        "answer": answer,
        "facts": facts,
        "base_option": None,
        "tradeoff_option": None,
        "narrative": {"summary": answer, "base_option_rationale": "", "tradeoff_option_rationale": "",
                      "key_tradeoffs": []},
        "narrative_source": "deterministic_direct_answer",
        "supporting_clauses": clauses,
        "negotiation": [],
        "is_fallback_mode": False,
        "warnings": warnings,
        "disclaimers": [LEGAL_DISCLAIMER,
                        FRESHNESS_DISCLAIMER_TEMPLATE.format(
                            oldest_date=freshness["oldest_verified_date"] or "an unknown date",
                            max_age_days=FRESHNESS_MAX_AGE_DAYS),
                        DETERMINISTIC_FIGURES_NOTE],
        "requires_human_review": False,
        "human_review_reasons": [],
    }
    log = [ctx.log(f"Direct {intent} answer for {[f['product_key'] for f in facts]} computed with "
                   f"calculate_net_impact (deposit=MXN {deposit:,.2f}, transfers/month={transfers}); "
                   f"{len(clauses)} RECA clause(s); fresh={freshness['data_freshness_flag']}.")]
    return {
        "user_profile": {"deposit_mxn": deposit, "deposit_stated": deposit_stated, "monthly_transfers": transfers,
                         "preferred_channel": channel, "hard_constraints": [], "objectives": [],
                         "missing_inputs": [] if deposit_stated else ["deposit_mxn"]},
        "api_structured_data": [{**r, "hard_constraint_checks": {}} for r in records],
        "rag_retrieved_chunks": chunks,
        "data_freshness_flag": freshness["data_freshness_flag"],
        "draft_recommendation": draft,
        "guardrail_warnings": warnings,
        "review_approved": True,  # deterministic catalog facts; no generated narrative to critique
        "requires_human_review": False,
        "confidence_score": 1.0 if freshness["data_freshness_flag"] else 0.5,
        "trajectory_log": log,
        "execution_metadata": {"freshness_report": dict(freshness), "route": "direct_answer"},
    }
