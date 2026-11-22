"""Consultative Advisory Agent: transparent synthesis and abstention protocol.

The recommendation skeleton (options, figures, status, warnings, disclaimers)
is assembled deterministically from evaluator output. The LLM only writes the
narrative, which the Evaluator then audits in ``review_node``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from agents.common import NodeContext, SquadResources
from agents.evaluator import HIGHER_BALANCE_STAGE
from agents.llm import LLMUnavailableError
from guardrails import AbstentionDecision, assess_abstention, benefit_conflicts, unresolved_conflicts
from prompts.advisory_prompts import (
    ADVISORY_SYSTEM_PROMPT,
    BENEFIT_UNVERIFIED_TEMPLATE,
    DETERMINISTIC_FIGURES_NOTE,
    FALLBACK_DISCLAIMER,
    FRESHNESS_DISCLAIMER_TEMPLATE,
    LEGAL_DISCLAIMER,
    MISSING_DEPOSIT_WARNING,
    NEGOTIATION_INTRO,
    TOT_TIMEOUT_NOTICE,
    AdvisoryNarrative,
)
from state import FRESHNESS_MAX_AGE_DAYS, SquadState
from tools.catalog_search import baseline_products, eligible_products, nearest_alternatives

EVALUATION_INCOMPLETE_WARNING = (
    "The full comparison could not finish within the time budget. The Base Option shown is the best zero-fee "
    "baseline among the accounts that meet your requirements; it is provisional and other options may fit better."
)

_MAX_CLAUSES_PER_OPTION = 2
_EXCERPT_CHARS = 240


def _money(value: float) -> str:
    return f"MXN {value:,.2f}"


def _rate_is_estimate(api_record: Mapping[str, Any], deposit: float) -> bool:
    """True when the tier that pays this balance was filled in by DebitMatch (not the official table)."""
    for tier in sorted(api_record.get("yield_tiers") or [], key=lambda t: t["min_balance_mxn"], reverse=True):
        upper = tier.get("max_balance_mxn")
        if tier["min_balance_mxn"] <= deposit and (upper is None or deposit <= upper):
            return bool(tier.get("estimated"))
    return False


def build_option(branch: Mapping[str, Any], api_record: Mapping[str, Any]) -> Dict[str, Any]:
    """Option card fields of one scored branch (also used by the UI for any Top-5 account)."""
    impact = branch["net_impact"]
    breakdown = impact.get("breakdown") or {}
    return {
        "yield_rate_is_estimate": _rate_is_estimate(api_record, float(impact["deposit_evaluated_mxn"])),
        "product_key": branch["product_key"],
        "institution_name": branch["institution_name"],
        "product_name": branch["product_name"],
        "composite_score": branch["score"],
        "hard_constraints_met": branch["hard_constraints_met"],
        "hard_constraint_checks": branch["hard_constraint_checks"],
        "deposit_evaluated_mxn": impact["deposit_evaluated_mxn"],
        "applicable_yield_rate_percent": impact["applicable_yield_rate_percent"],
        "gross_annual_interest_mxn": impact["gross_annual_interest_mxn"],
        "isr_withheld_mxn": breakdown.get("isr_withheld_mxn", 0.0),
        "effective_monthly_fee_mxn": impact["effective_monthly_fee_mxn"],
        "fee_waived_by_balance": impact["fee_waived_by_balance"],
        "total_annual_fees_mxn": impact["total_annual_fees_mxn"],
        "net_annual_impact_mxn": impact["net_annual_impact_mxn"],
        "transfer_channel_used": breakdown.get("transfer_channel_used"),
        "minimum_balance_for_fee_waiver_mxn": api_record.get("minimum_balance_for_fee_waiver", 0.0),
        "qualitative_perks_summary": api_record.get("qualitative_perks_summary", ""),
        "includes_free_insurance": bool(api_record.get("has_free_insurance", False)),
        "includes_cashback_or_perks": bool(api_record.get("has_cashback_or_perks", False)),
        "last_verified_date": api_record.get("last_verified_date"),
    }


def select_options(state: SquadState) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Base Option ($0 fee / risk-free) and Positive Trade-off Option (higher net return)."""
    final = sorted(
        (b for b in state.get("tot_candidate_branches") or [] if b.get("final")),
        key=lambda b: b.get("final_rank", 999),
    )
    if not final:
        return None, None
    records = {r["catalog_key"]: r for r in state.get("api_structured_data") or []}

    zero_fee = [b for b in final if b["net_impact"]["total_annual_fees_mxn"] == 0.0]
    base_branch = zero_fee[0] if zero_fee else min(
        final, key=lambda b: (b["net_impact"]["total_annual_fees_mxn"], -b["score"])
    )
    base_net = base_branch["net_impact"]["net_annual_impact_mxn"]
    better = [
        b for b in final
        if b["product_key"] != base_branch["product_key"]
        and b["net_impact"]["net_annual_impact_mxn"] > max(base_net, 0.0)
    ]
    tradeoff_branch = max(better, key=lambda b: b["net_impact"]["net_annual_impact_mxn"]) if better else None

    base = build_option(base_branch, records.get(base_branch["product_key"], {}))
    tradeoff = build_option(tradeoff_branch, records.get(tradeoff_branch["product_key"], {})) if tradeoff_branch else None
    return base, tradeoff


def top_score_explanation(state: Mapping[str, Any], base: Optional[Mapping[str, Any]],
                          tradeoff: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Why the account with the highest match score is not recommended (None when it is the Base / Trade-off).

    Deterministic, from the same branches ``select_options`` used: the Base Option is the best-ranked account
    with no fees (or the lowest fees), the Positive Trade-off is the account with the highest net return above
    the Base Option and above zero. Every figure is copied from the calculator output.
    """
    final = sorted((b for b in state.get("tot_candidate_branches") or [] if b.get("final")),
                   key=lambda b: b.get("final_rank", 999))
    if not final or base is None:
        return None
    top = final[0]
    if top["product_key"] in {o["product_key"] for o in (base, tradeoff) if o}:
        return None
    impact = top["net_impact"]
    fees, net = float(impact["total_annual_fees_mxn"]), float(impact["net_annual_impact_mxn"])
    base_fees, base_net = float(base["total_annual_fees_mxn"]), float(base["net_annual_impact_mxn"])
    name = f"{top['institution_name']} {top['product_name']}"
    base_name = f"{base['institution_name']} {base['product_name']}"
    reasons: List[str] = []
    if fees > base_fees:
        rule = "the best-ranked account with no fees" if base_fees == 0 else "the account with the lowest fees"
        reasons.append(f"it charges {_money(fees)} in fees a year, so it cannot be the Base Option ({rule}: "
                       f"{base_name}, {_money(base_fees)} a year)")
    if net <= max(base_net, 0.0):
        reasons.append(f"its net annual impact is {_money(net)}" + (" (it costs you money every year)" if net < 0 else "")
                       + f", so it is not a Positive Trade-off either: that option must earn more than the Base "
                       f"Option's {_money(base_net)} and more than zero")
    elif tradeoff is not None:
        reasons.append(f"its net annual impact of {_money(net)} is below the {_money(tradeoff['net_annual_impact_mxn'])} "
                       f"of the Positive Trade-off ({tradeoff['institution_name']} {tradeoff['product_name']}, the "
                       f"highest net return)")
    if not reasons:  # same fees and a better net than the base, yet ranked below on ties: nothing to explain
        return None
    text = (f"Why not {name}, the highest match score ({float(top['score']):.2f}/100)? It matches your stated "
            f"priorities best, but " + "; and ".join(reasons) + ".")
    return {"product_key": top["product_key"], "institution_name": top["institution_name"],
            "product_name": top["product_name"], "composite_score": top["score"], "total_annual_fees_mxn": fees,
            "net_annual_impact_mxn": net, "text": text}


def select_higher_balance_option(state: Mapping[str, Any], base: Optional[Mapping[str, Any]]
                                 ) -> Optional[Dict[str, Any]]:
    """Best account that pays a yield only from a higher balance (yield objective), for one narrative sentence.

    Only accounts with a POSITIVE net annual impact at that balance qualify (a 3 % yield eaten by fees is not
    worth mentioning). Preference: the account that would beat the Base Option's current rate with the
    smallest extra balance; otherwise the qualifying account reachable with the smallest extra balance.
    Every figure is the calculator output at the unlock balance (computed by the evaluator); this is
    information, never the recommendation.
    """
    upgrades = sorted((b for b in state.get("tot_candidate_branches") or [] if b.get("stage") == HIGHER_BALANCE_STAGE
                       and b["net_impact"]["net_annual_impact_mxn"] > 0),
                      key=lambda b: (b["strategy"]["unlock_balance_mxn"], b.get("higher_balance_rank", 999)))
    if not upgrades:
        return None
    base_rate = float((base or {}).get("applicable_yield_rate_percent") or 0.0)
    better = [b for b in upgrades if b["net_impact"]["applicable_yield_rate_percent"] > base_rate]
    branch = (better or upgrades)[0]
    impact, strategy = branch["net_impact"], branch["strategy"]
    breakdown = impact.get("breakdown") or {}
    record = next((r for r in state.get("api_structured_data") or [] if r.get("catalog_key") == branch["product_key"]), {})
    return {
        "yield_rate_is_estimate": _rate_is_estimate(record, float(strategy["unlock_balance_mxn"])),
        "product_key": branch["product_key"],
        "institution_name": branch["institution_name"],
        "product_name": branch["product_name"],
        "stated_deposit_mxn": strategy["stated_deposit_mxn"],
        "unlock_balance_mxn": strategy["unlock_balance_mxn"],
        "extra_balance_mxn": round(strategy["unlock_balance_mxn"] - strategy["stated_deposit_mxn"], 2),
        "applicable_yield_rate_percent": impact["applicable_yield_rate_percent"],
        "gross_annual_interest_mxn": impact["gross_annual_interest_mxn"],
        "isr_withheld_mxn": breakdown.get("isr_withheld_mxn", 0.0),
        "total_annual_fees_mxn": impact["total_annual_fees_mxn"],
        "net_annual_impact_mxn": impact["net_annual_impact_mxn"],
        "beats_base_option_rate": bool(better),
        "base_option_rate_percent": base_rate,
    }


def higher_balance_sentence(option: Mapping[str, Any]) -> str:
    """Deterministic sentence for the higher-balance account (figures copied from the option)."""
    name = f"{option['institution_name']} {option['product_name']}"
    comparison = (f", above the {option['base_option_rate_percent']:.2f}% the Base Option pays at your balance"
                  if option["beats_base_option_rate"] else "")
    estimate = " (an estimated rate to be confirmed with the bank)" if option.get("yield_rate_is_estimate") else ""
    return (f"With a higher average balance of {_money(option['unlock_balance_mxn'])} "
            f"({_money(option['extra_balance_mxn'])} more than your {_money(option['stated_deposit_mxn'])}), "
            f"{name} would pay {option['applicable_yield_rate_percent']:.2f}% a year{estimate}{comparison}, for a net "
            f"annual impact of {_money(option['net_annual_impact_mxn'])}; at your current balance it pays no meaningful "
            "yield.")


def ensure_higher_balance_mention(narrative: Dict[str, Any], option: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Append the higher-balance sentence unless the narrative already names that account."""
    if not option:
        return narrative
    text = " ".join(str(v) for v in narrative.values() if not isinstance(v, list)).lower()
    text += " " + " ".join(str(t) for t in narrative.get("key_tradeoffs") or []).lower()
    if option["product_name"].lower() in text:
        return narrative
    return {**narrative, "key_tradeoffs": list(narrative.get("key_tradeoffs") or []) + [higher_balance_sentence(option)]}


def baseline_option(state: Mapping[str, Any], resources: SquadResources,
                    candidate_keys: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Best zero-fee, risk-free baseline (deterministic calculator figures).

    ``candidate_keys``: restrict to products that already passed the hard constraints (evaluation
    incomplete); otherwise the eligible catalog (hard-timeout Fallback Mode, possibly before profiling).
    """
    from agents.evaluator import structured_score  # local: evaluator imports guardrails/prompts only

    profile = state.get("user_profile") or {}
    overrides = (state.get("execution_metadata") or {}).get("profile_overrides") or {}
    deposit = float(profile.get("deposit_mxn", overrides.get("deposit_mxn") or 0.0) or 0.0)
    constraints = list(profile.get("hard_constraints") or overrides.get("hard_constraints") or [])
    transfers = int(profile.get("monthly_transfers", overrides.get("monthly_transfers") or 2))
    channel = profile.get("preferred_channel")
    scope = {**overrides, **profile}  # excluded banks / business accounts, even before profiling finished
    if candidate_keys is not None:
        pool = [resources.products_by_key[k] for k in candidate_keys if k in resources.products_by_key]
    else:
        pool = eligible_products(resources.catalog, scope)
    candidates = baseline_products(pool, constraints, deposit, transfers, channel)
    if not candidates:
        return None
    product, impact = candidates[0]
    records = {r["catalog_key"]: r for r in state.get("api_structured_data") or []}
    checks = (records.get(product.catalog_key) or {}).get("hard_constraint_checks") or {}
    weights = state.get("weighted_preference_vector")
    score = structured_score(product, weights, deposit, transfers, channel)[2] if weights else 0.0
    option = {
        "product_key": product.catalog_key, "institution_name": product.institution_name,
        "product_name": product.product_name, "composite_score": score,
        "hard_constraints_met": all(checks.values()) if checks else None,
        "hard_constraint_checks": checks, "deposit_evaluated_mxn": impact["deposit_evaluated_mxn"],
        "applicable_yield_rate_percent": impact["applicable_yield_rate_percent"],
        "gross_annual_interest_mxn": impact["gross_annual_interest_mxn"],
        "isr_withheld_mxn": (impact.get("breakdown") or {}).get("isr_withheld_mxn", 0.0),
        "effective_monthly_fee_mxn": impact["effective_monthly_fee_mxn"],
        "fee_waived_by_balance": impact["fee_waived_by_balance"],
        "total_annual_fees_mxn": impact["total_annual_fees_mxn"],
        "net_annual_impact_mxn": impact["net_annual_impact_mxn"],
        "transfer_channel_used": (impact.get("breakdown") or {}).get("transfer_channel_used"),
        "minimum_balance_for_fee_waiver_mxn": product.minimum_balance_for_fee_waiver,
        "qualitative_perks_summary": product.qualitative_perks_summary,
        "includes_free_insurance": product.has_free_insurance,
        "includes_cashback_or_perks": product.has_cashback_or_perks,
        "last_verified_date": product.last_verified_date.isoformat(),
    }
    option["yield_rate_is_estimate"] = _rate_is_estimate(product.model_dump(mode="json"), impact["deposit_evaluated_mxn"])
    return option


def supporting_clauses(state: Mapping[str, Any], keys: List[str]) -> List[Dict[str, Any]]:
    """RECA excerpts retrieved for each product key (also used by the UI for any Top-5 account)."""
    clauses: List[Dict[str, Any]] = []
    for key in keys:
        matched = [c for c in state.get("rag_retrieved_chunks") or [] if key in c.get("matched_product_keys", [])]
        for chunk in matched[:_MAX_CLAUSES_PER_OPTION]:
            meta = chunk.get("metadata") or {}
            clauses.append({
                "product_key": key,
                "chunk_id": chunk["chunk_id"],
                "institution_name": meta.get("institution_name"),
                "product_id": meta.get("product_id"),
                "document_type": meta.get("document_type"),
                "last_verified_date": meta.get("last_verified_date"),
                "excerpt": " ".join(str(chunk.get("text", "")).split())[:_EXCERPT_CHARS],
            })
    return clauses


def negotiation_options(state: SquadState, resources: SquadResources) -> List[Dict[str, Any]]:
    """Criteria negotiation (Agent Design Refinement): closest products and what to relax."""
    profile = state.get("user_profile") or {}
    catalog = eligible_products(resources.catalog, profile)
    return nearest_alternatives(catalog, list(profile.get("hard_constraints") or []),
                                float(profile.get("deposit_mxn") or 0.0), int(profile.get("monthly_transfers", 2)),
                                profile.get("preferred_channel"))


def template_narrative(
    base: Optional[Mapping[str, Any]], tradeoff: Optional[Mapping[str, Any]], abstain: bool,
    negotiation: Optional[List[Mapping[str, Any]]] = None, evaluation_incomplete: bool = False,
) -> Dict[str, Any]:
    """Deterministic narrative built only from verified figures (LLM fallback).

    Statements stay scoped to the accounts actually compared; ``evaluation_incomplete`` (candidates never
    scored within the budget) replaces the trade-off verdict with an explicit "not completed" notice.
    """
    hedge = "appears to be" if abstain else "is"
    if base is None:
        alternatives = [
            f"{n['institution_name']} {n['product_name']}: relax {', '.join(n['relax_labels'])} "
            f"(total annual fees {_money(n['total_annual_fees_mxn'])}, net annual impact "
            f"{_money(n['net_annual_impact_mxn'])})."
            for n in negotiation or []
        ]
        return {
            "summary": NEGOTIATION_INTRO if alternatives else
            "No product in the current catalog satisfies all of your hard constraints.",
            "base_option_rationale": ("Relaxing the constraint(s) listed for the closest alternative would make it "
                                      "eligible; tell me which requirement you can be flexible on.") if alternatives
            else "Consider relaxing one constraint (e.g. digital opening or payroll compatibility).",
            "tradeoff_option_rationale": "No trade-off can be evaluated until at least one product is eligible.",
            "key_tradeoffs": alternatives,
        }
    base_name = f"{base['institution_name']} {base['product_name']}"
    if evaluation_incomplete:
        summary = (f"{base_name} appears to be a suitable $0-fee baseline for your requirements, with total annual "
                   f"fees of {_money(base['total_annual_fees_mxn'])} and a net annual impact of "
                   f"{_money(base['net_annual_impact_mxn'])}; the full comparison did not finish in time.")
    else:
        summary = (f"{base_name} {hedge} the strongest $0-fee option for your profile, with total annual fees of "
                   f"{_money(base['total_annual_fees_mxn'])} and a net annual impact of "
                   f"{_money(base['net_annual_impact_mxn'])}.")
    base_rationale = (f"Effective monthly fee {_money(base['effective_monthly_fee_mxn'])}"
                      + (" (waived by your balance)" if base["fee_waived_by_balance"] else "")
                      + f"; composite match score {base['composite_score']:.2f}/100.")
    tradeoffs = [f"{base_name}: total annual fees {_money(base['total_annual_fees_mxn'])}, "
                 f"net annual impact {_money(base['net_annual_impact_mxn'])}."]
    if tradeoff:
        trade_name = f"{tradeoff['institution_name']} {tradeoff['product_name']}"
        trade_rationale = (f"{trade_name} {hedge} the higher-return alternative: gross interest "
                           f"{_money(tradeoff['gross_annual_interest_mxn'])}, ISR withheld "
                           f"{_money(tradeoff['isr_withheld_mxn'])}, fees {_money(tradeoff['total_annual_fees_mxn'])}, "
                           f"net {_money(tradeoff['net_annual_impact_mxn'])} per year.")
        tradeoffs.append(f"{trade_name}: net annual impact {_money(tradeoff['net_annual_impact_mxn'])} "
                         f"in exchange for its conditions and fees.")
    elif evaluation_incomplete:
        trade_rationale = ("The trade-off comparison was not completed within the time budget, so no Positive "
                           "Trade-off Option is proposed yet; ask again to run the full comparison.")
    elif abstain:
        trade_rationale = ("Based on the currently available data, which may be outdated or inconsistent, none of the "
                           "accounts compared for your requirements appears to earn a higher positive net return than "
                           "the Base Option; this should be re-validated before acting.")
    else:
        trade_rationale = ("None of the accounts compared for your requirements earns a higher positive net return "
                           "than the Base Option, so no Positive Trade-off Option is proposed.")
    return {"summary": summary, "base_option_rationale": base_rationale,
            "tradeoff_option_rationale": trade_rationale, "key_tradeoffs": tradeoffs}


def _warnings(state: SquadState, abstention_warning: Optional[str],
              options: Tuple[Optional[Dict[str, Any]], ...] = ()) -> List[str]:
    warnings: List[str] = []
    if abstention_warning:
        warnings.append(abstention_warning)
    for option in options:
        if option and option.get("yield_rate_is_estimate"):
            warnings.append(
                f"The {option['applicable_yield_rate_percent']:.2f}% yield used for {option['institution_name']} "
                f"{option['product_name']} at MXN {option['deposit_evaluated_mxn']:,.2f} is an estimate: the official "
                "rate table publishes no rate for that balance range. Confirm it with the bank before acting.")
    for conflict in unresolved_conflicts(state):
        warnings.append(
            f"Data conflict for {conflict['product_key']} ({conflict['type']}): catalog value "
            f"{conflict['api_value']} vs. RECA clause value {conflict['document_value']} "
            f"(source: {conflict.get('source_file')})."
        )
    meta = state.get("execution_metadata") or {}
    report = meta.get("freshness_report") or {}
    if report.get("stale_products"):
        warnings.append(f"Stale catalog data (> {FRESHNESS_MAX_AGE_DAYS} days): {', '.join(report['stale_products'])}.")
    for conflict in benefit_conflicts(state):
        warnings.append(BENEFIT_UNVERIFIED_TEMPLATE.format(
            benefit=conflict.get("benefit"), product=conflict.get("product_name") or conflict["product_key"],
            date=conflict.get("last_verified_date") or "the last verification date"))
    if state.get("is_fallback_mode"):
        warnings.append(FALLBACK_DISCLAIMER)
    if meta.get("degraded_sources"):
        warnings.append("RECA policy documents could not be retrieved for this analysis "
                        f"({'; '.join(meta['degraded_sources'])}); fine-print conditions were not verified.")
    if meta.get("tot_timeout"):
        warnings.append(TOT_TIMEOUT_NOTICE)
    if "deposit_mxn" in ((state.get("user_profile") or {}).get("missing_inputs") or []):
        warnings.append(MISSING_DEPOSIT_WARNING)
    return warnings


def verified_facts(state: SquadState, base: Optional[Dict[str, Any]], tradeoff: Optional[Dict[str, Any]],
                   decision: Any, evaluation_incomplete: bool = False,
                   higher_balance: Optional[Dict[str, Any]] = None,
                   top_choice: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The single facts object given to the narrative writer, the critic and the benchmark judge."""
    profile = state.get("user_profile") or {}
    compared = sum(1 for b in state.get("tot_candidate_branches") or [] if b.get("final"))
    return {
        "base_option": base,
        "tradeoff_option": tradeoff,
        # yield objective: account paying only from a higher balance (figures at that balance; not a recommendation)
        "higher_balance_option": higher_balance,
        # highest match score that is neither option, with the deterministic reason (None when it is an option)
        "top_score_not_recommended": top_choice,
        # Scope of the comparison: claims about "other accounts" may only refer to these compared candidates.
        "accounts_compared": compared,
        "comparison_completed": not evaluation_incomplete,
        "higher_net_return_alternative_found": tradeoff is not None,
        "preferences": state.get("weighted_preference_vector"),
        "archetype": state.get("archetype_prior"),
        "hard_constraints": profile.get("hard_constraints", []),
        "deposit_mxn": profile.get("deposit_mxn", 0.0),
        "monthly_transfers": profile.get("monthly_transfers"),
        "preferred_channel": profile.get("preferred_channel"),
        "abstain": decision.abstain,
        "abstention_reasons": decision.reasons,
    }


def advisory_node(state: SquadState, ctx: NodeContext, resources: SquadResources) -> Dict[str, Any]:
    decision = assess_abstention(state)
    base, tradeoff = select_options(state)
    candidates = [r["catalog_key"] for r in state.get("api_structured_data") or []]
    evaluation_incomplete = base is None and bool(candidates)
    extra_log: List[str] = []
    if evaluation_incomplete:
        # Candidates passed the hard constraints but were never scored (evaluator skipped / cut by the budget):
        # this is NOT a "no match"; fall back to the best zero-fee baseline among those candidates.
        base = baseline_option(state, resources, candidates)
        decision = AbstentionDecision(abstain=True, reasons=list(decision.reasons) + [
            "the full comparison did not finish within the execution time budget"])
        extra_log.append(ctx.log(f"Evaluation incomplete for {len(candidates)} constraint-compliant candidate(s); "
                                 f"provisional baseline {base['product_key'] if base else None} (not a 'no match')."))
    if base is None:
        status = "no_match"
    elif decision.abstain:
        status = "abstained_provisional"
    else:
        status = "definitive"

    revision = int(state.get("revision_count", 0))
    feedback = list(state.get("review_feedback") or []) if revision > 0 else []
    log = [ctx.log(f"Drafting recommendation (revision {revision}, status={status}); base="
                   f"{base['product_key'] if base else None}, tradeoff={tradeoff['product_key'] if tradeoff else None}.")]
    log = extra_log + log
    if decision.abstain:
        log.append(ctx.log(f"Abstention protocol ACTIVE: {decision.reasons}"))

    # Criteria negotiation only for a genuine no-match (no product passed the hard constraints).
    negotiation = negotiation_options(state, resources) if base is None and not candidates else []
    higher_balance = select_higher_balance_option(state, base) if base is not None else None
    if higher_balance:
        log.append(ctx.log(f"Higher-balance yield mention: {higher_balance['product_key']} "
                           f"{higher_balance['applicable_yield_rate_percent']:.2f}% from "
                           f"MXN {higher_balance['unlock_balance_mxn']:,.2f}."))
    top_choice = top_score_explanation(state, base, tradeoff)
    if top_choice:
        log.append(ctx.log(f"Highest score not recommended: {top_choice['text']}"))
    facts = verified_facts(state, base, tradeoff, decision, evaluation_incomplete, higher_balance, top_choice)
    narrative: Dict[str, Any]
    source = "template"
    if base is not None and resources.llm is not None and not ctx.has_budget(resources.advisory_llm_min_budget_s):
        log.append(ctx.log(f"Only {ctx.remaining_s():.2f}s left; using deterministic template to stay within budget."))
    if base is not None and resources.llm is not None and ctx.has_budget(resources.advisory_llm_min_budget_s):
        try:
            output = resources.llm.generate(
                AdvisoryNarrative,
                ADVISORY_SYSTEM_PROMPT,
                {"user_query": state["user_query"], "verified_facts": facts, "reviewer_feedback": feedback},
                timeout_s=resources.llm_timeout(ctx.remaining_s(), 5.0),
                run_name="advisory_llm",
            )
            narrative, source = output.model_dump(), "llm"
        except (LLMUnavailableError, ValueError) as exc:
            log.append(ctx.log(f"LLM narrative unavailable ({exc}); using deterministic template."))
            narrative = template_narrative(base, tradeoff, decision.abstain,
                                           evaluation_incomplete=evaluation_incomplete)
    else:
        if negotiation:
            log.append(ctx.log(f"Criteria negotiation: {len(negotiation)} closest alternative(s) "
                               f"{[(n['product_key'], n['relax_constraints']) for n in negotiation]}."))
        narrative = template_narrative(base, tradeoff, decision.abstain, negotiation,
                                       evaluation_incomplete=evaluation_incomplete)

    narrative = ensure_higher_balance_mention(narrative, higher_balance)
    report = (state.get("execution_metadata") or {}).get("freshness_report") or {}
    disclaimers = ([FALLBACK_DISCLAIMER] if state.get("is_fallback_mode") else []) + [
        LEGAL_DISCLAIMER,
        FRESHNESS_DISCLAIMER_TEMPLATE.format(
            oldest_date=report.get("oldest_verified_date") or "an unknown date", max_age_days=FRESHNESS_MAX_AGE_DAYS
        ),
        DETERMINISTIC_FIGURES_NOTE,
    ]
    warnings = _warnings(state, decision.warning, (base, tradeoff))
    if evaluation_incomplete and base is not None:
        warnings.append(EVALUATION_INCOMPLETE_WARNING)
    keys = [o["product_key"] for o in (base, tradeoff) if o]
    draft = {
        "status": status,
        "base_option": base,
        "tradeoff_option": tradeoff,
        "higher_balance_option": higher_balance,
        "narrative": narrative,
        "narrative_source": source,
        "verified_facts": facts,  # same object the writer saw; the critic and the benchmark judge reuse it
        "top_score_explanation": top_choice,
        "supporting_clauses": supporting_clauses(state, keys),
        "negotiation": negotiation,
        "is_fallback_mode": bool(state.get("is_fallback_mode")),
        "warnings": warnings,
        "disclaimers": disclaimers,
        "revision": revision,
    }
    log.append(ctx.log(f"Draft ready (narrative_source={source}, warnings={len(warnings)}, "
                       f"clauses={len(draft['supporting_clauses'])})."))
    return {"draft_recommendation": draft, "guardrail_warnings": warnings, "trajectory_log": log}
