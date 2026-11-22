"""Evaluator & Trade-Off Reconciliation Agent.

* Every MXN figure comes from ``calculate_net_impact`` (via
  ``tools.catalog_search.product_net_impact``); the LLM never computes money.
* Composite Match Score (0-100) = 100 * (w . NormalizedMetrics) - net-cost penalty (archetype-scaled,
  only when the account loses money over the year).
* ``evaluator_linear``   : one deterministic scoring pass (simple queries).
* ``evaluator_tot_beam`` : Tree-of-Thought beam search, b=5, D=3, with early
  stopping when a branch meets 100 % of hard constraints and scores > 85.
      D1 "product selection"   - one thought per candidate product
      D2 "usage strategy"      - expand each product into transfer-channel plans
      D3 "robustness check"    - balance stress test + evidence/freshness audit
* ``review_node``        : critic mode over the Advisory draft (guardrails).
* ``evaluator_node``     : revision step of the critique loop (increments N).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.common import NodeContext, SquadResources
from agents.llm import LLMUnavailableError
from guardrails import (
    MAX_REVISIONS,
    check_human_intervention_needed,
    human_review_reasons,
    verify_feature_claims,
    review_draft,
    benefit_conflicts,
    unresolved_conflicts,
)
from prompts.advisory_prompts import HITL_NOTICE
from prompts.evaluator_prompts import EVALUATOR_CRITIC_SYSTEM_PROMPT, CriticVerdict
from state import SquadState
from tools.catalog_search import MEANINGFUL_YIELD_PERCENT, evaluate_hard_constraints, product_net_impact, yield_unlock
from tools.schemas import SPEI_TRANSFER, BankProduct

# --- Scoring constants (absolute anchors keep scores comparable across queries) ---
YIELD_REFERENCE_RATE_PERCENT = 10.0  # net after-tax yield that earns a full yield metric
FEE_REFERENCE_ANNUAL_MXN = 2_400.0  # annual fees (MXN 200/month) that zero the fee metric
PERK_FLAG_WEIGHTS: Dict[str, float] = {
    "has_cashback_or_perks": 0.40,
    "has_free_insurance": 0.30,
    "has_virtual_card": 0.15,
    "has_mobile_app": 0.15,
}
# --- Net-cost penalty (design decision) ---
# An account that LOSES money over the year (net annual impact < 0 after yield, fees and ISR) is pushed down by
# up to NET_COST_MAX_PENALTY[archetype] points, linearly until the loss reaches NET_COST_REFERENCE_ANNUAL_MXN.
# Fees that the yield pays for are not penalised (the "positive trade-off" of Agent Design V1.3 stays intact).
# The archetype is the one nearest to the preference vector, so explicit user weights are covered too.
NET_COST_REFERENCE_ANNUAL_MXN = 600.0  # MXN 50/month of net loss reaches the full penalty
NET_COST_MAX_PENALTY: Dict[str, float] = {
    "Basic_Protected_User": 25.0,  # wants a zero-cost account: a net loss contradicts the goal
    "Strategic_Saver": 20.0,       # a yield that does not cover the fees defeats the purpose
    "Flow_Manager": 15.0,          # cost matters, so does the service around the payroll
    "Lifestyle_Maximizer": 5.0,    # may pay for perks on purpose
}

# --- Tree-of-Thought constants ---
BEAM_WIDTH = 5
MAX_DEPTH = 3
EARLY_STOP_SCORE = 85.0
CHANNEL_MISMATCH_PENALTY = 5.0
STRESS_BALANCE_FACTOR = 0.8
STRESS_WEIGHT = 0.3
EVIDENCE_BONUS = 2.0
STALE_OR_CONFLICT_PENALTY = 10.0  # applied at D3 to API/RAG-conflicted products
STALE_DOWNWEIGHT_FACTOR = 0.8  # ToT plan: branches with data > 30 days are downweighted by 20%
TIE_MARGIN = 5.0  # ToT plan: branches within 5 points are reranked by strict constraint adherence


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------- #
# Metrics & composite score
# ---------------------------------------------------------------------- #
def compute_metrics(product: BankProduct, impact: Mapping[str, Any], deposit: float) -> Dict[str, float]:
    """Normalise calculator output into [0, 1] metrics."""
    breakdown = impact.get("breakdown") or {}
    net_interest = impact["gross_annual_interest_mxn"] - breakdown.get("isr_withheld_mxn", 0.0)
    if deposit > 0:
        net_yield_percent = net_interest / deposit * 100
    else:
        net_yield_percent = impact["applicable_yield_rate_percent"]
    perks = sum(weight for flag, weight in PERK_FLAG_WEIGHTS.items() if getattr(product, flag))
    return {
        "yield_metric": round(_clamp(net_yield_percent / YIELD_REFERENCE_RATE_PERCENT), 4),
        "fee_metric": round(1.0 - _clamp(impact["total_annual_fees_mxn"] / FEE_REFERENCE_ANNUAL_MXN), 4),
        "perks_metric": round(_clamp(perks), 4),
        "net_after_tax_yield_percent": round(net_yield_percent, 4),
        "net_cost_mxn": round(max(0.0, -float(impact.get("net_annual_impact_mxn") or 0.0)), 2),
    }


def net_cost_penalty(weights: Mapping[str, float], metrics: Mapping[str, float]) -> float:
    """Points subtracted for a net annual loss, scaled by the archetype nearest to ``weights``."""
    net_cost = float(metrics.get("net_cost_mxn", 0.0))
    if net_cost <= 0:
        return 0.0
    from agents.profiling import classify_archetype  # local: profiling must not depend on the evaluator at import

    maximum = NET_COST_MAX_PENALTY[classify_archetype(weights)]
    return round(maximum * _clamp(net_cost / NET_COST_REFERENCE_ANNUAL_MXN), 2)


def composite_score(weights: Mapping[str, float], metrics: Mapping[str, float]) -> float:
    """100 * (w . metrics) minus the net-cost penalty, floored at 0."""
    score = 100.0 * (
        weights["yield_importance"] * metrics["yield_metric"]
        + weights["fee_importance"] * metrics["fee_metric"]
        + weights["perks_importance"] * metrics["perks_metric"]
    )
    return round(max(0.0, score - net_cost_penalty(weights, metrics)), 2)


def structured_score(product: BankProduct, weights: Mapping[str, float], deposit: float, transfers: int,
                     channel: Optional[str]) -> Tuple[Dict[str, Any], Dict[str, float], float]:
    """Calculator impact, metrics and composite score of one product (no RAG input).

    Same score the linear evaluator and the ToT D1 level use; research ranks its RAG
    finalists with it.
    """
    impact = product_net_impact(product, deposit, transfers, channel)
    metrics = compute_metrics(product, impact, deposit)
    return impact, metrics, composite_score(weights, metrics)


def _branch(
    branch_id: str,
    parent_id: Optional[str],
    depth: int,
    stage: str,
    product: BankProduct,
    strategy: Dict[str, Any],
    impact: Dict[str, Any],
    metrics: Dict[str, float],
    score: float,
    checks: Dict[str, bool],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "branch_id": branch_id,
        "parent_id": parent_id,
        "depth": depth,
        "stage": stage,
        "product_key": product.catalog_key,
        "institution_name": product.institution_name,
        "product_name": product.product_name,
        "strategy": strategy,
        "net_impact": impact,
        "metrics": metrics,
        "score": score,
        "hard_constraint_checks": checks,
        "hard_constraints_met": all(checks.values()) if checks else True,
        "rationale": rationale,
        "in_beam": False,
        "final": False,
    }


class _Evaluation:
    """Shared evaluation context for one graph run."""

    def __init__(self, state: SquadState, resources: SquadResources) -> None:
        self.profile = state["user_profile"]
        self.weights = state["weighted_preference_vector"]
        self.deposit = float(self.profile.get("deposit_mxn", 0.0))
        self.transfers = int(self.profile.get("monthly_transfers", 2))
        self.preferred_channel: Optional[str] = self.profile.get("preferred_channel")
        self.constraints: List[str] = list(self.profile.get("hard_constraints", []))
        self.objectives: List[str] = list(self.profile.get("objectives", []))
        self.products: List[BankProduct] = [
            resources.products_by_key[r["catalog_key"]]
            for r in state.get("api_structured_data") or []
            if r.get("catalog_key") in resources.products_by_key
        ]
        stale = set(((state.get("execution_metadata") or {}).get("freshness_report") or {}).get("stale_products") or [])
        conflicted = {c["product_key"] for c in unresolved_conflicts(state)}
        self.stale = stale
        self.conflicted = conflicted
        self.benefit_conflicts = benefit_conflicts(state)
        self.flagged = stale | conflicted
        self.evidence: Dict[str, int] = {}
        for chunk in state.get("rag_retrieved_chunks") or []:
            for key in chunk.get("matched_product_keys", []):
                if (chunk.get("metadata") or {}).get("product_id") != "ALL":
                    self.evidence[key] = self.evidence.get(key, 0) + 1

    def score(self, product: BankProduct, deposit: float, channel: Optional[str]) -> Tuple[Dict, Dict, float]:
        return structured_score(product, self.weights, deposit, self.transfers, channel)

    def tot_score(self, product: BankProduct, deposit: float, channel: Optional[str]) -> Tuple[Dict, Dict, float]:
        """ToT branch score: stale products are downweighted by 20% (ToT plan, pruning thresholds)."""
        impact, metrics, score = self.score(product, deposit, channel)
        if product.catalog_key in self.stale:
            score = round(score * STALE_DOWNWEIGHT_FACTOR, 2)
        return impact, metrics, score

    def adherence(self, branch: Mapping[str, Any]) -> Tuple[int, int, int, int]:
        """Strict constraint adherence used for tie resolution (higher is better)."""
        channel = (branch.get("strategy") or {}).get("transfer_channel")
        return (
            int(bool(branch.get("hard_constraints_met"))),
            int(branch["product_key"] not in self.flagged),
            int(self.preferred_channel is None or channel == self.preferred_channel),
            int(bool(self.evidence.get(branch["product_key"]))),
        )

    def checks(self, product: BankProduct) -> Dict[str, bool]:
        return evaluate_hard_constraints(product, self.constraints, self.deposit)


def _rank(branches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(branches, key=lambda b: (-b["score"], -b["net_impact"]["net_annual_impact_mxn"], b["product_key"]))


def _rank_with_ties(branches: List[Dict[str, Any]], evaluation: "_Evaluation") -> List[Dict[str, Any]]:
    """Score ranking, then branches within TIE_MARGIN points are reranked by strict constraint adherence."""
    ranked = _rank(branches)
    changed = True
    while changed:  # stable adjacent swaps; terminates because each swap strictly improves adherence order
        changed = False
        for i in range(len(ranked) - 1):
            a, b = ranked[i], ranked[i + 1]
            if abs(a["score"] - b["score"]) <= TIE_MARGIN and evaluation.adherence(b) > evaluation.adherence(a):
                ranked[i], ranked[i + 1] = b, a
                b.setdefault("tie_break", f"reranked above {a['branch_id']} (within {TIE_MARGIN:.0f} points, "
                                          f"stricter constraint adherence)")
                changed = True
    return ranked


def interchangeable(a: Mapping[str, Any], b: Mapping[str, Any], evaluation: "_Evaluation") -> bool:
    """Both branches are safe picks: every hard constraint met and neither stale nor in a data conflict.

    A near-tie between two such accounts is not uncertainty about the user's outcome - either one satisfies
    every stated requirement with verified data - so it must not lower the confidence.
    """
    return all(bool(x.get("hard_constraints_met")) and x["product_key"] not in evaluation.flagged for x in (a, b))


def separation_term(final: Sequence[Mapping[str, Any]], evaluation: "_Evaluation") -> float:
    """1.0 = clear winner (gap >= 10 points, no runner-up, or an interchangeable near-tie)."""
    if len(final) < 2:
        return 1.0
    best, second = final[0], final[1]
    gap = _clamp((best["score"] - second["score"]) / 10.0)
    if gap < 1.0 and interchangeable(best, second, evaluation):
        return 1.0
    return gap


def _confidence(final: Sequence[Mapping[str, Any]], state: SquadState, evaluation: _Evaluation) -> float:
    if not final:
        return 0.0
    best = final[0]
    separation = separation_term(final, evaluation)
    confidence = (
        0.45 * best["score"] / 100.0
        + 0.20 * (1.0 if best["hard_constraints_met"] else 0.0)
        + 0.15 * (1.0 if state.get("data_freshness_flag", True) else 0.0)
        + 0.10 * (1.0 if any(best["product_key"] in c.get("matched_product_keys", [])
                             for c in state.get("rag_retrieved_chunks") or []) else 0.0)
        + 0.10 * separation
    )
    if unresolved_conflicts(state):
        confidence -= 0.20
    return round(_clamp(confidence), 4)


HIGHER_BALANCE_STAGE = "higher_balance_yield"


def pays_yield_now(branch: Mapping[str, Any]) -> bool:
    """The account pays a meaningful yield at the balance the branch was evaluated at."""
    return float((branch.get("net_impact") or {}).get("applicable_yield_rate_percent") or 0.0) >= MEANINGFUL_YIELD_PERCENT


def higher_balance_branches(evaluation: _Evaluation, skip_keys: set) -> List[Dict[str, Any]]:
    """Yield objective: candidates that pay no meaningful yield at the stated balance but would from a higher one.

    Each branch is evaluated by the calculator AT the unlock balance (``strategy.unlock_balance_mxn``) and
    is never a final branch: it cannot become the Base / Trade-off option nor change the confidence. The
    UI lists them right after the accounts that already pay at the stated balance.
    """
    branches = []
    for product in evaluation.products:
        if product.catalog_key in skip_keys:
            continue
        unlock = yield_unlock(product, evaluation.deposit)
        if unlock is None:
            continue
        balance = unlock["unlock_balance_mxn"]
        checks = evaluate_hard_constraints(product, evaluation.constraints, balance)
        if checks and not all(checks.values()):
            continue
        impact, metrics, score = evaluation.score(product, balance, evaluation.preferred_channel)
        if impact.get("error") or impact["applicable_yield_rate_percent"] < MEANINGFUL_YIELD_PERCENT:
            continue
        extra = balance - evaluation.deposit
        branches.append(_branch(
            "", None, 0, HIGHER_BALANCE_STAGE, product,
            {"transfer_channel": impact["breakdown"]["transfer_channel_used"], "requires_higher_balance": True,
             "unlock_balance_mxn": balance, "unlock_rate_percent": unlock["unlock_rate_percent"],
             "stated_deposit_mxn": evaluation.deposit},
            impact, metrics, score, checks,
            f"Pays {impact['applicable_yield_rate_percent']:.2f}% from an average balance of MXN {balance:,.2f} "
            f"(MXN {extra:,.2f} more than the stated MXN {evaluation.deposit:,.2f}); 0 meaningful yield today. "
            "Figures are evaluated at that higher balance."))
    # positive net impact at the unlock balance first (a yield eaten by fees is listed last), then the
    # smallest extra balance, then the higher rate
    branches.sort(key=lambda b: (b["net_impact"]["net_annual_impact_mxn"] <= 0, b["strategy"]["unlock_balance_mxn"],
                                 -b["net_impact"]["applicable_yield_rate_percent"], b["product_key"]))
    for index, branch in enumerate(branches, start=1):
        branch["branch_id"] = f"U-{index}"
        branch["higher_balance_rank"] = index
    return branches


def _result(
    final: List[Dict[str, Any]],
    explored: List[Dict[str, Any]],
    state: SquadState,
    evaluation: _Evaluation,
    log: List[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    for rank, branch in enumerate(final, start=1):
        branch["final"] = True
        branch["final_rank"] = rank
    if "yield" in evaluation.objectives:
        upgrades = higher_balance_branches(evaluation, {b["product_key"] for b in final if pays_yield_now(b)})
        explored = explored + upgrades
        metadata["higher_balance_options"] = len(upgrades)
        if upgrades:
            log.append(f"[evaluator] Yield objective: {len(upgrades)} account(s) pay a yield only above the stated "
                       f"balance (listed after the accounts that pay now): "
                       + ", ".join(f"{b['product_key']} {b['net_impact']['applicable_yield_rate_percent']:.2f}% from "
                                   f"MXN {b['strategy']['unlock_balance_mxn']:,.0f}" for b in upgrades[:5]))
    best_score = final[0]["score"] if final else 0.0
    confidence = _confidence(final, state, evaluation)
    if final:
        log.append(f"[evaluator] Best: {final[0]['institution_name']} {final[0]['product_name']} "
                   f"score={best_score:.2f}/100, net=MXN {final[0]['net_impact']['net_annual_impact_mxn']:,.2f}, "
                   f"confidence={confidence:.2f}")
    else:
        log.append("[evaluator] No candidate satisfies the hard constraints; nothing to score.")
    return {
        "tot_candidate_branches": explored,
        "evaluator_score": best_score,
        "confidence_score": confidence,
        "trajectory_log": log,
        "execution_metadata": metadata,
    }


# ---------------------------------------------------------------------- #
# Linear evaluation
# ---------------------------------------------------------------------- #
def evaluator_linear_node(state: SquadState, ctx: NodeContext, resources: SquadResources) -> Dict[str, Any]:
    evaluation = _Evaluation(state, resources)
    branches = []
    for index, product in enumerate(evaluation.products, start=1):
        impact, metrics, score = evaluation.score(product, evaluation.deposit, evaluation.preferred_channel)
        branches.append(_branch(f"L-{index}", None, 1, "linear_scoring", product,
                                {"transfer_channel": impact["breakdown"]["transfer_channel_used"]},
                                impact, metrics, score, evaluation.checks(product),
                                "Single-pass deterministic scoring."))
    final = _rank(branches)
    log = [ctx.log(f"Linear evaluation of {len(final)} candidate(s) with calculate_net_impact "
                   f"(deposit=MXN {evaluation.deposit:,.2f}, transfers/month={evaluation.transfers}).")]
    for branch in final[:BEAM_WIDTH]:
        log.append(ctx.log(f"  {branch['product_key']}: score={branch['score']:.2f}, metrics={branch['metrics']}"))
    return _result(final, final, state, evaluation, log, {"evaluation_mode": "linear", "tot_depth_reached": 1,
                                                         "early_stopped": False})


# ---------------------------------------------------------------------- #
# Tree-of-Thought beam search
# ---------------------------------------------------------------------- #
def _spei_channels(product: BankProduct) -> List[str]:
    channels = [t.channel for t in product.transfer_fees_detail if t.transfer_type == SPEI_TRANSFER]
    return list(dict.fromkeys(channels))


def _select_beam(candidates: List[Dict[str, Any]], width: int, evaluation: "_Evaluation") -> List[Dict[str, Any]]:
    """Top-``width`` branches, at most one per product to keep the beam diverse."""
    beam: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for branch in _rank_with_ties(candidates, evaluation):
        if branch["product_key"] in seen:
            continue
        beam.append(branch)
        seen.add(branch["product_key"])
        if len(beam) == width:
            break
    for branch in beam:
        branch["in_beam"] = True
    return beam


# ToT levels (Tree-of-Thought Integration Plan, section 2.2, as adapted to the catalog):
#   D1 base debit account selection, D2 yields, D3 benefits / cashback / insurance & fee structure.
TOT_STAGES = {1: "product_selection", 2: "yield_assessment", 3: "benefits_and_fees"}
BENEFIT_UNVERIFIED_PENALTY = 2.0  # benefit stated in RECA but absent from the catalog


def _yield_thought(parent: Dict[str, Any], product: BankProduct, evaluation: "_Evaluation") -> Dict[str, Any]:
    """D2 - yields: net yield after ISR at the stated balance and under a 20% balance dip."""
    channel = parent["strategy"]["transfer_channel"]
    impact = parent["net_impact"]
    breakdown = impact.get("breakdown") or {}
    stress_deposit = round(evaluation.deposit * STRESS_BALANCE_FACTOR, 2)
    stress_impact, _, stress_score = evaluation.tot_score(product, stress_deposit, channel)
    robust = (1 - STRESS_WEIGHT) * parent["score"] + STRESS_WEIGHT * stress_score
    upper_tiers = sorted((t for t in product.yield_tiers if t.min_balance_mxn > evaluation.deposit),
                         key=lambda t: t.min_balance_mxn)
    notes = [
        f"rate {impact['applicable_yield_rate_percent']:.2f}%, gross interest MXN {impact['gross_annual_interest_mxn']:,.2f}, "
        f"ISR withheld MXN {breakdown.get('isr_withheld_mxn', 0.0):,.2f} "
        f"(net {parent['metrics'].get('net_after_tax_yield_percent', 0.0):.2f}% after tax)",
        f"stress balance MXN {stress_deposit:,.2f} -> rate {stress_impact['applicable_yield_rate_percent']:.2f}%, "
        f"score {stress_score:.2f}",
    ]
    if upper_tiers:
        notes.append(f"next yield tier from MXN {upper_tiers[0].min_balance_mxn:,.2f} "
                     f"at {upper_tiers[0].annual_yield_rate:.2f}%")
    elif not product.offers_yield:
        notes.append("no yield offered by this account")
    if product.catalog_key in evaluation.stale:
        notes.append(f"stale data (score downweighted x{STALE_DOWNWEIGHT_FACTOR})")
    strategy = {
        **parent["strategy"],
        "stress_deposit_mxn": stress_deposit,
        "yield_survives_stress": stress_impact["applicable_yield_rate_percent"] >= impact["applicable_yield_rate_percent"],
        "waiver_survives_stress": stress_impact["effective_monthly_fee_mxn"] <= impact["effective_monthly_fee_mxn"],
        "next_yield_tier_mxn": upper_tiers[0].min_balance_mxn if upper_tiers else None,
    }
    return _branch(f"{parent['branch_id']}.D2", parent["branch_id"], 2, TOT_STAGES[2], product, strategy,
                   impact, parent["metrics"], round(_clamp(robust, 0.0, 100.0), 2), parent["hard_constraint_checks"],
                   "Thought: what does this account really yield after ISR, and does it hold under a 20% balance "
                   "dip? " + "; ".join(notes))


def _benefits_and_fees_thoughts(parent: Dict[str, Any], product: BankProduct,
                                evaluation: "_Evaluation") -> List[Dict[str, Any]]:
    """D3 - benefits, cashback, insurance and fee structure: one thought per SPEI channel."""
    reference_channel = parent["strategy"]["transfer_channel"]
    _, _, reference_score = evaluation.tot_score(product, evaluation.deposit, reference_channel)
    perks = [label for flag, label in (("has_cashback_or_perks", "cashback/rewards"),
                                       ("has_free_insurance", "complimentary insurance"),
                                       ("has_virtual_card", "virtual card")) if getattr(product, flag)]
    unverified = sorted({c.get("benefit") for c in evaluation.benefit_conflicts if c["product_key"] == product.catalog_key})
    children = []
    for n, channel in enumerate(_spei_channels(product) or [reference_channel], start=1):
        impact, metrics, score = evaluation.tot_score(product, evaluation.deposit, channel)
        adjusted = parent["score"] + (score - reference_score)  # fee delta of this channel vs the D2 path
        notes = [f"SPEI via '{channel}': annual fees MXN {impact['total_annual_fees_mxn']:,.2f}",
                 f"perks: {', '.join(perks) or 'none listed'}"]
        if product.penalties_detail:
            notes.append(f"{len(product.penalties_detail)} penalty concept(s) in the catalog")
        if evaluation.preferred_channel and channel != evaluation.preferred_channel:
            adjusted -= CHANNEL_MISMATCH_PENALTY
            notes.append("not the user's preferred channel")
        if evaluation.evidence.get(product.catalog_key):
            adjusted += EVIDENCE_BONUS
            notes.append("benefits/fees confirmed by product-specific RECA or catalog benefit chunks")
        if product.catalog_key in evaluation.conflicted:
            adjusted -= STALE_OR_CONFLICT_PENALTY
            notes.append("unresolved API/RAG fee conflict")
        if unverified:
            adjusted -= BENEFIT_UNVERIFIED_PENALTY
            notes.append(f"benefit(s) in RECA but not in the catalog: {unverified}")
        strategy = {**parent["strategy"], "transfer_channel": channel}
        children.append(_branch(f"{parent['branch_id']}.D3-{n}", parent["branch_id"], 3, TOT_STAGES[3], product,
                                strategy, impact, metrics, round(_clamp(adjusted, 0.0, 100.0), 2),
                                parent["hard_constraint_checks"],
                                "Thought: what do the benefits, insurance, cashback and fee structure add? "
                                + "; ".join(notes)))
    return children


def _early_stop(beam: List[Dict[str, Any]]) -> bool:
    return bool(beam) and beam[0]["hard_constraints_met"] and beam[0]["score"] > EARLY_STOP_SCORE


def evaluator_tot_beam_node(state: SquadState, ctx: NodeContext, resources: SquadResources) -> Dict[str, Any]:
    evaluation = _Evaluation(state, resources)
    explored: List[Dict[str, Any]] = []
    log = [ctx.log(f"ToT beam search started: b={BEAM_WIDTH}, D={MAX_DEPTH}, early stop when constraints=100% "
                   f"and score>{EARLY_STOP_SCORE}; {len(evaluation.products)} root candidate(s).")]
    metadata: Dict[str, Any] = {"evaluation_mode": "tot_beam", "beam_width": BEAM_WIDTH, "max_depth": MAX_DEPTH}

    # D1 - product selection thoughts
    level: List[Dict[str, Any]] = []
    for index, product in enumerate(evaluation.products, start=1):
        impact, metrics, score = evaluation.tot_score(product, evaluation.deposit, evaluation.preferred_channel)
        level.append(_branch(f"D1-{index}", None, 1, "product_selection", product,
                             {"transfer_channel": impact["breakdown"]["transfer_channel_used"]},
                             impact, metrics, score, evaluation.checks(product),
                             "Thought: is this product a good fit under the default usage pattern?"))
    explored.extend(level)
    beam = _select_beam(level, BEAM_WIDTH, evaluation)
    depth_reached = 1
    if evaluation.stale & {b["product_key"] for b in level}:
        log.append(ctx.log(f"Stale-data downweight (x{STALE_DOWNWEIGHT_FACTOR}) applied to "
                           f"{sorted(evaluation.stale & {b['product_key'] for b in level})}."))
    log.append(ctx.log(f"D1 product_selection: {len(level)} thoughts -> beam "
                       f"{[(b['product_key'], b['score']) for b in beam]}"))

    for depth in (2, 3):
        if _early_stop(beam):
            log.append(ctx.log(f"EARLY STOP at depth {depth_reached}: {beam[0]['product_key']} meets 100% of hard "
                               f"constraints with score {beam[0]['score']:.2f} > {EARLY_STOP_SCORE}."))
            metadata["early_stopped"] = True
            break
        if not ctx.has_budget(0.5):
            log.append(ctx.log(f"Budget nearly exhausted; stopping ToT at depth {depth_reached} and returning the "
                               "highest-scoring valid branch."))
            metadata["tot_timeout"] = True
            break

        children: List[Dict[str, Any]] = []
        for parent in beam:
            product = resources.products_by_key[parent["product_key"]]
            if depth == 2:
                children.append(_yield_thought(parent, product, evaluation))
            else:
                children.extend(_benefits_and_fees_thoughts(parent, product, evaluation))
        explored.extend(children)
        beam = _select_beam(children, BEAM_WIDTH, evaluation)
        depth_reached = depth
        stage = TOT_STAGES[depth]
        log.append(ctx.log(f"D{depth} {stage}: {len(children)} thoughts -> beam "
                           f"{[(b['product_key'], b['score']) for b in beam]}"))

    metadata.setdefault("early_stopped", False)
    metadata["tot_depth_reached"] = depth_reached
    metadata["tot_branches_explored"] = len(explored)
    return _result(beam, explored, state, evaluation, log, metadata)


# ---------------------------------------------------------------------- #
# Critique loop: review (critic) and revision nodes
# ---------------------------------------------------------------------- #
def critic_narrative(draft: Mapping[str, Any]) -> Dict[str, Any]:
    """Narrative sent to the LLM critic, without the deterministic higher-balance sentence.

    That sentence is built from verified figures (``advisory.higher_balance_sentence``); reviewing it
    again only risks a false rejection.
    """
    from agents.advisory import higher_balance_sentence  # advisory imports this module

    narrative = dict(draft.get("narrative") or {})
    option = draft.get("higher_balance_option")
    if option:
        sentence = higher_balance_sentence(option)
        narrative["key_tradeoffs"] = [t for t in narrative.get("key_tradeoffs") or [] if t != sentence]
    return narrative


def _narrative_text(draft: Mapping[str, Any]) -> str:
    narrative = draft.get("narrative") or {}
    parts = [str(v) for k, v in narrative.items() if k != "key_tradeoffs"]
    parts += [str(t) for t in narrative.get("key_tradeoffs", [])]
    return " ".join(parts)


CRITIC_TIMEOUT_CAP_S = 4.5  # gpt-4o-mini critic latency is ~3-3.5 s


def _llm_critique(
    state: SquadState, ctx: NodeContext, resources: SquadResources, draft: Mapping[str, Any]
) -> Tuple[List[str], str]:
    """Grounded LLM critique: only issues quoting the draft verbatim are kept.

    Returns (issues, critic_status) where critic_status is one of
    skipped_no_llm | skipped_template | skipped_budget | unavailable | approved | rejected.
    """
    if resources.llm is None or not resources.enable_llm_critic:
        return [], "skipped_no_llm"
    if draft.get("narrative_source") != "llm":
        return [], "skipped_template"
    if not ctx.has_budget(resources.min_llm_call_budget_s + resources.llm_safety_margin_s):
        return [], "skipped_budget"
    # Same facts object the narrative writer received (advisory.verified_facts), so the critic never
    # rejects a figure the writer was legitimately given.
    facts = draft.get("verified_facts") or {
        "base_option": draft.get("base_option"),
        "tradeoff_option": draft.get("tradeoff_option"),
        "hard_constraints": (state.get("user_profile") or {}).get("hard_constraints", []),
        "abstain": draft.get("status") != "definitive",
    }
    try:
        verdict = resources.llm.generate(
            CriticVerdict,
            EVALUATOR_CRITIC_SYSTEM_PROMPT,
            {"verified_facts": facts, "draft_narrative": critic_narrative(draft)},
            timeout_s=resources.llm_timeout(ctx.remaining_s(), CRITIC_TIMEOUT_CAP_S),
            run_name="evaluator_critic_llm",
        )
    except (LLMUnavailableError, ValueError):
        return [], "unavailable"
    # LLM extracts feature claims; verification against catalog flags is deterministic.
    feature_issues = verify_feature_claims([c.model_dump() for c in verdict.feature_claims], draft, state)
    text = _narrative_text(draft).lower()
    grounded = [] if verdict.verdict != "reject" else [
        f"LLM critic ({issue.category}): {issue.explanation} [quote: '{issue.evidence_quote.strip()}']"
        for issue in verdict.issues
        if issue.evidence_quote.strip() and issue.evidence_quote.strip().lower() in text
    ]
    issues = grounded + feature_issues
    return issues, "rejected" if issues else "approved"


def review_node(state: SquadState, ctx: NodeContext, resources: SquadResources) -> Dict[str, Any]:
    draft = state.get("draft_recommendation")
    issues = review_draft(draft, state)
    critic_status = "skipped_deterministic_issues" if issues else "skipped_no_draft"
    if not issues and draft:
        issues, critic_status = _llm_critique(state, ctx, resources, draft)
    approved = not issues

    snapshot = dict(state)
    snapshot["review_approved"] = approved
    hitl = check_human_intervention_needed(snapshot)  # type: ignore[arg-type]
    reasons = human_review_reasons(snapshot)

    log = [ctx.log(f"Critic review of draft revision {state.get('revision_count', 0)}: "
                   f"{'APPROVED' if approved else 'REJECTED'} (llm_critic={critic_status})"
                   + (f" - issues: {issues}" if issues else ""))]
    if critic_status == "unavailable":
        log.append(ctx.log("WARNING: LLM critic timed out or failed; approval relies on deterministic checks only."))
    if hitl:
        log.append(ctx.log(f"HITL escalation required: {reasons}"))

    updated_draft = dict(draft) if draft else None
    if updated_draft is not None:
        updated_draft["review"] = {"approved": approved, "issues": issues, "revision_count": state.get("revision_count", 0)}
        updated_draft["requires_human_review"] = hitl
        updated_draft["human_review_reasons"] = reasons
        if hitl:
            updated_draft["hitl_notice"] = HITL_NOTICE.format(reasons="; ".join(reasons))
            if updated_draft.get("status") == "definitive":
                updated_draft["status"] = "pending_human_review"

    return {
        "review_approved": approved,
        "review_feedback": issues,
        "requires_human_review": hitl,
        "draft_recommendation": updated_draft,
        "trajectory_log": log,
        "execution_metadata": {"hitl_reasons": reasons, "llm_critic_status": critic_status},
    }


def evaluator_revision_node(state: SquadState, ctx: NodeContext, resources: SquadResources) -> Dict[str, Any]:
    """Return the rejected draft to the Advisory Agent with corrections (N += 1)."""
    revision = int(state.get("revision_count", 0)) + 1
    log = [ctx.log(f"Revision {revision}/{MAX_REVISIONS}: returning draft to advisory_node with "
                   f"{len(state.get('review_feedback') or [])} correction(s): {state.get('review_feedback')}")]
    return {"revision_count": revision, "trajectory_log": log}
