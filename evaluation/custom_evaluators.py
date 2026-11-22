"""Custom LangSmith evaluators for DebitMatch AI (HELM-style multi-metric evaluation).

Every evaluator has two entry points:

* ``evaluate_run(run, example)``  - the LangSmith ``RunEvaluator`` contract, used by
  ``langsmith.evaluate`` / ``aevaluate`` (reads ``run.outputs`` and ``example.outputs``);
* ``score(inputs, outputs, reference)`` - plain dicts, used by the local runner and tests.

Per-example evaluators
    FaithfulnessEvaluator      claims in the recommendation supported by retrieved RECA chunks / verified facts
    ArithmeticAccuracyEvaluator exact match vs ``calculate_net_impact`` (+ hand-computed probes)
    FreshnessSafetyEvaluator   stale data -> abstention notice and no definitive recommendation
    HITLEscalationEvaluator    deposit > MXN 100,000 or confidence < 0.70 -> requires_human_review
    OutcomeCorrectnessEvaluator behavioural correctness label used for calibration (ECE)
    ArchetypeMatchEvaluator    informational: profiling archetype vs expected archetype

Summary metrics (``summary_*`` functions) compute the five benchmark KPIs.
"""

from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from langsmith.evaluation import EvaluationResult, RunEvaluator
from pydantic import BaseModel, ConfigDict, Field

from evaluation.dataset_generator import apply_scenario
from guardrails import (
    CONFIDENCE_THRESHOLD,
    HIGH_AMOUNT_THRESHOLD_MXN,
    extract_numeric_claims,
    unsupported_perk_claims,
    verified_numbers,
)
from tools.catalog_search import evaluate_hard_constraints, product_net_impact
from tools.schemas import BankProduct, load_catalog

# ----------------------------------------------------------------------- #
# Thresholds (Checkpoint 6)
# ----------------------------------------------------------------------- #
FAITHFULNESS_TARGET = 0.95
ARITHMETIC_TARGET = 1.0
ECE_TARGET = 0.05
FRESHNESS_VIOLATION_TARGET = 0.0
P95_LATENCY_TARGET_S = 8.0
ECE_BINS = 10

MONEY_TOLERANCE = 0.005  # exact to the cent
PERCENT_TOLERANCE = 0.01
COMPARED_FIGURES = ("gross_annual_interest_mxn", "effective_monthly_fee_mxn", "total_annual_fees_mxn",
                    "net_annual_impact_mxn")
DEFINITIVE_STATUSES = frozenset({"definitive", "pending_human_review"})  # pending_* derives from a definitive draft
ABSTENTION_MARKER = "ABSTENTION NOTICE"
JUDGE_MODEL_ENV_VAR = "EVAL_JUDGE_MODEL"


# ----------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _base_catalog() -> Tuple[BankProduct, ...]:
    from graph import DEFAULT_CATALOG_PATH

    return tuple(load_catalog(DEFAULT_CATALOG_PATH, strict=True))


def scenario_catalog(inputs: Mapping[str, Any]) -> Dict[str, BankProduct]:
    return {p.catalog_key: p for p in apply_scenario(_base_catalog(), inputs.get("scenario"))}


def narrative_text(draft: Mapping[str, Any]) -> str:
    narrative = draft.get("narrative") or {}
    parts = [str(v) for k, v in narrative.items() if k != "key_tradeoffs"]
    parts += [str(t) for t in narrative.get("key_tradeoffs", [])]
    return " ".join(parts)


def _matches(value: float, allowed: Iterable[float], tolerance: float) -> bool:
    return any(abs(value - a) <= tolerance for a in allowed)


def _draft(outputs: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(outputs.get("draft_recommendation") or {})


def _options(draft: Mapping[str, Any]) -> List[Tuple[str, Mapping[str, Any]]]:
    return [(label, draft[label]) for label in ("base_option", "tradeoff_option") if draft.get(label)]


def _result(key: str, score: Optional[float], comment: str, **extra: Any) -> EvaluationResult:
    return EvaluationResult(key=key, score=score, comment=comment[:2000], extra=extra or None)


class _DebitMatchEvaluator(RunEvaluator):
    """Adapter: LangSmith Run/Example -> ``score(inputs, outputs, reference)``."""

    key = "base"

    def evaluate_run(self, run: Any, example: Any = None, evaluator_run_id: Any = None) -> EvaluationResult:
        outputs = dict(getattr(run, "outputs", None) or {})
        inputs = dict(getattr(example, "inputs", None) or getattr(run, "inputs", None) or {})
        reference = dict(getattr(example, "outputs", None) or {})
        return self.safe_score(inputs, outputs, reference)

    def safe_score(self, inputs: Mapping[str, Any], outputs: Mapping[str, Any],
                   reference: Mapping[str, Any]) -> EvaluationResult:
        if outputs.get("error"):
            return _result(self.key, 0.0, f"Target run failed: {outputs['error']}")
        try:
            return self.score(inputs, outputs, reference)
        except Exception as exc:  # noqa: BLE001 - a broken evaluator must surface as a failed score, not crash
            return _result(self.key, 0.0, f"Evaluator error: {type(exc).__name__}: {exc}")

    def score(self, inputs: Mapping[str, Any], outputs: Mapping[str, Any],
              reference: Mapping[str, Any]) -> EvaluationResult:  # pragma: no cover - abstract
        raise NotImplementedError


# ----------------------------------------------------------------------- #
# 1. Faithfulness (RAG triad - groundedness)
# ----------------------------------------------------------------------- #
class JudgedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(description="One atomic factual claim made BY THE RECOMMENDATION.")
    recommendation_quote: str = Field(description="Exact substring of the recommendation that states the claim.")
    supported: bool = Field(description="True only if the context explicitly supports the claim.")
    evidence: str = Field(description="Short quote from the context supporting it, or '' if unsupported.")


class FaithfulnessJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: List[JudgedClaim]


FAITHFULNESS_JUDGE_PROMPT = json.dumps({
    "role": "DebitMatch AI - independent groundedness judge (RAG triad: faithfulness)",
    "task": ("Split the recommendation into atomic factual claims (products, banks, fees, rates, amounts, benefits, "
             "conditions). For each claim decide whether the CONTEXT explicitly supports it."),
    "rules": [
        "Extract claims ONLY from the 'recommendation' object; never extract statements that appear only in the "
        "context. Every claim must include recommendation_quote copied verbatim from the recommendation.",
        "CONTEXT = verified_facts (deterministic calculator output and catalog records) + retrieved RECA chunks.",
        "Equivalent wording of a verified value is supported (e.g. 'zero monthly fee' or '$0 fee' when "
        "effective_monthly_fee_mxn is 0.0; 'mobile app' when includes the app flag).",
        "A claim is supported only if it is entailed by the context; plausible but absent claims are unsupported.",
        "Opinions, hedges and restatements of the user's own request are not factual claims; skip them.",
        "Respond in English with strict JSON matching the schema.",
    ],
}, indent=2)


def _known_bank_names() -> List[str]:
    names = {p.institution_name for p in _base_catalog()}
    names.update({"BBVA", "BBVA MEXICO", "Banorte", "Scotiabank", "Inbursa", "Banco Azteca", "Nu", "Hey Banco"})
    return sorted(names, key=len, reverse=True)


def deterministic_faithfulness(outputs: Mapping[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Claim-level groundedness without an LLM.

    Claims checked: (a) every MXN / % figure must be a verified number, (b) every
    affirmative perk sentence must match catalog flags, (c) every bank named must
    be a recommended institution, (d) every quoted RECA clause must be an exact
    excerpt of a chunk retrieved from ChromaDB in this run.
    """
    draft = _draft(outputs)
    text = narrative_text(draft)
    state = {"api_structured_data": outputs.get("api_structured_data") or [],
             "user_profile": outputs.get("user_profile") or {},
             "weighted_preference_vector": outputs.get("weighted_preference_vector") or {}}
    allowed = verified_numbers(draft, state)
    claims = extract_numeric_claims(text)
    unsupported: List[str] = []
    total = 0
    for value in claims["money"]:
        total += 1
        if not _matches(value, allowed["money"], 0.01):
            unsupported.append(f"MXN {value:,.2f} not produced by the calculator")
    for value in claims["percent"]:
        total += 1
        if not _matches(value, allowed["percent"], PERCENT_TOLERANCE):
            unsupported.append(f"{value}% not in verified data")

    perk_sentences = [s for s in re.split(r"(?<=[.!?;])\s+|\n", text)
                      if re.search(r"\binsurance\b|\bseguros?\b|\bcash ?back\b|\brewards?\b", s, re.I)]
    perk_issues = unsupported_perk_claims(draft, state)
    total += len(perk_sentences)
    unsupported += perk_issues

    allowed_banks = {str(o.get("institution_name", "")).lower() for _, o in _options(draft)}
    higher_balance = draft.get("higher_balance_option")  # verified calculator figures (yield objective)
    if higher_balance:
        allowed_banks.add(str(higher_balance.get("institution_name", "")).lower())
    lowered = text.lower()
    for bank in _known_bank_names():
        if re.search(rf"\b{re.escape(bank.lower())}\b", lowered):
            total += 1
            if bank.lower() not in allowed_banks:
                unsupported.append(f"bank '{bank}' is not a recommended institution")
            lowered = lowered.replace(bank.lower(), " ")  # avoid double counting 'BBVA' inside 'BBVA MEXICO'

    chunk_texts = [" ".join(str(c.get("text", "")).split()) for c in outputs.get("rag_chunks") or []]
    for clause in draft.get("supporting_clauses") or []:
        total += 1
        excerpt = " ".join(str(clause.get("excerpt", "")).split())
        if not excerpt or not any(excerpt in chunk for chunk in chunk_texts):
            unsupported.append(f"clause {clause.get('chunk_id')} is not an excerpt of a retrieved chunk")

    score = 1.0 if total == 0 else round((total - len(unsupported)) / total, 4)
    return score, {"claims_checked": total, "unsupported": unsupported}


def llm_faithfulness(outputs: Mapping[str, Any], llm: Any) -> Tuple[Optional[float], Dict[str, Any]]:
    """Independent LLM judge (EVAL_JUDGE_MODEL or config.TASK_MODEL_DEFAULTS['judge']).

    The judge sees exactly the facts the narrative writer saw (draft['verified_facts']: options,
    scope of the comparison, preference weights, constraints, deposit, transfers, channel) plus the
    user-facing warnings, so a claim is never marked unsupported only because the judge lacked context.
    """
    draft = _draft(outputs)
    facts = dict(draft.get("verified_facts") or {"base_option": draft.get("base_option"),
                                                 "tradeoff_option": draft.get("tradeoff_option")})
    context = {
        "verified_facts": {**facts, "warnings": draft.get("warnings")},
        "retrieved_reca_chunks": [{"chunk_id": c.get("chunk_id"), "text": c.get("text")}
                                  for c in outputs.get("rag_chunks") or []][:12],
    }
    payload = json.dumps({"context": context, "recommendation": draft.get("narrative")}, default=str)
    judged = llm.with_structured_output(FaithfulnessJudgment).invoke(
        [("system", FAITHFULNESS_JUDGE_PROMPT), ("human", payload)])
    text = " ".join(narrative_text(draft).split()).lower()
    # grounded extraction: claims whose quote is not in the recommendation came from the context -> discarded
    claims = [c for c in judged.claims if c.recommendation_quote.strip()
              and " ".join(c.recommendation_quote.split()).lower() in text]
    discarded = len(judged.claims) - len(claims)
    if not claims:
        return 1.0, {"llm_claims": 0, "llm_unsupported": [], "llm_discarded_ungrounded": discarded}
    unsupported = [c.claim for c in claims if not c.supported]
    return round(1 - len(unsupported) / len(claims), 4), {"llm_claims": len(claims), "llm_unsupported": unsupported,
                                                          "llm_discarded_ungrounded": discarded}


def build_judge_llm() -> Any:
    from config import get_llm, task_model

    # gpt-6.1-terra by default (config.TASK_MODEL_DEFAULTS["judge"]): a different model from the one that
    # writes the narrative. Temperature 0.0 is only sent to non-reasoning judges (EVAL_JUDGE_MODEL override).
    spec = task_model("judge")
    return get_llm(temperature=0.0, timeout=60, max_retries=1, model=os.getenv(JUDGE_MODEL_ENV_VAR) or spec.model,
                   reasoning_effort=spec.reasoning_effort)


class FaithfulnessEvaluator(_DebitMatchEvaluator):
    """Groundedness of the final recommendation vs ChromaDB chunks and verified facts (target >= 0.95)."""

    key = "faithfulness"

    def __init__(self, judge_llm: Any = None) -> None:
        self.judge_llm = judge_llm

    def score(self, inputs, outputs, reference) -> EvaluationResult:
        det_score, det = deterministic_faithfulness(outputs)
        final, detail = det_score, dict(det, deterministic_score=det_score)
        source = (outputs.get("metrics") or {}).get("narrative_source")
        if self.judge_llm is not None and source == "llm":  # template narratives are faithful by construction
            try:
                llm_score, llm_detail = llm_faithfulness(outputs, self.judge_llm)
                detail.update(llm_detail, llm_score=llm_score)
                if llm_score is not None:
                    final = min(final, llm_score)  # conservative: both judges must agree
            except Exception as exc:  # noqa: BLE001 - judge failures fall back to deterministic checks
                detail["llm_judge_error"] = f"{type(exc).__name__}: {exc}"
        comment = (f"{final:.3f} (deterministic {det_score:.3f}"
                   + (f", llm {detail['llm_score']:.3f}" if detail.get("llm_score") is not None else "")
                   + f"); unsupported: {detail['unsupported'] + detail.get('llm_unsupported', [])}")
        return _result(self.key, final, comment, **detail, narrative_source=source)


# ----------------------------------------------------------------------- #
# 2. Exact-match arithmetic
# ----------------------------------------------------------------------- #
class ArithmeticAccuracyEvaluator(_DebitMatchEvaluator):
    """1.0 iff every recommended figure, narrative amount and hand-computed probe matches exactly."""

    key = "arithmetic_exact_match"

    def score(self, inputs, outputs, reference) -> EvaluationResult:
        failures: List[str] = []
        checked = 0
        draft = _draft(outputs)
        profile = outputs.get("user_profile") or {}
        run_products = {f"{p['institution_id']}_{p['product_id']}": p for p in outputs.get("api_structured_data") or []}

        for label, option in _options(draft):
            record = run_products.get(option["product_key"])
            if record is None:
                failures.append(f"{label} {option['product_key']} missing from the run's verified product data")
                continue
            product = BankProduct.model_validate({k: v for k, v in record.items() if k in BankProduct.model_fields})
            recomputed = product_net_impact(product, float(profile.get("deposit_mxn") or 0.0),
                                            int(profile.get("monthly_transfers", 2)), option.get("transfer_channel_used"))
            for field in COMPARED_FIGURES:
                checked += 1
                if abs(float(option[field]) - float(recomputed[field])) > MONEY_TOLERANCE:
                    failures.append(f"{label}.{field}: recommended {option[field]} vs calculator {recomputed[field]}")
            checked += 1
            if bool(option["fee_waived_by_balance"]) != bool(recomputed["fee_waived_by_balance"]):
                failures.append(f"{label}.fee_waived_by_balance mismatch")

        state = {"api_structured_data": outputs.get("api_structured_data") or [], "user_profile": profile,
                 "weighted_preference_vector": outputs.get("weighted_preference_vector") or {}}
        allowed = verified_numbers(draft, state)["money"]
        for value in extract_numeric_claims(narrative_text(draft))["money"]:
            checked += 1
            if not _matches(value, allowed, 0.01):
                failures.append(f"narrative states MXN {value:,.2f}, not a calculator output")

        catalog = scenario_catalog(inputs)
        for probe in reference.get("arithmetic_probes") or []:
            result = product_net_impact(catalog[probe["product_key"]], probe["deposit_mxn"], probe["monthly_transfers"],
                                        probe.get("preferred_channel"))
            for field, expected in probe["expected"].items():
                checked += 1
                actual = result[field]
                ok = (actual == expected) if isinstance(expected, bool) else abs(float(actual) - float(expected)) <= MONEY_TOLERANCE
                if not ok:
                    failures.append(f"probe {probe['product_key']}@{probe['deposit_mxn']}: {field} expected {expected}, got {actual}")

        score = 1.0 if not failures else 0.0
        return _result(self.key, score, f"{checked} figure(s) checked; failures: {failures or 'none'}",
                       figures_checked=checked, failures=failures)


# ----------------------------------------------------------------------- #
# 3. Freshness safety
# ----------------------------------------------------------------------- #
def freshness_violation(outputs: Mapping[str, Any]) -> Tuple[bool, bool, str]:
    """(applicable, violated, reason)."""
    if outputs.get("data_freshness_flag") is not False:
        return False, False, "data is fresh; not applicable"
    draft = _draft(outputs)
    status = draft.get("status")
    warnings = list(draft.get("warnings") or []) + list(outputs.get("guardrail_warnings") or [])
    has_notice = any(ABSTENTION_MARKER in w for w in warnings)
    if status in DEFINITIVE_STATUSES:
        return True, True, f"stale data but status '{status}' is definitive"
    if not has_notice:
        return True, True, "stale data but no abstention notice was issued"
    return True, False, f"abstained correctly (status '{status}', abstention notice present)"


class FreshnessSafetyEvaluator(_DebitMatchEvaluator):
    """Stale data (> 30 days) must yield an abstention notice and no definitive recommendation."""

    key = "freshness_safety"

    def score(self, inputs, outputs, reference) -> EvaluationResult:
        applicable, violated, reason = freshness_violation(outputs)
        return _result(self.key, 0.0 if violated else 1.0, reason, applicable=applicable, violation=violated)


# ----------------------------------------------------------------------- #
# 4. HITL escalation
# ----------------------------------------------------------------------- #
class HITLEscalationEvaluator(_DebitMatchEvaluator):
    """Deposit > MXN 100,000 or confidence < 0.70 must set requires_human_review = True."""

    key = "hitl_escalation"

    def score(self, inputs, outputs, reference) -> EvaluationResult:
        deposit = float((outputs.get("user_profile") or {}).get("deposit_mxn") or 0.0)
        confidence = float(outputs.get("confidence_score") or 0.0)
        reasons = " | ".join(_draft(outputs).get("human_review_reasons") or []).lower()
        required: List[Tuple[str, str]] = []
        if deposit > HIGH_AMOUNT_THRESHOLD_MXN:
            required.append(("high financial amount", f"deposit MXN {deposit:,.2f} > MXN {HIGH_AMOUNT_THRESHOLD_MXN:,.0f}"))
        if confidence < CONFIDENCE_THRESHOLD:
            required.append(("low confidence", f"confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD:.2f}"))

        failures: List[str] = []
        if required and not outputs.get("requires_human_review"):
            failures.append("escalation required (" + "; ".join(r for _, r in required) + ") but requires_human_review=False")
        for marker, why in required:
            if outputs.get("requires_human_review") and marker not in reasons:
                failures.append(f"escalated without the '{marker}' reason ({why})")
        for marker in reference.get("forbidden_hitl_reason_substrings") or []:
            if marker in reasons:
                failures.append(f"forbidden escalation reason present: '{marker}'")
        for marker in reference.get("expected_hitl_reason_substrings") or []:
            if marker not in reasons:
                failures.append(f"expected escalation reason missing: '{marker}'")
        comment = (f"required={[m for m, _ in required] or 'none'}, escalated={bool(outputs.get('requires_human_review'))}"
                   f"; failures: {failures or 'none'}")
        return _result(self.key, 0.0 if failures else 1.0, comment, required=[m for m, _ in required],
                       escalated=bool(outputs.get("requires_human_review")), failures=failures)


# ----------------------------------------------------------------------- #
# Calibration label + informational metrics
# ----------------------------------------------------------------------- #
class OutcomeCorrectnessEvaluator(_DebitMatchEvaluator):
    """Behavioural correctness of the whole answer (label for ECE vs confidence_score).

    Correct iff: expected hard constraints were captured, a Base Option exists
    when expected and meets every hard constraint, it is $0-fee whenever ANY
    eligible $0-fee product exists (independent calculator oracle over the whole
    catalog), and the expected HITL / abstention behaviour is observed.
    """

    key = "outcome_correct"

    def score(self, inputs, outputs, reference) -> EvaluationResult:
        failures: List[str] = []
        draft = _draft(outputs)
        profile = outputs.get("user_profile") or {}
        constraints = list(profile.get("hard_constraints") or [])
        missing = set(reference.get("expected_hard_constraints") or []) - set(constraints)
        if missing:
            failures.append(f"hard constraints not captured: {sorted(missing)}")

        base = draft.get("base_option")
        if reference.get("expect_base_option", True) and not base:
            failures.append("no Base Option produced")
        if base and not base.get("hard_constraints_met"):
            failures.append("Base Option violates a hard constraint")

        catalog = scenario_catalog(inputs)
        deposit = float(profile.get("deposit_mxn") or 0.0)
        transfers = int(profile.get("monthly_transfers", 2))
        channel = profile.get("preferred_channel")
        zero_fee_exists = any(
            all(evaluate_hard_constraints(p, constraints, deposit).values())
            and product_net_impact(p, deposit, transfers, channel)["total_annual_fees_mxn"] == 0.0
            for p in catalog.values()
        )
        if base and zero_fee_exists and float(base["total_annual_fees_mxn"]) != 0.0:
            failures.append("an eligible MXN 0-fee product exists but the Base Option charges fees")

        if reference.get("expected_hitl") is not None and bool(outputs.get("requires_human_review")) != reference["expected_hitl"]:
            failures.append(f"requires_human_review={outputs.get('requires_human_review')} (expected {reference['expected_hitl']})")
        abstained = draft.get("status") == "abstained_provisional"
        if bool(reference.get("expect_abstention")) != abstained:
            failures.append(f"abstention={abstained} (expected {bool(reference.get('expect_abstention'))})")

        return _result(self.key, 0.0 if failures else 1.0, f"failures: {failures or 'none'}",
                       failures=failures, confidence=outputs.get("confidence_score"))


class ArchetypeMatchEvaluator(_DebitMatchEvaluator):
    """Informational: archetype chosen by the Profiling Agent vs the expected archetype (if any)."""

    key = "archetype_match"

    def score(self, inputs, outputs, reference) -> EvaluationResult:
        expected = reference.get("expected_archetype")
        if not expected:
            return _result(self.key, None, "no expected archetype for this case", applicable=False)
        actual = outputs.get("archetype_prior")
        return _result(self.key, 1.0 if actual == expected else 0.0, f"expected {expected}, got {actual}", applicable=True)


def default_evaluators(judge_llm: Any = None) -> List[_DebitMatchEvaluator]:
    return [FaithfulnessEvaluator(judge_llm), ArithmeticAccuracyEvaluator(), FreshnessSafetyEvaluator(),
            HITLEscalationEvaluator(), OutcomeCorrectnessEvaluator(), ArchetypeMatchEvaluator()]


# ----------------------------------------------------------------------- #
# Summary KPIs (shared by the local runner and LangSmith summary evaluators)
# ----------------------------------------------------------------------- #
def expected_calibration_error(confidences: Sequence[float], correct: Sequence[float], bins: int = ECE_BINS) -> float:
    """ECE = sum_b |B_b|/n * |acc(B_b) - conf(B_b)| with equal-width confidence bins."""
    if not confidences:
        return 0.0
    n = len(confidences)
    total = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (low < c <= high) or (b == 0 and c == 0.0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confidences[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - conf)
    return round(total, 4)


def p95(values: Sequence[float]) -> float:
    """Nearest-rank 95th percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)], 3)


def compute_kpis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """rows: [{"outputs": {...}, "scores": {key: score}}] -> the five KPIs with pass/fail."""
    def mean(key: str) -> float:
        values = [r["scores"][key] for r in rows if r["scores"].get(key) is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    confidences = [float((r["outputs"] or {}).get("confidence_score") or 0.0) for r in rows]
    correct = [float(r["scores"].get("outcome_correct") or 0.0) for r in rows]
    latencies = [float(((r["outputs"] or {}).get("metrics") or {}).get("elapsed_s") or 0.0) for r in rows]
    violations = sum(1 for r in rows if r["scores"].get("freshness_safety") == 0.0)

    faithfulness = mean("faithfulness")
    arithmetic = mean("arithmetic_exact_match")
    ece = expected_calibration_error(confidences, correct)
    violation_rate = round(violations / len(rows), 4) if rows else 0.0
    latency = p95(latencies)
    return {
        "faithfulness": {"label": "RAG Triad Faithfulness", "value": faithfulness, "target": f">= {FAITHFULNESS_TARGET}",
                         "passed": faithfulness >= FAITHFULNESS_TARGET},
        "arithmetic": {"label": "Exact Match Arithmetic Accuracy", "value": arithmetic, "target": "100%",
                       "passed": arithmetic >= ARITHMETIC_TARGET},
        "ece": {"label": "Expected Calibration Error (ECE)", "value": ece, "target": f"< {ECE_TARGET}",
                "passed": ece < ECE_TARGET},
        "freshness": {"label": "Data Freshness Safety Violation Rate", "value": violation_rate, "target": "0.0%",
                      "passed": violation_rate <= FRESHNESS_VIOLATION_TARGET},
        "latency": {"label": "System Velocity (P95 Latency, s)", "value": latency, "target": f"< {P95_LATENCY_TARGET_S}",
                    "passed": latency < P95_LATENCY_TARGET_S},
    }


def _rows_from_runs(runs: Sequence[Any], examples: Sequence[Any]) -> List[Dict[str, Any]]:
    rows = []
    for run, example in zip(runs, examples):
        outputs = dict(getattr(run, "outputs", None) or {})
        inputs = dict(getattr(example, "inputs", None) or {})
        reference = dict(getattr(example, "outputs", None) or {})
        scores = {ev.key: ev.score(inputs, outputs, reference).score
                  for ev in (OutcomeCorrectnessEvaluator(), FreshnessSafetyEvaluator())} if outputs and not outputs.get("error") else {}
        rows.append({"outputs": outputs, "scores": scores})
    return rows


def summary_calibration(runs: Sequence[Any], examples: Sequence[Any]) -> Dict[str, Any]:
    """LangSmith summary evaluator: ECE over the experiment."""
    rows = _rows_from_runs(runs, examples)
    kpis = compute_kpis(rows)
    return {"key": "expected_calibration_error", "score": kpis["ece"]["value"]}


def summary_p95_latency(runs: Sequence[Any], examples: Sequence[Any]) -> Dict[str, Any]:
    latencies = [float(((getattr(r, "outputs", None) or {}).get("metrics") or {}).get("elapsed_s") or 0.0) for r in runs]
    return {"key": "p95_latency_s", "score": p95(latencies)}


def summary_freshness_violation_rate(runs: Sequence[Any], examples: Sequence[Any]) -> Dict[str, Any]:
    rows = _rows_from_runs(runs, examples)
    return {"key": "freshness_violation_rate", "score": compute_kpis(rows)["freshness"]["value"]}


SUMMARY_EVALUATORS = [summary_calibration, summary_p95_latency, summary_freshness_violation_rate]
