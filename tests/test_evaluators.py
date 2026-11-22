"""Mutation tests for the evaluation harness: evaluators must CATCH injected faults.

A real (deterministic) squad output is captured once per scenario, then mutated
with known defects. Every evaluator must score the clean output 1.0 and the
faulty one 0.0 / < 1.0. Also checks the LangSmith RunEvaluator adapter with real
``langsmith.schemas.Run`` / ``Example`` objects and the summary KPIs.

Usage::

    .venv\\Scripts\\python.exe tests\\test_evaluators.py            # deterministic checks only
    .venv\\Scripts\\python.exe tests\\test_evaluators.py --live     # + LLM faithfulness judge sensitivity (OpenAI)
"""

from __future__ import annotations

import copy
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # tests/ sits one level below the project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from langsmith.schemas import Example, Run  # noqa: E402

from evaluation.custom_evaluators import (  # noqa: E402
    ArithmeticAccuracyEvaluator,
    FaithfulnessEvaluator,
    FreshnessSafetyEvaluator,
    HITLEscalationEvaluator,
    OutcomeCorrectnessEvaluator,
    build_judge_llm,
    compute_kpis,
    expected_calibration_error,
    p95,
    summary_calibration,
)
from evaluation.dataset_generator import REFERENCE_DATE, build_cases  # noqa: E402
from evaluation.langsmith_runner import SquadBenchmarkTarget  # noqa: E402
from graph import build_default_resources  # noqa: E402

_CACHE: Dict[str, Any] = {}


def _case(case_id: str) -> Dict[str, Any]:
    return next(c for c in build_cases() if c["inputs"]["case_id"] == case_id)


def _output(case_id: str) -> Dict[str, Any]:
    if "target" not in _CACHE:
        _CACHE["target"] = SquadBenchmarkTarget(build_default_resources(use_llm=False, reference_date=REFERENCE_DATE))
    if case_id not in _CACHE:
        _CACHE[case_id] = _CACHE["target"].run(_case(case_id)["inputs"])
    return copy.deepcopy(_CACHE[case_id])


def _score(evaluator: Any, case_id: str, outputs: Dict[str, Any]) -> float:
    case = _case(case_id)
    return evaluator.safe_score(case["inputs"], outputs, case["outputs"]).score


# --------------------------------------------------------------------------- #
def test_clean_outputs_pass_every_evaluator() -> None:
    for case_id in ("arch_lifestyle_maximizer", "edge_santander_threshold_3000", "safety_stale_product", "hitl_150k"):
        outputs = _output(case_id)
        for evaluator in (FaithfulnessEvaluator(), ArithmeticAccuracyEvaluator(), FreshnessSafetyEvaluator(),
                          HITLEscalationEvaluator(), OutcomeCorrectnessEvaluator()):
            assert _score(evaluator, case_id, outputs) == 1.0, (case_id, evaluator.key)


def test_arithmetic_catches_tampered_figure_and_invented_amount() -> None:
    tampered = _output("arch_lifestyle_maximizer")
    tampered["draft_recommendation"]["base_option"]["net_annual_impact_mxn"] += 0.01  # one cent
    assert _score(ArithmeticAccuracyEvaluator(), "arch_lifestyle_maximizer", tampered) == 0.0

    invented = _output("arch_lifestyle_maximizer")
    invented["draft_recommendation"]["narrative"]["summary"] += " You will save MXN 1,234.56 per year."
    assert _score(ArithmeticAccuracyEvaluator(), "arch_lifestyle_maximizer", invented) == 0.0


def test_arithmetic_probes_are_independent_of_the_run() -> None:
    case = _case("edge_santander_threshold_3000")
    broken = copy.deepcopy(case["outputs"])
    broken["arithmetic_probes"][0]["expected"]["effective_monthly_fee_mxn"] = 140.0  # wrong ground truth
    result = ArithmeticAccuracyEvaluator().safe_score(case["inputs"], _output(case["inputs"]["case_id"]), broken)
    assert result.score == 0.0 and "probe 40014_060014@3000.0" in result.comment


def test_faithfulness_catches_unsupported_claims() -> None:
    outputs = _output("arch_lifestyle_maximizer")
    outputs["draft_recommendation"]["narrative"]["summary"] += (
        " It also includes free life insurance, and BBVA MEXICO offers a similar account.")
    assert _score(FaithfulnessEvaluator(), "arch_lifestyle_maximizer", outputs) < 0.95

    fake_clause = _output("arch_lifestyle_maximizer")
    fake_clause["draft_recommendation"]["supporting_clauses"].append(
        {"chunk_id": "fabricated", "excerpt": "The bank pays a 12% bonus on all balances."})
    assert _score(FaithfulnessEvaluator(), "arch_lifestyle_maximizer", fake_clause) < 1.0


def test_freshness_flags_definitive_answer_on_stale_data() -> None:
    outputs = _output("safety_stale_product")
    assert outputs["data_freshness_flag"] is False
    outputs["draft_recommendation"]["status"] = "definitive"
    assert _score(FreshnessSafetyEvaluator(), "safety_stale_product", outputs) == 0.0

    silent = _output("safety_stale_product")
    silent["draft_recommendation"]["warnings"] = []
    silent["guardrail_warnings"] = []
    assert _score(FreshnessSafetyEvaluator(), "safety_stale_product", silent) == 0.0


def test_hitl_flags_missing_escalation_and_boundary() -> None:
    outputs = _output("hitl_150k")
    outputs["requires_human_review"] = False
    assert _score(HITLEscalationEvaluator(), "hitl_150k", outputs) == 0.0

    low_conf = _output("arch_lifestyle_maximizer")
    low_conf["confidence_score"] = 0.55  # < 0.70 but not escalated
    assert _score(HITLEscalationEvaluator(), "arch_lifestyle_maximizer", low_conf) == 0.0

    boundary = _output("hitl_boundary_100k")
    boundary["draft_recommendation"]["human_review_reasons"] = ["high financial amount (MXN 100,000.00)"]
    assert _score(HITLEscalationEvaluator(), "hitl_boundary_100k", boundary) == 0.0


def test_langsmith_run_evaluator_adapter_and_summary() -> None:
    case = _case("hitl_150k")
    outputs = _output("hitl_150k")
    now = datetime.now(timezone.utc)
    run = Run(id=uuid.uuid4(), name="DebitMatchSquad", start_time=now, end_time=now, run_type="chain",
              inputs=case["inputs"], outputs=outputs, trace_id=uuid.uuid4())
    example = Example(id=uuid.uuid4(), dataset_id=uuid.uuid4(), inputs=case["inputs"], outputs=case["outputs"],
                      created_at=now, metadata=case["metadata"])
    for evaluator in (FaithfulnessEvaluator(), ArithmeticAccuracyEvaluator(), FreshnessSafetyEvaluator(),
                      HITLEscalationEvaluator(), OutcomeCorrectnessEvaluator()):
        result = evaluator.evaluate_run(run, example)
        assert result.key == evaluator.key and result.score == 1.0, (result.key, result.comment)
    summary = summary_calibration([run], [example])
    assert summary["key"] == "expected_calibration_error" and 0.0 <= summary["score"] <= 1.0

    failed = Run(id=uuid.uuid4(), name="x", start_time=now, run_type="chain", inputs={}, outputs={"error": "boom"},
                 trace_id=uuid.uuid4())
    assert ArithmeticAccuracyEvaluator().evaluate_run(failed, example).score == 0.0


def test_kpi_math() -> None:
    assert expected_calibration_error([0.9, 0.9, 0.9, 0.9], [1, 1, 1, 1]) == 0.1
    assert expected_calibration_error([1.0, 0.0], [1, 0]) == 0.0
    assert p95([float(i) for i in range(1, 21)]) == 19.0 and p95([3.0]) == 3.0
    rows = [{"outputs": {"confidence_score": 1.0, "metrics": {"elapsed_s": 1.0}},
             "scores": {"faithfulness": 1.0, "arithmetic_exact_match": 1.0, "outcome_correct": 1.0,
                        "freshness_safety": 1.0}}]
    assert all(k["passed"] for k in compute_kpis(rows).values())
    rows[0]["scores"]["freshness_safety"] = 0.0
    assert compute_kpis(rows)["freshness"]["passed"] is False


def test_live_llm_judge_detects_hallucination() -> None:
    """Sensitivity of the LLM faithfulness judge (requires OPENAI_API_KEY)."""
    judge = FaithfulnessEvaluator(build_judge_llm())
    outputs = _output("arch_lifestyle_maximizer")
    outputs["metrics"]["narrative_source"] = "llm"  # judge only runs on LLM narratives
    clean = _score(judge, "arch_lifestyle_maximizer", outputs)
    outputs["draft_recommendation"]["narrative"]["base_option_rationale"] = (
        "It pays a guaranteed 6% annual yield and includes free travel insurance worldwide.")
    case = _case("arch_lifestyle_maximizer")
    result = judge.safe_score(case["inputs"], outputs, case["outputs"])
    assert clean >= 0.95, clean
    assert result.score < 0.95 and (result.extra or {}).get("llm_unsupported"), result.comment


TESTS: List[Callable[[], None]] = [
    test_clean_outputs_pass_every_evaluator,
    test_arithmetic_catches_tampered_figure_and_invented_amount,
    test_arithmetic_probes_are_independent_of_the_run,
    test_faithfulness_catches_unsupported_claims,
    test_freshness_flags_definitive_answer_on_stale_data,
    test_hitl_flags_missing_escalation_and_boundary,
    test_langsmith_run_evaluator_adapter_and_summary,
    test_kpi_math,
]


def main() -> int:
    tests = TESTS + ([test_live_llm_judge_detects_hallucination] if "--live" in sys.argv else [])
    failures = 0
    for test in tests:
        started = time.perf_counter()
        try:
            test()
            print(f"PASS  {test.__name__}  ({time.perf_counter() - started:.1f}s)")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} evaluator tests passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
