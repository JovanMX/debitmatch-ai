"""DebitMatch AI evaluation harness (HELM-style KPIs + LangSmith experiments).

Submodules are imported lazily so ``python -m evaluation.dataset_generator`` runs cleanly.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ArithmeticAccuracyEvaluator": "evaluation.custom_evaluators",
    "FaithfulnessEvaluator": "evaluation.custom_evaluators",
    "FreshnessSafetyEvaluator": "evaluation.custom_evaluators",
    "HITLEscalationEvaluator": "evaluation.custom_evaluators",
    "OutcomeCorrectnessEvaluator": "evaluation.custom_evaluators",
    "compute_kpis": "evaluation.custom_evaluators",
    "expected_calibration_error": "evaluation.custom_evaluators",
    "DATASET_NAME": "evaluation.dataset_generator",
    "build_cases": "evaluation.dataset_generator",
    "run_benchmark": "evaluation.langsmith_runner",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'evaluation' has no attribute '{name}'")
