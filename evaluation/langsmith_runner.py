"""Benchmark runner: LangSmith ``aevaluate()`` experiments with a local fallback.

Modes
    langsmith - dataset ``debitmatch_e2e_benchmark_v1`` is created/updated in LangSmith and the
                squad is evaluated with ``langsmith.aevaluate`` (experiment + per-run feedback +
                summary metrics). Requires LANGCHAIN_API_KEY in .env (read through config.py).
    local     - identical target, evaluators and KPIs, executed without network access to LangSmith.
    auto      - langsmith when a key is configured, otherwise local.

The target invokes the compiled LangGraph ``app_graph`` asynchronously (one worker thread per
case via ``asyncio.to_thread``) and records token usage (prompt/completion), per-node latency
(``timing_ms.*`` from the trajectory) and the compact final state the evaluators need.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from agents.profiling import PROFILE_OVERRIDES_KEY
from evaluation.custom_evaluators import SUMMARY_EVALUATORS, build_judge_llm, compute_kpis, default_evaluators
from evaluation.dataset_generator import DATASET_NAME, REFERENCE_DATE, apply_scenario, build_cases, push_to_langsmith
from graph import build_default_resources, build_graph, run_squad

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "evaluation" / "reports"
DEFAULT_EXPERIMENT_PREFIX = "debitmatch-e2e"
_CHUNK_TEXT_LIMIT = 1_200


# ------------------------------------------------------------------------- #
# LangSmith client
# ------------------------------------------------------------------------- #
def build_langsmith_client() -> Any:
    """LangSmith client from config.get_langsmith_settings(); None when no key is configured."""
    from config import get_langsmith_settings

    settings = get_langsmith_settings()
    if not settings["enabled"]:
        return None
    from langsmith import Client

    return Client(api_key=settings["api_key"], api_url=settings["endpoint"])


# ------------------------------------------------------------------------- #
# Target
# ------------------------------------------------------------------------- #
def _tokens(usage: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    prompt = sum(int(u.get("input_tokens", 0) or 0) for u in usage.values())
    completion = sum(int(u.get("output_tokens", 0) or 0) for u in usage.values())
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
            "by_model": {model: dict(u) for model, u in usage.items()}}


def compact_output(state: Mapping[str, Any], tokens: Mapping[str, Any]) -> Dict[str, Any]:
    """JSON-serialisable slice of the final SquadState used by the evaluators."""
    meta = state.get("execution_metadata") or {}
    draft = state.get("draft_recommendation") or {}
    return {
        "status": draft.get("status"),
        "draft_recommendation": draft,
        "data_freshness_flag": state.get("data_freshness_flag"),
        "is_fallback_mode": bool(state.get("is_fallback_mode")),
        "requires_human_review": bool(state.get("requires_human_review")),
        "review_approved": bool(state.get("review_approved")),
        "revision_count": int(state.get("revision_count") or 0),
        "confidence_score": float(state.get("confidence_score") or 0.0),
        "evaluator_score": float(state.get("evaluator_score") or 0.0),
        "archetype_prior": state.get("archetype_prior"),
        "weighted_preference_vector": state.get("weighted_preference_vector"),
        "user_profile": state.get("user_profile"),
        "data_conflicts": state.get("data_conflicts") or [],
        "guardrail_warnings": state.get("guardrail_warnings") or [],
        "api_structured_data": state.get("api_structured_data") or [],
        "rag_chunks": [{"chunk_id": c.get("chunk_id"), "text": str(c.get("text", ""))[:_CHUNK_TEXT_LIMIT],
                        "metadata": c.get("metadata"), "matched_product_keys": c.get("matched_product_keys")}
                       for c in state.get("rag_retrieved_chunks") or []],
        "trajectory_log": list(state.get("trajectory_log") or []),
        "metrics": {
            "elapsed_s": meta.get("elapsed_s"),
            "within_budget": meta.get("within_budget"),
            "timed_out": bool(meta.get("timed_out")),
            "route": meta.get("route"),
            "llm_model": meta.get("llm_model"),
            "profiling_source": meta.get("profiling_source"),
            "llm_critic_status": meta.get("llm_critic_status"),
            "narrative_source": draft.get("narrative_source"),
            "timing_ms": {k.split(".", 1)[1]: v for k, v in meta.items() if k.startswith("timing_ms.")},
            "tokens": dict(tokens),
        },
    }


class SquadBenchmarkTarget:
    """Async LangSmith target: runs one benchmark case through ``app_graph``."""

    def __init__(self, resources: Any) -> None:
        self.resources = resources
        self._graphs: Dict[str, Any] = {}

    def _resources_for(self, inputs: Mapping[str, Any]) -> Any:
        reference = date.fromisoformat(inputs.get("reference_date") or REFERENCE_DATE.isoformat())
        return dataclasses.replace(self.resources, catalog=apply_scenario(self.resources.catalog, inputs.get("scenario")),
                                   reference_date=reference)

    def _graph_for(self, inputs: Mapping[str, Any], resources: Any) -> Any:
        key = json.dumps([inputs.get("scenario") or {}, inputs.get("reference_date")], sort_keys=True)
        if key not in self._graphs:
            self._graphs[key] = build_graph(resources)  # app_graph bound to the scenario's resources
        return self._graphs[key]

    def run(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            resources = self._resources_for(inputs)
            app_graph = self._graph_for(inputs, resources)
            overrides = dict(inputs.get("profile_overrides") or {})
            state = run_squad(inputs["query"], resources, compiled_graph=app_graph,
                              weighted_preference_vector=overrides.get("weights") or None,
                              run_metadata={PROFILE_OVERRIDES_KEY: overrides,
                                            "benchmark_case": inputs.get("case_id"), "dataset": DATASET_NAME})
            usage = (state.get("execution_metadata") or {}).get("token_usage") or {}
            return compact_output(state, _tokens(usage.get("by_model") or {}))
        except Exception as exc:  # noqa: BLE001 - a failed case is scored 0, never aborts the benchmark
            logger.exception("Benchmark case %s failed", inputs.get("case_id"))
            return {"error": f"{type(exc).__name__}: {exc}",
                    "metrics": {"elapsed_s": round(time.perf_counter() - started, 3), "tokens": {}}}

    async def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return await asyncio.to_thread(self.run, inputs)


# ------------------------------------------------------------------------- #
# Runs
# ------------------------------------------------------------------------- #
def _row(case: Mapping[str, Any], outputs: Mapping[str, Any], results: Sequence[Any]) -> Dict[str, Any]:
    scores = {r.key: r.score for r in results}
    comments = {r.key: r.comment for r in results}
    if outputs.get("error"):
        scores = {key: 0.0 for key in scores}
    return {"case_id": case["inputs"]["case_id"], "category": case["inputs"]["category"],
            "inputs": case["inputs"], "outputs": dict(outputs), "scores": scores, "comments": comments}


async def run_local(target: SquadBenchmarkTarget, cases: Sequence[Mapping[str, Any]], evaluators: Sequence[Any],
                    concurrency: int = 1) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(case: Mapping[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            outputs = await target(case["inputs"])
        results = [ev.safe_score(case["inputs"], outputs, case["outputs"]) for ev in evaluators]
        return _row(case, outputs, results)

    return list(await asyncio.gather(*(one(case) for case in cases)))


async def run_langsmith(target: SquadBenchmarkTarget, cases: Sequence[Mapping[str, Any]], evaluators: Sequence[Any],
                        client: Any, concurrency: int, experiment_prefix: str,
                        metadata: Mapping[str, Any]) -> Dict[str, Any]:
    from langsmith import aevaluate

    sync = push_to_langsmith(client)
    wanted = {case["inputs"]["case_id"]: case for case in cases}
    examples = [ex for ex in client.list_examples(dataset_name=DATASET_NAME)
                if str((ex.metadata or {}).get("case_id")) in wanted]
    results = await aevaluate(target, data=examples, evaluators=list(evaluators), summary_evaluators=SUMMARY_EVALUATORS,
                              experiment_prefix=experiment_prefix, max_concurrency=max(1, concurrency), client=client,
                              metadata=dict(metadata), description="DebitMatch AI end-to-end benchmark (Checkpoint 6)")
    rows: List[Dict[str, Any]] = []
    async for item in results:
        example, run = item["example"], item["run"]
        case = wanted[str((example.metadata or {}).get("case_id"))]
        rows.append(_row(case, run.outputs or {}, item["evaluation_results"]["results"]))
    return {"rows": rows, "dataset_sync": sync, "experiment_name": getattr(results, "experiment_name", None)}


# ------------------------------------------------------------------------- #
# Orchestration
# ------------------------------------------------------------------------- #
def run_benchmark(
    dataset_size: int = 15,
    mode: str = "auto",
    use_llm: bool = True,
    judge: str = "auto",
    concurrency: int = 1,
    experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX,
    report_dir: Optional[Path] = REPORT_DIR,
) -> Dict[str, Any]:
    """Run the benchmark and return the report (KPIs, per-case rows, token totals)."""
    cases = build_cases(dataset_size)
    client = build_langsmith_client() if mode in {"auto", "langsmith"} else None
    if mode == "langsmith" and client is None:
        raise RuntimeError("Mode 'langsmith' requires LANGCHAIN_API_KEY in .env (see .env.example).")
    resolved_mode = "langsmith" if client is not None else "local"

    resources = build_default_resources(use_llm=use_llm, reference_date=REFERENCE_DATE)
    judge_llm = None
    if judge == "llm" or (judge == "auto" and resources.llm is not None):
        try:
            judge_llm = build_judge_llm()
        except ValueError as exc:  # no OpenAI key
            if judge == "llm":
                raise
            logger.info("LLM judge unavailable (%s); deterministic faithfulness only.", exc)
    evaluators = default_evaluators(judge_llm)
    target = SquadBenchmarkTarget(resources)
    metadata = {"dataset": DATASET_NAME, "dataset_size": len(cases), "llm_model": getattr(resources.llm, "model_name", None),
                "judge_model": getattr(judge_llm, "model_name", None), "reference_date": REFERENCE_DATE.isoformat()}

    started = time.perf_counter()
    extra: Dict[str, Any] = {}
    if resolved_mode == "langsmith":
        outcome = asyncio.run(run_langsmith(target, cases, evaluators, client, concurrency, experiment_prefix, metadata))
        rows = outcome.pop("rows")
        extra.update(outcome)
    else:
        rows = asyncio.run(run_local(target, cases, evaluators, concurrency))
    rows.sort(key=lambda r: [c["inputs"]["case_id"] for c in cases].index(r["case_id"]))

    report = build_report(rows, resolved_mode, metadata, round(time.perf_counter() - started, 2), extra)
    if report_dir is not None:
        report["report_path"] = str(save_report(report, report_dir))
    return report


def build_report(rows: Sequence[Mapping[str, Any]], mode: str, metadata: Mapping[str, Any], wall_time_s: float,
                 extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    kpis = compute_kpis(rows)

    def rate(key: str) -> Optional[float]:
        values = [r["scores"].get(key) for r in rows if r["scores"].get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    tokens = [((r["outputs"].get("metrics") or {}).get("tokens") or {}) for r in rows]
    node_ms: Dict[str, List[float]] = {}
    for r in rows:
        for node, ms in ((r["outputs"].get("metrics") or {}).get("timing_ms") or {}).items():
            node_ms.setdefault(node, []).append(float(ms))
    return {
        "dataset": DATASET_NAME,
        "mode": mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": wall_time_s,
        "metadata": dict(metadata),
        "kpis": kpis,
        "passed": all(k["passed"] for k in kpis.values()),
        "informational": {
            "outcome_accuracy": rate("outcome_correct"),
            "hitl_escalation_pass_rate": rate("hitl_escalation"),
            "archetype_accuracy": rate("archetype_match"),
            "mean_confidence": round(sum(float(r["outputs"].get("confidence_score") or 0) for r in rows) / max(1, len(rows)), 4),
            "errors": [r["case_id"] for r in rows if r["outputs"].get("error")],
        },
        "tokens": {
            "prompt_tokens": sum(int(t.get("prompt_tokens", 0) or 0) for t in tokens),
            "completion_tokens": sum(int(t.get("completion_tokens", 0) or 0) for t in tokens),
            "total_tokens": sum(int(t.get("total_tokens", 0) or 0) for t in tokens),
        },
        "node_latency_ms_mean": {node: round(sum(v) / len(v), 1) for node, v in node_ms.items()},
        "cases": [{
            "case_id": r["case_id"], "category": r["category"],
            "status": r["outputs"].get("status"), "route": (r["outputs"].get("metrics") or {}).get("route"),
            "requires_human_review": r["outputs"].get("requires_human_review"),
            "confidence": r["outputs"].get("confidence_score"),
            "elapsed_s": (r["outputs"].get("metrics") or {}).get("elapsed_s"),
            "tokens": ((r["outputs"].get("metrics") or {}).get("tokens") or {}).get("total_tokens"),
            "narrative_source": (r["outputs"].get("metrics") or {}).get("narrative_source"),
            "scores": r["scores"], "comments": r["comments"], "error": r["outputs"].get("error"),
        } for r in rows],
        **dict(extra or {}),
    }


def save_report(report: Mapping[str, Any], report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"benchmark_{report['mode']}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
