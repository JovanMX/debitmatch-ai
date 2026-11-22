"""End-to-end benchmark CLI for DebitMatch AI (Checkpoint 6 KPIs).

Examples::

    .venv\\Scripts\\python.exe tests\\test_e2e_benchmark.py --dataset_size 15
    .venv\\Scripts\\python.exe tests\\test_e2e_benchmark.py --dataset_size 15 --mode local --offline
    .venv\\Scripts\\python.exe tests\\test_e2e_benchmark.py --mode langsmith --concurrency 2 -v

Exit code: 0 = BENCHMARK PASSED, 1 = BENCHMARK FAILED, 2 = configuration error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # tests/ sits one level below the project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")  # emoji verdicts on Windows consoles

from evaluation.dataset_generator import BENCHMARK_CASES, DATASET_NAME  # noqa: E402
from evaluation.langsmith_runner import DEFAULT_EXPERIMENT_PREFIX, REPORT_DIR, run_benchmark  # noqa: E402

PASS_BANNER = "🟢 BENCHMARK PASSED"
FAIL_BANNER = "🔴 BENCHMARK FAILED"


def _fmt(key: str, value: Any) -> str:
    if value is None:
        return "n/a"
    if key == "freshness":
        return f"{float(value) * 100:.1f}%"
    if key == "arithmetic":
        return f"{float(value) * 100:.1f}%"
    if key == "latency":
        return f"{float(value):.2f}s"
    return f"{float(value):.4f}"


def render_summary(report: Mapping[str, Any]) -> str:
    lines = []
    meta = report["metadata"]
    lines.append("")
    lines.append(f"DebitMatch AI - E2E Benchmark · dataset '{report['dataset']}' ({meta['dataset_size']} cases) · "
                 f"mode={report['mode']} · llm={meta.get('llm_model') or 'deterministic'} · "
                 f"judge={meta.get('judge_model') or 'deterministic'}")
    header = f"{'Metric / KPI':<40} {'Result':>10} {'Target':>10}   Status"
    lines += ["=" * len(header) + "=====", header, "-" * len(header) + "-----"]
    for key, kpi in report["kpis"].items():
        status = "✅ PASS" if kpi["passed"] else "❌ FAIL"
        lines.append(f"{kpi['label']:<40} {_fmt(key, kpi['value']):>10} {kpi['target']:>10}   {status}")
    lines.append("=" * len(header) + "=====")
    info = report["informational"]
    tokens = report["tokens"]
    lines.append(f"Informational: outcome accuracy {info['outcome_accuracy']}, HITL pass rate "
                 f"{info['hitl_escalation_pass_rate']}, archetype accuracy {info['archetype_accuracy']}, "
                 f"mean confidence {info['mean_confidence']}, errors {info['errors'] or 'none'}")
    lines.append(f"Tokens: prompt {tokens['prompt_tokens']:,} · completion {tokens['completion_tokens']:,} · "
                 f"total {tokens['total_tokens']:,} · wall time {report['wall_time_s']}s")
    if report.get("node_latency_ms_mean"):
        lines.append("Mean node latency (ms): " + ", ".join(f"{n} {ms:,.0f}" for n, ms in report["node_latency_ms_mean"].items()))
    if report.get("experiment_name"):
        lines.append(f"LangSmith experiment: {report['experiment_name']}")
    if report.get("report_path"):
        lines.append(f"JSON report: {report['report_path']}")
    lines.append("")
    lines.append(PASS_BANNER if report["passed"] else FAIL_BANNER)
    return "\n".join(lines)


def render_cases(report: Mapping[str, Any]) -> str:
    header = (f"{'case_id':<30} {'route':<19} {'status':<22} {'HITL':<5} {'conf':>5} {'out':>4} {'faith':>6} "
              f"{'arith':>5} {'fresh':>5} {'hitl':>5} {'time':>6} {'tok':>6}")
    lines = ["", header, "-" * len(header)]

    def s(value: Any) -> str:
        return "-" if value is None else f"{float(value):.2f}".rstrip("0").rstrip(".")

    for case in report["cases"]:
        sc = case["scores"]
        lines.append(
            f"{case['case_id']:<30} {str(case['route']):<19} {str(case['status']):<22} "
            f"{'yes' if case['requires_human_review'] else 'no':<5} {s(case['confidence']):>5} {s(sc.get('outcome_correct')):>4} "
            f"{s(sc.get('faithfulness')):>6} {s(sc.get('arithmetic_exact_match')):>5} {s(sc.get('freshness_safety')):>5} "
            f"{s(sc.get('hitl_escalation')):>5} {s(case['elapsed_s']):>6} {str(case['tokens'] or 0):>6}")
        failing = {k: v for k, v in case["comments"].items() if sc.get(k) == 0.0}
        for key, comment in failing.items():
            lines.append(f"    ↳ {key}: {comment}")
        if case.get("error"):
            lines.append(f"    ↳ ERROR: {case['error']}")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run the DebitMatch AI E2E benchmark on '{DATASET_NAME}'.")
    parser.add_argument("--dataset_size", type=int, default=len(BENCHMARK_CASES),
                        help=f"number of benchmark cases (1-{len(BENCHMARK_CASES)})")
    parser.add_argument("--mode", choices=("auto", "local", "langsmith"), default="auto",
                        help="langsmith = aevaluate() experiment (needs LANGCHAIN_API_KEY); auto falls back to local")
    parser.add_argument("--offline", action="store_true", help="deterministic squad (no OpenAI calls)")
    parser.add_argument("--judge", choices=("auto", "llm", "none"), default="auto",
                        help="faithfulness LLM judge (auto = when the squad uses the LLM)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="parallel cases (keep 1 for faithful latency measurements)")
    parser.add_argument("--experiment-prefix", default=DEFAULT_EXPERIMENT_PREFIX)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("-v", "--verbose", action="store_true", help="per-case table with failure details")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        report = run_benchmark(
            dataset_size=args.dataset_size,
            mode=args.mode,
            use_llm=not args.offline,
            judge="none" if args.offline and args.judge == "auto" else args.judge,
            concurrency=args.concurrency,
            experiment_prefix=args.experiment_prefix,
            report_dir=args.report_dir,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"Configuration error: {exc}")
        return 2
    if args.verbose:
        print(render_cases(report))
    print(render_summary(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
