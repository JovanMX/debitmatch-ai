"""Run one query through the agent squad and print what each stage decided.

Shows the inferred profile, the route, the top candidates with their scores and
annual net impact, the two recommended options and the review outcome, without
opening the Streamlit UI.

Usage (from the project root)::

    .venv\\Scripts\\python.exe scripts\\run_demo.py
    .venv\\Scripts\\python.exe scripts\\run_demo.py "I want an account with cashback" --deposit 30000
    .venv\\Scripts\\python.exe scripts\\run_demo.py --trace
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from graph import build_default_resources, build_graph, run_squad  # noqa: E402

DEFAULT_QUERY = "I want a digital debit account with zero monthly fee and some yield. I keep about 20,000 pesos."
TOP_N = 5


def _money(value: Optional[float]) -> str:
    return "n/a" if value is None else f"MXN {value:,.2f}"


def _option_line(label: str, option: Optional[Mapping[str, Any]]) -> str:
    if not option:
        return f"  {label:<12} none"
    return (f"  {label:<12} {option['institution_name']} {option['product_name']} "
            f"(net {_money(option.get('net_annual_impact_mxn'))}/year, "
            f"fees {_money(option.get('total_annual_fees_mxn'))}, "
            f"rate {option.get('applicable_yield_rate_percent', 0.0):.2f}%)")


def print_report(result: Mapping[str, Any], show_trace: bool) -> None:
    profile = result.get("user_profile") or {}
    weights = result.get("weighted_preference_vector") or {}
    meta = result.get("execution_metadata") or {}
    draft = result.get("draft_recommendation") or {}

    print("\n== Profile")
    print(f"  archetype    {profile.get('archetype')} (source: {profile.get('profiling_source')})")
    print(f"  deposit      {_money(profile.get('deposit_mxn'))}")
    print("  weights      " + ", ".join(f"{k.replace('_importance', '')}={v:.2f}" for k, v in weights.items()))
    print(f"  constraints  {profile.get('hard_constraints') or []}")
    print(f"  route        {result.get('query_complexity')}")

    finals = sorted((b for b in result.get("tot_candidate_branches") or [] if b.get("final")),
                    key=lambda b: b.get("final_rank", 99))
    if finals:
        print(f"\n== Top {min(TOP_N, len(finals))} candidates (of {len(finals)} ranked)")
        for branch in finals[:TOP_N]:
            impact: Dict[str, Any] = branch.get("net_impact") or {}
            print(f"  #{branch['final_rank']:<2} {branch['score']:6.2f}  {branch['product_key']:<28} "
                  f"net {_money(impact.get('net_annual_impact_mxn'))}")

    print(f"\n== Recommendation ({draft.get('status')})")
    print(_option_line("Base", draft.get("base_option")))
    print(_option_line("Trade-off", draft.get("tradeoff_option")))
    explanation = draft.get("top_score_explanation")
    if explanation:
        print(f"  note         {explanation['text']}")
    summary = (draft.get("narrative") or {}).get("summary")
    if summary:
        print(f"\n  {summary}")

    print("\n== Review")
    print(f"  approved     {result.get('review_approved')} after {result.get('revision_count', 0)} revision(s)")
    print(f"  confidence   {result.get('confidence_score', 0.0):.2f}")
    print(f"  human review {result.get('requires_human_review')}")
    for reason in draft.get("human_review_reasons") or []:
        print(f"    - {reason}")
    for warning in result.get("guardrail_warnings") or []:
        print(f"  warning      {warning}")
    tokens = (meta.get("token_usage") or {}).get("total_tokens", 0)
    print(f"  elapsed      {meta.get('elapsed_s')} s (budget {meta.get('time_budget_s')} s), tokens {tokens}")

    if show_trace:
        print("\n== Trajectory log")
        for line in result.get("trajectory_log") or []:
            print(f"  {line}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run one DebitMatch AI query from the terminal.")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--deposit", type=float, help="Average balance in MXN (overrides the one in the query).")
    parser.add_argument("--trace", action="store_true", help="Print the full trajectory log.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print("Loading catalog, document index and models...")
    resources = build_default_resources()
    overrides = {"profile_overrides": {"deposit_mxn": args.deposit}} if args.deposit is not None else None
    print(f"Query: {args.query}")
    result = run_squad(args.query, resources, build_graph(resources), run_metadata=overrides)
    print_report(result, args.trace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
