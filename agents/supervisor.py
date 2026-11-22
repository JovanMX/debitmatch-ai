"""Supervisor / Orchestrator Agent: flow control, routing, timeout and observability.

* :func:`supervised` wraps every node: enforces the strict execution budget
  (default 10 s), injects a :class:`NodeContext`, records per-node timings and
  converts unexpected exceptions into logged, recoverable failures.
* :func:`supervisor_node` inspects the state and chooses linear evaluation vs.
  Tree-of-Thought beam search.
* :func:`router_edge` / :func:`review_router_edge` are the conditional edges.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Literal, Tuple

from langgraph.graph import END

from agents.common import NodeContext, SquadResources
from guardrails import MAX_REVISIONS
from state import SquadState

logger = logging.getLogger(__name__)

NodeFn = Callable[[SquadState, NodeContext, SquadResources], Dict[str, Any]]

# Nodes that must still run after the budget is exhausted so the user always
# receives a safe (abstaining) answer and the HITL flag is evaluated.
ALWAYS_RUN_NODES = frozenset({"advisory_node", "review_node"})


def supervised(name: str, fn: NodeFn, resources: SquadResources) -> Callable[[SquadState], Dict[str, Any]]:
    """Wrap a node with deadline enforcement, timing and error containment."""

    def wrapper(state: SquadState) -> Dict[str, Any]:
        metadata = state.get("execution_metadata") or {}
        meta_updates: Dict[str, Any] = {}
        deadline = metadata.get("deadline_epoch")
        if deadline is None:  # graph invoked directly, not via run_squad()
            deadline = time.time() + resources.time_budget_s
            meta_updates["deadline_epoch"] = deadline

        timed_out = bool(metadata.get("timed_out")) or time.time() > deadline
        if timed_out:
            meta_updates["timed_out"] = True
            if name not in ALWAYS_RUN_NODES:
                return {
                    "trajectory_log": [f"[{name}] SKIPPED: execution budget of {resources.time_budget_s:.1f}s exceeded."],
                    "execution_metadata": meta_updates,
                }

        ctx = NodeContext(node_name=name, deadline_epoch=deadline, timed_out=timed_out)
        started = time.perf_counter()
        try:
            updates = fn(state, ctx, resources)
        except Exception as exc:  # noqa: BLE001 - contain node failures, never crash the squad
            logger.exception("Node %s failed", name)
            updates = {
                "trajectory_log": [ctx.log(f"ERROR {type(exc).__name__}: {exc}")],
                "execution_metadata": {f"error.{name}": f"{type(exc).__name__}: {exc}"},
            }
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        updates = dict(updates)
        updates["execution_metadata"] = {
            **meta_updates,
            **(updates.get("execution_metadata") or {}),
            f"timing_ms.{name}": elapsed_ms,
        }
        updates["trajectory_log"] = list(updates.get("trajectory_log") or []) + [
            f"[{name}] completed in {elapsed_ms:.1f} ms (budget remaining {ctx.remaining_s():.2f}s)"
        ]
        return updates

    wrapper.__name__ = name
    return wrapper


def decide_complexity(state: SquadState) -> Tuple[Literal["simple", "hybrid"], str]:
    """Simple -> linear scoring; hybrid -> ToT beam search over competing objectives."""
    profile = state.get("user_profile") or {}
    objectives = set(profile.get("objectives") or [])
    candidates = len(state.get("api_structured_data") or [])

    if candidates <= 1:
        return "simple", f"only {candidates} candidate(s) survive hard constraints; nothing to explore"
    if len(objectives) >= 3:
        return "hybrid", f"three competing objectives {sorted(objectives)}"
    if "yield" in objectives and len(objectives) >= 2:
        return "hybrid", f"yield must be traded off against {sorted(objectives - {'yield'})}"
    if state.get("archetype_prior") == "Flow_Manager" and len(objectives) >= 2:
        return "hybrid", "payroll profile balancing multiple objectives"
    return "simple", f"single dominant objective {sorted(objectives) or ['general']}"


def supervisor_node(state: SquadState, ctx: NodeContext, resources: SquadResources) -> Dict[str, Any]:
    complexity, reason = decide_complexity(state)
    route = "evaluator_tot_beam" if complexity == "hybrid" else "evaluator_linear"
    log: List[str] = [
        ctx.log(f"Routing decision: {complexity.upper()} query -> {route} ({reason})."),
        ctx.log(
            f"State snapshot: archetype={state.get('archetype_prior')}, candidates={len(state.get('api_structured_data') or [])}, "
            f"rag_chunks={len(state.get('rag_retrieved_chunks') or [])}, fresh={state.get('data_freshness_flag')}, "
            f"conflicts={len(state.get('data_conflicts') or [])}, budget_remaining={ctx.remaining_s():.2f}s"
        ),
    ]
    return {
        "query_complexity": complexity,
        "trajectory_log": log,
        "execution_metadata": {"route": route, "route_reason": reason},
    }


def router_edge(state: SquadState) -> str:
    return "evaluator_tot_beam" if state.get("query_complexity") == "hybrid" else "evaluator_linear"


def review_router_edge(state: SquadState) -> str:
    """Revise while the draft is rejected and N < N_max; otherwise finish (approved or HITL)."""
    if state.get("review_approved"):
        return END
    if (state.get("execution_metadata") or {}).get("timed_out"):
        return END
    if int(state.get("revision_count", 0)) < MAX_REVISIONS:
        return "evaluator_node"
    return END
