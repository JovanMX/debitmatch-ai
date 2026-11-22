"""LangGraph topology for the DebitMatch AI agent squad.

    START -> intent_router_node --(product fact / benefit question about a named product)--> direct_answer_node -> END
                                --(recommendation)--> profiling_node
    profiling_node -> research_node -> supervisor_node -> router_edge
                      router_edge --(simple)--> evaluator_linear   --+
                      router_edge --(hybrid)--> evaluator_tot_beam --+--> advisory_node -> review_node
    review_node -> review_router_edge --(needs revision & N < 2)--> evaluator_node -> advisory_node
                                      --(approved OR HITL / N_max)--> END

``review_node`` is the Evaluator Agent in critic mode; ``evaluator_node`` is
its revision step (increments ``revision_count``).

Usage::

    resources = build_default_resources()
    final_state = run_squad("I want a digital debit account with zero monthly fee.", resources)
"""

from __future__ import annotations

import contextvars
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.graph import END, START, StateGraph

from agents.advisory import advisory_node, baseline_option
from agents.common import DEFAULT_TIME_BUDGET_S, SquadResources
from agents.evaluator import evaluator_linear_node, evaluator_revision_node, evaluator_tot_beam_node, review_node
from agents.llm import build_default_llm
from agents.profiling import profiling_node
from agents.research import rag_query_variants, research_node
from agents.router import direct_answer_node, intent_edge, intent_router_node
from agents.supervisor import review_router_edge, router_edge, supervised, supervisor_node
from guardrails import human_review_reasons
from prompts.advisory_prompts import ABSTENTION_WARNING, FALLBACK_DISCLAIMER, HITL_NOTICE, LEGAL_DISCLAIMER, TOT_TIMEOUT_NOTICE
from state import SquadState, create_initial_state

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.json"
DEFAULT_DOCS_PATH = PROJECT_ROOT / "data" / "rag_docs"
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
HARD_TIMEOUT_GRACE_S = 2.0  # outer guard on top of the cooperative per-node budget
RECURSION_LIMIT = 30


def build_graph(resources: SquadResources):
    """Compile the squad StateGraph bound to ``resources``."""
    graph = StateGraph(SquadState)
    nodes = {
        "intent_router_node": intent_router_node,
        "direct_answer_node": direct_answer_node,
        "profiling_node": profiling_node,
        "research_node": research_node,
        "supervisor_node": supervisor_node,
        "evaluator_linear": evaluator_linear_node,
        "evaluator_tot_beam": evaluator_tot_beam_node,
        "advisory_node": advisory_node,
        "review_node": review_node,
        "evaluator_node": evaluator_revision_node,
    }
    for name, fn in nodes.items():
        graph.add_node(name, supervised(name, fn, resources))

    graph.add_edge(START, "intent_router_node")
    graph.add_conditional_edges("intent_router_node", intent_edge,
                                {"direct_answer_node": "direct_answer_node", "profiling_node": "profiling_node"})
    graph.add_edge("direct_answer_node", END)
    graph.add_edge("profiling_node", "research_node")
    graph.add_edge("research_node", "supervisor_node")
    graph.add_conditional_edges(
        "supervisor_node",
        router_edge,
        {"evaluator_linear": "evaluator_linear", "evaluator_tot_beam": "evaluator_tot_beam"},
    )
    graph.add_edge("evaluator_linear", "advisory_node")
    graph.add_edge("evaluator_tot_beam", "advisory_node")
    graph.add_edge("advisory_node", "review_node")
    graph.add_conditional_edges("review_node", review_router_edge, {"evaluator_node": "evaluator_node", END: END})
    graph.add_edge("evaluator_node", "advisory_node")
    return graph.compile()


def build_default_resources(
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    chroma_path: Optional[Path] = DEFAULT_CHROMA_PATH,
    docs_path: Path = DEFAULT_DOCS_PATH,
    use_llm: bool = True,
    reference_date: Optional[date] = None,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
) -> SquadResources:
    """Load catalog, RAG engine (ingesting if empty) and LLM - outside the query budget."""
    from config import configure_tracing
    from tools.rag_engine import InstitutionalRAGEngine, institution_names_from_catalog
    from tools.schemas import load_catalog

    tracing = configure_tracing()
    catalog = load_catalog(catalog_path, strict=True)
    engine = InstitutionalRAGEngine(persist_directory=chroma_path)
    if engine.count() == 0 and docs_path.is_dir():
        engine.ingest_directory(docs_path, institution_names=institution_names_from_catalog(catalog))
    from tools.schemas import API_BENEFIT_DOC_TYPE

    if engine.count_document_type(API_BENEFIT_DOC_TYPE) == 0:
        engine.add_benefit_texts(catalog)  # atomic API benefit chunks (one per benefit entry)
    engine.embed_query("warm-up")  # load the embedding model now, not inside the 10 s budget
    engine.warm_up()  # in-memory vector index + conflict-audit keys, also outside the budget

    llm = build_default_llm() if use_llm else None
    logger.info("Resources ready: %d products, %d chunks, llm=%s, tracing=%s",
                len(catalog), engine.count(), getattr(llm, "model_name", None), tracing)
    return SquadResources(
        catalog=catalog,
        rag_engine=engine,
        llm=llm,
        reference_date=reference_date,
        time_budget_s=time_budget_s,
        extra={"langsmith_tracing": tracing},
    )


def _baseline_option(state: Mapping[str, Any], resources: SquadResources) -> Optional[Dict[str, Any]]:
    """Fallback Mode: best cached zero-fee baseline account (deterministic calculator figures)."""
    option = baseline_option(state, resources)
    if option is not None:  # hard timeout: the score is not part of a ranking, keep the historical neutral value
        option.update(composite_score=0.0, hard_constraints_met=None, hard_constraint_checks={})
    return option


def _timeout_fallback(state: SquadState, budget: float, resources: Optional[SquadResources] = None) -> Dict[str, Any]:
    """Safe answer when even the outer hard timeout fires (cached baseline + disclaimers)."""
    fallback: Dict[str, Any] = dict(state)
    fallback["execution_metadata"] = {**state["execution_metadata"], "timed_out": True, "hard_timeout": True}
    reasons = human_review_reasons(fallback)
    base = _baseline_option(state, resources) if resources is not None else None
    warnings = [ABSTENTION_WARNING.format(reasons=f"the analysis exceeded its {budget:.0f}-second budget"),
                FALLBACK_DISCLAIMER] + ([TOT_TIMEOUT_NOTICE] if base else [])
    fallback.update(
        requires_human_review=True,
        review_approved=False,
        is_fallback_mode=True,
        guardrail_warnings=warnings,
        draft_recommendation={
            "status": "abstained_provisional",
            "base_option": base,
            "tradeoff_option": None,
            "narrative": {"summary": (f"Provisional baseline: {base['institution_name']} {base['product_name']} has "
                                      f"total annual fees of MXN {base['total_annual_fees_mxn']:,.2f}.") if base else "",
                          "key_tradeoffs": []},
            "narrative_source": "fallback_baseline",
            "supporting_clauses": [],
            "negotiation": [],
            "is_fallback_mode": True,
            "warnings": warnings,
            "disclaimers": [FALLBACK_DISCLAIMER, LEGAL_DISCLAIMER],
            "requires_human_review": True,
            "human_review_reasons": reasons,
            "hitl_notice": HITL_NOTICE.format(reasons="; ".join(reasons)),
        },
        trajectory_log=list(state["trajectory_log"]) + [
            f"[supervisor] HARD TIMEOUT after {budget:.1f}s; FALLBACK MODE with cached baseline "
            f"{base['product_key'] if base else 'none'}; abstaining."],
    )
    return fallback


ProgressCallback = Callable[[str, Mapping[str, Any]], None]
_PROGRESS_POLL_S = 0.05


def _stream_with_progress(
    graph: Any,
    state: SquadState,
    config: Dict[str, Any],
    pool: ThreadPoolExecutor,
    on_progress: ProgressCallback,
    hard_deadline: float,
) -> Dict[str, Any]:
    """Stream the graph in a worker thread; relay node updates to the caller's thread."""
    events: "queue.SimpleQueue[Tuple[str, Mapping[str, Any]]]" = queue.SimpleQueue()

    def stream() -> Dict[str, Any]:
        final: Dict[str, Any] = dict(state)
        for mode, chunk in graph.stream(state, config, stream_mode=["updates", "values"]):
            if mode == "values":
                final = dict(chunk)
            else:
                for node, update in (chunk or {}).items():
                    events.put((node, update or {}))
        return final

    def relay() -> None:
        while True:
            try:
                node, update = events.get_nowait()
            except queue.Empty:
                return
            try:
                on_progress(node, update)
            except Exception:  # noqa: BLE001 - a UI callback must never break the squad
                logger.exception("Progress callback failed for node %s", node)

    future = pool.submit(contextvars.copy_context().run, stream)
    while not future.done():
        relay()
        if time.perf_counter() > hard_deadline:
            raise FutureTimeout()
        time.sleep(_PROGRESS_POLL_S)
    relay()
    return future.result()


def run_squad(
    user_query: str,
    resources: SquadResources,
    compiled_graph: Any = None,
    weighted_preference_vector: Optional[Mapping[str, float]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Execute one query through the squad under the strict time budget.

    ``on_progress(node_name, node_update)`` - optional; when given, the graph is
    streamed and the callback runs in the CALLER's thread after each node
    (safe for UI frameworks such as Streamlit that forbid cross-thread writes).
    """
    graph = compiled_graph or build_graph(resources)
    budget = resources.time_budget_s
    started_wall = time.time()
    engine = getattr(resources, "rag_engine", None)
    if hasattr(engine, "prefetch_query_embeddings"):  # overlap the embedding round trip with routing + profiling
        try:
            engine.prefetch_query_embeddings(rag_query_variants(user_query.strip()))
        except Exception:  # noqa: BLE001 - an optimisation only; research embeds on demand otherwise
            logger.warning("Query embedding prefetch failed", exc_info=True)
    state = create_initial_state(
        user_query,
        weighted_preference_vector=weighted_preference_vector,
        execution_metadata={
            "deadline_epoch": started_wall + budget,
            "time_budget_s": budget,
            "llm_model": getattr(resources.llm, "model_name", None),
            "reference_date": resources.effective_reference_date().isoformat(),
            **(run_metadata or {}),
        },
    )
    config = {
        "run_name": "DebitMatchSquad",
        "tags": ["debitmatch-ai", "capstone-cmu"],
        "metadata": {"llm_model": getattr(resources.llm, "model_name", "deterministic"), **(run_metadata or {})},
        "recursion_limit": RECURSION_LIMIT,
    }

    started = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="squad")
    usage_cb = get_usage_metadata_callback()
    usage = usage_cb.__enter__()  # token usage of every LLM call in this run (copied into worker threads)
    try:
        try:
            if on_progress is None:
                future = pool.submit(contextvars.copy_context().run, graph.invoke, state, config)
                result: Dict[str, Any] = dict(future.result(timeout=budget + HARD_TIMEOUT_GRACE_S))
            else:
                result = _stream_with_progress(graph, state, config, pool, on_progress,
                                               started + budget + HARD_TIMEOUT_GRACE_S)
        except FutureTimeout:
            logger.error("Squad exceeded hard timeout of %.1fs", budget + HARD_TIMEOUT_GRACE_S)
            result = _timeout_fallback(state, budget, resources)
    finally:
        usage_cb.__exit__(None, None, None)
        pool.shutdown(wait=False, cancel_futures=True)

    elapsed = round(time.perf_counter() - started, 3)
    by_model = {model: dict(u) for model, u in (usage.usage_metadata or {}).items()}
    prompt_tokens = sum(int(u.get("input_tokens", 0) or 0) for u in by_model.values())
    completion_tokens = sum(int(u.get("output_tokens", 0) or 0) for u in by_model.values())
    result["execution_metadata"] = {
        **result.get("execution_metadata", {}),
        "token_usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens, "by_model": by_model},
        "elapsed_s": elapsed,
        "within_budget": elapsed <= budget,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return result
