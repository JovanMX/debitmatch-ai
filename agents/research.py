"""Market & Policy Research Agent: structured filter, then RECA retrieval for the finalists.

Branch A (structured API): filters the catalog by the user's hard constraints.
Branch B (vector RAG):     MMR (k=4) retrieval with metadata filters (institution_id,
                           product_id, document_type) for the FINALISTS only: the top
                           ``RAG_FINALISTS`` candidates by the evaluator's structured score,
                           plus any candidate within ``FINALIST_MARGIN`` points of the last
                           one (RECA evidence can move a product by EVIDENCE_BONUS and tie
                           reranking acts within TIE_MARGIN, so those could still win).
Branch B gets the remaining budget minus ``DOWNSTREAM_RESERVE_S``, so a slow retrieval
never starves the evaluator / advisory nodes. Results are joined, then freshness and
API-vs-RAG conflicts are evaluated.
"""

from __future__ import annotations

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Dict, List, Sequence, Tuple

from agents.common import NodeContext, SquadResources
from agents.evaluator import EVIDENCE_BONUS, STALE_DOWNWEIGHT_FACTOR, TIE_MARGIN, structured_score
from guardrails import API_BENEFIT_DOC_TYPE, detect_api_rag_conflicts
from state import SquadState, compute_data_freshness
from tools.catalog_search import baseline_products, eligible_products, filter_products, profile_is_for_minor
from tools.schemas import BankProduct

logger = logging.getLogger(__name__)

MMR_K = 4
MMR_FETCH_K = 20
MMR_LAMBDA = 0.5
RAG_FINALISTS = 10
FINALIST_MARGIN = EVIDENCE_BONUS + TIE_MARGIN  # points a non-finalist could still recover downstream
MAX_RAG_FINALISTS = 20
DOWNSTREAM_RESERVE_S = 1.5  # evaluator + advisory template always get this much time

_BASE_DOC_TYPES = ("terminos_y_condiciones_generales", "terminos_y_condiciones_producto")
_PERK_DOC_TYPES = ("bases_promociones_cashback", "poliza_seguros")
_QUERY_HINTS = {
    "fee": "monthly maintenance fee, minimum average balance, fee waiver conditions",
    "yield": "interest earned, yield on balance, return conditions",
    "perks": "cashback rewards, merchant discounts, complimentary insurance coverage",
}


def _document_types(objectives: Sequence[str]) -> List[str]:
    types = list(_BASE_DOC_TYPES) + [API_BENEFIT_DOC_TYPE]  # atomic API benefit texts (RAG design, section 4)
    if "perks" in objectives:
        types.extend(_PERK_DOC_TYPES)
    return types


def _rag_query(user_query: str, objectives: Sequence[str]) -> str:
    # Fixed hint order (not the objectives' order) so the text is one of rag_query_variants() -> prefetchable.
    hints = "; ".join(hint for objective, hint in _QUERY_HINTS.items() if objective in objectives)
    return f"{user_query} | {hints}" if hints else user_query


def rag_query_variants(user_query: str) -> List[str]:
    """Every retrieval query research can build for ``user_query`` (one per subset of objectives with hints).

    Objectives are only known after profiling, so ``run_squad`` embeds all variants up front in a single
    background API call (8 short texts) and research awaits the one it needs instead of paying the round trip.
    """
    keys = list(_QUERY_HINTS)
    subsets = [[k for bit, k in enumerate(keys) if mask >> bit & 1] for mask in range(2 ** len(keys))]
    return list(dict.fromkeys(_rag_query(user_query, subset) for subset in subsets))


def _record(product: BankProduct, checks: Dict[str, bool]) -> Dict[str, Any]:
    record = product.model_dump(mode="json")
    record["catalog_key"] = product.catalog_key
    record["hard_constraint_checks"] = checks
    return record


def _branch_a_structured(
    catalog: Sequence[BankProduct], constraints: Sequence[str], deposit: float
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, bool]]]:
    passing, checks = filter_products(catalog, constraints, deposit)
    return [_record(product, checks[product.catalog_key]) for product in passing], checks


def _fallback_baseline(
    catalog: Sequence[BankProduct], constraints: Sequence[str], deposit: float, transfers: int, channel: Any
) -> List[Dict[str, Any]]:
    """Fallback Mode (Multi-Agent plan, section 5): cached zero-fee, risk-free baseline accounts."""
    records = []
    for product, _ in baseline_products(catalog, constraints, deposit, transfers, channel):
        checks = filter_products([product], constraints, deposit)[1][product.catalog_key]
        if all(checks.values()):
            records.append(_record(product, checks))
    return records


def select_rag_finalists(records: Sequence[Dict[str, Any]], state: SquadState,
                         resources: SquadResources, stale: Sequence[str] = ()) -> List[BankProduct]:
    """Candidates that need RECA evidence: top ``RAG_FINALISTS`` by structured score + near ties,
    plus every candidate that owns product-specific RECA chunks (API-vs-RECA conflict audit).

    ``stale`` products get the evaluator's STALE_DOWNWEIGHT_FACTOR, as in the ToT ranking.
    """
    stale_keys = set(stale)
    profile = state["user_profile"]
    weights = state["weighted_preference_vector"]
    deposit = float(profile.get("deposit_mxn", 0.0))
    transfers = int(profile.get("monthly_transfers", 2))
    channel = profile.get("preferred_channel")
    scored = []
    for record in records:
        product = resources.products_by_key.get(record["catalog_key"])
        if product is not None:
            _, _, score = structured_score(product, weights, deposit, transfers, channel)
            if product.catalog_key in stale_keys:
                score *= STALE_DOWNWEIGHT_FACTOR
            scored.append((score, product))
    scored.sort(key=lambda item: (-item[0], item[1].catalog_key))
    if len(scored) <= RAG_FINALISTS:
        return [product for _, product in scored]
    cutoff = scored[RAG_FINALISTS - 1][0] - FINALIST_MARGIN
    finalists = [product for score, product in scored if score >= cutoff][:MAX_RAG_FINALISTS]
    # Conflict audit: a product with its OWN RECA document is always checked, even when the catalog figures
    # rank it low (a wrong catalog fee could be what hides it). These documents are few, so this is cheap.
    engine = resources.rag_engine
    audited = engine.product_specific_keys() if engine is not None else set()
    chosen = {p.catalog_key for p in finalists}
    finalists += [product for _, product in scored if product.catalog_key in audited and product.catalog_key not in chosen]
    return finalists


def _branch_b_rag(
    resources: SquadResources, query: str, objectives: Sequence[str], catalog: Sequence[BankProduct]
) -> Dict[str, List[Dict[str, Any]]]:
    """MMR retrieval for the given products (the finalists; joined with Branch A afterwards)."""
    engine = resources.rag_engine
    if engine is None or engine.count() == 0:
        return {}
    embedding = engine.embed_query(query)  # embed once, reuse for every filtered search
    doc_types = _document_types(objectives)
    by_product: Dict[str, List[Dict[str, Any]]] = {}
    for product in catalog:
        by_product[product.catalog_key] = engine.query_mmr(
            query,
            institution_id=product.institution_id,
            product_id=product.product_id,
            document_types=doc_types,
            k=MMR_K,
            fetch_k=MMR_FETCH_K,
            lambda_mult=MMR_LAMBDA,
            query_embedding=embedding,
        )
    return by_product


def research_node(state: SquadState, ctx: NodeContext, resources: SquadResources) -> Dict[str, Any]:
    profile = state["user_profile"]
    constraints: List[str] = list(profile.get("hard_constraints", []))
    deposit = float(profile.get("deposit_mxn", 0.0))
    objectives: List[str] = list(profile.get("objectives", []))
    rag_query = _rag_query(state["user_query"], objectives)
    excluded = list(profile.get("excluded_institutions") or [])
    catalog = eligible_products(resources.catalog, profile)
    log: List[str] = [ctx.log(f"Research started: Branch A (catalog, {len(catalog)} products), then Branch B "
                              f"(MMR k={MMR_K} over the in-memory ChromaDB snapshot) for the top-{RAG_FINALISTS} finalists + near ties.")]
    business = sum(p.is_business_account for p in resources.catalog)
    if business and not profile.get("business_account"):
        log.append(ctx.log(f"{business} business account(s) (PFAE / persona moral) excluded: personal-banking request."))
    minor_only = sum(p.for_minor_under_guardianship for p in resources.catalog)
    if minor_only and not profile_is_for_minor(profile):
        log.append(ctx.log(f"{minor_only} account(s) for a minor under guardianship excluded: the holder is not a "
                           "minor (say e.g. 'for my 10-year-old son' to include them)."))
    if excluded:
        log.append(ctx.log(f"Institutions excluded by the user: {excluded}."))
    metadata: Dict[str, Any] = {}
    fallback_reasons: List[str] = []  # structured source failed -> Fallback Mode
    degraded: List[str] = []  # RAG source failed -> recommendation without RECA evidence

    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="research")
    try:
        future_a = pool.submit(contextvars.copy_context().run, _branch_a_structured, catalog, constraints, deposit)
        try:
            api_records, checks = future_a.result(timeout=max(ctx.remaining_s(), 0.1))
        except (FutureTimeout, Exception) as exc:  # noqa: BLE001 - structured source failed -> Fallback Mode
            timed_out = isinstance(exc, FutureTimeout)
            if timed_out:
                metadata["timed_out"] = True
            api_records = _fallback_baseline(catalog, constraints, deposit, int(profile.get("monthly_transfers", 2)),
                                             profile.get("preferred_channel"))
            checks = {r["catalog_key"]: r["hard_constraint_checks"] for r in api_records}
            fallback_reasons.append("structured catalog query " + ("timed out" if timed_out else f"failed ({exc})"))
            log.append(ctx.log(f"Branch A {'exceeded the execution budget' if timed_out else 'failed'}; FALLBACK MODE: "
                               f"{len(api_records)} cached zero-fee baseline account(s) selected."))
        catalog_stale = compute_data_freshness(api_records, [], reference_date=resources.effective_reference_date())
        finalists = select_rag_finalists(api_records, state, resources, catalog_stale["stale_products"])
        log.append(ctx.log(f"RAG finalists: {len(finalists)} of {len(api_records)} candidate(s) "
                           f"(top {RAG_FINALISTS} by structured score, + ties within {FINALIST_MARGIN:.0f} points, "
                           "+ candidates with product-specific RECA documents for the conflict audit)."))
        future_b = pool.submit(contextvars.copy_context().run, _branch_b_rag, resources, rag_query, objectives,
                               finalists)
        rag_budget = ctx.remaining_s() - DOWNSTREAM_RESERVE_S
        try:
            chunks_by_product = future_b.result(timeout=max(rag_budget, 0.1))
        except FutureTimeout:
            # The evaluator still runs (no global timed_out): the advice abstains on missing RECA evidence instead.
            chunks_by_product = {}
            metadata["rag_timed_out"] = True
            degraded.append("RECA document retrieval timed out")
            log.append(ctx.log(f"Branch B exceeded its {max(rag_budget, 0.1):.2f}s share of the budget "
                               f"({DOWNSTREAM_RESERVE_S:.1f}s kept for evaluation and advice); continuing without "
                               "RAG evidence."))
        except Exception as exc:  # noqa: BLE001 - RAG failure must not kill the structured path
            chunks_by_product = {}
            metadata["rag_error"] = f"{type(exc).__name__}: {exc}"
            degraded.append("RECA document retrieval failed")
            log.append(ctx.log(f"Branch B failed ({type(exc).__name__}: {exc}); continuing without RAG evidence."))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    rejected = sorted(k for k, c in checks.items() if not all(c.values()))
    log.append(ctx.log(f"Branch A: {len(api_records)} product(s) satisfy hard constraints {constraints or '[none]'}; "
                       f"{len(rejected)} rejected."))

    # Join: keep RAG evidence only for surviving candidates, deduplicated by chunk_id.
    joined: Dict[str, Dict[str, Any]] = {}
    for record in api_records:
        for chunk in chunks_by_product.get(record["catalog_key"], []):
            entry = joined.setdefault(chunk["chunk_id"], {**chunk, "matched_product_keys": []})
            entry["matched_product_keys"].append(record["catalog_key"])
    rag_chunks = sorted(joined.values(), key=lambda c: (-c["relevance_score"], c["chunk_id"]))
    log.append(ctx.log(f"Branch B: {len(rag_chunks)} unique MMR chunk(s) retrieved for the finalists "
                       f"(document_types={_document_types(objectives)})."))

    freshness = compute_data_freshness(api_records, rag_chunks, reference_date=resources.effective_reference_date())
    conflicts = detect_api_rag_conflicts(api_records, rag_chunks)
    log.append(ctx.log(
        f"Freshness check (> 30 days = stale): flag={freshness['data_freshness_flag']}, "
        f"oldest={freshness['oldest_verified_date']}, stale_products={freshness['stale_products']}, "
        f"stale_chunks={len(freshness['stale_chunks'])}"
    ))
    if conflicts:
        log.append(ctx.log(f"API vs RAG conflicts detected: {[(c['product_key'], c['type']) for c in conflicts]}"))

    if fallback_reasons:
        log.append(ctx.log(f"Fallback Mode active: {fallback_reasons}."))
    metadata.update({"freshness_report": dict(freshness), "rejected_products": rejected, "rag_query": rag_query,
                     "fallback_reasons": fallback_reasons, "degraded_sources": degraded})
    return {
        "api_structured_data": api_records,
        "rag_retrieved_chunks": rag_chunks,
        "is_fallback_mode": bool(fallback_reasons),
        "data_freshness_flag": freshness["data_freshness_flag"],
        "data_conflicts": conflicts,
        "trajectory_log": log,
        "execution_metadata": metadata,
    }
