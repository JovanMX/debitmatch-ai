"""Infrastructure acceptance tests for DebitMatch AI.

Run with:  python test_infra.py
Exit code is 0 only when every test passes. No pytest dependency required
(the ``test_*`` functions are also pytest-compatible).
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import time
import traceback
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, get_type_hints

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # tests/ sits one level below the project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import TypeAdapter, ValidationError  # noqa: E402

from state import SquadState, compute_data_freshness, create_initial_state, is_data_fresh  # noqa: E402
from tools import calculate_net_impact, load_catalog  # noqa: E402
from tools.calculator import IsrParameters  # noqa: E402
from tools.rag_engine import (  # noqa: E402
    ALL_PRODUCTS,
    CHROMA_DEFAULT_EMBEDDING,
    PDF_LIBRARY_MAP,
    SCOPED_PRODUCTS,
    InstitutionalRAGEngine,
    institution_names_from_catalog,
)
from tools.schemas import (  # noqa: E402
    CHANNEL_BRANCH,
    CHANNEL_MOBILE_APP,
    BankProduct,
    CatalogConsistencyError,
    audit_catalog,
    audit_product,
)

CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.json"
RAG_DOCS_PATH = PROJECT_ROOT / "data" / "rag_docs"

SANTANDER_ID = "40014"
LIKEU_PRO_ID = "060014"
HSBC_ID = "40021"

NESTED_ARRAYS = ("yield_tiers", "monthly_fees_detail", "transfer_fees_detail", "penalties_detail")
EXPECTED_RESULT_KEYS = {
    "deposit_evaluated_mxn",
    "applicable_yield_rate_percent",
    "gross_annual_interest_mxn",
    "effective_monthly_fee_mxn",
    "fee_waived_by_balance",
    "total_annual_fees_mxn",
    "net_annual_impact_mxn",
    "is_net_positive",
    "error",
}


def _find_product(institution_id: str, product_id: str) -> Dict[str, Any]:
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    for product in raw:
        if product["institution_id"] == institution_id and product["product_id"] == product_id:
            return product
    raise AssertionError(f"Product {institution_id}_{product_id} not found in catalog.")


# --------------------------------------------------------------------------- #
# 1. Catalog loading & structure
# --------------------------------------------------------------------------- #
def test_catalog_structure() -> None:
    products = load_catalog(CATALOG_PATH)  # full Pydantic validation
    assert len(products) > 0, "Catalog is empty."

    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, list) and len(raw) == len(products)
    for item in raw:
        for key in NESTED_ARRAYS:
            assert isinstance(item.get(key), list), f"{item['product_id']}.{key} must be a list."

    keys = {p.catalog_key for p in products}
    for required in (f"{SANTANDER_ID}_{LIKEU_PRO_ID}", "37166_DEB0107", "40021_BASICAGRAL25", "40002_SwitchBanamex"):
        assert required in keys, f"Missing product {required}."

    likeu = next(p for p in products if p.catalog_key == f"{SANTANDER_ID}_{LIKEU_PRO_ID}")
    exempt_fees = [f for f in likeu.monthly_fees_detail if f.has_minimum_balance_exemption]
    assert exempt_fees and exempt_fees[0].cost_mxn == 140.0
    assert exempt_fees[0].waiver_minimum_average_balance_mxn == 3000.0

    # Schema must reject malformed products (defensive contract check).
    broken = dict(raw[0])
    broken["monthly_maintenance_fee"] = -1
    try:
        TypeAdapter(type(products[0])).validate_python(broken)
    except ValidationError:
        pass
    else:
        raise AssertionError("Negative fee was accepted by the schema.")


def test_catalog_consistency_audit() -> None:
    products = load_catalog(CATALOG_PATH, strict=True)  # current catalog has no hard errors
    found = {(i.catalog_key, i.code) for i in audit_catalog(products)}
    for expected in (
        ("40014_3120", "BALANCE_FEE_WITHOUT_WAIVER"),
        ("40014_72213", "BALANCE_FEE_WITHOUT_WAIVER"),
        ("40014_72214", "BALANCE_FEE_WITHOUT_WAIVER"),
        ("37166_DEB0108", "PAYROLL_NAME_FLAG"),
        (f"{SANTANDER_ID}_{LIKEU_PRO_ID}", "NO_DIGITAL_SPEI_ENTRY"),
    ):
        assert expected in found, f"Expected audit warning {expected} not raised."

    # A summary field that contradicts its detail array is a hard error.
    raw = _find_product(SANTANDER_ID, LIKEU_PRO_ID)
    tampered = BankProduct.model_validate({**raw, "monthly_maintenance_fee": 99.0})
    codes = {i.code for i in audit_product(tampered) if i.severity == "error"}
    assert codes == {"FEE_SUM_MISMATCH"}, codes

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        bad_catalog = Path(tmp) / "catalog.json"
        bad_catalog.write_text(json.dumps([{**raw, "spei_transfer_fee": 1.0}]), encoding="utf-8")
        try:
            load_catalog(bad_catalog, strict=True)
        except CatalogConsistencyError as exc:
            assert exc.issues[0].code == "SPEI_FEE_MISMATCH"
        else:
            raise AssertionError("strict=True did not raise on an inconsistent catalog.")
        assert len(load_catalog(bad_catalog, strict=False)) == 1  # non-strict only logs


# --------------------------------------------------------------------------- #
# 2. Deterministic fee-waiver calculation
# --------------------------------------------------------------------------- #
def test_calculator_fee_waiver() -> None:
    likeu = _find_product(SANTANDER_ID, LIKEU_PRO_ID)
    kwargs = {
        "yield_tiers": likeu["yield_tiers"],
        "monthly_fees_detail": likeu["monthly_fees_detail"],
        "estimated_monthly_transfers": 2,
        "spei_fee_per_transfer": likeu["spei_transfer_fee"],
    }

    waived = calculate_net_impact(deposit=3500.0, **kwargs)
    assert set(waived) == EXPECTED_RESULT_KEYS, f"Unexpected keys: {set(waived) ^ EXPECTED_RESULT_KEYS}"
    assert waived["error"] is None, waived["error"]
    assert waived["effective_monthly_fee_mxn"] == 0.0
    assert waived["fee_waived_by_balance"] is True
    # branch SPEI MXN 110 (LikeU Pro fee clause): 110 * 2 * 12, maintenance waived
    assert waived["total_annual_fees_mxn"] == 2640.0

    charged = calculate_net_impact(deposit=1500.0, **kwargs)
    assert charged["error"] is None, charged["error"]
    assert charged["effective_monthly_fee_mxn"] == 140.0
    assert charged["fee_waived_by_balance"] is False
    assert charged["total_annual_fees_mxn"] == 4320.0  # 140*12 + 110*2*12
    assert charged["net_annual_impact_mxn"] == -4320.0
    assert charged["is_net_positive"] is False

    # Boundary: exactly the threshold qualifies for the waiver.
    assert calculate_net_impact(deposit=3000.0, **kwargs)["fee_waived_by_balance"] is True

    # Determinism: identical inputs -> identical outputs.
    assert calculate_net_impact(deposit=1500.0, **kwargs) == charged


def test_calculator_yield_tiers_and_errors() -> None:
    tiers = [
        {"tier_level": 1, "min_balance_mxn": 0.0, "max_balance_mxn": 10000.0, "annual_yield_rate": 5.0, "payout_frequency": "daily"},
        {"tier_level": 2, "min_balance_mxn": 10000.01, "max_balance_mxn": None, "annual_yield_rate": 10.0, "payout_frequency": "daily"},
    ]
    low = calculate_net_impact(deposit=5000.0, yield_tiers=tiers)
    assert low["applicable_yield_rate_percent"] == 5.0
    assert low["gross_annual_interest_mxn"] == 250.0
    assert low["is_net_positive"] is True

    high = calculate_net_impact(deposit=50000.0, yield_tiers=tiers)  # open-ended top tier
    assert high["applicable_yield_rate_percent"] == 10.0
    assert high["gross_annual_interest_mxn"] == 5000.0

    for bad_call in (
        lambda: calculate_net_impact(deposit=-1.0),
        lambda: calculate_net_impact(deposit="1000"),  # type: ignore[arg-type]
        lambda: calculate_net_impact(deposit=float("nan")),
        lambda: calculate_net_impact(deposit=100.0, estimated_monthly_transfers=-3),
        lambda: calculate_net_impact(deposit=100.0, monthly_fees_detail=[{"service_concept": "x"}]),
    ):
        result = bad_call()
        assert isinstance(result["error"], str) and result["error"], "Expected an error message."
        assert set(result) == EXPECTED_RESULT_KEYS


def test_calculator_channel_aware_transfers() -> None:
    smart = _find_product(SANTANDER_ID, "06140925")  # SPEI: branch 100, app/web 0
    base = {"monthly_fees_detail": smart["monthly_fees_detail"], "estimated_monthly_transfers": 2,
            "spei_fee_per_transfer": smart["spei_transfer_fee"], "transfer_fees_detail": smart["transfer_fees_detail"],
            "include_breakdown": True}

    app = calculate_net_impact(deposit=5000.0, preferred_channel=CHANNEL_MOBILE_APP, **base)
    assert app["breakdown"]["transfer_fee_source"] == "exact_channel"
    assert app["breakdown"]["annual_transfer_fees_mxn"] == 0.0
    assert app["total_annual_fees_mxn"] == 240.0  # only the 20 MXN monthly fee

    branch = calculate_net_impact(deposit=5000.0, preferred_channel=CHANNEL_BRANCH, **base)
    assert branch["breakdown"]["annual_transfer_fees_mxn"] == 2400.0
    assert branch["total_annual_fees_mxn"] == 2640.0

    # Legacy behaviour is unchanged when transfer_fees_detail is not supplied.
    legacy = calculate_net_impact(5000.0, None, smart["monthly_fees_detail"], 2, smart["spei_transfer_fee"])
    assert legacy["total_annual_fees_mxn"] == 2640.0 and "breakdown" not in legacy

    # LikeU Pro lists SPEI only at the branch -> app users fall back to it, with a note.
    likeu = _find_product(SANTANDER_ID, LIKEU_PRO_ID)
    fallback = calculate_net_impact(deposit=3500.0, monthly_fees_detail=likeu["monthly_fees_detail"],
                                    transfer_fees_detail=likeu["transfer_fees_detail"],
                                    preferred_channel=CHANNEL_MOBILE_APP, include_breakdown=True)
    assert fallback["breakdown"]["transfer_fee_source"] == "fallback_cheapest_listed_channel"
    assert fallback["breakdown"]["transfer_channel_used"] == CHANNEL_BRANCH and fallback["breakdown"]["notes"]

    # Free monthly allowance reduces billable transfers.
    allowance = [{"transfer_type": "Interbank SPEI Transfer", "channel": CHANNEL_MOBILE_APP,
                  "cost_per_transaction_mxn": 5.0, "free_transactions_per_month": 3}]
    free = calculate_net_impact(deposit=0.0, estimated_monthly_transfers=5, transfer_fees_detail=allowance,
                                preferred_channel=CHANNEL_MOBILE_APP, include_breakdown=True)
    assert free["breakdown"]["billable_transfers_per_month"] == 2
    assert free["total_annual_fees_mxn"] == 120.0  # 2 * 5 * 12


def test_calculator_penalties_and_isr() -> None:
    likeu = _find_product(SANTANDER_ID, LIKEU_PRO_ID)
    with_penalty = calculate_net_impact(
        deposit=3500.0, monthly_fees_detail=likeu["monthly_fees_detail"], estimated_monthly_transfers=0,
        penalties_detail=likeu["penalties_detail"],
        expected_penalty_events_per_year={"Account Inactivity Fee": 1}, include_breakdown=True,
    )
    assert with_penalty["breakdown"]["annual_expected_penalties_mxn"] == 50.0
    assert with_penalty["total_annual_fees_mxn"] == 50.0
    no_events = calculate_net_impact(deposit=3500.0, penalties_detail=likeu["penalties_detail"])
    assert no_events["total_annual_fees_mxn"] == 0.0  # per_event penalties default to 0 events

    tier = [{"tier_level": 1, "min_balance_mxn": 0.0, "max_balance_mxn": None,
             "annual_yield_rate": 10.0, "payout_frequency": "monthly"}]
    taxed = calculate_net_impact(deposit=300_000.0, yield_tiers=tier, estimated_monthly_transfers=0,
                                 include_breakdown=True)
    assert taxed["gross_annual_interest_mxn"] == 30_000.0
    assert taxed["breakdown"]["isr_withheld_mxn"] == 2_700.0  # 0.90 % of capital
    assert taxed["net_annual_impact_mxn"] == 27_300.0

    exempt = calculate_net_impact(deposit=100_000.0, yield_tiers=tier, include_breakdown=True)
    assert exempt["breakdown"]["isr_exempt"] is True  # <= 5 annual UMAs (213,973.20 MXN)
    assert exempt["net_annual_impact_mxn"] == 10_000.0

    low_tier = [{**tier[0], "annual_yield_rate": 0.5}]
    capped = calculate_net_impact(deposit=300_000.0, yield_tiers=low_tier, include_breakdown=True)
    assert capped["breakdown"]["isr_withheld_mxn"] == 1_500.0  # capped at interest earned
    assert capped["net_annual_impact_mxn"] == 0.0

    untaxed = calculate_net_impact(deposit=300_000.0, yield_tiers=tier, apply_isr_withholding=False)
    assert untaxed["net_annual_impact_mxn"] == 30_000.0
    custom = calculate_net_impact(deposit=300_000.0, yield_tiers=tier,
                                  isr_parameters=IsrParameters(retention_rate_percent=0.5, exempt_uma_multiple=0))
    assert custom["net_annual_impact_mxn"] == 28_500.0


# --------------------------------------------------------------------------- #
# 3. RAG retrieval with metadata filters
# --------------------------------------------------------------------------- #
def _build_engine(persist_dir: Path, docs_root: Path) -> tuple[InstitutionalRAGEngine, Dict[str, int]]:
    engine = InstitutionalRAGEngine(persist_directory=persist_dir)
    names = institution_names_from_catalog(load_catalog(CATALOG_PATH))
    stats = engine.ingest_directory(docs_root, institution_names=names)
    return engine, stats


def test_rag_metadata_filtering() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine, stats = _build_engine(Path(tmp) / "chroma", RAG_DOCS_PATH)
        assert stats["documents_indexed"] == 7 and stats["chunks_upserted"] > 0, stats

        results = engine.query_qualitative_docs(
            "What fee is charged if my average balance falls below the minimum?",
            institution_id=SANTANDER_ID,
            product_id=LIKEU_PRO_ID,
            top_k=3,
        )
        assert results, "No chunks retrieved for Santander LikeU Pro."
        for chunk in results:
            meta = chunk["metadata"]
            assert meta["institution_id"] == SANTANDER_ID
            assert meta["product_id"] in {LIKEU_PRO_ID, ALL_PRODUCTS}
            for key in ("institution_name", "document_type", "applies_to_all_products", "last_verified_date"):
                assert key in meta, f"Missing metadata key {key}."
        assert "140.00" in results[0]["text"], f"Top chunk is not the waiver clause: {results[0]['text'][:120]}"

        # Multilingual embeddings: a Spanish question must hit the English clause.
        spanish = engine.query_qualitative_docs(
            "¿Cuánto me cobran si mi saldo promedio mensual queda por debajo del mínimo?",
            institution_id=SANTANDER_ID,
            product_id=LIKEU_PRO_ID,
            top_k=1,
        )
        assert spanish and "140.00" in spanish[0]["text"], f"Spanish query missed: {spanish[0]['text'][:120]}"

        # Institution-only filter must never leak other banks.
        hsbc = engine.query_qualitative_docs("insurance coverage exclusions", institution_id=HSBC_ID, top_k=5)
        assert hsbc and all(c["metadata"]["institution_id"] == HSBC_ID for c in hsbc)
        assert hsbc[0]["metadata"]["document_type"] == "poliza_seguros"

        # Another Santander product sees ONLY the shared framework contract.
        other = engine.query_qualitative_docs("inactivity fee", institution_id=SANTANDER_ID, product_id="3120", top_k=5)
        assert other and all(c["metadata"]["product_id"] == ALL_PRODUCTS for c in other)

        # Re-ingestion is idempotent (deterministic IDs + upsert).
        before = engine.count()
        engine.ingest_directory(RAG_DOCS_PATH)
        assert engine.count() == before, "Re-ingestion duplicated chunks."

        # Input validation.
        for bad in ({"query": "  "}, {"query": "fees", "top_k": 0}):
            try:
                engine.query_qualitative_docs(**bad)  # type: ignore[arg-type]
            except ValueError:
                continue
            raise AssertionError(f"Invalid input accepted: {bad}")

        # Opening the same collection with a different embedding model must fail loudly...
        try:
            InstitutionalRAGEngine(persist_directory=Path(tmp) / "chroma", embedding_model=CHROMA_DEFAULT_EMBEDDING)
        except ValueError:
            pass
        else:
            raise AssertionError("Embedding-model mismatch was not detected.")
        # ...unless the caller explicitly resets it.
        rebuilt = InstitutionalRAGEngine(
            persist_directory=Path(tmp) / "chroma", embedding_model=CHROMA_DEFAULT_EMBEDDING, reset=True
        )
        assert rebuilt.count() == 0


def test_rag_framework_contract_stored_once() -> None:
    """The same 'Contrato Marco' in two product folders is stored once per bank."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        corpus = Path(tmp) / "rag_docs"
        source = RAG_DOCS_PATH / f"{SANTANDER_ID}_{LIKEU_PRO_ID}"
        shutil.copytree(source, corpus / source.name)

        second = corpus / f"{SANTANDER_ID}_3120"
        second.mkdir()
        shutil.copy(source / "framework_contract.txt", second / "framework_contract.txt")
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        manifest["product_id"] = "3120"
        manifest["documents"] = [d for d in manifest["documents"] if d["applies_to_all_products"]]
        (second / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        engine, stats = _build_engine(Path(tmp) / "chroma", corpus)
        assert stats["documents_deduplicated"] == 1, stats

        shared = engine.collection.get(where={"product_id": ALL_PRODUCTS}, include=["metadatas"])
        hashes = {m["content_sha256"] for m in shared["metadatas"]}
        indices = [m["chunk_index"] for m in shared["metadatas"]]
        assert len(hashes) == 1 and len(indices) == len(set(indices)), "Framework contract stored more than once."


PDFS_PATH = PROJECT_ROOT / "data" / "pdfs"


def test_pdf_library_scopes() -> None:
    """Official PDFs: stored once, retrieved for exactly the products the relation document lists."""
    full_map = json.loads((PDFS_PATH / PDF_LIBRARY_MAP).read_text(encoding="utf-8"))
    wanted = {"Santander_folleto-super-cuenta-debito-likeu-pro.pdf",   # owner 060014 (LikeU Pro), shared with 2 more
              "Santander_folleto-basica-nomina.pdf",                   # owner 3072-1, shared with LikeU
              "Sabadell_Caratula_de_Contrato_Multiple_de_Cuentas_Basica.pdf"}  # ALL Sabadell products
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        library = Path(tmp) / "pdfs"
        library.mkdir()
        docs = [d for d in full_map["documents"] if d["file"] in wanted]
        for doc in docs:
            shutil.copy(PDFS_PATH / doc["file"], library / doc["file"])
        (library / PDF_LIBRARY_MAP).write_text(json.dumps({**full_map, "documents": docs}), encoding="utf-8")

        products = load_catalog(CATALOG_PATH)
        known: Dict[str, set] = {}
        for product in products:
            known.setdefault(product.institution_id, set()).add(product.product_id)
        engine = InstitutionalRAGEngine(persist_directory=Path(tmp) / "chroma")
        stats = engine.ingest_pdf_library(library, institution_names_from_catalog(products), known)
        assert stats["pdfs_indexed"] == 3 and not stats["unknown_products"] and not stats["missing_files"], stats
        again = engine.ingest_pdf_library(library, institution_names_from_catalog(products), known)
        assert engine.count() == stats["pdf_chunks_upserted"] == again["pdf_chunks_upserted"], "re-ingest duplicated"

        def sources(institution: str, product: str) -> set:
            hits = engine.query_mmr("comisiones por manejo de cuenta y saldo mínimo", institution_id=institution,
                                    product_id=product, k=10, fetch_k=40)
            return {h["metadata"]["source_file"] for h in hits}

        likeu, nomina = "pdfs/Santander_folleto-super-cuenta-debito-likeu-pro.pdf", "pdfs/Santander_folleto-basica-nomina.pdf"
        # The ID matrix also lists the Básica Nómina leaflet in LikeU's promotions column (owner-shared).
        assert {likeu, nomina} <= sources(SANTANDER_ID, "06001409")
        assert nomina in sources(SANTANDER_ID, "3072-1") and likeu not in sources(SANTANDER_ID, "3072-1")
        assert not sources(SANTANDER_ID, "3120"), "SuperNómina is not in the relation of either leaflet"
        assert sources("40156", "20") == {"pdfs/Sabadell_Caratula_de_Contrato_Multiple_de_Cuentas_Basica.pdf"}

        meta = engine.collection.get(where={"source_file": likeu}, include=["metadatas"])["metadatas"]
        assert {m["product_id"] for m in meta} == {"060014"}  # LikeU Pro's own leaflet: figure checks for it only
        assert sorted(meta[0]["product_scope"]) == ["060014", "06001409", "06140925"]
        assert meta[0]["document_title"] == "Folleto Súper Cuenta Débito LikeU Pro"
        specific = engine.collection.get(where={"source_file": nomina}, include=["metadatas"])["metadatas"]
        assert {m["product_id"] for m in specific} == {"3072-1"}  # its own leaflet: eligible for figure checks
        assert sorted(specific[0]["product_scope"]) == ["060014", "06001409", "06140925", "3072-1"]  # stored once


# --------------------------------------------------------------------------- #
# 4. SquadState instantiation
# --------------------------------------------------------------------------- #
def test_squad_state() -> None:
    state = create_initial_state(
        "I get paid weekly and keep about 3,500 MXN. Which payroll account avoids fees?",
        archetype_prior="payroll_saver",
    )
    hints = get_type_hints(SquadState)
    assert set(state) == set(hints), f"Key mismatch: {set(state) ^ set(hints)}"
    TypeAdapter(SquadState).validate_python(state, strict=True)  # runtime type check

    assert isinstance(state["weighted_preference_vector"], dict)
    assert abs(sum(state["weighted_preference_vector"].values()) - 1.0) < 1e-9
    assert state["trajectory_log"] == [] and state["draft_recommendation"] is None
    assert isinstance(state["evaluator_score"], float) and isinstance(state["data_freshness_flag"], bool)

    for bad_prefs in ({"yield_importance": 0.9, "fee_importance": 0.9, "perks_importance": 0.2}, {"yield_importance": 1.0}):
        try:
            create_initial_state("query", weighted_preference_vector=bad_prefs)
        except ValueError:
            continue
        raise AssertionError(f"Invalid preference vector accepted: {bad_prefs}")

    reference = date(2026, 11, 13)
    assert is_data_fresh("2026-11-01", reference) is True
    assert is_data_fresh("2026-09-01", reference) is False
    assert is_data_fresh("not-a-date", reference) is False


def test_data_freshness_includes_rag_chunks() -> None:
    reference = date(2026, 11, 13)
    products = [{"institution_id": SANTANDER_ID, "product_id": LIKEU_PRO_ID, "last_verified_date": "2026-11-01"}]
    fresh_chunk = {"chunk_id": "a", "metadata": {"last_verified_date": "2026-11-01"}}
    stale_chunk = {"chunk_id": "b", "metadata": {"last_verified_date": "2026-08-01"}}
    undated_chunk = {"chunk_id": "c", "metadata": {}}

    ok = compute_data_freshness(products, [fresh_chunk], reference)
    assert ok["data_freshness_flag"] is True and not ok["stale_chunks"]

    stale = compute_data_freshness(products, [fresh_chunk, stale_chunk, undated_chunk], reference)
    assert stale["data_freshness_flag"] is False
    assert stale["stale_chunks"] == ["b", "c"] and stale["stale_products"] == []
    assert stale["oldest_verified_date"] == "2026-08-01"


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
TESTS: List[Callable[[], None]] = [
    test_catalog_structure,
    test_catalog_consistency_audit,
    test_calculator_fee_waiver,
    test_calculator_yield_tiers_and_errors,
    test_calculator_channel_aware_transfers,
    test_calculator_penalties_and_isr,
    test_rag_metadata_filtering,
    test_rag_framework_contract_stored_once,
    test_pdf_library_scopes,
    test_squad_state,
    test_data_freshness_includes_rag_chunks,
]


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    failures = 0
    for test in TESTS:
        started = time.perf_counter()
        try:
            test()
        except Exception:  # noqa: BLE001 - runner must report every failure
            failures += 1
            print(f"[FAIL] {test.__name__}")
            traceback.print_exc()
        else:
            print(f"[PASS] {test.__name__} ({time.perf_counter() - started:.2f}s)")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
