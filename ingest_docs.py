"""CLI: build / refresh the persistent ChromaDB index of policy documents.

Examples::

    python ingest_docs.py                       # incremental, idempotent upsert
    python ingest_docs.py --reset               # drop and rebuild (e.g. new model)
    python ingest_docs.py --audit-catalog       # also print catalog consistency issues
    python ingest_docs.py --query "comision por saldo minimo" --institution 40014 --product 060014
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from state import compute_data_freshness  # noqa: E402
from tools.rag_engine import (  # noqa: E402
    PDF_LIBRARY_MAP,
    InstitutionalRAGEngine,
    default_embedding_model,
    institution_names_from_catalog,
)
from tools.schemas import audit_catalog, load_catalog  # noqa: E402

DEFAULT_DOCS = PROJECT_ROOT / "data" / "rag_docs"
DEFAULT_PDFS = PROJECT_ROOT / "data" / "pdfs"  # official RECA PDFs + document_map.json (product relation)
DEFAULT_DB = PROJECT_ROOT / "data" / "chroma_db"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "catalog.json"


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index DebitMatch AI policy documents into ChromaDB.")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS, help="rag_docs root folder.")
    parser.add_argument("--pdfs", type=Path, default=DEFAULT_PDFS,
                        help="Folder with official PDFs and document_map.json ('' to skip).")
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_DB, help="ChromaDB directory.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="catalog.json (institution names).")
    parser.add_argument("--embedding-model", default=default_embedding_model(),
                        help="OpenAI embedding model (default text-embedding-3-small).")
    parser.add_argument("--reset", action="store_true", help="Drop the collection before ingesting.")
    parser.add_argument("--audit-catalog", action="store_true", help="Print catalog consistency findings.")
    parser.add_argument("--query", help="Optional smoke-test query after ingestion.")
    parser.add_argument("--institution", help="institution_id filter for --query.")
    parser.add_argument("--product", help="product_id filter for --query.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        # Non-strict: an inconsistent catalog must not block document indexing.
        products = load_catalog(args.catalog, strict=False)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR loading catalog: {exc}", file=sys.stderr)
        return 2

    if args.audit_catalog:
        issues = audit_catalog(products)
        print(f"Catalog audit: {len(issues)} finding(s)")
        for issue in issues:
            print(f"  [{issue.severity.upper():7}] {issue.catalog_key:<24} {issue.code}: {issue.message}")

    try:
        engine = InstitutionalRAGEngine(
            persist_directory=args.persist_dir,
            embedding_model=args.embedding_model,
            reset=args.reset,
        )
        stats = engine.ingest_directory(args.docs, institution_names=institution_names_from_catalog(products))
        stats["benefit_chunks_upserted"] = engine.add_benefit_texts(products)  # atomic API benefit texts
        stats["benefit_chunks_purged"] = engine.purge_benefit_texts(
            (p.institution_id, p.product_id) for p in products)  # products removed from the catalog
        if args.pdfs and str(args.pdfs) not in {"", "."} and (args.pdfs / PDF_LIBRARY_MAP).is_file():
            known: dict = {}
            for product in products:
                known.setdefault(product.institution_id, set()).add(product.product_id)
            pdf_stats = engine.ingest_pdf_library(args.pdfs, institution_names_from_catalog(products), known)
            stats["pdfs_indexed"] = pdf_stats["pdfs_indexed"]
            stats["pdf_chunks_upserted"] = pdf_stats["pdf_chunks_upserted"]
            for label in pdf_stats["unknown_products"]:
                print(f"WARNING: document_map product not in catalog: {label}", file=sys.stderr)
            for source in pdf_stats["purged_files"]:
                print(f"Removed stale chunks of a PDF no longer in document_map: {source}")
            for name in pdf_stats["missing_files"]:
                print(f"WARNING: PDF listed in document_map not found: {name}", file=sys.stderr)
            if args.verbose:
                for name, count in pdf_stats["per_file"].items():
                    print(f"  {count:5d} chunks  {name}")
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Indexed into {args.persist_dir} (model: {engine.embedding_model}): "
        f"{json.dumps(stats)} | collection size: {engine.count()} chunks"
    )

    all_chunks = engine.collection.get(include=["metadatas"])
    report = compute_data_freshness(
        [],
        [{"chunk_id": i, "metadata": m} for i, m in zip(all_chunks["ids"], all_chunks["metadatas"])],
        reference_date=date.today(),
    )
    print(
        f"Document freshness: {'OK' if report['data_freshness_flag'] else 'STALE'} "
        f"(oldest verification: {report['oldest_verified_date']}, stale chunks: {len(report['stale_chunks'])})"
    )

    if args.query:
        results = engine.query_qualitative_docs(args.query, args.institution, args.product, args.top_k)
        for rank, chunk in enumerate(results, start=1):
            meta = chunk["metadata"]
            print(f"\n#{rank} score={chunk['relevance_score']} {meta['institution_name']} "
                  f"product={meta['product_id']} type={meta['document_type']}")
            print(f"   {chunk['text'][:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
