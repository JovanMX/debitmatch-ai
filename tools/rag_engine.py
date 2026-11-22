"""Semantic retrieval over bank legal documents (RECA contracts, fine print).

Corpus layout (``data/rag_docs``)::

    rag_docs/
        <institution_id>_<product_id>/
            manifest.json          # optional but recommended, see DocumentManifest
            contrato_marco.txt
            terminos_producto.md
            ...

Storage efficiency: a document flagged ``applies_to_all_products`` (e.g. the
bank's framework contract, "Contrato Marco") is stored with
``product_id == "ALL"`` and chunk IDs derived from *institution + content
hash*. The same contract dropped into several product folders therefore maps
to the same IDs and is stored exactly once per institution.

Re-ingestion is idempotent: chunk IDs are deterministic and written with
``upsert``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import unicodedata
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import chromadb
import numpy as np
from chromadb.api.types import EmbeddingFunction
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

COLLECTION_NAME = "debitmatch_policy_docs"
ALL_PRODUCTS = "ALL"
SCOPED_PRODUCTS = "SCOPED"  # stored once, backs the product IDs listed in ``product_scope``
PDF_LIBRARY_MAP = "document_map.json"

# RAG and Retrieval Design Integration V1, section 2: dense encoder text-embedding-3-small (OpenAI,
# 1536-dim, multilingual, 8,191-token input) -> supports the specified 400-token chunks.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_PREFIX = "text-embedding-"
# RAG design, section 4: "Fixed-Size Chunking with 400 tokens and 50-token overlap".
CHUNK_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
TOKEN_ENCODING = "cl100k_base"  # tokenizer of the text-embedding-3 family
# Chroma's bundled ONNX all-MiniLM-L6-v2: used by the unit tests on throwaway collections only.
CHROMA_DEFAULT_EMBEDDING = "chroma-default"
SUPPORTED_EXTENSIONS = frozenset({".txt", ".md"})
_FOLDER_PATTERN = re.compile(r"^(?P<institution_id>[^_]+)_(?P<product_id>.+)$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.;:!?])\s+")


# ------------------------------------------------------------------------- #
# Manifest contracts
# ------------------------------------------------------------------------- #
class DocumentEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str = Field(min_length=1)
    document_type: str = Field(min_length=1)
    applies_to_all_products: bool = False
    last_verified_date: Optional[date] = None  # overrides the folder-level date


class DocumentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    institution_id: str = Field(min_length=1)
    institution_name: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    last_verified_date: date
    documents: List[DocumentEntry] = Field(default_factory=list)


class PdfDocumentEntry(BaseModel):
    """One official PDF and the catalog products it backs (``data/pdfs/document_map.json``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str = Field(min_length=1)
    institution_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    document_type: str = Field(min_length=1)
    products: Union[Literal["ALL"], List[str]]
    product_specific: bool = False  # the PDF is THIS product's own document (enables figure cross-checks)
    # Product whose OWN document this is while it also backs the other ``products`` (stored once:
    # product_id=owner -> owner keeps figure cross-checks; the rest see it through product_scope).
    owner_product: Optional[str] = None
    last_verified_date: date
    relation_columns: List[str] = Field(default_factory=list)  # T&C / promotions & cashback / insurance
    notes: Optional[str] = None


class PdfLibraryMap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    documents: List[PdfDocumentEntry]


class ChunkMetadata(BaseModel):
    """Metadata contract stored with every chunk in ChromaDB."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    institution_id: str
    institution_name: str
    product_id: str  # concrete product ID, "ALL" (whole institution) or "SCOPED" (see product_scope)
    document_type: str
    applies_to_all_products: bool
    last_verified_date: str  # ISO date
    source_file: str
    chunk_index: int
    content_sha256: str
    benefit_type: str = "general"  # insurance | cashback | fee_waiver | digital | virtual_card | payroll | general
    # SCOPED documents (one PDF shared by a subset of an institution's products): the product IDs it backs.
    product_scope: Optional[List[str]] = None
    document_title: Optional[str] = None  # official title from the product/document relation


_BENEFIT_TYPE_RULES = (
    ("insurance", re.compile(r"insurance|seguro|p[oó]liza", re.IGNORECASE)),
    ("cashback", re.compile(r"cash ?back|rewards?|perks?|discounts?|promoci", re.IGNORECASE)),
    ("virtual_card", re.compile(r"virtual card|tarjeta digital", re.IGNORECASE)),
    ("payroll", re.compile(r"payroll|n[oó]mina|direct deposit", re.IGNORECASE)),
    ("fee_waiver", re.compile(r"fee|commission|comisi[oó]n|waived|exent", re.IGNORECASE)),
    ("digital", re.compile(r"digital|online|mobile|app\b", re.IGNORECASE)),
)
_DOCUMENT_BENEFIT_TYPES = {"poliza_seguros": "insurance", "bases_promociones_cashback": "cashback"}


def classify_benefit_type(text: str, document_type: str = "") -> str:
    """Metadata tag used for benefit-type filtering (RAG design, section 4)."""
    if document_type in _DOCUMENT_BENEFIT_TYPES:
        return _DOCUMENT_BENEFIT_TYPES[document_type]
    for benefit_type, pattern in _BENEFIT_TYPE_RULES:
        if pattern.search(text):
            return benefit_type
    return "general"


def split_benefit_entries(summary: str) -> List[str]:
    """Atomic chunking of a semi-structured benefit list: one benefit per chunk.

    A comma followed by exactly three digits is a thousands separator ("$40,000 mxn"), not a list separator.
    """
    entries = [e.strip(" .;") for e in re.split(r",(?!\d{3}(?!\d))|;|\n", summary or "")]
    return [e for e in entries if len(e) >= 3]


# ------------------------------------------------------------------------- #
# PDF extraction
# ------------------------------------------------------------------------- #
# Glyphs some bank PDFs emit for ligatures ("ÚlƟma" -> "Última"); NFKC handles the standard ones (ﬁ, ﬂ).
_PDF_GLYPH_FIXES = {"Ɵ": "ti", "Ō": "fi", "ƪ": "ft", "­": ""}


def extract_pdf_text(path: str | Path) -> str:
    """Plain text of a PDF (pypdf), one paragraph per page, whitespace and ligatures normalised."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency documented in requirements.txt
        raise ImportError("PDF ingestion requires pypdf: pip install 'pypdf>=5.0'") from exc
    pages: List[str] = []
    for page in PdfReader(str(path)).pages:
        raw = unicodedata.normalize("NFKC", page.extract_text() or "")
        for glyph, replacement in _PDF_GLYPH_FIXES.items():
            raw = raw.replace(glyph, replacement)
        text = " ".join(raw.split())
        if text:
            pages.append(text)
    return "\n\n".join(pages)


# ------------------------------------------------------------------------- #
# Chunking
# ------------------------------------------------------------------------- #
def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    """Paragraph-aware, deterministic character chunker.

    Packs whole paragraphs up to ``chunk_size``; paragraphs that are too long
    are split on sentence boundaries, and as a last resort on hard character
    windows. Consecutive chunks share up to ``chunk_overlap`` trailing chars so
    a clause cut at a boundary remains retrievable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return []

    units: List[str] = []
    for paragraph in re.split(r"\n\s*\n", normalized):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                step = chunk_size - chunk_overlap
                units.extend(sentence[i : i + chunk_size] for i in range(0, len(sentence), step))

    chunks: List[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            # Start the overlap on a word boundary to avoid half words.
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            candidate = f"{tail} {unit}".strip() if tail else unit
            current = candidate if len(candidate) <= chunk_size else unit
        else:
            current = unit
    if current:
        chunks.append(current)
    return chunks


def chunk_text_tokens(text: str, max_tokens: int = CHUNK_TOKENS, overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
                      encoding: str = TOKEN_ENCODING) -> List[str]:
    """Fixed-size token windows (RAG design: 400 tokens, 50-token overlap), counted with tiktoken."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be >= 0 and smaller than max_tokens.")
    import tiktoken

    normalized = "\n\n".join(" ".join(p.split()) for p in re.split(r"\n\s*\n", re.sub(r"\r\n?", "\n", text)))
    normalized = normalized.strip()
    if not normalized:
        return []
    encoder = tiktoken.get_encoding(encoding)
    tokens = encoder.encode(normalized)
    step = max_tokens - overlap_tokens
    chunks = []
    for start in range(0, len(tokens), step):
        chunk = encoder.decode(tokens[start: start + max_tokens]).strip()
        if chunk:
            chunks.append(chunk)
        if start + max_tokens >= len(tokens):
            break
    return chunks


def default_embedding_model() -> str:
    """Embedding model of the production index."""
    return DEFAULT_EMBEDDING_MODEL


# ------------------------------------------------------------------------- #
# Embeddings
# ------------------------------------------------------------------------- #
_EMBEDDING_CACHE: Dict[str, EmbeddingFunction[Any]] = {}


def build_embedding_function(model_name: Optional[str] = None) -> EmbeddingFunction[Any]:
    """Return a (process-cached) Chroma embedding function for ``model_name``.

    ``text-embedding-*`` -> OpenAI embeddings (key read from OPENAI_API_KEY at call time; it is never
    stored in the collection config). ``CHROMA_DEFAULT_EMBEDDING`` -> Chroma's bundled ONNX model,
    for the unit tests.

    Raises:
        ValueError: OPENAI_API_KEY is not configured, or the model is not supported.
    """
    model_name = model_name or default_embedding_model()
    if model_name in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[model_name]

    from chromadb.utils import embedding_functions

    if model_name.startswith(OPENAI_EMBEDDING_PREFIX):
        from dotenv import load_dotenv

        load_dotenv()
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key or key == "COLOCAR-AQUI-APIKEY":
            raise ValueError(f"Embedding model '{model_name}' needs OPENAI_API_KEY in .env.")
        function: EmbeddingFunction[Any] = embedding_functions.OpenAIEmbeddingFunction(  # type: ignore[assignment]
            model_name=model_name, api_key_env_var="OPENAI_API_KEY")
    elif model_name == CHROMA_DEFAULT_EMBEDDING:
        function = embedding_functions.DefaultEmbeddingFunction()  # type: ignore[assignment]
    else:
        raise ValueError(f"Unsupported embedding model '{model_name}' (expected an OpenAI text-embedding-* model).")
    _EMBEDDING_CACHE[model_name] = function
    return function


# ------------------------------------------------------------------------- #
# In-memory read index
# ------------------------------------------------------------------------- #
class _MemoryIndex:
    """Read-only snapshot of the collection (vectors + metadata) for exact, filtered search in numpy.

    The corpus is small (~1.5k chunks x 1,536 dims ~ 9 MB), so one matrix product over every chunk is far
    cheaper than Chroma's per-query fixed cost (SQLite metadata filter + copying vectors back to Python).
    The filter mirrors :meth:`InstitutionalRAGEngine.build_where_filter` exactly.
    """

    def __init__(self, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[Dict[str, Any]],
                 embeddings: Any) -> None:
        self.ids = list(ids)
        self.documents = list(documents)
        self.metadatas = [dict(m or {}) for m in metadatas]
        self.embeddings = np.asarray(embeddings, dtype=np.float32).reshape(len(self.ids), -1)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.unit = self.embeddings / norms
        self.institution = np.array([m.get("institution_id") for m in self.metadatas], dtype=object)
        self.product = np.array([m.get("product_id") for m in self.metadatas], dtype=object)
        self.document_type = np.array([m.get("document_type") for m in self.metadatas], dtype=object)
        self.in_scope: Dict[str, np.ndarray] = {}
        for i, meta in enumerate(self.metadatas):
            for product_id in meta.get("product_scope") or ():
                self.in_scope.setdefault(str(product_id), np.zeros(len(self.ids), dtype=bool))[i] = True

    def __len__(self) -> int:
        return len(self.ids)

    def mask(self, institution_id: Optional[str], product_id: Optional[str],
             document_types: Optional[Sequence[str]]) -> np.ndarray:
        keep = np.ones(len(self.ids), dtype=bool)
        if institution_id:
            keep &= self.institution == institution_id
        if product_id:
            owned = (self.product == product_id) | (self.product == ALL_PRODUCTS)
            scoped = self.in_scope.get(product_id)
            keep &= owned | scoped if scoped is not None else owned
        if document_types:
            keep &= np.isin(self.document_type, list(document_types))
        return keep

    def search(self, query_vector: Sequence[float], n_results: int, institution_id: Optional[str],
               product_id: Optional[str], document_types: Optional[Sequence[str]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Nearest chunks by cosine distance (best first), in the ``_format_results`` shape, plus their vectors."""
        rows = np.flatnonzero(self.mask(institution_id, product_id, document_types))
        if rows.size == 0:
            return [], np.empty((0, self.embeddings.shape[1]), dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32)
        query = query / (np.linalg.norm(query) or 1.0)
        distances = 1.0 - self.unit[rows] @ query
        order = np.argsort(distances, kind="stable")[:n_results]
        chosen = rows[order]
        results = [{"chunk_id": self.ids[i], "text": self.documents[i], "metadata": dict(self.metadatas[i]),
                    "distance": float(d), "relevance_score": round(1.0 - float(d), 4)}
                   for i, d in zip(chosen, distances[order])]
        return results, self.embeddings[chosen]


# ------------------------------------------------------------------------- #
# Engine
# ------------------------------------------------------------------------- #
QUERY_EMBEDDING_CACHE_SIZE = 256


class InstitutionalRAGEngine:
    """ChromaDB-backed retriever for institutional policy documents."""

    def __init__(
        self,
        persist_directory: Optional[str | Path] = None,
        collection_name: str = COLLECTION_NAME,
        embedding_model: Optional[str] = None,
        embedding_function: Optional[EmbeddingFunction[Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        client: Optional[Any] = None,
        reset: bool = False,
        chunk_unit: str = "tokens",
    ) -> None:
        """
        Args:
            persist_directory: On-disk Chroma location. ``None`` -> in-memory client.
            collection_name: Chroma collection name.
            embedding_model: Model name recorded in the collection metadata; defaults to
                ``default_embedding_model()`` (text-embedding-3-small, multilingual).
            embedding_function: Custom embedder (overrides ``embedding_model``
                loading; ``embedding_model`` is still used as its label).
            chunk_size / chunk_overlap: Chunking parameters in ``chunk_unit``
                (defaults: 400 / 50 tokens, RAG design section 4; 800 / 120 for characters).
            client: Pre-built Chroma client (takes precedence; useful for DI/tests).
            reset: Drop any existing collection before opening it (e.g. to
                switch embedding model).
            chunk_unit: "tokens" (tiktoken windows) or "chars" (paragraph-aware characters).

        Raises:
            ValueError: the existing collection was built with another embedding model.
        """
        if chunk_unit not in {"tokens", "chars"}:
            raise ValueError("chunk_unit must be 'tokens' or 'chars'.")
        tokens = chunk_unit == "tokens"
        chunk_size = chunk_size if chunk_size is not None else (CHUNK_TOKENS if tokens else 800)
        chunk_overlap = chunk_overlap if chunk_overlap is not None else (CHUNK_OVERLAP_TOKENS if tokens else 120)
        # Validate chunking params eagerly so misconfiguration fails at startup.
        (chunk_text_tokens if tokens else chunk_text)("probe", chunk_size, chunk_overlap)
        self._chunk_unit = chunk_unit
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._collection_name = collection_name
        embedding_model = embedding_model or default_embedding_model()
        self._embedding_model = embedding_model
        self._embedding_function = embedding_function or build_embedding_function(embedding_model)

        if client is not None:
            self._client = client
        elif persist_directory is not None:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_directory))
        else:
            self._client = chromadb.EphemeralClient()

        # Retrieval reads from an in-memory snapshot (see _MemoryIndex); every write through this engine
        # invalidates it. Set to False to query Chroma directly (used to check both paths agree).
        self.use_memory_index = True
        self._memory_index: Optional[_MemoryIndex] = None
        self._index_lock = threading.Lock()
        # Query embeddings: LRU of futures, so a prefetched (background) embedding is awaited, not repeated.
        self._query_embeddings: "OrderedDict[str, Future]" = OrderedDict()
        self._embedding_lock = threading.Lock()
        self._embedding_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed")

        if reset:
            self._drop_collection()
        self._collection = self._open_collection()

    def _invalidate_index(self) -> None:
        self._memory_index = None

    def _snapshot(self) -> _MemoryIndex:
        index = self._memory_index
        if index is not None:
            return index
        with self._index_lock:
            if self._memory_index is None:
                data = self._collection.get(include=["documents", "metadatas", "embeddings"])
                embeddings = data.get("embeddings")
                self._memory_index = _MemoryIndex(data.get("ids") or [], data.get("documents") or [],
                                                  data.get("metadatas") or [],
                                                  embeddings if embeddings is not None else [])
                logger.info("In-memory RAG index loaded: %d chunks.", len(self._memory_index))
            return self._memory_index

    def warm_up(self) -> None:
        """Load the in-memory index and the conflict-audit key set now (outside any query time budget)."""
        if self.use_memory_index:
            self._snapshot()
        self.product_specific_keys()

    def _drop_collection(self) -> None:
        self._invalidate_index()
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:  # noqa: BLE001 - collection may not exist; Chroma raises varying types
            logger.debug("Collection %s did not exist; nothing to drop.", self._collection_name)

    def _open_collection(self) -> Any:
        collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine", "embedding_model": self._embedding_model},
        )
        stored_model = (collection.metadata or {}).get("embedding_model")
        if stored_model is not None and stored_model != self._embedding_model:
            raise ValueError(
                f"Collection '{self._collection_name}' was built with '{stored_model}', "
                f"not '{self._embedding_model}'. Vectors are incompatible: re-ingest with reset()."
            )
        return collection

    # ------------------------------------------------------------------ #
    # Introspection / maintenance
    # ------------------------------------------------------------------ #
    @property
    def collection(self) -> Any:
        return self._collection

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        """Drop and recreate the (now empty) collection."""
        self._drop_collection()
        self._collection = self._open_collection()
        self._invalidate_index()

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_directory(
        self,
        docs_root: str | Path,
        institution_names: Optional[Dict[str, str]] = None,
        default_verified_date: Optional[date] = None,
    ) -> Dict[str, int]:
        """Index every ``<institution_id>_<product_id>`` folder under ``docs_root``.

        Args:
            docs_root: Root folder (``data/rag_docs``).
            institution_names: ``institution_id -> name`` fallback (e.g. from the
                catalog) for folders without a manifest.
            default_verified_date: Fallback date for folders without a manifest.

        Returns:
            Ingestion stats: ``documents_indexed``, ``documents_deduplicated``,
            ``chunks_upserted``, ``folders_skipped``.
        """
        root = Path(docs_root)
        if not root.is_dir():
            raise FileNotFoundError(f"RAG docs root not found: {root}")

        stats = {"documents_indexed": 0, "documents_deduplicated": 0, "chunks_upserted": 0, "folders_skipped": 0}
        seen_shared_docs: set[tuple[str, str]] = set()  # (institution_id, content_sha256)

        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            manifest = self._load_manifest(folder, institution_names or {}, default_verified_date)
            if manifest is None:
                stats["folders_skipped"] += 1
                continue

            for entry in manifest.documents:
                doc_path = folder / entry.file
                if not doc_path.is_file() or doc_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    logger.warning("Skipping missing/unsupported document: %s", doc_path)
                    continue

                text = doc_path.read_text(encoding="utf-8")
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

                if entry.applies_to_all_products:
                    key = (manifest.institution_id, content_hash)
                    if key in seen_shared_docs:
                        stats["documents_deduplicated"] += 1
                        continue
                    seen_shared_docs.add(key)

                stats["chunks_upserted"] += self.add_document(
                    text=text,
                    institution_id=manifest.institution_id,
                    institution_name=manifest.institution_name,
                    product_id=manifest.product_id,
                    document_type=entry.document_type,
                    applies_to_all_products=entry.applies_to_all_products,
                    last_verified_date=entry.last_verified_date or manifest.last_verified_date,
                    source_file=f"{folder.name}/{entry.file}",
                )
                stats["documents_indexed"] += 1

        logger.info("RAG ingestion finished: %s", stats)
        return stats

    def add_document(
        self,
        text: str,
        institution_id: str,
        institution_name: str,
        product_id: str,
        document_type: str,
        applies_to_all_products: bool,
        last_verified_date: date | str,
        source_file: str = "inline",
        product_scope: Optional[Sequence[str]] = None,
        document_title: Optional[str] = None,
    ) -> int:
        """Chunk and upsert one document. Returns the number of chunks written.

        ``product_scope`` (2+ product IDs of the same institution) stores the document ONCE as
        ``product_id="SCOPED"``; retrieval for any listed product includes it. When ``product_id``
        is a concrete ID inside the scope, it is kept (owner document shared with the others).
        """
        scope = sorted({str(p) for p in product_scope or []}) or None
        if applies_to_all_products:
            effective_product_id, scope = ALL_PRODUCTS, None
        elif scope:
            effective_product_id = product_id if product_id in scope else SCOPED_PRODUCTS
        else:
            effective_product_id = product_id
        verified_iso = (
            last_verified_date.isoformat()
            if isinstance(last_verified_date, date)
            else date.fromisoformat(last_verified_date).isoformat()
        )
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        chunker = chunk_text_tokens if self._chunk_unit == "tokens" else chunk_text
        chunks = chunker(text, self._chunk_size, self._chunk_overlap)
        if not chunks:
            logger.warning("Document %s is empty; nothing indexed.", source_file)
            return 0

        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for index, _chunk in enumerate(chunks):
            metadata = ChunkMetadata(
                institution_id=institution_id,
                institution_name=institution_name,
                product_id=effective_product_id,
                document_type=document_type,
                applies_to_all_products=applies_to_all_products,
                last_verified_date=verified_iso,
                source_file=source_file,
                chunk_index=index,
                content_sha256=content_hash,
                benefit_type=classify_benefit_type(_chunk, document_type),
                product_scope=scope,
                document_title=document_title,
            )
            # Shared docs: ID ignores the product -> stored once per institution.
            ids.append(f"{institution_id}:{effective_product_id}:{content_hash[:16]}:{index:04d}")
            metadatas.append(metadata.model_dump(exclude_none=True))  # Chroma rejects None values

        self._collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        self._invalidate_index()
        return len(chunks)

    def add_benefit_texts(self, products: Iterable[Any]) -> int:
        """Index each catalog product's semi-structured benefit list as ATOMIC chunks.

        RAG design, section 4: "API texts: Atomic Chunking, treating each benefit
        entry as a single, indivisible chunk to preserve list integrity". Previous
        benefit chunks of the product are replaced (the catalog text may change).
        """
        from tools.schemas import API_BENEFIT_DOC_TYPE

        written = 0
        self._invalidate_index()
        for product in products:
            self._collection.delete(where={"$and": [{"document_type": API_BENEFIT_DOC_TYPE},
                                                    {"institution_id": product.institution_id},
                                                    {"product_id": product.product_id}]})
            entries = split_benefit_entries(product.qualitative_perks_summary)
            if not entries:
                continue
            ids, documents, metadatas = [], [], []
            for index, entry in enumerate(entries):
                text = f"{product.institution_name} {product.product_name}: {entry}."
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                ids.append(f"{product.institution_id}:{product.product_id}:benefit:{content_hash[:16]}:{index:04d}")
                documents.append(text)
                metadatas.append(ChunkMetadata(
                    institution_id=product.institution_id, institution_name=product.institution_name,
                    product_id=product.product_id, document_type=API_BENEFIT_DOC_TYPE,
                    applies_to_all_products=False, last_verified_date=product.last_verified_date.isoformat(),
                    source_file="catalog.json#qualitative_perks_summary", chunk_index=index,
                    content_sha256=content_hash, benefit_type=classify_benefit_type(entry),
                ).model_dump(exclude_none=True))
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            written += len(ids)
        self._invalidate_index()
        return written

    def product_specific_keys(self) -> set:
        """``institution_product`` keys that own product-specific RECA chunks (not ALL / SCOPED / catalog texts).

        Only these chunks feed the API-vs-RECA figure cross-checks, so research always retrieves them for any
        candidate that has one. Cached until the collection size changes.
        """
        from tools.schemas import API_BENEFIT_DOC_TYPE

        size = self.count()
        cached = getattr(self, "_product_specific_cache", None)
        if cached is not None and cached[0] == size:
            return cached[1]
        found = self._collection.get(
            where={"$and": [{"product_id": {"$nin": [ALL_PRODUCTS, SCOPED_PRODUCTS]}},
                            {"document_type": {"$ne": API_BENEFIT_DOC_TYPE}}]},
            include=["metadatas"])
        keys = {f"{m['institution_id']}_{m['product_id']}" for m in found.get("metadatas") or []}
        self._product_specific_cache = (size, keys)
        return keys

    def purge_benefit_texts(self, catalog_keys: Iterable[Tuple[str, str]]) -> int:
        """Delete benefit chunks of products no longer in the catalog (removed or merged variants)."""
        from tools.schemas import API_BENEFIT_DOC_TYPE

        keep = {(str(i), str(p)) for i, p in catalog_keys}
        stored = self._collection.get(where={"document_type": API_BENEFIT_DOC_TYPE}, include=["metadatas"])
        stale = [cid for cid, meta in zip(stored["ids"], stored["metadatas"])
                 if (meta.get("institution_id"), meta.get("product_id")) not in keep]
        if stale:
            self._collection.delete(ids=stale)
            self._invalidate_index()
        return len(stale)

    def ingest_pdf_library(
        self,
        pdf_root: str | Path,
        institution_names: Optional[Dict[str, str]] = None,
        known_products: Optional[Dict[str, set]] = None,
        map_file: str = PDF_LIBRARY_MAP,
    ) -> Dict[str, Any]:
        """Index official RECA PDFs following ``<pdf_root>/document_map.json``.

        The map encodes the product <-> document relation ("Relación Consolidada de Productos del
        Catálogo y Documentación Legal de Respaldo"). Each PDF is stored once:
        ``products: "ALL"`` -> institution-wide; one product with ``product_specific: true`` -> that
        product (eligible for API-vs-RECA figure checks); ``owner_product`` -> stored under the owner
        (figure checks for it only) and shared with the other ``products`` through ``product_scope``;
        otherwise ``SCOPED`` + ``product_scope``. Re-ingestion replaces the previous chunks of each file (the extraction may change).

        Args:
            known_products: ``institution_id -> {product_id}`` from the catalog; unknown IDs are
                reported and dropped from the scope (never silently indexed).
        """
        root = Path(pdf_root)
        library = PdfLibraryMap.model_validate_json((root / map_file).read_text(encoding="utf-8"))
        self._invalidate_index()  # deletes below bypass add_document
        stats: Dict[str, Any] = {"pdfs_indexed": 0, "pdf_chunks_upserted": 0, "unknown_products": [],
                                 "missing_files": [], "per_file": {}, "purged_files": []}
        # Files dropped or renamed in the map must not keep serving stale chunks.
        mapped = {f"pdfs/{doc.file}" for doc in library.documents}
        indexed = self._collection.get(include=["metadatas"])["metadatas"] or []
        for source in sorted({m.get("source_file", "") for m in indexed} - mapped):
            if source.startswith("pdfs/"):
                self._collection.delete(where={"source_file": source})
                stats["purged_files"].append(source)
        # File names may be stored decomposed on disk (e.g. "Débito" from a download) while the map
        # spells them composed: resolve by NFC-normalised name.
        on_disk = {unicodedata.normalize("NFC", p.name): p for p in root.iterdir() if p.is_file()}
        for doc in library.documents:
            path = root / doc.file
            if not path.is_file():
                path = on_disk.get(unicodedata.normalize("NFC", doc.file), path)
            if not path.is_file():
                stats["missing_files"].append(doc.file)
                continue
            catalog_ids = (known_products or {}).get(doc.institution_id)
            scope: List[str] = []
            if doc.products != ALL_PRODUCTS:
                for product_id in doc.products:
                    if catalog_ids is not None and product_id not in catalog_ids:
                        stats["unknown_products"].append(f"{doc.institution_id}_{product_id} ({doc.file})")
                    else:
                        scope.append(product_id)
            source = f"pdfs/{doc.file}"
            self._collection.delete(where={"source_file": source})
            text = extract_pdf_text(path)
            owner = doc.owner_product if doc.owner_product in scope else None
            if doc.owner_product and owner is None:
                stats["unknown_products"].append(f"{doc.institution_id}_{doc.owner_product} (owner, {doc.file})")
            single = (doc.product_specific and len(scope) == 1) or (owner is not None and len(scope) == 1)
            written = self.add_document(
                text=text,
                institution_id=doc.institution_id,
                institution_name=(institution_names or {}).get(doc.institution_id, doc.institution_id),
                product_id=scope[0] if single else (owner or SCOPED_PRODUCTS),
                document_type=doc.document_type,
                applies_to_all_products=doc.products == ALL_PRODUCTS,
                last_verified_date=doc.last_verified_date,
                source_file=source,
                product_scope=None if single or doc.products == ALL_PRODUCTS else (scope or ["__unmapped__"]),
                document_title=doc.title,
            )
            stats["pdfs_indexed"] += 1
            stats["pdf_chunks_upserted"] += written
            stats["per_file"][doc.file] = written
        self._invalidate_index()
        return stats

    def count_document_type(self, document_type: str) -> int:
        result = self._collection.get(where={"document_type": document_type}, include=[])
        return len(result.get("ids") or [])

    @staticmethod
    def _load_manifest(
        folder: Path,
        institution_names: Dict[str, str],
        default_verified_date: Optional[date],
    ) -> Optional[DocumentManifest]:
        manifest_path = folder / "manifest.json"
        if manifest_path.is_file():
            try:
                return DocumentManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            except ValidationError as exc:
                logger.error("Invalid manifest %s: %s", manifest_path, exc)
                return None

        # Fallback: infer IDs from the folder name; every file is product-specific.
        match = _FOLDER_PATTERN.match(folder.name)
        if match is None:
            logger.warning("Folder %s does not follow <institution_id>_<product_id>; skipped.", folder)
            return None
        institution_id = match["institution_id"]
        files = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        return DocumentManifest(
            institution_id=institution_id,
            institution_name=institution_names.get(institution_id, institution_id),
            product_id=match["product_id"],
            last_verified_date=default_verified_date or date.today(),
            documents=[DocumentEntry(file=name, document_type=Path(name).stem) for name in files],
        )

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_where_filter(
        institution_id: Optional[str] = None,
        product_id: Optional[str] = None,
        document_types: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Compose the Chroma ``where`` clause.

        institution AND (product OR "ALL" OR product in product_scope) AND document_type IN (...);
        every part is optional.
        """
        clauses: List[Dict[str, Any]] = []
        if institution_id:
            clauses.append({"institution_id": {"$eq": institution_id}})
        if product_id:
            clauses.append({"$or": [{"product_id": {"$in": [product_id, ALL_PRODUCTS]}},
                                    {"product_scope": {"$contains": product_id}}]})
        if document_types:
            clauses.append({"document_type": {"$in": list(document_types)}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def embed_query(self, query: str, timeout: Optional[float] = None) -> List[float]:
        """Embed a query (cached): reuses a finished or in-flight (prefetched) embedding of the same text."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        text = query.strip()
        with self._embedding_lock:
            future = self._query_embeddings.get(text)
            cached = future is not None
            if cached:
                self._query_embeddings.move_to_end(text)
            else:
                future = self._register_embeddings([text])[text]
        if not cached:
            self._compute_embeddings({text: future})  # cache miss: embed in the caller's thread
        return list(future.result(timeout=timeout))

    def prefetch_query_embeddings(self, queries: Iterable[str]) -> None:
        """Start embedding the given texts in ONE background API call; later ``embed_query`` calls await it."""
        texts = list(dict.fromkeys(q.strip() for q in queries if isinstance(q, str) and q.strip()))
        with self._embedding_lock:
            pending = self._register_embeddings([t for t in texts if t not in self._query_embeddings])
        if pending:
            self._embedding_pool.submit(self._compute_embeddings, pending)

    def _register_embeddings(self, texts: Sequence[str]) -> Dict[str, Future]:
        """Create cache entries (caller holds ``_embedding_lock``); evicts the least recently used ones."""
        futures = {text: Future() for text in texts}
        for text, future in futures.items():
            self._query_embeddings[text] = future
        while len(self._query_embeddings) > QUERY_EMBEDDING_CACHE_SIZE:
            self._query_embeddings.popitem(last=False)
        return futures

    def _compute_embeddings(self, futures: Dict[str, Future]) -> None:
        texts = list(futures)
        try:
            vectors = self._embedding_function(texts)
            for text, vector in zip(texts, vectors):
                futures[text].set_result(tuple(float(x) for x in vector))
        except BaseException as exc:  # noqa: BLE001 - surfaced to every waiter; failed entries are not cached
            with self._embedding_lock:
                for text in texts:
                    if self._query_embeddings.get(text) is futures[text]:
                        del self._query_embeddings[text]
            for future in futures.values():
                if not future.done():
                    future.set_exception(exc)

    def query_mmr(
        self,
        query: str,
        institution_id: Optional[str] = None,
        product_id: Optional[str] = None,
        document_types: Optional[Sequence[str]] = None,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        query_embedding: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Maximal Marginal Relevance search with metadata filters.

        Fetches ``fetch_k`` nearest chunks, then greedily selects ``k`` that
        balance relevance to the query (``lambda_mult``) against redundancy
        with already-selected chunks (``1 - lambda_mult``). Result dicts match
        :meth:`query_qualitative_docs` plus ``mmr_rank``.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if fetch_k < k:
            raise ValueError("fetch_k must be >= k.")
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError("lambda_mult must be within [0, 1].")

        index = self._snapshot() if self.use_memory_index else None
        total = len(index) if index is not None else self.count()
        if total == 0:
            logger.warning("Collection is empty; did you run ingest_directory()?")
            return []

        query_vector = list(query_embedding) if query_embedding is not None else self.embed_query(query)
        if index is not None:
            candidates, embeddings = index.search(query_vector, min(fetch_k, total), institution_id, product_id,
                                                  document_types)
        else:
            response = self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(fetch_k, total),
                where=self.build_where_filter(institution_id, product_id, document_types),
                include=["documents", "metadatas", "distances", "embeddings"],
            )
            candidates = list(self._format_results(response))
            raw_embeddings = response.get("embeddings")
            embeddings = raw_embeddings[0] if raw_embeddings is not None and len(raw_embeddings) else []
        if not candidates:
            return []

        selected = maximal_marginal_relevance(query_vector, embeddings, k=k, lambda_mult=lambda_mult)
        results: List[Dict[str, Any]] = []
        for rank, index in enumerate(selected, start=1):
            results.append({**candidates[index], "mmr_rank": rank})
        return results

    def query_qualitative_docs(
        self,
        query: str,
        institution_id: Optional[str] = None,
        product_id: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Semantic search restricted by institution / product metadata.

        Returns a list (best first) of dicts with keys ``chunk_id``, ``text``,
        ``metadata``, ``distance`` and ``relevance_score`` (1 - cosine distance).

        Raises:
            ValueError: empty query or non-positive ``top_k``.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        total = self.count()
        if total == 0:
            logger.warning("Collection is empty; did you run ingest_directory()?")
            return []

        response = self._collection.query(
            query_texts=[query.strip()],
            n_results=min(top_k, total),
            where=self.build_where_filter(institution_id, product_id),
            include=["documents", "metadatas", "distances"],
        )
        return list(self._format_results(response))

    @staticmethod
    def _format_results(response: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            yield {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": dict(metadata or {}),
                "distance": float(distance),
                "relevance_score": round(1.0 - float(distance), 4),
            }


def maximal_marginal_relevance(
    query_embedding: Sequence[float],
    candidate_embeddings: Any,
    k: int = 4,
    lambda_mult: float = 0.5,
) -> List[int]:
    """Return indices of ``k`` candidates selected by MMR (cosine similarity)."""
    matrix = np.asarray(candidate_embeddings, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return []
    query = np.asarray(query_embedding, dtype=float)

    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms

    docs = _normalize(matrix)
    relevance = docs @ _normalize(query[None, :])[0]
    redundancy = docs @ docs.T

    selected: List[int] = [int(np.argmax(relevance))]
    while len(selected) < min(k, len(docs)):
        remaining = [i for i in range(len(docs)) if i not in selected]
        scores = [
            lambda_mult * relevance[i] - (1.0 - lambda_mult) * float(np.max(redundancy[i, selected]))
            for i in remaining
        ]
        selected.append(remaining[int(np.argmax(scores))])
    return selected


def institution_names_from_catalog(catalog: Iterable[Any]) -> Dict[str, str]:
    """Helper: build ``institution_id -> institution_name`` from catalog products."""
    return {p.institution_id: p.institution_name for p in catalog}
