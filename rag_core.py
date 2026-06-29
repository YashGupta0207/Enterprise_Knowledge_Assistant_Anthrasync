"""
rag_core.py — Shared RAG pipeline for the Enterprise Knowledge Assistant.

Both the Streamlit UI (chat_pdf.py) and the FastAPI service (api.py) import
from this module so that retrieval, ranking, and prompting logic live in
exactly one place.

Pipeline implemented here:
    1. PDF ingestion (pdfplumber, with PyPDF2 fallback) -> page-level text + tables
    2. Chunking (RecursiveCharacterTextSplitter)
    3. Embedding (sentence-transformers/all-MiniLM-L6-v2, local — no API cost)
    4. Indexing (FAISS for dense vectors, BM25 for sparse/keyword matching)
    5. Hybrid retrieval: FAISS similarity search + BM25 keyword search, fused
       with weighted Reciprocal Rank Fusion (RRF)
    6. Diversity re-ranking via Maximal Marginal Relevance (MMR) to reduce
       redundant chunks from the same page/section
    7. Answer generation via an OpenRouter-hosted LLM through LangChain
    8. Confidence scoring derived from actual embedding similarity (cosine
       distance converted to a 0-1 score), not keyword overlap
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ───────────────────────────── LOGGING ──────────────────────────────── #
# Centralised logger. Both api.py and chat_pdf.py call configure_logging()
# once at startup; library code below just calls logging.getLogger(__name__).

LOG_FILE = os.getenv("EKA_LOG_FILE", "eka.log")


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:  # avoid duplicate handlers on Streamlit re-runs
        return
    root.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)
    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        # e.g. read-only filesystem in some deployment targets — degrade to
        # console-only logging rather than crashing the app.
        root.warning("Could not open log file %s; logging to console only.", LOG_FILE)


log = logging.getLogger("eka.rag_core")

# ───────────────────────────── CONFIG ───────────────────────────────── #

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# NOTE on model choice: "openrouter/free" is OpenRouter's auto-routed free
# tier. It dynamically picks among whatever free models are currently
# available, which means response style/quality can vary run to run and the
# exact model is not guaranteed. This is a deliberate cost/availability
# trade-off for this project (see README "Known Limitations"). For
# production use, pin to a specific model id (e.g. one returned by
# GET https://openrouter.ai/api/v1/models filtered to pricing.prompt == "0").
LLM_MODEL = os.getenv("EKA_LLM_MODEL", "openrouter/free")

FAISS_INDEX_PATH = os.getenv("EKA_FAISS_INDEX_PATH", "faiss_index")
BM25_INDEX_PATH = os.path.join(FAISS_INDEX_PATH, "bm25_corpus.json")
INDEX_META_PATH = os.path.join(FAISS_INDEX_PATH, "indexed_files.json")

EMBEDDING_MODEL_NAME = os.getenv("EKA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

TOP_K_DENSE = int(os.getenv("EKA_TOP_K_DENSE", "8"))     # candidates from FAISS
TOP_K_SPARSE = int(os.getenv("EKA_TOP_K_SPARSE", "8"))   # candidates from BM25
TOP_K_FUSED = int(os.getenv("EKA_TOP_K_FUSED", "8"))     # candidates after RRF fusion
TOP_K_FINAL = int(os.getenv("EKA_TOP_K_FINAL", "4"))     # candidates after MMR rerank, sent to LLM

CHUNK_SIZE = int(os.getenv("EKA_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("EKA_CHUNK_OVERLAP", "150"))
MAX_HISTORY_TURNS = int(os.getenv("EKA_MAX_HISTORY_TURNS", "6"))

RRF_K = 60          # standard Reciprocal Rank Fusion smoothing constant
MMR_LAMBDA = 0.5    # 1.0 = pure relevance, 0.0 = pure diversity; 0.5 balances
                    # the two roughly evenly, which empirically does a better
                    # job of dropping near-duplicate chunks (e.g. the same
                    # clause repeated across two overlapping chunk windows)
                    # than higher values, which let relevance dominate almost
                    # entirely unless similarity is extremely high.

MAX_QUESTION_LENGTH = int(os.getenv("EKA_MAX_QUESTION_LENGTH", "2000"))


# ───────────────────────────── ERRORS ───────────────────────────────── #

class RAGError(Exception):
    """Base class for all recoverable errors raised by this module."""


class IndexNotFoundError(RAGError):
    """Raised when no FAISS/BM25 index exists yet."""


class EmptyDocumentError(RAGError):
    """Raised when a PDF yields no extractable text."""


class InvalidQuestionError(RAGError):
    """Raised when the incoming question fails validation."""


class LLMConfigError(RAGError):
    """Raised when the LLM cannot be constructed (e.g. missing API key)."""


class LLMGenerationError(RAGError):
    """Raised when the LLM call itself fails (timeout, API error, etc.)."""


# ───────────────────────────── VALIDATION ───────────────────────────── #

def validate_question(question: str) -> str:
    """Validate and normalise a user question. Raises InvalidQuestionError."""
    if question is None:
        raise InvalidQuestionError("Question cannot be empty.")
    question = question.strip()
    if not question:
        raise InvalidQuestionError("Question cannot be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise InvalidQuestionError(
            f"Question is too long ({len(question)} chars). "
            f"Limit is {MAX_QUESTION_LENGTH} characters."
        )
    return question


def validate_history(history: list[dict] | None) -> list[dict]:
    """Defensively clean a chat history list coming from a client."""
    if not history:
        return []
    cleaned = []
    for m in history:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()})
    return cleaned


# ─────────────────────────── INDEX METADATA ─────────────────────────── #

def load_indexed_files() -> set[str]:
    """Return the set of filenames already present in the index."""
    if os.path.exists(INDEX_META_PATH):
        try:
            with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not read index metadata (%s); treating as empty.", e)
            return set()
    return set()


def save_indexed_files(names: set[str]) -> None:
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(names), f)


def index_exists() -> bool:
    return os.path.exists(FAISS_INDEX_PATH) and bool(load_indexed_files())


# ───────────────────────────── EMBEDDINGS ───────────────────────────── #

_embeddings_singleton = None


def get_embeddings():
    """Lazily construct and cache the embedding model (expensive to load)."""
    global _embeddings_singleton
    if _embeddings_singleton is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        log.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embeddings_singleton = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings_singleton


# ───────────────────────────────── LLM ──────────────────────────────── #

def get_llm():
    if not OPENROUTER_API_KEY:
        raise LLMConfigError("OPENROUTER_API_KEY not configured. Set it in your .env file.")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=1200,
        timeout=30,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Enterprise Knowledge Assistant",
        },
    )


# ─────────────────────────────── PDF → TEXT ─────────────────────────── #

def extract_text_with_metadata(pdf_files: list) -> list[dict]:
    """
    Extract text (and flattened table rows) per page from a list of
    file-like PDF objects. Returns a list of {"text", "source", "page"}.
    Falls back from pdfplumber (better table/layout handling) to PyPDF2
    if pdfplumber fails on a given file.
    """
    pages: list[dict] = []
    for pdf in pdf_files:
        filename = getattr(pdf, "name", "unknown.pdf")
        pdf.seek(0)
        extracted_via_plumber = False
        try:
            import pdfplumber
            with pdfplumber.open(pdf) as reader:
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    try:
                        tables = page.extract_tables()
                        for table in (tables or []):
                            for row in (table or []):
                                row_text = " | ".join(
                                    str(cell).strip() for cell in (row or []) if cell
                                )
                                if row_text:
                                    text += "\n" + row_text
                    except Exception as e:
                        log.debug("Table extraction failed on %s p%d: %s", filename, page_num, e)
                    text = text.strip()
                    if text:
                        pages.append({"text": text, "source": filename, "page": page_num})
            extracted_via_plumber = True
        except Exception as e:
            log.warning("pdfplumber failed for %s (%s); trying PyPDF2 fallback.", filename, e)

        if not extracted_via_plumber:
            try:
                from PyPDF2 import PdfReader
                pdf.seek(0)
                reader = PdfReader(pdf)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if text:
                        pages.append({"text": text, "source": filename, "page": page_num})
            except Exception as e2:
                log.error("Could not read %s with either extractor: %s", filename, e2)

    if not pages:
        log.warning("No extractable text found across %d file(s).", len(pdf_files))
    return pages


# ──────────────────────────────── CHUNKING ──────────────────────────── #

def chunk_pages(pages: list[dict]) -> tuple[list[str], list[dict]]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts, metadatas = [], []
    for p in pages:
        chunks = splitter.split_text(p["text"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"source": p["source"], "page": p["page"]})
    return texts, metadatas


# ──────────────────────────────── BM25 STORE ────────────────────────── #
# A lightweight sparse-retrieval companion to FAISS. We persist the raw
# corpus (texts + metadata) as JSON and rebuild the BM25 index in memory on
# load — BM25 indexing over a few thousand chunks is sub-second, so this
# keeps the on-disk format simple and dependency-free (no custom binary
# serialisation to maintain).

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _load_bm25_corpus() -> tuple[list[str], list[dict]]:
    if not os.path.exists(BM25_INDEX_PATH):
        return [], []
    try:
        with open(BM25_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("texts", []), data.get("metadatas", [])
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Could not read BM25 corpus (%s); treating as empty.", e)
        return [], []


def _save_bm25_corpus(texts: list[str], metadatas: list[dict]) -> None:
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    with open(BM25_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "metadatas": metadatas}, f)


def _build_bm25_index(texts: list[str]):
    from rank_bm25 import BM25Okapi
    tokenized = [_tokenize(t) for t in texts]
    return BM25Okapi(tokenized)


# ──────────────────────────────── VECTOR STORE ──────────────────────── #

def build_vector_store(texts: list[str], metadatas: list[dict]):
    from langchain_community.vectorstores import FAISS
    embeddings = get_embeddings()
    vs = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vs.save_local(FAISS_INDEX_PATH)
    _save_bm25_corpus(texts, metadatas)
    log.info("Built new FAISS+BM25 index with %d chunks.", len(texts))
    return vs


def append_to_vector_store(texts: list[str], metadatas: list[dict]):
    from langchain_community.vectorstores import FAISS
    embeddings = get_embeddings()
    vs = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    vs.add_texts(texts, metadatas=metadatas)
    vs.save_local(FAISS_INDEX_PATH)

    old_texts, old_metadatas = _load_bm25_corpus()
    _save_bm25_corpus(old_texts + texts, old_metadatas + metadatas)
    log.info("Appended %d chunks to existing index.", len(texts))
    return vs


def load_vector_store():
    from langchain_community.vectorstores import FAISS
    if not os.path.exists(FAISS_INDEX_PATH):
        raise IndexNotFoundError(
            "No document index found. Upload and process documents first."
        )
    embeddings = get_embeddings()
    try:
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        log.error("Failed to load FAISS index: %s", e)
        raise IndexNotFoundError(f"Index exists but could not be loaded: {e}") from e


# ──────────────────────────── HYBRID RETRIEVAL ──────────────────────── #

@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: Any
    dense_score: float | None = None   # cosine similarity, higher = better (0..1)
    sparse_score: float | None = None  # raw BM25 score, higher = better
    fused_score: float = 0.0           # RRF-combined rank score
    embedding: list[float] | None = field(default=None, repr=False)


def _dense_search(vs, query: str, k: int) -> list[RetrievedChunk]:
    """FAISS similarity search with relevance scores (cosine similarity)."""
    results = vs.similarity_search_with_relevance_scores(query, k=k)
    chunks = []
    for doc, score in results:
        # langchain's relevance score is already normalised to roughly 0..1
        # for cosine-similarity-backed FAISS indexes; clip defensively.
        norm_score = max(0.0, min(1.0, float(score)))
        chunks.append(RetrievedChunk(
            text=doc.page_content,
            source=doc.metadata.get("source", "Unknown"),
            page=doc.metadata.get("page", "?"),
            dense_score=norm_score,
        ))
    return chunks


def _sparse_search(query: str, k: int) -> list[RetrievedChunk]:
    """BM25 keyword search over the same corpus, run independently of FAISS."""
    texts, metadatas = _load_bm25_corpus()
    if not texts:
        return []
    bm25 = _build_bm25_index(texts)
    scores = bm25.get_scores(_tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    chunks = []
    for i in ranked_idx:
        if scores[i] <= 0:
            continue
        meta = metadatas[i]
        chunks.append(RetrievedChunk(
            text=texts[i],
            source=meta.get("source", "Unknown"),
            page=meta.get("page", "?"),
            sparse_score=float(scores[i]),
        ))
    return chunks


def _reciprocal_rank_fusion(
    dense: list[RetrievedChunk], sparse: list[RetrievedChunk], k: int = RRF_K
) -> list[RetrievedChunk]:
    """
    Fuse dense and sparse ranked lists using Reciprocal Rank Fusion:
    score(d) = sum over lists containing d of 1 / (k + rank(d))
    RRF is rank-based (not score-based), which avoids having to normalise
    BM25 scores and cosine similarities onto the same scale.
    """
    def key(c: RetrievedChunk):
        return (c.source, c.page, c.text[:50])

    fused: dict[tuple, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense, start=1):
        kk = key(chunk)
        if kk not in fused:
            fused[kk] = chunk
        fused[kk].fused_score += 1.0 / (k + rank)

    for rank, chunk in enumerate(sparse, start=1):
        kk = key(chunk)
        if kk not in fused:
            fused[kk] = chunk
        else:
            # carry sparse score onto the existing (dense) record
            fused[kk].sparse_score = chunk.sparse_score
        fused[kk].fused_score += 1.0 / (k + rank)

    return sorted(fused.values(), key=lambda c: c.fused_score, reverse=True)


def _mmr_rerank(
    vs, query: str, candidates: list[RetrievedChunk], top_n: int, lambda_mult: float = MMR_LAMBDA
) -> list[RetrievedChunk]:
    """
    Maximal Marginal Relevance re-ranking over the fused candidate set.
    Selects chunks that are individually relevant to the query AND
    mutually dissimilar from each other, reducing redundant near-duplicate
    chunks (e.g. the same clause repeated across overlapping chunks) in the
    final context window sent to the LLM.
    """
    if len(candidates) <= top_n:
        return candidates

    import numpy as np
    embeddings_model = get_embeddings()

    query_vec = np.array(embeddings_model.embed_query(query))
    doc_vecs = np.array(embeddings_model.embed_documents([c.text for c in candidates]))

    def cosine(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)

    relevance = [cosine(query_vec, v) for v in doc_vecs]

    selected_idx: list[int] = []
    remaining_idx = list(range(len(candidates)))

    while remaining_idx and len(selected_idx) < top_n:
        best_idx, best_score = None, float("-inf")
        for i in remaining_idx:
            diversity_penalty = 0.0
            if selected_idx:
                diversity_penalty = max(cosine(doc_vecs[i], doc_vecs[j]) for j in selected_idx)
            mmr_score = lambda_mult * relevance[i] - (1 - lambda_mult) * diversity_penalty
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, i
        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)

    return [candidates[i] for i in selected_idx]


def hybrid_retrieve(question: str, top_k: int = TOP_K_FINAL) -> list[RetrievedChunk]:
    """
    Full retrieval pipeline: dense (FAISS) + sparse (BM25) -> RRF fusion ->
    MMR diversity rerank -> top_k chunks for the LLM context window.
    """
    vs = load_vector_store()

    dense = _dense_search(vs, question, k=TOP_K_DENSE)
    sparse = _sparse_search(question, k=TOP_K_SPARSE)
    fused = _reciprocal_rank_fusion(dense, sparse)[:TOP_K_FUSED]

    log.info(
        "Retrieval for %r: %d dense, %d sparse, %d after fusion",
        question[:60], len(dense), len(sparse), len(fused),
    )

    if not fused:
        return []

    reranked = _mmr_rerank(vs, question, fused, top_n=top_k)
    return reranked


# ─────────────────────────────── PROMPT ─────────────────────────────── #

SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant. Your job is to answer employee questions accurately using ONLY the context provided below.

Rules you must follow:
1. Answer strictly from the context. Do NOT use outside knowledge.
2. If the answer is not present in the context, respond with:
   "I could not find this information in the provided documents."
3. Be concise and professional.
4. Do not fabricate page numbers, policies, or figures.
5. If the question is ambiguous, answer what you can and note the ambiguity.
6. When you state a fact, you may reference it like [Chunk 1] to indicate which context block it came from.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Conversation so far:
{history}

Question: {question}

Answer:"""


def build_prompt():
    from langchain_core.prompts import PromptTemplate
    return PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "history", "question"],
    )


def format_history(messages: list[dict]) -> str:
    messages = validate_history(messages)
    lines = []
    for m in messages[-(MAX_HISTORY_TURNS * 2):]:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines) if lines else "None"


def build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Chunk {i}] Source: {c.source} | Page: {c.page}\n{c.text}")
    return "\n\n".join(parts)


# ─────────────────────────── CONFIDENCE SCORING ─────────────────────── #

def compute_confidence(chunks: list[RetrievedChunk]) -> float:
    """
    Confidence is derived from the actual dense-retrieval similarity scores
    (cosine similarity between the query embedding and each retrieved chunk
    embedding, as reported by FAISS), not from naive keyword overlap.

    We use the mean of the top retrieved chunks' dense scores, with a small
    bonus for retrieval agreement (chunks that were found by BOTH dense and
    sparse search are more reliably relevant), capped at 0.99 since we never
    want to claim full certainty for a generative system.
    """
    if not chunks:
        return 0.0

    dense_scores = [c.dense_score for c in chunks if c.dense_score is not None]
    base = sum(dense_scores) / len(dense_scores) if dense_scores else 0.3

    agreement_bonus = 0.0
    both_found = sum(1 for c in chunks if c.dense_score is not None and c.sparse_score is not None)
    if chunks:
        agreement_bonus = 0.1 * (both_found / len(chunks))

    return round(min(base + agreement_bonus, 0.99), 2)


# ─────────────────────────────── ANSWER ─────────────────────────────── #

@dataclass
class AnswerResult:
    answer: str
    sources: list[dict]
    confidence: float
    latency_ms: int = 0
    retrieved_chunks: int = 0


def answer_question(question: str, history_messages: list[dict] | None = None) -> AnswerResult:
    """
    End-to-end RAG: validate -> retrieve (hybrid + rerank) -> generate -> score.
    Raises RAGError subclasses on failure; callers (api.py / chat_pdf.py)
    translate these into HTTP responses or UI messages.
    """
    start = time.monotonic()
    question = validate_question(question)

    if not index_exists():
        raise IndexNotFoundError(
            "No document index found. Upload and process documents first."
        )

    chunks = hybrid_retrieve(question, top_k=TOP_K_FINAL)

    if not chunks:
        return AnswerResult(
            answer="I could not find this information in the provided documents.",
            sources=[],
            confidence=0.0,
            latency_ms=int((time.monotonic() - start) * 1000),
            retrieved_chunks=0,
        )

    context = build_context(chunks)
    history_str = format_history(history_messages or [])

    chain = build_prompt() | get_llm()
    try:
        from langchain_core.output_parsers import StrOutputParser
        chain = chain | StrOutputParser()
        answer = chain.invoke({"context": context, "history": history_str, "question": question})
    except Exception as e:
        log.error("LLM generation failed for question %r: %s", question[:60], e)
        raise LLMGenerationError(f"The language model failed to generate an answer: {e}") from e

    seen, sources = set(), []
    for c in chunks:
        key = (c.source, c.page)
        if key not in seen:
            seen.add(key)
            sources.append({"document": c.source, "page": c.page})

    confidence = compute_confidence(chunks)
    latency_ms = int((time.monotonic() - start) * 1000)
    log.info(
        "Answered %r in %dms, confidence=%.2f, sources=%d",
        question[:60], latency_ms, confidence, len(sources),
    )

    return AnswerResult(
        answer=answer,
        sources=sources,
        confidence=confidence,
        latency_ms=latency_ms,
        retrieved_chunks=len(chunks),
    )
