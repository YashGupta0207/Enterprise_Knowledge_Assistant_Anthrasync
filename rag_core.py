"""
Shared RAG pipeline for the Enterprise Knowledge Assistant.

The Streamlit UI (chat_pdf.py) and FastAPI service (api.py) import this
module directly, so this file owns retrieval, ranking, prompting, answer
generation, and confidence scoring while preserving the public API expected
by those two callers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from statistics import mean
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging

LOG_FILE = os.getenv("EKA_LOG_FILE", "eka.log")


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        root.warning("Could not open log file %s; logging to console only.", LOG_FILE)


log = logging.getLogger("eka.rag_core")

# ---------------------------------------------------------------------------
# Configuration

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("EKA_LLM_MODEL", "openrouter/free")

FAISS_INDEX_PATH = os.getenv("EKA_FAISS_INDEX_PATH", "faiss_index")
BM25_INDEX_PATH = os.path.join(FAISS_INDEX_PATH, "bm25_corpus.json")
INDEX_META_PATH = os.path.join(FAISS_INDEX_PATH, "indexed_files.json")

# A compact model keeps local startup practical. Override with a stronger
# embedding model, for example BAAI/bge-base-en-v1.5, when resources allow it.
EMBEDDING_MODEL_NAME = os.getenv(
    "EKA_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

TOP_K_DENSE = int(os.getenv("EKA_TOP_K_DENSE", "24"))
TOP_K_SPARSE = int(os.getenv("EKA_TOP_K_SPARSE", "24"))
TOP_K_FUSED = int(os.getenv("EKA_TOP_K_FUSED", "16"))
TOP_K_FINAL = int(os.getenv("EKA_TOP_K_FINAL", "5"))

CHUNK_SIZE = int(os.getenv("EKA_CHUNK_SIZE", "1100"))
CHUNK_OVERLAP = int(os.getenv("EKA_CHUNK_OVERLAP", "220"))
MAX_HISTORY_TURNS = int(os.getenv("EKA_MAX_HISTORY_TURNS", "6"))
MAX_QUESTION_LENGTH = int(os.getenv("EKA_MAX_QUESTION_LENGTH", "2000"))

RRF_K = int(os.getenv("EKA_RRF_K", "50"))
DENSE_RRF_WEIGHT = float(os.getenv("EKA_DENSE_RRF_WEIGHT", "1.15"))
SPARSE_RRF_WEIGHT = float(os.getenv("EKA_SPARSE_RRF_WEIGHT", "1.00"))
MMR_LAMBDA = float(os.getenv("EKA_MMR_LAMBDA", "0.68"))

SOURCE_SNIPPET_MAX_CHARS = int(os.getenv("EKA_SOURCE_SNIPPET_MAX_CHARS", "1800"))
LOW_CONFIDENCE_ANSWER_PATTERNS = (
    "could not find",
    "not present in the context",
    "not provided in the documents",
    "not enough information",
    "insufficient information",
)

# ---------------------------------------------------------------------------
# Errors


class RAGError(Exception):
    """Base class for recoverable errors raised by this module."""


class IndexNotFoundError(RAGError):
    """Raised when no FAISS/BM25 index exists yet."""


class EmptyDocumentError(RAGError):
    """Raised when a PDF yields no extractable text."""


class InvalidQuestionError(RAGError):
    """Raised when the incoming question fails validation."""


class LLMConfigError(RAGError):
    """Raised when the LLM cannot be constructed."""


class LLMGenerationError(RAGError):
    """Raised when the LLM call itself fails."""


# ---------------------------------------------------------------------------
# Validation


def validate_question(question: str) -> str:
    if question is None:
        raise InvalidQuestionError("Question cannot be empty.")
    question = re.sub(r"\s+", " ", question).strip()
    if not question:
        raise InvalidQuestionError("Question cannot be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise InvalidQuestionError(
            f"Question is too long ({len(question)} chars). "
            f"Limit is {MAX_QUESTION_LENGTH} characters."
        )
    return question


def validate_history(history: list[dict] | None) -> list[dict]:
    if not history:
        return []
    cleaned = []
    for message in history:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()})
    return cleaned


# ---------------------------------------------------------------------------
# Index metadata


def load_indexed_files() -> set[str]:
    if os.path.exists(INDEX_META_PATH):
        try:
            with open(INDEX_META_PATH, "r", encoding="utf-8") as file:
                return set(json.load(file))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read index metadata (%s); treating as empty.", exc)
    return set()


def save_indexed_files(names: set[str]) -> None:
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    with open(INDEX_META_PATH, "w", encoding="utf-8") as file:
        json.dump(sorted(names), file, ensure_ascii=False, indent=2)


def index_exists() -> bool:
    return os.path.exists(FAISS_INDEX_PATH) and bool(load_indexed_files())


def _clear_runtime_caches() -> None:
    _load_bm25_corpus_cached.cache_clear()
    _build_bm25_index_cached.cache_clear()
    _load_vector_store_cached.cache_clear()


# ---------------------------------------------------------------------------
# Embeddings and LLM

_embeddings_singleton = None


def get_embeddings():
    """Lazily construct and cache the local embedding model."""
    global _embeddings_singleton
    if _embeddings_singleton is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        log.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embeddings_singleton = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_singleton


def get_llm():
    if not OPENROUTER_API_KEY:
        raise LLMConfigError("OPENROUTER_API_KEY not configured. Set it in your .env file.")

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=1200,
        timeout=30,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Enterprise Knowledge Assistant",
        },
    )


# ---------------------------------------------------------------------------
# PDF extraction


def _clean_extracted_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_with_metadata(pdf_files: list) -> list[dict]:
    """
    Extract text and flattened table rows per page from file-like PDF objects.
    Returns [{"text", "source", "page"}].
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
                    text_parts = [page.extract_text(x_tolerance=1, y_tolerance=3) or ""]
                    try:
                        for table in page.extract_tables() or []:
                            for row in table or []:
                                cells = [str(cell).strip() for cell in (row or []) if cell]
                                if cells:
                                    text_parts.append(" | ".join(cells))
                    except Exception as exc:
                        log.debug("Table extraction failed on %s p%d: %s", filename, page_num, exc)

                    text = _clean_extracted_text("\n".join(text_parts))
                    if text:
                        pages.append({"text": text, "source": filename, "page": page_num})
            extracted_via_plumber = True
        except Exception as exc:
            log.warning("pdfplumber failed for %s (%s); trying PyPDF2 fallback.", filename, exc)

        if not extracted_via_plumber:
            try:
                from PyPDF2 import PdfReader

                pdf.seek(0)
                reader = PdfReader(pdf)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = _clean_extracted_text(page.extract_text() or "")
                    if text:
                        pages.append({"text": text, "source": filename, "page": page_num})
            except Exception as exc:
                log.error("Could not read %s with either extractor: %s", filename, exc)

    if not pages:
        log.warning("No extractable text found across %d file(s).", len(pdf_files))
    return pages


# ---------------------------------------------------------------------------
# Chunking


def chunk_pages(pages: list[dict]) -> tuple[list[str], list[dict]]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator=True,
    )

    texts: list[str] = []
    metadatas: list[dict] = []
    seen_hashes: set[str] = set()

    for page in pages:
        chunks = splitter.split_text(page["text"])
        for chunk_index, chunk in enumerate(chunks, start=1):
            cleaned = _clean_extracted_text(chunk)
            if len(cleaned) < 40:
                continue
            digest = hashlib.sha1(
                f"{page['source']}:{page['page']}:{_normalize_for_key(cleaned)}".encode("utf-8")
            ).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            texts.append(cleaned)
            metadatas.append(
                {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk": chunk_index,
                    "chunk_hash": digest,
                }
            )

    return texts, metadatas


# ---------------------------------------------------------------------------
# BM25 store

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "i", "in", "is", "it", "its", "may", "of", "on", "or", "our", "that",
    "the", "their", "there", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "you", "your",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9_./-]*", text.lower())
    expanded: list[str] = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        expanded.append(token)
        if "-" in token or "/" in token:
            expanded.extend(part for part in re.split(r"[-/]", token) if part and part not in STOPWORDS)
    return expanded


def _normalize_for_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


@lru_cache(maxsize=1)
def _load_bm25_corpus_cached(index_mtime: float) -> tuple[tuple[str, ...], tuple[dict, ...]]:
    del index_mtime
    if not os.path.exists(BM25_INDEX_PATH):
        return (), ()
    try:
        with open(BM25_INDEX_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        return tuple(data.get("texts", [])), tuple(data.get("metadatas", []))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read BM25 corpus (%s); treating as empty.", exc)
        return (), ()


def _bm25_mtime() -> float:
    try:
        return os.path.getmtime(BM25_INDEX_PATH)
    except OSError:
        return 0.0


def _load_bm25_corpus() -> tuple[list[str], list[dict]]:
    texts, metadatas = _load_bm25_corpus_cached(_bm25_mtime())
    return list(texts), list(metadatas)


def _save_bm25_corpus(texts: list[str], metadatas: list[dict]) -> None:
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    with open(BM25_INDEX_PATH, "w", encoding="utf-8") as file:
        json.dump({"texts": texts, "metadatas": metadatas}, file, ensure_ascii=False)
    _clear_runtime_caches()


@lru_cache(maxsize=1)
def _build_bm25_index_cached(index_mtime: float):
    from rank_bm25 import BM25Okapi

    texts, _ = _load_bm25_corpus_cached(index_mtime)
    return BM25Okapi([_tokenize(text) for text in texts]) if texts else None


def _build_bm25_index():
    return _build_bm25_index_cached(_bm25_mtime())


# ---------------------------------------------------------------------------
# Vector store


def build_vector_store(texts: list[str], metadatas: list[dict]):
    from langchain_community.vectorstores import FAISS

    if not texts:
        raise EmptyDocumentError("No text chunks were produced from the uploaded documents.")

    vector_store = FAISS.from_texts(texts, embedding=get_embeddings(), metadatas=metadatas)
    vector_store.save_local(FAISS_INDEX_PATH)
    _save_bm25_corpus(texts, metadatas)
    log.info("Built new FAISS+BM25 index with %d chunks.", len(texts))
    return vector_store


def append_to_vector_store(texts: list[str], metadatas: list[dict]):
    if not texts:
        raise EmptyDocumentError("No text chunks were produced from the uploaded documents.")

    vector_store = load_vector_store()
    vector_store.add_texts(texts, metadatas=metadatas)
    vector_store.save_local(FAISS_INDEX_PATH)

    old_texts, old_metadatas = _load_bm25_corpus()
    _save_bm25_corpus(old_texts + texts, old_metadatas + metadatas)
    _clear_runtime_caches()
    log.info("Appended %d chunks to existing index.", len(texts))
    return vector_store


@lru_cache(maxsize=1)
def _load_vector_store_cached(index_mtime: float):
    del index_mtime
    from langchain_community.vectorstores import FAISS

    if not os.path.exists(FAISS_INDEX_PATH):
        raise IndexNotFoundError("No document index found. Upload and process documents first.")
    try:
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        log.error("Failed to load FAISS index: %s", exc)
        raise IndexNotFoundError(f"Index exists but could not be loaded: {exc}") from exc


def _faiss_mtime() -> float:
    candidates = [
        os.path.join(FAISS_INDEX_PATH, "index.faiss"),
        os.path.join(FAISS_INDEX_PATH, "index.pkl"),
    ]
    mtimes = []
    for path in candidates:
        try:
            mtimes.append(os.path.getmtime(path))
        except OSError:
            pass
    return max(mtimes) if mtimes else 0.0


def load_vector_store():
    return _load_vector_store_cached(_faiss_mtime())


# ---------------------------------------------------------------------------
# Retrieval


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: Any
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    query_variants: set[str] = field(default_factory=set)
    embedding: list[float] | None = field(default=None, repr=False)


def _chunk_key(chunk: RetrievedChunk) -> tuple[str, Any, str]:
    return (chunk.source, chunk.page, _normalize_for_key(chunk.text)[:220])


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _normalize_dense_score(score: float) -> float:
    score = float(score)
    if math.isnan(score):
        return 0.0
    if 0.0 <= score <= 1.0:
        return score
    if -1.0 <= score <= 1.0:
        return (score + 1.0) / 2.0
    # Some vector store methods expose squared L2 distance; convert it gently.
    return 1.0 / (1.0 + max(score, 0.0))


def _question_keywords(question: str) -> list[str]:
    tokens = _tokenize(question)
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            keywords.append(token)
    return keywords


def _expand_query(question: str, history_messages: list[dict] | None = None) -> list[str]:
    variants = [question]
    keywords = _question_keywords(question)
    if keywords:
        variants.append(" ".join(keywords))

    nounish = [token for token in keywords if len(token) >= 4]
    if nounish and nounish != keywords:
        variants.append(" ".join(nounish))

    history = validate_history(history_messages)
    previous_user_messages = [m["content"] for m in history if m["role"] == "user"]
    if previous_user_messages and len(question.split()) <= 10:
        variants.append(f"{previous_user_messages[-1]} {question}")

    unique: list[str] = []
    seen = set()
    for variant in variants:
        normalized = _normalize_for_key(variant)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(variant)
    return unique[:4]


def _dense_search(vs, query: str, k: int, query_label: str) -> list[RetrievedChunk]:
    try:
        results = vs.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        raw_results = vs.similarity_search_with_score(query, k=k)
        results = [(doc, _normalize_dense_score(score)) for doc, score in raw_results]

    chunks: list[RetrievedChunk] = []
    for rank, (doc, score) in enumerate(results, start=1):
        chunks.append(
            RetrievedChunk(
                text=doc.page_content,
                source=doc.metadata.get("source", "Unknown"),
                page=doc.metadata.get("page", "?"),
                dense_score=_normalize_dense_score(float(score)),
                dense_rank=rank,
                query_variants={query_label},
            )
        )
    return chunks


def _sparse_search(query: str, k: int, query_label: str) -> list[RetrievedChunk]:
    texts, metadatas = _load_bm25_corpus()
    if not texts:
        return []

    bm25 = _build_bm25_index()
    if bm25 is None:
        return []

    scores = bm25.get_scores(_tokenize(query))
    max_score = max(scores) if len(scores) else 0.0
    if max_score <= 0:
        return []

    ranked_idx = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:k]
    chunks: list[RetrievedChunk] = []
    for rank, index in enumerate(ranked_idx, start=1):
        if scores[index] <= 0:
            continue
        meta = metadatas[index]
        chunks.append(
            RetrievedChunk(
                text=texts[index],
                source=meta.get("source", "Unknown"),
                page=meta.get("page", "?"),
                sparse_score=float(scores[index] / max_score),
                sparse_rank=rank,
                query_variants={query_label},
            )
        )
    return chunks


def _merge_retrieval_signal(target: RetrievedChunk, incoming: RetrievedChunk) -> None:
    if incoming.dense_score is not None:
        target.dense_score = max(target.dense_score or 0.0, incoming.dense_score)
    if incoming.sparse_score is not None:
        target.sparse_score = max(target.sparse_score or 0.0, incoming.sparse_score)
    if incoming.dense_rank is not None:
        target.dense_rank = min(target.dense_rank or incoming.dense_rank, incoming.dense_rank)
    if incoming.sparse_rank is not None:
        target.sparse_rank = min(target.sparse_rank or incoming.sparse_rank, incoming.sparse_rank)
    target.query_variants.update(incoming.query_variants)


def _reciprocal_rank_fusion(
    dense_lists: list[list[RetrievedChunk]],
    sparse_lists: list[list[RetrievedChunk]],
    k: int = RRF_K,
) -> list[RetrievedChunk]:
    fused: dict[tuple[str, Any, str], RetrievedChunk] = {}

    def add_ranked(chunks: list[RetrievedChunk], weight: float) -> None:
        for rank, chunk in enumerate(chunks, start=1):
            key = _chunk_key(chunk)
            if key not in fused:
                fused[key] = chunk
            else:
                _merge_retrieval_signal(fused[key], chunk)
            fused[key].fused_score += weight / (k + rank)

    for chunks in dense_lists:
        add_ranked(chunks, DENSE_RRF_WEIGHT)
    for chunks in sparse_lists:
        add_ranked(chunks, SPARSE_RRF_WEIGHT)

    return sorted(fused.values(), key=lambda chunk: chunk.fused_score, reverse=True)


def _dedupe_near_duplicates(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    unique: list[RetrievedChunk] = []
    seen_exact: set[str] = set()

    for chunk in chunks:
        normalized = _normalize_for_key(chunk.text)
        exact_key = normalized[:500]
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)

        too_similar = False
        token_set = set(normalized.split())
        for existing in unique:
            existing_tokens = set(_normalize_for_key(existing.text).split())
            if not token_set or not existing_tokens:
                continue
            overlap = len(token_set & existing_tokens) / max(1, min(len(token_set), len(existing_tokens)))
            same_page = chunk.source == existing.source and chunk.page == existing.page
            if overlap > (0.92 if same_page else 0.96):
                too_similar = True
                break
        if not too_similar:
            unique.append(chunk)

    return unique


def _cosine(a, b) -> float:
    import numpy as np

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def _mmr_rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_n: int,
    lambda_mult: float = MMR_LAMBDA,
) -> list[RetrievedChunk]:
    if len(candidates) <= top_n:
        return candidates

    import numpy as np

    embeddings_model = get_embeddings()
    query_vec = np.array(embeddings_model.embed_query(query))
    doc_vecs = np.array(embeddings_model.embed_documents([candidate.text for candidate in candidates]))

    relevance = []
    for candidate, doc_vec in zip(candidates, doc_vecs):
        semantic = _clamp((_cosine(query_vec, doc_vec) + 1.0) / 2.0)
        lexical = candidate.sparse_score or 0.0
        fused = min(candidate.fused_score * 30.0, 1.0)
        relevance.append((0.70 * semantic) + (0.20 * lexical) + (0.10 * fused))
        candidate.embedding = doc_vec.tolist()
        candidate.dense_score = max(candidate.dense_score or 0.0, semantic)

    selected_idx: list[int] = []
    remaining_idx = list(range(len(candidates)))

    while remaining_idx and len(selected_idx) < top_n:
        best_idx = remaining_idx[0]
        best_score = float("-inf")
        for index in remaining_idx:
            diversity_penalty = 0.0
            if selected_idx:
                diversity_penalty = max(_cosine(doc_vecs[index], doc_vecs[selected]) for selected in selected_idx)
            mmr_score = (lambda_mult * relevance[index]) - ((1.0 - lambda_mult) * diversity_penalty)
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = index
        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)

    return [candidates[index] for index in selected_idx]


def hybrid_retrieve(
    question: str,
    top_k: int = TOP_K_FINAL,
    history_messages: list[dict] | None = None,
) -> list[RetrievedChunk]:
    """
    Dense FAISS + sparse BM25 retrieval with query expansion, weighted RRF,
    duplicate removal, and MMR diversity reranking.
    """
    vector_store = load_vector_store()
    query_variants = _expand_query(question, history_messages)

    dense_lists = [
        _dense_search(vector_store, variant, TOP_K_DENSE, f"q{i}")
        for i, variant in enumerate(query_variants)
    ]
    sparse_lists = [
        _sparse_search(variant, TOP_K_SPARSE, f"q{i}")
        for i, variant in enumerate(query_variants)
    ]

    fused = _reciprocal_rank_fusion(dense_lists, sparse_lists)
    fused = _dedupe_near_duplicates(fused)[:TOP_K_FUSED]

    log.info(
        "Retrieval for %r: variants=%d dense=%d sparse=%d fused=%d",
        question[:60],
        len(query_variants),
        sum(len(items) for items in dense_lists),
        sum(len(items) for items in sparse_lists),
        len(fused),
    )

    if not fused:
        return []

    return _mmr_rerank(question, fused, top_n=top_k)


# ---------------------------------------------------------------------------
# Prompting

SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant. Answer the user's question using ONLY the provided context.

Required behavior:
1. If the answer is not clearly supported by the context, say: "I could not find this information in the provided documents."
2. Do not use outside knowledge, assumptions, or invented figures.
3. Be concise, direct, and professional.
4. Cite supporting context with [Chunk N] when stating factual claims.
5. If multiple chunks disagree, mention the disagreement instead of choosing a side.
6. If the question is ambiguous, answer only the supported interpretation and briefly note the ambiguity.

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
    for message in messages[-(MAX_HISTORY_TURNS * 2):]:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines) if lines else "None"


def build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        text = chunk.text[:SOURCE_SNIPPET_MAX_CHARS].strip()
        parts.append(f"[Chunk {index}] Source: {chunk.source} | Page: {chunk.page}\n{text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Confidence scoring


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def compute_confidence(chunks: list[RetrievedChunk], answer: str | None = None) -> float:
    """
    Estimate answer confidence from retrieval quality, corroboration, and
    diversity. This is a calibrated retrieval confidence, not a claim that
    the generated answer is mathematically certain.
    """
    if not chunks:
        return 0.0

    if answer and any(pattern in answer.lower() for pattern in LOW_CONFIDENCE_ANSWER_PATTERNS):
        return 0.12

    dense_scores = [score for score in (chunk.dense_score for chunk in chunks) if score is not None]
    sparse_scores = [score for score in (chunk.sparse_score for chunk in chunks) if score is not None]

    best_dense = max(dense_scores) if dense_scores else 0.0
    avg_dense = mean(dense_scores[: min(3, len(dense_scores))]) if dense_scores else 0.0
    best_sparse = max(sparse_scores) if sparse_scores else 0.0

    agreement = sum(
        1 for chunk in chunks if chunk.dense_score is not None and chunk.sparse_score is not None
    ) / len(chunks)
    source_diversity = len({(chunk.source, chunk.page) for chunk in chunks}) / len(chunks)
    variant_coverage = len(set().union(*(chunk.query_variants for chunk in chunks))) if chunks else 0
    variant_bonus = min(variant_coverage / 3.0, 1.0)

    retrieval_strength = (
        (0.45 * best_dense)
        + (0.25 * avg_dense)
        + (0.12 * best_sparse)
        + (0.10 * agreement)
        + (0.05 * source_diversity)
        + (0.03 * variant_bonus)
    )

    # Map the useful retrieval range into a human-facing score. Good matches
    # should land in the 0.80-0.95 band; weak matches remain visibly low.
    confidence = 0.18 + (0.80 * _sigmoid((retrieval_strength - 0.58) * 6.5))

    if best_dense < 0.42 and best_sparse < 0.25:
        confidence = min(confidence, 0.45)
    elif best_dense < 0.55 and agreement < 0.25:
        confidence = min(confidence, 0.68)

    return round(_clamp(confidence, 0.0, 0.97), 2)


def _compute_answer_aware_confidence(
    question: str,
    chunks: list[RetrievedChunk],
    answer: str,
) -> float:
    base = compute_confidence(chunks, answer=answer)
    if base <= 0.15:
        return base

    question_terms = {token for token in _tokenize(question) if len(token) >= 4}
    context_terms = set(_tokenize(" ".join(chunk.text for chunk in chunks[:3])))
    if question_terms:
        term_coverage = len(question_terms & context_terms) / len(question_terms)
        if term_coverage < 0.35:
            base = min(base, 0.62)
        elif term_coverage > 0.70:
            base = min(0.97, base + 0.03)

    citation_count = len(re.findall(r"\[Chunk\s+\d+\]", answer, flags=re.I))
    if citation_count == 0 and len(answer.split()) > 20:
        base = min(base, 0.78)

    return round(_clamp(base, 0.0, 0.97), 2)


# ---------------------------------------------------------------------------
# Answering


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict]
    confidence: float
    latency_ms: int = 0
    retrieved_chunks: int = 0


def _select_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    seen: set[tuple[str, Any]] = set()
    sources: list[dict] = []
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"document": chunk.source, "page": chunk.page})
    return sources


def answer_question(question: str, history_messages: list[dict] | None = None) -> AnswerResult:
    start = time.monotonic()
    question = validate_question(question)
    history_messages = validate_history(history_messages)

    if not index_exists():
        raise IndexNotFoundError("No document index found. Upload and process documents first.")

    chunks = hybrid_retrieve(question, top_k=TOP_K_FINAL, history_messages=history_messages)
    if not chunks:
        return AnswerResult(
            answer="I could not find this information in the provided documents.",
            sources=[],
            confidence=0.0,
            latency_ms=int((time.monotonic() - start) * 1000),
            retrieved_chunks=0,
        )

    context = build_context(chunks)
    history_str = format_history(history_messages)
    chain = build_prompt() | get_llm()

    try:
        from langchain_core.output_parsers import StrOutputParser

        answer = (chain | StrOutputParser()).invoke(
            {"context": context, "history": history_str, "question": question}
        )
        answer = str(answer).strip()
    except Exception as exc:
        log.error("LLM generation failed for question %r: %s", question[:60], exc)
        raise LLMGenerationError(f"The language model failed to generate an answer: {exc}") from exc

    confidence = _compute_answer_aware_confidence(question, chunks, answer)
    latency_ms = int((time.monotonic() - start) * 1000)
    sources = _select_sources(chunks)

    log.info(
        "Answered %r in %dms, confidence=%.2f, chunks=%d, sources=%d",
        question[:60],
        latency_ms,
        confidence,
        len(chunks),
        len(sources),
    )

    return AnswerResult(
        answer=answer,
        sources=sources,
        confidence=confidence,
        latency_ms=latency_ms,
        retrieved_chunks=len(chunks),
    )
