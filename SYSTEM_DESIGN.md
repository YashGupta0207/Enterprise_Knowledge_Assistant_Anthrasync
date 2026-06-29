# System Design Document

## High-Level Architecture

The Enterprise Knowledge Assistant is a RAG application with three layers:

1. Interface layer: `chat_pdf.py` provides a Streamlit chat UI, and `api.py` provides a FastAPI REST API.
2. RAG core layer: `rag_core.py` owns ingestion, chunking, indexing, retrieval, prompting, generation, confidence scoring, and errors.
3. Storage layer: FAISS stores dense vectors on disk, and a JSON BM25 corpus stores sparse retrieval text and metadata.

This split keeps UI and API code thin. Any retrieval improvement made in `rag_core.py` automatically benefits both Streamlit and FastAPI.

## Data Flow

1. A user uploads PDFs in Streamlit.
2. `extract_text_with_metadata()` extracts text per page and preserves document/page metadata.
3. `chunk_pages()` splits text into overlapping chunks and attaches metadata.
4. `build_vector_store()` or `append_to_vector_store()` writes chunks to FAISS and BM25 storage.
5. When a user asks a question, `hybrid_retrieve()` expands the query and searches both FAISS and BM25.
6. Weighted Reciprocal Rank Fusion merges dense and sparse results.
7. Duplicate removal and MMR reranking choose the final context chunks.
8. The LLM receives only the selected context, recent history, and current question.
9. The app returns an answer, sources, confidence, latency, and retrieved chunk count.

## Component Explanation

- `chat_pdf.py`: Handles file upload, indexing button, chat history, source display, confidence display, and reset controls.
- `api.py`: Exposes `/health` and `/ask`, validates request/response models, and maps RAG errors to HTTP status codes.
- `rag_core.py`: Implements the full reusable RAG pipeline.
- FAISS index: Provides dense semantic retrieval for meaning-based matches.
- BM25 corpus: Provides keyword retrieval for exact names, numbers, codes, and policy terms.
- OpenRouter LLM: Generates final natural-language answers from retrieved context.

## Retrieval Strategy

The system uses hybrid retrieval because enterprise questions often combine semantic intent with exact terms. Dense search handles paraphrases, while BM25 handles exact words and numeric policy details. RRF avoids fragile score normalization by combining rank positions instead of raw scores.

## Scalability Considerations

The current system is suitable for a take-home assignment and small internal knowledge bases. For larger deployments:

- Replace local FAISS with a managed vector database or shared vector service.
- Replace JSON BM25 storage with OpenSearch, Elasticsearch, or another persistent sparse index.
- Add authentication and per-document authorization.
- Add background indexing jobs instead of indexing inside the Streamlit request cycle.
- Add request-level rate limiting and LLM concurrency controls.
- Store evaluation traces and user feedback for continuous improvement.

## Assumptions

- Uploaded PDFs contain extractable text.
- The user provides a valid `OPENROUTER_API_KEY`.
- Documents are non-confidential or are handled in a secure local environment.
- The assignment evaluation corpus is small enough for local FAISS and BM25.
