# Enterprise Knowledge Assistant

A production-oriented Retrieval Augmented Generation (RAG) assistant for asking questions over internal PDF documents. The app supports PDF upload, text extraction, indexing, hybrid retrieval, cited answers, confidence scoring, a Streamlit chat UI, and an optional FastAPI endpoint.

## Features

- Upload one or more PDF documents from the Streamlit sidebar.
- Extract page-level text and table rows using `pdfplumber`, with `PyPDF2` fallback.
- Chunk documents with metadata for source document and page number.
- Build a persistent FAISS dense vector index and a BM25 sparse keyword corpus.
- Retrieve with hybrid search: dense semantic search plus BM25 keyword search.
- Fuse rankings with weighted Reciprocal Rank Fusion.
- Reduce duplicate context with MMR reranking.
- Generate concise answers with source references.
- Return confidence scores based on retrieval quality and answer support.
- Expose the same RAG core through both Streamlit and FastAPI.

## Architecture

```text
PDF Uploads
    |
    v
Document Extraction
pdfplumber + PyPDF2 fallback
page text + table rows
    |
    v
Chunking
RecursiveCharacterTextSplitter
1100 chars / 220 overlap
source + page metadata
    |
    +-------------------------+
    |                         |
    v                         v
FAISS Dense Index         BM25 Sparse Corpus
Sentence Transformers     rank-bm25
semantic retrieval        keyword retrieval
top 24 candidates         top 24 candidates
    |                         |
    +-----------+-------------+
                |
                v
Weighted Reciprocal Rank Fusion
top 16 fused candidates
                |
                v
Near-Duplicate Removal + MMR Reranking
top 5 context chunks
                |
                v
Prompt Assembly
context + chat history + question
                |
                v
OpenRouter Chat Model
                |
                v
Answer + Sources + Confidence
                |
        +-------+-------+
        v               v
Streamlit UI       FastAPI /ask
```

`rag_core.py` owns the full RAG pipeline. `chat_pdf.py` is only the Streamlit interface, and `api.py` is only the HTTP interface. This keeps retrieval and answer behavior consistent across both entry points.

## Setup

Use Python 3.11 or newer.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key
```

You can use any OpenRouter-compatible model by setting:

```env
EKA_LLM_MODEL=openrouter/free
```

## Run The Streamlit App

```bash
streamlit run chat_pdf.py
```

Then:

1. Upload PDF files in the sidebar.
2. Click the process/index button.
3. Ask questions in the chat input.
4. Open the sources expander to inspect source document and page references.

## Run The API

```bash
uvicorn api:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is the refund policy?\"}"
```

Example response:

```json
{
  "answer": "Refunds are allowed within 30 days. [Chunk 1]",
  "sources": [
    {
      "document": "Customer_Policy.pdf",
      "page": 5
    }
  ],
  "confidence": 0.91,
  "retrieved_chunks": 5,
  "latency_ms": 1840
}
```

## Configuration

All settings are optional and can be overridden with environment variables.

| Variable | Default | Purpose |
|---|---:|---|
| `EKA_LLM_MODEL` | `openrouter/free` | OpenRouter chat model |
| `EKA_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `EKA_FAISS_INDEX_PATH` | `faiss_index` | Persistent index directory |
| `EKA_TOP_K_DENSE` | `24` | Dense candidates from FAISS |
| `EKA_TOP_K_SPARSE` | `24` | Sparse candidates from BM25 |
| `EKA_TOP_K_FUSED` | `16` | Candidates after rank fusion |
| `EKA_TOP_K_FINAL` | `5` | Chunks sent to the LLM |
| `EKA_CHUNK_SIZE` | `1100` | Chunk size in characters |
| `EKA_CHUNK_OVERLAP` | `220` | Chunk overlap in characters |
| `EKA_LOG_FILE` | `eka.log` | Log file path |

## Design Decisions

### Chunking

The app uses `RecursiveCharacterTextSplitter` with 1100-character chunks and 220-character overlap. This keeps policy or FAQ sections large enough to preserve context while still small enough for precise retrieval. The overlap helps preserve sentences that cross chunk boundaries.

### Embeddings

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`. It runs locally on CPU, avoids embedding API costs, and is fast enough for a take-home assignment or small internal document collection. A stronger model can be configured with `EKA_EMBEDDING_MODEL`.

### Hybrid Retrieval

Dense retrieval is good for semantic matches, while BM25 is better for exact terms such as policy names, acronyms, numbers, and product codes. The system retrieves from both FAISS and BM25, then combines the ranked lists with weighted Reciprocal Rank Fusion.

### Reranking

After fusion, near-duplicate chunks are removed and Maximal Marginal Relevance reranking selects a diverse final context. This prevents the LLM from receiving several overlapping chunks that all say the same thing.

### Prompting

The prompt instructs the model to answer only from retrieved context, cite facts with `[Chunk N]`, avoid invented information, and use a fixed refusal when the answer is not supported by the documents.

### Confidence

Confidence is a retrieval-quality estimate. It uses dense similarity, sparse keyword support, dense/sparse agreement, source diversity, query-variant coverage, and answer-aware checks. If the answer says the information was not found, confidence is intentionally low.

## Evaluation Approach

For this assignment, evaluate the system with a small set of representative PDFs and questions:

| Case Type | Example | Expected Check |
|---|---|---|
| Direct fact lookup | "What is the employee leave policy?" | Correct answer and correct source page |
| Exact term lookup | "What is the refund period?" | BM25 retrieves exact policy wording |
| Multi-document question | "Compare onboarding and support escalation steps." | Sources include all relevant documents |
| Ambiguous question | "What is the policy?" | Answer notes ambiguity or uses available context carefully |
| Unanswerable question | "What is the CEO's home address?" | System refuses instead of hallucinating |

Suggested metrics:

- Retrieval Recall@5: whether the expected source page appears in the final sources.
- Source precision: whether returned sources are relevant.
- Answer coverage: whether key expected facts appear in the answer.
- Hallucination resistance: whether unsupported questions receive the refusal response.
- Latency: time returned in `latency_ms`.
- Confidence sanity: high for supported answers, low for refusals.

## Assignment Coverage

| Requirement | Implementation |
|---|---|
| Document ingestion | Streamlit PDF uploader |
| Text extraction | `pdfplumber` with `PyPDF2` fallback |
| Chunking and metadata | `chunk_pages()` in `rag_core.py` |
| Embeddings | Sentence Transformers |
| Searchable index | FAISS plus persisted BM25 corpus |
| Semantic search | FAISS similarity search |
| Hybrid search bonus | FAISS + BM25 + RRF |
| Reranking bonus | MMR reranking |
| Conversation memory bonus | Recent chat history included in prompt and retrieval expansion |
| Source citation | Document and page returned in UI/API |
| UI | Streamlit chat app |
| API preferred | FastAPI `/ask` and `/health` |
| Documentation | README plus `SYSTEM_DESIGN.md` |

## Known Limitations

- Scanned image-only PDFs are not OCR processed.
- `openrouter/free` can route to different free models, so generated wording may vary.
- Confidence is not a calibrated probability of correctness.
- The local FAISS/BM25 setup is suitable for small to medium document sets, not multi-tenant enterprise scale.
- There is no authentication or document-level access control.
- The API and Streamlit app share the on-disk index, not in-memory state.

## Future Improvements

- Add OCR for scanned PDFs.
- Add a formal automated evaluation harness with stored test cases.
- Add cross-encoder reranking for higher precision.
- Add user feedback on answers.
- Add authentication and per-document permissions.
- Move FAISS/BM25 to a managed vector database/search service for horizontal scaling.
- Pin a deterministic production LLM instead of `openrouter/free`.
