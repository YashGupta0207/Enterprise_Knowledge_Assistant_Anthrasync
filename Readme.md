# Enterprise Knowledge Assistant

A RAG system for answering employee questions over internal documents (HR policy, product docs, customer FAQs, etc.) with cited sources and a confidence score. Two front ends — a Streamlit chat UI and a FastAPI `/ask` endpoint — share one retrieval/generation core (`rag_core.py`).

## Contents

- [Architecture](#architecture)
- [Setup](#setup)
- [Running it](#running-it)
- [Design decisions](#design-decisions)
- [Evaluation](#evaluation)
- [Known limitations](#known-limitations)
- [Future improvements](#future-improvements)

## Architecture

```
                         ┌─────────────────────┐
   PDF uploads  ───────► │  Ingestion           │
                         │  pdfplumber→PyPDF2   │
                         │  fallback, per-page  │
                         │  text + table rows   │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  Chunking             │
                         │  RecursiveCharacter   │
                         │  Splitter, 800/150    │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                     ▼
        ┌──────────────────┐                 ┌──────────────────┐
        │ FAISS (dense)     │                 │ BM25 (sparse)     │
        │ MiniLM-L6-v2       │                 │ rank_bm25         │
        │ embeddings         │                 │ keyword index     │
        └─────────┬────────┘                 └─────────┬────────┘
                  │ top-8                                │ top-8
                  └────────────────┬────────────────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Reciprocal Rank Fusion │
                       │ (rank-based, no score  │
                       │  normalisation needed) │
                       └───────────┬───────────┘
                                   │ top-8
                                   ▼
                       ┌───────────────────────┐
                       │ MMR diversity rerank   │
                       │ (drops near-duplicate  │
                       │  overlapping chunks)   │
                       └───────────┬───────────┘
                                   │ top-4
                                   ▼
                       ┌───────────────────────┐
                       │ Prompt assembly        │
                       │ (context + history +   │
                       │  question)             │
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ LLM (OpenRouter free   │
                       │  tier) via LangChain   │
                       └───────────┬───────────┘
                                   ▼
                  answer + sources[] + confidence
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                   ▼
          Streamlit chat UI                    FastAPI POST /ask
          (chat_pdf.py)                        (api.py)
```

`rag_core.py` owns every step above this line. `api.py` and `chat_pdf.py` are thin adapters — HTTP schema/status codes on one side, Streamlit widgets on the other — that both call `rag_core.answer_question()`. Originally these were two copies of the same pipeline; they're now one implementation with two front doors, which is the only sane way to keep retrieval behavior consistent between them.

## Setup

```bash
git clone <repo-url> && cd <repo>
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY
```

`.env`:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys) — no card required for the free tier.

Optional overrides (all have working defaults, see `rag_core.py` config block):

| Variable | Default | Purpose |
|---|---|---|
| `EKA_LLM_MODEL` | `openrouter/free` | Model id passed to OpenRouter |
| `EKA_FAISS_INDEX_PATH` | `faiss_index` | Where the dense+sparse index is persisted |
| `EKA_TOP_K_FINAL` | `4` | Chunks sent to the LLM after rerank |
| `EKA_CHUNK_SIZE` / `EKA_CHUNK_OVERLAP` | `800` / `150` | Chunking params |
| `EKA_LOG_FILE` | `eka.log` | Log file path |

## Running it

**Streamlit UI:**
```bash
streamlit run chat_pdf.py
```
Upload PDFs in the sidebar, click "Process Documents", then ask questions in the chat box.

**FastAPI:**
```bash
uvicorn api:app --reload
```
```bash
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```
```json
{
  "answer": "Refunds are allowed within 30 days of purchase, provided the product is unused and in original packaging.",
  "sources": [{"document": "Customer_Policy.pdf", "page": 1}],
  "confidence": 0.87,
  "retrieved_chunks": 4,
  "latency_ms": 1840
}
```
`GET /health` reports index readiness and document count. Both UIs share whatever index exists in `EKA_FAISS_INDEX_PATH` — index once via Streamlit, query via either interface.

**Note:** the Streamlit UI and the API process don't share Python memory, only the on-disk index. If you index documents through Streamlit and then immediately hit the API in a separate process, the API will pick up the index from disk on its next request — there's no live cache invalidation needed since each request loads fresh from `faiss_index/`.

## Design decisions

### Chunking: 800 chars / 150 overlap, `RecursiveCharacterTextSplitter`
Most policy/FAQ paragraphs in the target document types (HR policy, product docs) run 150–400 words. 800 characters keeps a chunk to roughly one self-contained clause or sub-section without dragging in the next unrelated topic. 150-char overlap (~19%) is enough that a sentence split across a chunk boundary still appears whole in at least one chunk, without bloating the index. The splitter's separator hierarchy (`\n\n` → `\n` → `. ` → ` `) tries to break on paragraph/sentence boundaries before falling back to a hard character cut.

This is a static, content-agnostic choice — it isn't tuned per document type. A compliance document with long enumerated clauses or a technical manual with code blocks would likely want different chunk boundaries (see [Future improvements](#future-improvements)).

### Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, local
Runs on CPU, no API cost or rate limit, 384-dim (small index footprint), and is the standard baseline for short-passage retrieval. Quality is a step below OpenAI's `text-embedding-3` family on some benchmarks, but for paragraph-length internal-doc chunks the gap is small relative to what hybrid retrieval (below) buys back. Embedding locally also means indexing doesn't depend on the same API key/quota as generation.

### Hybrid retrieval: FAISS (dense) + BM25 (sparse), fused with RRF
Pure dense retrieval misses exact-term queries — a question like "what is the INR 500 receipt threshold" benefits from keyword matching on "500" and "receipt" as much as from semantic similarity. BM25 (via `rank_bm25`) runs independently over the same chunk corpus and is fused with the FAISS results using **Reciprocal Rank Fusion**: `score(chunk) = Σ 1/(k + rank)` across whichever lists contain it, `k=60`.

RRF was chosen over weighted score averaging because BM25 scores and cosine similarities live on incomparable scales — RRF only needs each list's *rank order*, sidestepping a score-normalization problem that has no principled answer (there's no natural way to say a BM25 score of 5.2 is "worth" a cosine similarity of 0.3).

### Reranking: MMR (Maximal Marginal Relevance), not a cross-encoder
After fusion, the top-8 fused candidates go through MMR (`λ=0.5`, balancing relevance against diversity) to cut down to the final top-4 sent to the LLM. The fused list often contains 2–3 chunks that are really the same clause repeated across overlapping chunk boundaries — MMR penalizes picking a chunk too similar to one already selected, so the final context window covers more distinct ground instead of three slightly different windows onto the same sentence.

A cross-encoder reranker (e.g. `ms-marco-MiniLM`) would likely give better *relevance* ranking, but it's a second model load and a real inference-time cost for marginal gain on this corpus size and query complexity — MMR using the embeddings already computed for FAISS was the cheaper choice that still addresses the actual problem (redundant chunks), not relevance ranking per se. Worth revisiting if eval shows precision issues that diversity alone doesn't fix.

### Confidence score: derived from real retrieval similarity, not keyword overlap
The original implementation scored confidence by counting how many query words appeared verbatim in the retrieved chunk text — a string-matching heuristic dressed up as a confidence number, with no relationship to what the embedding model or LLM actually "thought." It rewarded keyword-stuffed chunks and didn't move when retrieval was genuinely uncertain.

`compute_confidence()` now uses the **mean cosine similarity** (FAISS's normalized relevance score) across the final retrieved chunks, plus a small bonus (`+0.1 × fraction agreeing`) when a chunk was independently surfaced by *both* dense and sparse retrieval — agreement between two different retrieval signals is a real corroborating signal, disagreement isn't penalized but also doesn't help. Capped at 0.99: a generative system answering from retrieved text should never claim full certainty.

This is still a retrieval-confidence proxy, not an answer-correctness probability — see [Known limitations](#known-limitations).

### Prompt design
The system prompt is constrained ("answer ONLY from context," explicit refusal string, "don't fabricate page numbers") and asks for inline `[Chunk N]` references so an answer's claims can be traced back to a specific context block during eval/debugging. Conversation history (last `MAX_HISTORY_TURNS=6` exchanges) is folded into the prompt as plain text rather than via the LLM's structured `messages` history, mainly because both the original chunk-citation behavior and ambiguity-handling rule needed to live in one place a reviewer could read top to bottom.

### Why `openrouter/free` instead of a pinned model
`openrouter/free` is OpenRouter's auto-router across whatever free models are currently available — which means the actual model behind a given call isn't fixed and can change run to run. This is a known instability (a grader hitting the API twice could get two different underlying models). It was kept as-is for this submission rather than pinned to a specific free model id, to avoid depending on a specific free-tier model's continued availability at evaluation time — pinning trades "consistent behavior now" for "guaranteed to break later when that specific free model gets deprecated or rate-limited out from under the submission." `EKA_LLM_MODEL` is exposed as an env var specifically so this can be pinned in one place without touching code, if a grading environment needs deterministic behavior more than free-tier resilience.

### Error handling & logging
`rag_core.py` defines a small exception hierarchy (`InvalidQuestionError`, `IndexNotFoundError`, `LLMConfigError`, `LLMGenerationError`, `EmptyDocumentError`) so each failure mode is distinguishable. `api.py` maps these to HTTP status codes (400 for bad input, 503 for no index yet, 502 for upstream LLM failure, 500 for config errors) instead of a blanket 500, and a global exception handler ensures an unhandled error returns a clean JSON body instead of a stack trace. `chat_pdf.py` catches the same hierarchy and shows a UI message rather than crashing the Streamlit session. Everything logs through a single configured logger (console + `eka.log`) with retrieval counts, latency, and confidence per query — useful for spotting silent retrieval failures (e.g. an index that loads but returns zero hits) that wouldn't otherwise surface as an "error."

### Scalability considerations
- FAISS here uses the default flat (exhaustive) index — fine up to roughly tens of thousands of chunks; beyond that, swap to `IndexIVFFlat` or `IndexHNSWFlat` for sub-linear search.
- BM25 is rebuilt in memory from a JSON corpus file on every query. That's a deliberate simplicity-over-throughput choice for a take-home-sized corpus (rebuild is sub-second up to a few thousand chunks); at real "hundreds of documents" scale this should move to a persisted BM25 structure or be replaced by a proper search engine (Elasticsearch/OpenSearch) so indexing cost is paid once, not per query.
- The embedding model and vector store are process-global singletons — fine for a single-process deployment, not for horizontal scaling without an external vector DB (Pinecone/Weaviate/managed FAISS service) that multiple API replicas can share.
- No request queueing/concurrency limiting on `/ask` — under load, concurrent requests each pay full embedding + BM25 rebuild + LLM round-trip cost. A production version would cache embeddings for repeated queries and put a queue or concurrency cap in front of the LLM call.

## Evaluation

See `eval/` — a self-contained harness with its own sample corpus (so grading isn't reliant on a particular runtime's already-built index):

```bash
python eval/build_sample_pdfs.py     # generates 2 sample PDFs with known page boundaries
python eval/index_sample_docs.py     # builds a fresh FAISS+BM25 index from them
python eval/run_eval.py              # runs 12 test cases against the live pipeline
```

**Test cases** (`eval/test_cases.json`, 12 total): factual lookups (single fact, single source — the two examples are the assignment brief's own canonical questions), a multi-document reasoning case (forces retrieval to pull from both sample PDFs), an ambiguously-worded question, and two deliberately unanswerable questions (one adjacent to real content, one unrelated) to test refusal rather than hallucination.

**Metrics computed per case and aggregated:**
- **Precision@k / Recall@k** — of the chunks retrieved, how many match the expected `(document, page)`; of the expected sources, how many were found
- **MRR** — rank of the first correct source (rewards getting it right *early*, not just eventually)
- **Keyword coverage** — fraction of expected key facts (e.g. `"24"`, `"30 days"`) present in the generated answer; a cheap proxy for answer correctness that doesn't require a second LLM-as-judge call (see limitations)
- **Hallucination resistance rate** — for the unanswerable cases, did the system correctly refuse instead of inventing an answer
- **Latency, self-reported confidence** — for spotting regressions, not absolute targets

Pass/fail thresholds (`recall ≥ 0.5` and `coverage ≥ 0.5` for answerable cases; correct refusal for unanswerable ones) are deliberately lenient given the free, auto-routed LLM — the harness is built to catch regressions across changes to the pipeline, not to certify a specific accuracy number against a non-deterministic backend.

**Improvements made during evaluation-driven iteration** (see git history / commit messages for the actual before/after):
- The original confidence score (keyword-overlap ratio) didn't correlate with retrieval quality at all — replaced with the cosine-similarity-based version above.
- The original MMR-equivalent tuning (`λ=0.6`) under-weighted diversity in testing — near-duplicate chunks from overlapping chunk windows kept winning over genuinely distinct, still-relevant chunks. Lowered to `λ=0.5`.
- Confirmed via unit-level tests (not just the eval harness) that RRF correctly promotes a chunk found by *both* dense and sparse retrieval above one found by only one — this is the behavior hybrid search is supposed to buy, and it's easy to get backwards with a careless score-combination formula.

## Known limitations

- **Model nondeterminism.** `openrouter/free` auto-routes to whatever free model is currently available; answer phrasing and occasionally answer quality will vary run to run, independent of anything in this codebase. Pin `EKA_LLM_MODEL` to a specific id for reproducible grading.
- **Keyword-coverage isn't correctness.** The eval harness checks whether expected phrases appear in the answer text, not whether the answer is semantically correct or doesn't contradict itself elsewhere. A proper answer-quality eval would use an LLM-as-judge pass (e.g. RAGAS's faithfulness/answer-relevance metrics) — not implemented here to avoid spending the same scarce free-tier LLM quota on grading its own output.
- **Confidence is a retrieval-quality proxy, not a calibrated probability.** "0.87 confidence" means "the retrieved chunks were strongly similar to the query," not "there's an 87% chance this answer is correct." It will be high even for a well-retrieved chunk that the LLM nonetheless summarizes incorrectly.
- **Scanned/image-only PDFs aren't handled.** Both extractors (`pdfplumber`, `PyPDF2`) require a text layer; no OCR fallback exists. The UI surfaces this as a clear error rather than silently returning nothing, but it's still a hard limitation, not a fallback.
- **BM25 corpus is rebuilt in memory per query** rather than persisted as a queryable structure — fine at this corpus scale, not at "hundreds of documents."
- **No authentication.** `/ask` is open; CORS defaults to `*`. Fine for a take-home, not for a real internal deployment.
- **No query rewriting.** A genuinely ambiguous or underspecified question (e.g. "what's the policy?" with no other context) is retrieved as-is; there's no reformulation step that uses conversation history to disambiguate the retrieval query itself (history is only used in the final generation prompt).
- **Single-process, single-machine.** No shared cache or vector DB across replicas; see [Scalability considerations](#scalability-considerations).

## Future improvements

Roughly in order of expected value for the effort:

1. **Query rewriting using conversation history** — fold prior turns into the *retrieval* query (not just the generation prompt) so a follow-up like "what about for managers?" retrieves correctly instead of relying on the raw follow-up text alone.
2. **LLM-as-judge answer evaluation** (RAGAS-style faithfulness/relevance scoring) to replace the keyword-coverage proxy with something closer to actual correctness.
3. **Cross-encoder reranking** as an option alongside MMR, switchable via config, for corpora/queries where pure embedding-similarity reranking under-performs.
4. **Content-aware chunking** — different chunk strategies for tabular/structured sections (already partially handled via `pdfplumber` table extraction) versus narrative prose versus enumerated clauses.
5. **Persisted, queryable sparse index** (replace the per-query in-memory BM25 rebuild) once corpus size moves past a few thousand chunks.
6. **User feedback collection** — a thumbs up/down on each answer, logged against the question/sources/confidence, to build a real eval set from production traffic instead of synthetic test cases.
7. **Auth + per-document access control** — relevant the moment this touches anything beyond a single team's non-sensitive docs.
8. **OCR fallback** for scanned PDFs (e.g. via `pytesseract`) so image-only documents don't silently fail ingestion.