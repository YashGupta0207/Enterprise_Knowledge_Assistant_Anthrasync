# Five-Minute Demo Script

## 1. Introduction

Show the Streamlit app and explain that it is an Enterprise Knowledge Assistant for internal PDFs.

## 2. Upload And Index

Upload one or more sample PDFs from the sidebar. Click the indexing button and point out that the app extracts text, chunks it, embeds it, and stores it in FAISS plus BM25.

## 3. Ask A Direct Question

Ask a factual question such as:

```text
What is the employee leave policy?
```

Show the answer, confidence, and source document/page.

## 4. Ask An Exact-Term Question

Ask a question with a number, acronym, or policy name. Explain that BM25 helps retrieve exact terms while FAISS handles semantic similarity.

## 5. Ask An Unsupported Question

Ask something not present in the documents. Show that the assistant refuses instead of hallucinating.

## 6. Architecture Walkthrough

Briefly explain:

- Streamlit UI calls `rag_core.answer_question()`.
- FastAPI exposes the same core through `/ask`.
- Retrieval is FAISS + BM25 + RRF + MMR.
- Prompting restricts the model to retrieved context.
- Sources and confidence are returned with every answer.

## 7. Limitations And Next Steps

Mention no OCR for scanned PDFs, no authentication, and that production scale would use a managed vector/search service.
