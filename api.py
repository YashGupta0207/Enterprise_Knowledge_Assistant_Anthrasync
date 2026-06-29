"""
api.py  –  FastAPI REST endpoint for the Enterprise Knowledge Assistant.

Run with:  uvicorn api:app --reload

All retrieval/generation logic lives in rag_core.py; this file is
responsible for HTTP concerns only: request/response schemas, status
codes, and translating internal errors into appropriate HTTP responses.
"""

import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import rag_core
from rag_core import (
    EmptyDocumentError,
    IndexNotFoundError,
    InvalidQuestionError,
    LLMConfigError,
    LLMGenerationError,
    RAGError,
)

rag_core.configure_logging()
log = logging.getLogger("eka.api")

# ─────────────────────────── APP ────────────────────────────── #

app = FastAPI(
    title="Enterprise Knowledge Assistant API",
    description="RAG-based Q&A over internal company documents.",
    version="2.0.0",
)

# CORS is wide open ("*") for local development / take-home evaluation
# convenience. In a real deployment, restrict allow_origins to the actual
# frontend domain(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("EKA_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────── SCHEMAS ────────────────────────── #

class HistoryMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=rag_core.MAX_QUESTION_LENGTH)
    history: list[HistoryMessage] | None = None


class SourceRef(BaseModel):
    document: str
    page: int | str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    confidence: float
    retrieved_chunks: int = 0
    latency_ms: int = 0


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    indexed_documents: int


class ErrorResponse(BaseModel):
    error: str
    detail: str


# ─────────────────────── GLOBAL ERROR HANDLING ──────────────── #

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all so unexpected errors never leak a raw 500 traceback to clients."""
    log.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "An unexpected error occurred."},
    )


# ─────────────────────────── ROUTES ─────────────────────────── #

@app.get("/health", response_model=HealthResponse)
def health():
    indexed = rag_core.load_indexed_files()
    return HealthResponse(
        status="ok",
        index_ready=bool(indexed),
        indexed_documents=len(indexed),
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
def ask(body: AskRequest):
    history = [m.model_dump() for m in body.history] if body.history else []

    try:
        result = rag_core.answer_question(body.question, history)
    except InvalidQuestionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IndexNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except LLMConfigError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except LLMGenerationError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except EmptyDocumentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RAGError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AskResponse(
        answer=result.answer,
        sources=[SourceRef(**s) for s in result.sources],
        confidence=result.confidence,
        retrieved_chunks=result.retrieved_chunks,
        latency_ms=result.latency_ms,
    )
