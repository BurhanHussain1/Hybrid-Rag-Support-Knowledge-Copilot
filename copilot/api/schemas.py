"""Request and response shapes for the API.

Responses reuse the models the pipeline already produces - `CopilotAnswer`,
`RetrievalResult` - rather than defining parallel API-only copies. Two definitions
of the same thing drift apart, and then the dashboard renders a field the API
stopped sending.

Requests do get their own models, because an HTTP request is genuinely a
different thing from an internal function call: it comes from outside, so it
needs validation and sensible bounds. `top_k` capped at 20 is not arbitrary -
without a ceiling, one request asking for 10,000 chunks would hand the reranker
10,000 cross-encoder pairs and stall the service.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["dense", "sparse", "hybrid", "rerank"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Mode = Field(default="rerank", description="Retrieval strategy")
    top_k: int = Field(default=5, ge=1, le=20, description="Chunks passed to the model")
    verify: bool = Field(default=True, description="Run citation verification")

    # Metadata filters, matching the fields stored in the Qdrant payload.
    source_name: str | None = None
    doc_type: str | None = None
    access_level: str | None = None

    def filters(self) -> dict[str, str] | None:
        active = {
            "source_name": self.source_name,
            "doc_type": self.doc_type,
            "access_level": self.access_level,
        }
        active = {k: v for k, v in active.items() if v}
        return active or None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "why is my pod stuck in pending", "mode": "rerank", "top_k": 5},
                {"question": "what is the parental leave policy", "access_level": "internal"},
            ]
        }
    }


class SearchRequest(BaseModel):
    """Retrieval only, no generation.

    Exists so the dashboard can show the dense-vs-hybrid comparison without
    paying for an LLM call on every toggle - and so you can debug retrieval
    without generation noise in the way.
    """

    query: str = Field(min_length=1, max_length=1000)
    mode: Mode = "rerank"
    top_k: int = Field(default=10, ge=1, le=50)
    source_name: str | None = None
    doc_type: str | None = None
    access_level: str | None = None

    def filters(self) -> dict[str, str] | None:
        active = {
            "source_name": self.source_name,
            "doc_type": self.doc_type,
            "access_level": self.access_level,
        }
        active = {k: v for k, v in active.items() if v}
        return active or None


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    score: float
    text: str
    breadcrumb: str
    source_name: str
    doc_type: str
    url: str | None = None
    age_days: int | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    found_by: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    mode: str
    hits: list[SearchHit]
    timings_ms: dict[str, float]
    coverage: dict[str, int] = Field(
        default_factory=dict, description="How many hits came from each retriever"
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    qdrant_reachable: bool
    collection_exists: bool
    indexed_chunks: int
    bm25_loaded: bool
    models_warm: bool
    llm_configured: bool
    detail: list[str] = Field(default_factory=list)
