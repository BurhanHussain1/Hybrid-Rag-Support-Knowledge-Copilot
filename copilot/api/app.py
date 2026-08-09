"""The FastAPI application.

Two things here are worth more attention than the routing, because they are the
difference between a service that works and one that only works in a demo.

**1. Models are loaded at startup, not on the first request.**

Running `ask.py` costs ~34 seconds before it answers anything: loading bge-small,
unpickling a 43 MB BM25 index, loading the cross-encoder. In a CLI you pay that
once per invocation and shrug. In a service, the first user after every deploy
would wait 34 seconds and probably time out.

So the lifespan handler loads everything and runs one throwaway query before the
server accepts traffic. Startup gets slower; every request gets fast. That is the
right trade for anything long-lived.

**2. Endpoints are `def`, not `async def`.**

This looks like a mistake in a FastAPI app and is not. `async def` handlers run
directly on the event loop, and everything we do - torch inference, BM25 scoring,
a blocking HTTP call to OpenAI - is synchronous CPU or blocking IO. Putting that
on the event loop would freeze the entire server for the duration of every
request, including the health check.

A plain `def` handler is dispatched to a worker threadpool instead, so one slow
request cannot block the others. The rule: `async def` only if the body is
genuinely awaitable all the way down.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from copilot import __version__
from copilot.api.schemas import (
    AskRequest,
    HealthResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from copilot.config import settings
from copilot.generation.llm import MissingAPIKey
from copilot.generation.pipeline import Copilot, CopilotAnswer
from copilot.retrieval.bm25_index import BM25Index
from copilot.retrieval.vector_store import VectorStore

# Process-wide state, populated during startup. A module-level dict rather than
# globals scattered around, so the health endpoint can report exactly what is
# loaded without guessing.
state: dict = {"copilot": None, "warm": False, "startup_seconds": 0.0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models before serving, release nothing on shutdown (the OS will)."""
    started = time.perf_counter()
    print("starting up: loading models and indexes...")

    # Wait for Qdrant before touching it.
    #
    # Needed for docker-compose: `depends_on` only guarantees the qdrant container
    # has been *started*, not that it is accepting connections. Without this the
    # API's warmup races Qdrant's boot, loses roughly half the time, and comes up
    # reporting degraded until someone restarts it.
    store = VectorStore()
    for attempt in range(30):
        if store.ping():
            break
        if attempt == 0:
            print(f"  waiting for Qdrant at {settings.qdrant_url}...")
        time.sleep(2)
    else:
        print(f"  Qdrant never became reachable at {settings.qdrant_url}")

    copilot = Copilot()

    try:
        # A real query, not just a model load. This forces the embedder, the BM25
        # unpickle and the cross-encoder all to initialise, and it surfaces a
        # broken index at boot instead of on a user's first question.
        copilot.retriever.retrieve("startup warmup query", mode="rerank", top_k=1)
        state["warm"] = True
    except Exception as exc:  # noqa: BLE001
        # Serve anyway: /health will report degraded, which is more useful than a
        # container that crash-loops and tells nobody why.
        print(f"  warmup failed: {type(exc).__name__}: {exc}")
        state["warm"] = False

    state["copilot"] = copilot
    state["startup_seconds"] = time.perf_counter() - started
    print(f"ready in {state['startup_seconds']:.1f}s")

    yield


app = FastAPI(
    title="Support Knowledge Copilot",
    version=__version__,
    lifespan=lifespan,
    description=(
        "Answers support questions from internal documentation using hybrid retrieval, "
        "then verifies that every citation actually supports the claim attached to it.\n\n"
        "Interactive docs at `/docs`."
    ),
)

# Wide open because this runs locally and the Streamlit dashboard in Step 7 calls
# it from a different port. A deployed version would list real origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _copilot() -> Copilot:
    instance = state.get("copilot")
    if instance is None:
        raise HTTPException(status_code=503, detail="service still starting up")
    return instance


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Is the service actually able to answer questions?

    Deliberately more than a 200 OK. A health check that passes while the vector
    store is empty is worse than none: it tells your orchestrator everything is
    fine while every answer is a refusal.
    """
    detail: list[str] = []

    store = VectorStore()
    reachable = store.ping()
    exists = reachable and store.exists()
    count = store.count() if exists else 0

    if not reachable:
        detail.append(f"Qdrant unreachable at {settings.qdrant_url} - run: docker compose up -d")
    elif not exists:
        detail.append(f"collection '{settings.qdrant_collection}' missing - run: python index.py --rebuild")
    elif count == 0:
        detail.append("collection exists but is empty")

    bm25_path = BM25Index.path(settings.chunk_strategy)
    bm25_loaded = bm25_path.exists()
    if not bm25_loaded:
        detail.append(f"{bm25_path.name} missing - run: python index.py --bm25-only")

    llm_configured = bool(settings.openai_api_key) and not settings.openai_api_key.startswith("sk-replace")
    if not llm_configured:
        detail.append("OPENAI_API_KEY not set - retrieval works, generation will fail")

    healthy = reachable and exists and count > 0 and bm25_loaded and llm_configured

    return HealthResponse(
        status="ok" if healthy else "degraded",
        qdrant_reachable=reachable,
        collection_exists=exists,
        indexed_chunks=count,
        bm25_loaded=bm25_loaded,
        models_warm=bool(state.get("warm")),
        llm_configured=llm_configured,
        detail=detail,
    )


@app.post("/ask", response_model=CopilotAnswer, tags=["copilot"])
def ask(request: AskRequest) -> CopilotAnswer:
    """Answer a question with verified citations and a confidence breakdown.

    Note that a refusal is a normal 200 response with `answered: false`, not an
    error. The client asked a valid question and got an honest answer; nothing
    went wrong. Reserving non-2xx for actual faults keeps monitoring meaningful.
    """
    try:
        return _copilot().ask(
            request.question,
            mode=request.mode,
            filters=request.filters(),
            top_k=request.top_k,
            verify=request.verify,
        )
    except MissingAPIKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/search", response_model=SearchResponse, tags=["copilot"])
def search(request: SearchRequest) -> SearchResponse:
    """Retrieval only, no LLM call. Used by the dashboard's comparison toggle."""
    result = _copilot().retriever.retrieve(
        request.query, mode=request.mode, filters=request.filters(), top_k=request.top_k
    )

    return SearchResponse(
        query=request.query,
        mode=request.mode,
        hits=[
            SearchHit(
                rank=hit.rank,
                chunk_id=hit.chunk_id,
                score=hit.score,
                text=hit.text,
                breadcrumb=hit.breadcrumb,
                source_name=hit.source_name,
                doc_type=hit.doc_type,
                url=hit.url,
                age_days=hit.age_days,
                dense_rank=hit.dense_rank,
                sparse_rank=hit.sparse_rank,
                rerank_score=hit.rerank_score,
                found_by=hit.found_by(),
            )
            for hit in result.chunks
        ],
        timings_ms={k: round(v, 1) for k, v in result.timings_ms.items()},
        coverage=result.coverage(),
    )


@app.get("/stats", tags=["ops"])
def stats() -> dict:
    """What is indexed, and with what settings."""
    store = VectorStore()
    return {
        "version": __version__,
        "startup_seconds": round(state.get("startup_seconds", 0.0), 1),
        "models_warm": bool(state.get("warm")),
        "collection": settings.qdrant_collection,
        "indexed_chunks": store.count() if store.ping() and store.exists() else 0,
        "chunk_strategy": settings.chunk_strategy,
        "models": {
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
            "llm": settings.llm_model,
            "judge": settings.judge_model or f"{settings.llm_model} (same as generator)",
        },
        "retrieval": {
            "dense_top_k": settings.dense_top_k,
            "sparse_top_k": settings.sparse_top_k,
            "rrf_k": settings.rrf_k,
            "dense_weight": settings.dense_weight,
            "sparse_weight": settings.sparse_weight,
            "rerank_top_n": settings.rerank_top_n,
            "final_top_k": settings.final_top_k,
        },
        "min_confidence": settings.min_confidence,
    }
