"""Central configuration for the Support Knowledge Copilot.

Every tunable value in this project lives here, and every value can be
overridden from `.env`. That matters more than it sounds: the whole point of
this project is producing an evaluation number that changes when you change a
setting. If chunk size is hardcoded in three files, you can never honestly say
"chunk size 800 beat chunk size 400" - you would not know what actually ran.

Usage anywhere in the codebase:

    from copilot.config import settings
    print(settings.final_top_k)
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives at <repo>/copilot/config.py, so parents[1] is the repo root.
# Deriving this from __file__ means the code works no matter which directory
# you run it from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
REPORTS_DIR = PROJECT_ROOT / "reports"


class Settings(BaseSettings):
    """Typed settings, loaded from environment variables and `.env`.

    pydantic-settings validates on startup: a bad value fails immediately with a
    clear message instead of surfacing as a confusing error deep inside a
    retrieval call an hour later.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unrelated variables already in your shell
    )

    # --- Secrets ----------------------------------------------------------
    openai_api_key: str = ""

    # --- Qdrant -----------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "support_chunks"

    # --- Models -----------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gpt-4o-mini"

    # bge-small outputs 384-dimensional vectors. Qdrant needs this at collection
    # creation time and will reject vectors of any other size, which is a useful
    # guardrail against accidentally swapping the embedding model mid-index.
    embedding_dim: int = 384

    # --- Chunking ---------------------------------------------------------
    chunk_strategy: Literal["heading", "fixed"] = "heading"
    chunk_size: int = 800       # characters, not tokens - simpler to reason about
    chunk_overlap: int = 150    # only used by the "fixed" strategy
    min_chunk_chars: int = 200  # below this a chunk has too little context to be useful
    max_chunk_chars: int = 1600  # hard ceiling; oversized heading sections get split

    # --- Retrieval --------------------------------------------------------
    dense_top_k: int = 20       # candidates from vector search
    sparse_top_k: int = 20      # candidates from BM25
    rrf_k: int = 60             # RRF smoothing constant; 60 is the paper default
    dense_weight: float = 1.0   # bump to trust semantic search more
    sparse_weight: float = 1.0  # bump to trust keyword search more
    rerank_top_n: int = 20      # how many fused candidates the reranker scores
    final_top_k: int = 5        # how many chunks actually reach the LLM

    # --- Generation -------------------------------------------------------
    llm_temperature: float = 0.0  # deterministic: required for meaningful evals
    max_answer_tokens: int = 800

    # --- Citation verification -------------------------------------------
    # Separate setting from llm_model on purpose. Using one model to write an
    # answer and then to grade its own citations is circular: it is predisposed
    # to agree with itself. Pointing this at a different model is the honest fix,
    # and having it as a knob means the limitation is visible rather than buried.
    judge_model: str = ""  # empty = use llm_model, and say so in the report

    # --- Confidence -------------------------------------------------------
    min_confidence: float = 0.35  # below this the assistant refuses to answer

    # Weights for the four confidence components. They should sum to 1.0;
    # `ConfidenceScorer` normalises them if they do not, so experimenting is safe.
    weight_retrieval: float = 0.25
    weight_citation_support: float = 0.40  # the strongest signal, so the largest weight
    weight_grounding: float = 0.20
    weight_completeness: float = 0.15

    # How much to subtract when every cited document is over two years old.
    staleness_penalty: float = 0.10


settings = Settings()
