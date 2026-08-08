"""Support Knowledge Copilot with verified citations.

Package layout:
    config       central settings, loaded from .env
    ingest       loaders, metadata extraction, chunking
    retrieval    dense, sparse, RRF fusion, reranking
    generation   grounded answers, citation verification, confidence
    evaluation   golden set, metrics, report generation
"""

__version__ = "0.1.0"
