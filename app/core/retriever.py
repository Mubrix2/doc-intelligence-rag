# app/core/retriever.py
import logging
from dataclasses import dataclass

from app.config import TOP_K_RESULTS
from app.core.embedder import embed_texts
from app.db.vector_store import search_vectors

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned from vector search, with its relevance score."""
    text: str
    source: str
    page_number: int
    score: float


def retrieve(
    query: str,
    top_k: int = TOP_K_RESULTS,
    source_filter: str | None = None,
) -> list[RetrievedChunk]:
    """
    Embed a query and retrieve the most relevant chunks from Qdrant.

    Args:
        query: The user's question in plain English.
        top_k: How many chunks to return.
        source_filter: If provided, only search within this document.

    Returns:
        List of RetrievedChunk ordered by relevance (highest score first).
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")

    # Step 1 — embed the query using the same model used during ingestion
    query_vector = embed_texts([query])[0]

    # Step 2 — search Qdrant for nearest neighbours
    raw_results = search_vectors(
        query_vector=query_vector,
        top_k=top_k,
        source_filter=source_filter,
    )

    # Step 3 — map raw Qdrant results to clean RetrievedChunk objects
    chunks = [
        RetrievedChunk(
            text=result.payload["text"],
            source=result.payload["source"],
            page_number=result.payload["page_number"],
            score=round(result.score, 4),
        )
        for result in raw_results
    ]

    logger.info(
        f"Retrieved {len(chunks)} chunks for query '{query[:60]}...' "
        f"(filter: {source_filter or 'none'})"
    )
    return chunks