# app/services/rag_service.py
import logging

from app.config import TOP_K_RESULTS
from app.core.retriever import retrieve
from app.core.generator import generate_answer, RAGResponse

logger = logging.getLogger(__name__)


def answer_question(
    query: str,
    source_filter: str | None = None,
    top_k: int = TOP_K_RESULTS,
) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks then generate a cited answer.

    Args:
        query: The user's question in plain English.
        source_filter: Optional filename to restrict search to one document.
        top_k: Number of chunks to retrieve.

    Returns:
        Dict ready to be returned as an API response.
    """
    if not query.strip():
        raise ValueError("Question cannot be empty")

    # Step 1 — retrieve
    chunks = retrieve(query=query, top_k=top_k, source_filter=source_filter)

    if not chunks:
        logger.warning(f"No chunks retrieved for query: '{query[:60]}'")
        return {
            "answer": "No relevant content was found in the uploaded documents.",
            "citations": [],
            "confidence": "low",
            "chunks_retrieved": 0,
        }

    # Step 2 — generate
    response: RAGResponse = generate_answer(query=query, chunks=chunks)

    return {
        "answer": response.answer,
        "citations": [c.model_dump() for c in response.citations],
        "confidence": response.confidence,
        "chunks_retrieved": len(chunks),
    }