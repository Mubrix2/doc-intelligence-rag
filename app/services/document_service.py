# app/services/document_service.py
import logging

from app.core.chunker import parse_and_chunk_pdf
from app.core.embedder import embed_chunks
from app.db.vector_store import (
    delete_document,
    ensure_collection_exists,
    list_documents,
    upsert_chunks,
)

logger = logging.getLogger(__name__)


def ingest_document(file_bytes: bytes, filename: str) -> dict:
    """
    Full ingestion pipeline for a single PDF:
    parse → chunk → embed → store in Qdrant.

    If the document already exists, it is deleted and re-ingested.
    Returns a summary dict for the API response.
    """
    ensure_collection_exists()

    # If document already exists, remove old vectors first
    existing = list_documents()
    if filename in existing:
        logger.info(f"'{filename}' already exists — deleting old vectors before re-ingestion")
        delete_document(filename)

    # Parse and chunk
    chunks = parse_and_chunk_pdf(file_bytes, filename)

    # Embed chunks (returns list of (TextChunk, vector) pairs)
    chunk_vector_pairs = embed_chunks(chunks)

    # Store in Qdrant
    upsert_chunks(chunk_vector_pairs, source=filename)

    summary = {
        "filename": filename,
        "chunks_stored": len(chunks),
        "status": "success",
    }
    logger.info(f"Ingestion complete: {summary}")
    return summary


def get_all_documents() -> list[str]:
    """Return all document names currently stored."""
    ensure_collection_exists()
    return list_documents()


def remove_document(filename: str) -> dict:
    """Delete all vectors for a specific document."""
    delete_document(filename)
    return {"filename": filename, "status": "deleted"}