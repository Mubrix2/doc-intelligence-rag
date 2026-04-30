# app/db/vector_store.py
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import COLLECTION_NAME, QDRANT_API_KEY, QDRANT_URL

logger = logging.getLogger(__name__)

VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimension

# Lazy singleton — one client, reused for every request
_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Create and cache the Qdrant client connection."""
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        logger.info("Qdrant client connected")
    return _client


def ensure_collection_exists() -> None:
    """
    Create the Qdrant collection if it does not already exist.
    Also ensures a payload index exists on 'source' for filtered search.
    Safe to call on every app startup.
    """
    from qdrant_client.models import PayloadSchemaType

    client = get_client()
    existing_names = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in existing_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection '{COLLECTION_NAME}'")
    else:
        logger.info(f"Qdrant collection '{COLLECTION_NAME}' already exists — skipping creation")

    # Create payload index on 'source' field so filtered search works.
    # This is idempotent — safe to call even if the index already exists.
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="source",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    logger.info("Payload index on 'source' verified")


def upsert_chunks(chunk_vector_pairs: list[tuple], source: str) -> None:
    """
    Store a list of (TextChunk, vector) pairs in Qdrant.
    Each point gets a UUID and carries the chunk text + metadata as payload.
    """
    client = get_client()

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk.text,
                "source": chunk.source,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            },
        )
        for chunk, vector in chunk_vector_pairs
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info(f"Upserted {len(points)} vectors for document '{source}'")


def delete_document(source: str) -> None:
    """
    Remove all vectors belonging to a specific document.
    Called when a user re-uploads or deletes a document.
    """
    client = get_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        ),
    )
    logger.info(f"Deleted all vectors for document '{source}'")


def list_documents() -> list[str]:
    """
    Return a list of unique document names stored in the collection.
    Uses scroll to retrieve all payloads without a search query.
    """
    client = get_client()
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        limit=1000,
    )
    sources = list({point.payload["source"] for point in results if point.payload})
    return sorted(sources)


def search_vectors(
    query_vector: list[float],
    top_k: int,
    source_filter: str | None = None,
) -> list:
    client = get_client()

    filter_ = None
    if source_filter:
        filter_ = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=filter_,
        with_payload=True,
    )
    return results.points