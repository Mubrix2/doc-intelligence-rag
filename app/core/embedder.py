# app/core/embedder.py
import logging
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL
from app.core.chunker import TextChunk

logger = logging.getLogger(__name__)

# Lazy singleton — model loads once, reused for every request
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' — first call only")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded and cached")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of strings into a list of embedding vectors.
    normalize_embeddings=True ensures vectors have unit length (required for cosine similarity).
    """
    model = get_model()
    vectors = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=32,
    )
    return vectors.tolist()


def embed_chunks(chunks: list[TextChunk]) -> list[tuple[TextChunk, list[float]]]:
    """
    Embed a list of TextChunks.
    Returns pairs of (chunk, vector) so metadata stays attached to its vector.
    """
    texts = [chunk.text for chunk in chunks]
    vectors = embed_texts(texts)
    paired = list(zip(chunks, vectors))
    logger.info(f"Embedded {len(paired)} chunks")
    return paired
