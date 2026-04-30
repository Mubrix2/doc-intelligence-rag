# app/core/embedder.py
import logging
from fastembed import TextEmbedding
from app.config import EMBEDDING_MODEL
from app.core.chunker import TextChunk

logger = logging.getLogger(__name__)

_model: TextEmbedding | None = None


def get_model() -> TextEmbedding:
    """Load the FastEmbed model once and cache it."""
    global _model
    if _model is None:
        logger.info(f"Loading FastEmbed model '{EMBEDDING_MODEL}' — first call only")
        _model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("FastEmbed model loaded and cached")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of strings into embedding vectors.
    FastEmbed returns a generator — convert to list immediately.
    Vectors are already normalized for cosine similarity.
    """
    model = get_model()
    embeddings = list(model.embed(texts))
    return [embedding.tolist() for embedding in embeddings]


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