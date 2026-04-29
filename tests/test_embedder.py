# tests/test_embedder.py
from app.core.embedder import embed_texts, embed_chunks
from app.core.chunker import TextChunk


def test_embed_texts_returns_vectors():
    texts = ["This is a test sentence.", "Another sentence here."]
    vectors = embed_texts(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384  # all-MiniLM-L6-v2 dimension


def test_embed_texts_normalized():
    """Normalized vectors should have magnitude close to 1.0."""
    import math
    vectors = embed_texts(["test sentence"])
    magnitude = math.sqrt(sum(v ** 2 for v in vectors[0]))
    assert abs(magnitude - 1.0) < 0.001


def test_embed_chunks_preserves_metadata():
    chunks = [
        TextChunk(text="hello world", source="test.pdf", page_number=1, chunk_index=0),
        TextChunk(text="another chunk", source="test.pdf", page_number=2, chunk_index=1),
    ]
    pairs = embed_chunks(chunks)
    assert len(pairs) == 2
    assert pairs[0][0].source == "test.pdf"
    assert len(pairs[0][1]) == 384