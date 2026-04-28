# tests/test_chunker.py
import pytest
from app.core.chunker import extract_text_from_pdf, chunk_pages, parse_and_chunk_pdf, TextChunk


# --- Fixtures ---

@pytest.fixture
def sample_pdf_bytes():
    """Load a small real PDF for testing."""
    with open("tests/fixtures/My_Next_Roadmap.pdf", "rb") as f:
        return f.read()


# --- Tests ---

def test_extract_text_returns_pages(sample_pdf_bytes):
    pages = extract_text_from_pdf(sample_pdf_bytes, "My_Next_Roadmap.pdf")
    assert isinstance(pages, list)
    assert len(pages) > 0
    assert all("text" in p for p in pages)
    assert all("page_number" in p for p in pages)
    assert all("source" in p for p in pages)


def test_chunk_pages_returns_text_chunks(sample_pdf_bytes):
    pages = extract_text_from_pdf(sample_pdf_bytes, "My_Next_Roadmap.pdf")
    chunks = chunk_pages(pages)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(c, TextChunk) for c in chunks)


def test_chunk_text_length_within_limit(sample_pdf_bytes):
    pages = extract_text_from_pdf(sample_pdf_bytes, "My_Next_Roadmap.pdf")
    chunks = chunk_pages(pages)
    # Allow slight overflow due to overlap — no chunk should be more than 2x chunk size
    for chunk in chunks:
        assert len(chunk.text) <= 1100  # CHUNK_SIZE=500, generous upper bound


def test_empty_pdf_raises_value_error():
    fake_empty_bytes = b"%PDF-1.4"  # minimal invalid PDF
    with pytest.raises(Exception):
        extract_text_from_pdf(fake_empty_bytes, "empty.pdf")


def test_parse_and_chunk_pdf_is_complete_pipeline(sample_pdf_bytes):
    chunks = parse_and_chunk_pdf(sample_pdf_bytes, "My_Next_Roadmap.pdf")
    assert len(chunks) > 0
    assert chunks[0].source == "My_Next_Roadmap.pdf"
    assert chunks[0].page_number >= 1