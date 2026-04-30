# app/core/chunker.py
import io
import logging
from dataclasses import dataclass, field

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A single chunk of text with its origin metadata."""
    text: str
    source: str          # original filename
    page_number: int     # 1-based page number
    chunk_index: int     # position of this chunk within the document


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Extract text from a PDF file provided as raw bytes.
    Returns a list of dicts, one per page: {text, page_number, source}
    """
    pages = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        logger.info(f"Opened '{filename}' — {len(pdf.pages)} pages found")

        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if not text or not text.strip():
                logger.warning(f"Page {i + 1} of '{filename}' returned no text — skipping")
                continue

            pages.append({
                "text": text.strip(),
                "page_number": i + 1,
                "source": filename,
            })

    if not pages:
        raise ValueError(f"No readable text found in '{filename}'. The PDF may be scanned or image-based.")

    logger.info(f"Extracted text from {len(pages)} pages in '{filename}'")
    return pages


def chunk_pages(pages: list[dict]) -> list[TextChunk]:
    """
    Split page texts into overlapping chunks using RecursiveCharacterTextSplitter.
    Each chunk carries forward the page number and source filename.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[TextChunk] = []
    chunk_index = 0

    for page in pages:
        raw_chunks = splitter.split_text(page["text"])

        for raw_chunk in raw_chunks:
            if not raw_chunk.strip():
                continue

            all_chunks.append(TextChunk(
                text=raw_chunk.strip(),
                source=page["source"],
                page_number=page["page_number"],
                chunk_index=chunk_index,
            ))
            chunk_index += 1

    logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks


def parse_and_chunk_pdf(file_bytes: bytes, filename: str) -> list[TextChunk]:
    """
    Public entry point. Combines extraction and chunking in one call.
    This is what the rest of the app will import and use.
    """
    pages = extract_text_from_pdf(file_bytes, filename)
    chunks = chunk_pages(pages)
    return chunks