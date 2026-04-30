# app/core/generator.py
import logging
from pydantic import BaseModel, Field
import instructor
from groq import Groq

from app.config import GROQ_API_KEY, LLM_MODEL
from app.core.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


# --- Structured output schema ---

class Citation(BaseModel):
    """A single source used in generating the answer."""
    source: str = Field(description="The filename of the source document")
    page_number: int = Field(description="The page number within the document")
    relevant_excerpt: str = Field(
        description="A short direct excerpt from the source that supports the answer"
    )


class RAGResponse(BaseModel):
    """The complete structured response from the RAG system."""
    answer: str = Field(
        description="A clear, complete answer to the user's question based only on the provided context"
    )
    citations: list[Citation] = Field(
        description="List of sources used. Only include sources that directly contributed to the answer."
    )
    confidence: str = Field(
        description="One of: 'high', 'medium', or 'low'. "
                    "High means the answer is fully supported by the context. "
                    "Low means the context was insufficient."
    )


# --- Groq client (patched with instructor) ---

# Lazy singleton — client created once
_client = None


def get_client():
    global _client
    if _client is None:
        raw_client = Groq(api_key=GROQ_API_KEY)
        _client = instructor.from_groq(raw_client, mode=instructor.Mode.JSON)
        logger.info("Groq client initialised with instructor")
    return _client


# --- Prompt builder ---

def _build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """
    Construct the context block from retrieved chunks.
    Each chunk is labelled with its source and page so the LLM can cite correctly.
    """
    if not chunks:
        return "No relevant context was found in the uploaded documents."

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        block = (
            f"[Source {i}]\n"
            f"File: {chunk.source}\n"
            f"Page: {chunk.page_number}\n"
            f"Content: {chunk.text}"
        )
        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)


SYSTEM_PROMPT = """You are a precise document analyst. Your job is to answer questions
using ONLY the context provided to you below. 

Rules you must follow without exception:
1. If the answer is present in the context, answer clearly and completely.
2. If the answer is NOT in the context, say exactly: "The provided documents do not contain 
   enough information to answer this question." Do not guess or use outside knowledge.
3. Always cite the specific sources you used by referencing the file name and page number.
4. Never fabricate citations. Only cite sources that you directly used in your answer.
5. Keep your answer concise but complete — do not pad with unnecessary text."""


# --- Main generation function ---

def generate_answer(query: str, chunks: list[RetrievedChunk]) -> RAGResponse:
    """
    Generate a structured answer with citations from retrieved chunks.

    Args:
        query: The user's original question.
        chunks: Retrieved chunks from the vector search.

    Returns:
        RAGResponse containing answer, citations, and confidence level.
    """
    client = get_client()
    context = _build_prompt(query, chunks)

    user_message = f"""Context from uploaded documents:

{context}

---

Question: {query}

Answer the question using only the context above. Cite your sources."""

    logger.info(f"Sending query to Groq: '{query[:60]}...' with {len(chunks)} context chunks")

    try:
        response: RAGResponse = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_model=RAGResponse,
            max_tokens=1000,
            temperature=0.1,  # low temperature = factual, less creative
        )
        logger.info(f"Answer generated. Confidence: {response.confidence}. Citations: {len(response.citations)}")
        return response

    except Exception as e:
        logger.error(f"Groq generation failed: {e}")
        raise