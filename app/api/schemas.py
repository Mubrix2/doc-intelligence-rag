# app/api/schemas.py
from pydantic import BaseModel, Field


# --- Document endpoints ---

class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_stored: int
    status: str
    message: str


class DocumentListResponse(BaseModel):
    documents: list[str]
    total: int


class DocumentDeleteResponse(BaseModel):
    filename: str
    status: str


# --- Chat endpoint ---

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        description="The question to ask about your documents",
        examples=["What are the key findings in the report?"]
    )
    source_filter: str | None = Field(
        default=None,
        description="Optional: restrict search to a specific document filename"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (1–20)"
    )


class CitationResponse(BaseModel):
    source: str
    page_number: int
    relevant_excerpt: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    confidence: str
    chunks_retrieved: int