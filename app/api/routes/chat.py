# app/api/routes/chat.py
import logging

from fastapi import APIRouter, HTTPException, status

from app.api.schemas import ChatRequest, ChatResponse, CitationResponse
from app.services.rag_service import answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "/ask",
    response_model=ChatResponse,
    summary="Ask a question about your uploaded documents",
)
async def ask_question(request: ChatRequest):
    """
    Submit a question and receive an answer with citations.
    Optionally filter to search within a specific document only.
    """
    try:
        result = answer_question(
            query=request.question,
            source_filter=request.source_filter,
            top_k=request.top_k,
        )

        citations = [
            CitationResponse(
                source=c["source"],
                page_number=c["page_number"],
                relevant_excerpt=c["relevant_excerpt"],
            )
            for c in result["citations"]
        ]

        return ChatResponse(
            answer=result["answer"],
            citations=citations,
            confidence=result["confidence"],
            chunks_retrieved=result["chunks_retrieved"],
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate an answer. Please try again.",
        )