# app/api/routes/documents.py
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.schemas import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.services.document_service import (
    get_all_documents,
    ingest_document,
    remove_document,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_CONTENT_TYPES = {"application/pdf"}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF document",
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF file. The system will parse, chunk, embed, and store it
    in the vector database. If the document already exists it will be replaced.
    """
    # Validate file type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF files are accepted. Received: {file.content_type}",
        )

    # Read file into memory
    file_bytes = await file.read()

    # Validate file size
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB",
        )

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    try:
        result = ingest_document(file_bytes=file_bytes, filename=file.filename)
        return DocumentUploadResponse(
            filename=result["filename"],
            chunks_stored=result["chunks_stored"],
            status=result["status"],
            message=f"'{file.filename}' successfully ingested with {result['chunks_stored']} chunks",
        )
    except ValueError as e:
        # Raised by chunker when PDF has no readable text (e.g. scanned image PDF)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Ingestion failed for '{file.filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion failed. Please try again.",
        )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
)
async def list_documents():
    """Return all document filenames currently stored in the vector database."""
    try:
        documents = get_all_documents()
        return DocumentListResponse(documents=documents, total=len(documents))
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document list",
        )


@router.delete(
    "/{filename}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document from the vector database",
)
async def delete_document(filename: str):
    """Remove all vectors associated with a specific document."""
    try:
        result = remove_document(filename)
        return DocumentDeleteResponse(
            filename=result["filename"],
            status=result["status"],
        )
    except Exception as e:
        logger.error(f"Failed to delete '{filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document '{filename}'",
        )