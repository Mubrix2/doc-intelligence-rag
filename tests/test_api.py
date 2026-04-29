# tests/test_api.py
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import create_app

client = TestClient(create_app())


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch("app.api.routes.documents.ingest_document")
def test_upload_pdf_success(mock_ingest):
    mock_ingest.return_value = {
        "filename": "sample.pdf",
        "chunks_stored": 12,
        "status": "success",
    }
    with open("tests/fixtures/sample.pdf", "rb") as f:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("sample.pdf", f, "application/pdf")},
        )
    assert response.status_code == 201
    assert response.json()["chunks_stored"] == 12


def test_upload_non_pdf_rejected():
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", b"some text", "text/plain")},
    )
    assert response.status_code == 415


@patch("app.api.routes.documents.get_all_documents")
def test_list_documents(mock_list):
    mock_list.return_value = ["doc1.pdf", "doc2.pdf"]
    response = client.get("/api/v1/documents/")
    assert response.status_code == 200
    assert response.json()["total"] == 2


@patch("app.api.routes.chat.answer_question")
def test_ask_question_success(mock_answer):
    mock_answer.return_value = {
        "answer": "The report found revenue grew by 20%.",
        "citations": [
            {
                "source": "report.pdf",
                "page_number": 3,
                "relevant_excerpt": "Revenue grew by 20% in Q3."
            }
        ],
        "confidence": "high",
        "chunks_retrieved": 5,
    }
    response = client.post(
        "/api/v1/chat/ask",
        json={"question": "What did the report find about revenue?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["confidence"] == "high"
    assert len(data["citations"]) == 1


def test_ask_question_too_short():
    response = client.post(
        "/api/v1/chat/ask",
        json={"question": "Hi"},
    )
    assert response.status_code == 422  # Pydantic min_length validation