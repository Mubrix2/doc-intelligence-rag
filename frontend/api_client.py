# frontend/api_client.py
import logging
import requests
from config import API_BASE_URL

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 60  # generous — LLM calls can take time


def upload_document(file_bytes: bytes, filename: str) -> dict:
    """Upload a PDF to the backend for ingestion."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/documents/upload",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. The document may be large — please try again."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to the backend API. Is it running?"}
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e))
        return {"success": False, "error": detail}


def list_documents() -> dict:
    """Fetch all uploaded document names."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/documents/",
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to the backend API."}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": str(e)}


def delete_document(filename: str) -> dict:
    """Delete a document from the vector store."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/v1/documents/{filename}",
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": str(e)}


def ask_question(
    question: str,
    source_filter: str | None = None,
    top_k: int = 5,
) -> dict:
    """Send a question and receive a structured answer with citations."""
    try:
        payload = {
            "question": question,
            "top_k": top_k,
        }
        if source_filter and source_filter != "All documents":
            payload["source_filter"] = source_filter

        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat/ask",
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "The request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to the backend API."}
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e))
        return {"success": False, "error": detail}


def check_health() -> bool:
    """Return True if the backend API is reachable."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )
        return response.status_code == 200
    except Exception:
        return False