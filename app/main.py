# app/main.py
import logging
import logging.config

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents
from app.config import APP_ENV
from app.db.vector_store import ensure_collection_exists

# --- Logging setup (once, at startup) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- App factory ---
def create_app() -> FastAPI:
    app = FastAPI(
        title="Document Intelligence RAG API",
        description=(
            "Upload PDF documents and ask questions. "
            "Answers are generated with citations pointing to the exact source page."
        ),
        version="1.0.0",
        docs_url="/docs",       # Swagger UI at /docs
        redoc_url="/redoc",     # ReDoc UI at /redoc
    )

    # CORS — allow Streamlit frontend to talk to this API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten this to your Streamlit URL in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")

    # Startup event
    @app.on_event("startup")
    async def startup():
        logger.info(f"Starting Document Intelligence RAG API | env={APP_ENV}")
        ensure_collection_exists()
        logger.info("Qdrant collection verified. API ready.")

    # Health check
    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "env": APP_ENV}

    return app


app = create_app()