# app/main.py
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents
from app.config import APP_ENV
from app.db.vector_store import ensure_collection_exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting Document Intelligence RAG API | env={APP_ENV}")
    ensure_collection_exists()
    logger.info("Qdrant collection verified. API ready.")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Document Intelligence RAG API",
        description=(
            "Upload PDF documents and ask questions. "
            "Answers are generated with citations pointing to the exact source page."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")

    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "env": APP_ENV}

    return app


app = create_app()