# doc-intelligence-rag

# Document Intelligence RAG System

A production-ready AI system that lets you upload PDF documents and ask questions in plain English. Every answer includes citations pointing to the exact page and document the information came from.

Built to demonstrate a complete, end-to-end AI engineering pipeline — from document ingestion to vector search to LLM generation — deployed on cloud infrastructure.

**Live Demo:** [your-app.streamlit.app](https://doc-intelligence-rag-bcbbpwznmtmgzkivxryjrg.streamlit.app)  
**API Docs:** [your-api.onrender.com/docs](https://doc-intelligence-rag-1.onrender.com/docs)

---

## What It Does

1. Upload one or more PDF documents through the web interface
2. The system parses, chunks, embeds, and stores them in a vector database
3. Ask any question in plain English
4. The system retrieves the most relevant sections and generates a precise answer
5. Every answer includes citations: document name, page number, and the exact excerpt used

---

## System Architecture

PDF Upload
│
▼
[FastAPI Backend]
│
├── pdfplumber → extracts text page by page
│
├── RecursiveCharacterTextSplitter → chunks text with overlap
│
├── sentence-transformers (all-MiniLM-L6-v2) → embeds chunks as vectors
│
└── Qdrant Cloud → stores vectors + metadata (source, page number)
User Question
│
▼
[FastAPI Backend]
│
├── sentence-transformers → embeds the question
│
├── Qdrant → retrieves top-K most similar chunks
│
└── Groq (Llama 3.3 70B) + Instructor → generates structured answer with citations
│
▼
[Streamlit Frontend] → displays answer + expandable citations

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| PDF Parsing | pdfplumber | Accurate text extraction per page |
| Chunking | LangChain RecursiveCharacterTextSplitter | Overlapping chunks preserving context |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, free, 384-dim vectors |
| Vector DB | Qdrant Cloud | Managed vector storage with payload filtering |
| LLM | Groq API — Llama 3.3 70B | Fast inference, structured output |
| Structured Output | Instructor + Pydantic | Enforces citation schema from LLM |
| Backend | FastAPI | REST API with automatic validation and docs |
| Frontend | Streamlit | Interactive web interface |
| Containerisation | Docker + Docker Compose | Environment parity across dev and production |
| Backend Hosting | Render (free tier) | FastAPI deployment |
| Frontend Hosting | Streamlit Community Cloud | Streamlit deployment |

---

## Key Engineering Decisions

**Why overlapping chunks?**  
A sentence split across two chunks loses meaning. 50-character overlap ensures no context is severed at a boundary.

**Why a local embedding model instead of OpenAI?**  
`all-MiniLM-L6-v2` is free, runs on CPU, and produces quality 384-dimension vectors sufficient for document retrieval. The architecture allows swapping to any embedding provider without changing the rest of the codebase.

**Why Qdrant over FAISS?**  
Qdrant supports native payload filtering — restricting search to a specific document — without manual post-processing. It also persists independently of the API server, so vectors survive server restarts.

**Why Instructor for structured output?**  
Without structured output enforcement, LLMs return free text that must be parsed with fragile regex. Instructor patches the Groq client to validate responses against a Pydantic model and retries automatically on malformed output.

**RAG limitation acknowledged:**  
For questions requiring information spread across many pages, higher `top_k` values improve recall at the cost of slightly longer generation time. This is a known tradeoff in retrieval-augmented generation systems.

---

## Project Structure

```
doc-intelligence-rag/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── documents.py     # Upload, list, delete endpoints
│   │   │   └── chat.py          # Ask question endpoint
│   │   └── schemas.py           # Request/response Pydantic models
│   ├── core/
│   │   ├── chunker.py           # PDF parsing and text chunking
│   │   ├── embedder.py          # sentence-transformers embedding
│   │   ├── retriever.py         # Vector search and ranking
│   │   └── generator.py         # LLM generation with citations
│   ├── db/
│   │   └── vector_store.py      # Qdrant CRUD operations
│   ├── services/
│   │   ├── document_service.py  # Ingestion pipeline orchestration
│   │   └── rag_service.py       # Retrieval + generation orchestration
│   ├── config.py                # Single source of truth for all settings
│   └── main.py                  # FastAPI app factory and startup
├── frontend/
│   ├── app.py                   # Streamlit UI
│   ├── api_client.py            # HTTP client for backend communication
│   └── config.py                # Frontend configuration
├── tests/
│   ├── test_chunker.py
│   ├── test_embedder.py
│   └── test_api.py
├── Dockerfile                   # Backend container
├── Dockerfile.frontend          # Frontend container
├── docker-compose.yml           # Local multi-service orchestration
├── .env.example                 # Environment variable template
└── requirements.txt
```

## Running Locally

### Prerequisites
- Python 3.11+
- A [Qdrant Cloud](https://cloud.qdrant.io) free account
- A [Groq](https://console.groq.com) free API key

### Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/doc-intelligence-rag.git
cd doc-intelligence-rag
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**
```bash
cp .env.example .env
```

Open `.env` and fill in your values:

GROQ_API_KEY=your_groq_api_key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

**5. Start the backend**
```bash
uvicorn app.main:app --reload --port 8000
```

**6. Start the frontend** (new terminal, venv active)
```bash
cd frontend
streamlit run app.py
```

Visit `http://localhost:8501` — upload a PDF and start asking questions.

### Running with Docker

```bash
docker compose up --build
```

Backend available at `http://localhost:8000/docs`  
Frontend available at `http://localhost:8501`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/documents/upload` | Upload and ingest a PDF |
| `GET` | `/api/v1/documents/` | List all uploaded documents |
| `DELETE` | `/api/v1/documents/{filename}` | Remove a document |
| `POST` | `/api/v1/chat/ask` | Ask a question |
| `GET` | `/health` | Health check |

Full interactive documentation available at `/docs` (Swagger UI).

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Place a PDF named `sample.pdf` in `tests/fixtures/` before running.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ | — | Groq API key |
| `QDRANT_URL` | ✅ | — | Qdrant cluster URL |
| `QDRANT_API_KEY` | ✅ | — | Qdrant API key |
| `COLLECTION_NAME` | ❌ | `doc_intelligence` | Qdrant collection name |
| `EMBEDDING_MODEL` | ❌ | `all-MiniLM-L6-v2` | HuggingFace model name |
| `CHUNK_SIZE` | ❌ | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | ❌ | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | ❌ | `10` | Chunks retrieved per query |
| `LLM_MODEL` | ❌ | `llama-3.3-70b-versatile` | Groq model name |
| `APP_ENV` | ❌ | `development` | Environment name |

---

## Author

**Mubarak Olalekan Oladipo**  
AI Software Engineer  
[GitHub](https://github.com/Mubrix2) · [LinkedIn](https://linkedin.com/in/mubarak-oladipo/)

