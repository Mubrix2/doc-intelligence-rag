# Dockerfile

# ── Stage 1: base image ────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevents Python from buffering stdout/stderr (important for logs)
ENV PYTHONUNBUFFERED=1

# ── Stage 2: install dependencies ─────────────────────────────────────────
# Copy requirements FIRST — before copying your code.
# This is critical for Docker layer caching:
# if your code changes but requirements.txt does not,
# Docker reuses the cached dependency layer and does not reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Pre-download the sentence-transformers model during build
# so the container does not download it on first request
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Stage 3: copy application code ────────────────────────────────────────
COPY app/ ./app/

# ── Stage 4: expose and run ───────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]