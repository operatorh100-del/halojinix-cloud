# ScornSpine RunPod Serverless Dockerfile
# RT31800: Cloud-native RAG deployment for A100 80GB
#
# Build: RunPod auto-builds from GitHub
# GPU: A100 80GB ($0.00076/s scale-to-zero)
#
# Using slim base to reduce image size

FROM runpod/base:0.6.2-cuda12.1.0

LABEL maintainer="Halojinix Triad"
LABEL description="ScornSpine RAG + COCONUT Latent Reasoning for RunPod Serverless"
LABEL version="1.0.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Cache directories (inside container)
ENV HF_HOME=/runpod-volume/cache/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/runpod-volume/cache/sentence_transformers

# RunPod worker config
ENV RUNPOD_DEBUG=false

WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install dependencies in correct order:
# 1. RunPod from PyPI (not on PyTorch index)
RUN pip install --no-cache-dir runpod>=1.5.0

# 2. PyTorch with CUDA wheels (separate index)
RUN pip install --no-cache-dir torch==2.2.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. All other ML dependencies from PyPI
RUN pip install --no-cache-dir \
    numpy>=1.26.0 \
    sentence-transformers>=2.7.0 \
    transformers>=4.38.0 \
    huggingface-hub>=0.21.0 \
    fastapi>=0.109.0 \
    uvicorn>=0.27.0 \
    pydantic>=2.6.0 \
    httpx>=0.26.0 \
    qdrant-client>=1.7.0 \
    rank_bm25>=0.2.2 \
    scikit-learn>=1.4.0 \
    nltk>=3.8.0 \
    tqdm>=4.66.0 \
    PyYAML>=6.0.0 \
    python-dotenv>=1.0.0 \
    aiohttp>=3.9.0

# Copy all source files
COPY . .

# Create proper package structure for imports
RUN mkdir -p /app/scornspine /app/haloscorn/scornspine /app/haloscorn/latent_space && \
    # Create __init__.py files
    touch /app/scornspine/__init__.py && \
    touch /app/haloscorn/__init__.py && \
    touch /app/haloscorn/scornspine/__init__.py && \
    touch /app/haloscorn/latent_space/__init__.py && \
    # Copy spine.py to scornspine module
    cp /app/spine.py /app/scornspine/spine.py && \
    cp /app/vertebrae.py /app/scornspine/vertebrae.py && \
    cp /app/embeddings_cache.py /app/scornspine/embeddings_cache.py && \
    cp /app/embedding_cache.py /app/scornspine/embedding_cache.py && \
    cp /app/bm25_index.py /app/scornspine/bm25_index.py && \
    cp /app/qdrant_fallback.py /app/scornspine/qdrant_fallback.py && \
    cp /app/reranker.py /app/scornspine/reranker.py && \
    cp /app/semantic_chunker.py /app/scornspine/semantic_chunker.py && \
    cp /app/memory.py /app/scornspine/memory.py && \
    cp /app/server.py /app/scornspine/server.py && \
    # Copy latent_space module
    cp /app/latent_space/*.py /app/haloscorn/latent_space/ || true && \
    # Also symlink to haloscorn/scornspine for backward compat
    cp /app/scornspine/*.py /app/haloscorn/scornspine/

# Pre-download embedding model (avoid cold start delay)
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
    print('Downloading embedding model...'); \
    SentenceTransformer('intfloat/multilingual-e5-base'); \
    print('Model cached!')"

# Expose for local testing (RunPod doesn't need this)
EXPOSE 7782

# RunPod handler entrypoint
CMD ["python3", "-u", "runpod_handler.py"]
