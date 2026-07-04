# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder
# Install all Python dependencies into a virtual environment so the final
# image only copies what it needs (no build tools, no pip cache).
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System packages required to compile certain Python wheels (e.g. OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy only requirements first to leverage Docker layer caching.
COPY requirements.txt .

# Create a venv and install dependencies into it.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime
# Lean image — copy venv from builder, no build tools needed at runtime.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="SameShot"
LABEL description="SameShot — Near Duplicate Image Detection API"

# Runtime system libraries only (OpenCV needs libGL + libGLib at runtime too).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security.
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source code.
COPY . .

# Ensure the runtime directories exist and are owned by appuser.
# These are also mounted as Docker volumes for persistence.
RUN mkdir -p uploads/pool uploads/query dinov2_faiss_store \
    && chown -R appuser:appuser /app

# Switch to non-root user.
USER appuser

# Expose the API port.
EXPOSE 8000

# Health check — curl is not available; use Python's http.client instead.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')" || exit 1

# Start the application.
# --workers 1 : single worker — the DINOv2 model is loaded once per process.
# --timeout-keep-alive 30 : keeps connections alive for up to 30 s.
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]
