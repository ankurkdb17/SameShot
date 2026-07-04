"""
app.py
------
FastAPI application factory and startup sequence.

This is the single entry point for the application.
Run with: uvicorn app:app --host 0.0.0.0 --port 8000

Startup sequence
----------------
1. Configure structured logging for the whole process.
2. Create upload directories if they don't exist.
3. Load the DINOv2 model (once — from models/dino_model.py).
4. Instantiate DuplicateDetectionPipeline with the loaded model.
5. Inject the pipeline into the route handlers.
6. Mount static file serving for the uploads directory.
7. Register the API router.

Nothing else belongs here. Business logic lives in services/.
Route handlers live in routes/. Persistence lives in storage/.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import CORS_ORIGINS, POOL_DIR, QUERY_DIR, STORE_DIR, UPLOAD_DIR

# ---------------------------------------------------------------------------
# Logging — configured before any other import so every module's logger
# inherits this setup from the start.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SameShot — Near Duplicate Image Detection",
    description=(
        "Detects near-duplicate and exactly-same images using DINOv2 embeddings, "
        "pHash pre-filtering, FAISS retrieval, and SIFT geometric verification."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------

for directory in (POOL_DIR, QUERY_DIR, STORE_DIR):
    os.makedirs(directory, exist_ok=True)
    logger.debug("Directory ready: %s", directory)

# Static file serving — frontend fetches images via /images/<filename>.
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

# ---------------------------------------------------------------------------
# Model and pipeline startup
# ---------------------------------------------------------------------------

logger.info("Loading DINOv2 model...")

# This import triggers the single model load. Every service that needs the
# model imports MODEL, TRANSFORM, DEVICE from here — never loads it again.
from models.dino_model import DEVICE, MODEL, TRANSFORM

logger.info("DINOv2 model loaded on device '%s'.", DEVICE)

from services.pipeline import DuplicateDetectionPipeline
from routes.images import router, set_pipeline

pipeline = DuplicateDetectionPipeline(MODEL, TRANSFORM, DEVICE)
set_pipeline(pipeline)
logger.info("Detection pipeline ready.")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(router)
logger.info("SameShot API started. Listening for requests.")

# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
