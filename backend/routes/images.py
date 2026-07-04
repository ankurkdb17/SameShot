"""
routes/images.py
----------------
All HTTP route handlers for the SameShot API.

Design
------
Every handler does exactly three things:
  1. Extract inputs from the request.
  2. Call into a service or the index store.
  3. Return a JSON-serialisable dict.

No business logic lives here. No ML calls. Handlers are thin by design so
that the routes file reads like a specification, not an implementation.

Client input handling
----------------------
Every filename that originates from the client (UploadFile.filename or a
path parameter) is passed through security.safe_filename() before it is
used to build a filesystem path, and every upload is checked against
config.ALLOWED_UPLOAD_EXTENSIONS and config.MAX_UPLOAD_SIZE_BYTES before
being written to disk. See security.py for why this matters.

Endpoints
---------
POST   /upload/pool            Upload one or more reference images.
DELETE /delete/pool/{filename} Remove a reference image from pool and index.
POST   /upload/query           Replace the current query image.
DELETE /delete/query           Clear the query image.
GET    /analyze                Run the detection pipeline on the current query.
POST   /reset                  Wipe all images and the FAISS index.
"""

import glob
import logging
import os
import shutil
from typing import List

import faiss
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from config import (
    ALLOWED_UPLOAD_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES,
    POOL_DIR,
    POOL_IMAGE_EXTENSIONS,
    QUERY_DIR,
)
from models.dino_model import DEVICE, MODEL, TRANSFORM
from security import has_allowed_extension, safe_filename
from services.embedding_service import embed_batch
from services.hash_service import compute_hash_hex
from services.pipeline import DuplicateDetectionPipeline
from storage import index_store
from storage.index_store import IndexData

logger = logging.getLogger(__name__)

# The router is registered on the FastAPI app in app.py.
router = APIRouter()

# The pipeline singleton is injected by app.py after model load.
# Routes call _get_pipeline() so the injection point is testable.
_pipeline: DuplicateDetectionPipeline | None = None


def set_pipeline(pipeline: DuplicateDetectionPipeline) -> None:
    """Called once from app.py startup after the model is loaded."""
    global _pipeline
    _pipeline = pipeline


def _get_pipeline() -> DuplicateDetectionPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised — app startup incomplete.")
    return _pipeline


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------

@router.post("/upload/pool")
async def upload_pool(files: List[UploadFile] = File(...)):
    """
    Append one or more images to the reference pool and index them immediately.

    Each filename is sanitised, checked against the allowed extension list,
    and size-limited before being saved to POOL_DIR. Valid images are then
    added to the FAISS index via index_store.append_one(). If the index does
    not yet exist (first upload), _build_pool_index() builds it from scratch.

    Returns
    -------
    JSON with lists of filenames added to disk / index, and any rejected
    filenames with a reason.
    """
    saved_files   = []
    indexed_files = []
    rejected      = []

    for file in files:
        filename = safe_filename(file.filename or "")

        if not has_allowed_extension(filename, ALLOWED_UPLOAD_EXTENSIONS):
            logger.warning("upload_pool: rejected '%s' — disallowed extension.", filename)
            rejected.append({"file": filename, "reason": "unsupported file type"})
            continue

        file_path = os.path.join(POOL_DIR, filename)

        if os.path.exists(file_path):
            logger.debug("upload_pool: '%s' already on disk — skipping.", filename)
            continue

        written = await _save_upload_within_limit(file, file_path)
        if not written:
            rejected.append({"file": filename, "reason": "file exceeds size limit"})
            continue

        saved_files.append(filename)
        logger.info("upload_pool: saved '%s'.", filename)

        # If no index exists yet, build it from the whole pool directory.
        # _build_pool_index embeds every file in the pool (including this one),
        # so we must NOT call append_one afterwards — it would hit the duplicate
        # guard and incorrectly report this file as un-indexed.
        if not index_store.is_initialised():
            logger.info("upload_pool: index absent — building from pool directory.")
            _build_pool_index()
            if index_store.is_initialised():
                indexed_files.append(filename)
            continue

        # Fast incremental add — no full rebuild required.
        vector = embed_batch([file_path], MODEL, TRANSFORM, DEVICE, batch_size=1)
        if vector.size == 0:
            logger.warning("upload_pool: embedding failed for '%s' — not indexed.", filename)
            continue

        phash = compute_hash_hex(file_path)
        added = index_store.append_one(file_path, vector, phash)
        if added:
            indexed_files.append(filename)
            logger.info("upload_pool: indexed '%s'.", filename)

    return {
        "added_to_disk":  saved_files,
        "added_to_index": indexed_files,
        "rejected":       rejected,
        "count":          len(saved_files),
    }


@router.delete("/delete/pool/{filename}")
def delete_pool_image(filename: str):
    """
    Remove a reference image from disk and from the FAISS index.

    The filename path parameter is sanitised before use so a client cannot
    supply "../" segments to reach files outside POOL_DIR.

    Uses index_store.remove_one() which rebuilds the index without the
    deleted vector. No model inference required.
    """
    safe_name   = safe_filename(filename)
    file_path   = os.path.join(POOL_DIR, safe_name)
    file_deleted = False

    if os.path.exists(file_path):
        os.remove(file_path)
        file_deleted = True
        logger.info("delete_pool_image: deleted '%s' from disk.", safe_name)

    index_updated = index_store.remove_one(safe_name)

    if file_deleted or index_updated:
        return {
            "status":        "deleted",
            "file":          safe_name,
            "disk_deleted":  file_deleted,
            "index_updated": index_updated,
        }

    logger.warning("delete_pool_image: '%s' not found on disk or in index.", safe_name)
    return {"status": "not_found", "detail": "File not found on disk or in index."}


# ---------------------------------------------------------------------------
# Query image management
# ---------------------------------------------------------------------------

@router.post("/upload/query")
async def upload_query(file: UploadFile = File(...)):
    """
    Replace the current query image.

    Deletes any existing query image first so there is always exactly
    one query image on disk. Filename is sanitised and validated the
    same way as pool uploads.
    """
    filename = safe_filename(file.filename or "")

    if not has_allowed_extension(filename, ALLOWED_UPLOAD_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    _clear_directory(QUERY_DIR)

    file_path = os.path.join(QUERY_DIR, filename)
    written = await _save_upload_within_limit(file, file_path)
    if not written:
        raise HTTPException(status_code=413, detail="File exceeds the maximum upload size.")

    logger.info("upload_query: query image set to '%s'.", filename)
    return {"status": "updated", "file": filename}


@router.delete("/delete/query")
def delete_query_image():
    """Clear the query image folder."""
    _clear_directory(QUERY_DIR)
    logger.info("delete_query_image: query folder cleared.")
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@router.get("/analyze")
def analyze():
    """
    Run the duplicate detection pipeline on the current query image.

    Pipeline stages (see services/pipeline.py for details):
      Validate → Load index → [pHash ║ Embed] → FAISS → SIFT → Rank → Return

    Returns
    -------
    {"results": list[dict]}
    Each dict: {"name": str, "score": float, "status": str}
    """
    # Resolve query image.
    query_files = glob.glob(os.path.join(QUERY_DIR, "*"))
    if not query_files:
        raise HTTPException(status_code=400, detail="No query image found on server.")
    query_path = query_files[0]

    # Ensure the index is up to date with what's in the pool folder.
    # _sync_pool_index is incremental — it only adds new images.
    _sync_pool_index()

    logger.info("analyze: running pipeline on '%s'.", query_path)
    try:
        results = _get_pipeline().run(query_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("analyze: pipeline raised an unexpected error.")
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info("analyze: pipeline returned %d result(s).", len(results))
    return {"results": results}


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

@router.post("/reset")
def reset_backend():
    """
    Clear all images and the FAISS index (full reset).

    Deletes pool images, query images, and all four store files.
    The application is ready for a fresh upload immediately after.
    """
    _clear_directory(POOL_DIR)
    _clear_directory(QUERY_DIR)
    index_store.clear()

    logger.info("reset_backend: pool, query, and index cleared.")
    return {
        "status":        "fully_reset",
        "pool_cleared":  True,
        "query_cleared": True,
        "faiss_cleared": True,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _save_upload_within_limit(file: UploadFile, destination: str) -> bool:
    """
    Stream an UploadFile to disk, aborting if it exceeds MAX_UPLOAD_SIZE_BYTES.

    Reads in chunks rather than loading the whole file into memory, and
    deletes any partial file written before the limit was hit.

    Returns
    -------
    bool
        True if the file was written successfully within the size limit,
        False if it was rejected for being too large (partial file removed).
    """
    chunk_size = 1024 * 1024  # 1 MB
    total_written = 0

    with open(destination, "wb") as buf:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_written += len(chunk)
            if total_written > MAX_UPLOAD_SIZE_BYTES:
                buf.close()
                os.remove(destination)
                logger.warning(
                    "_save_upload_within_limit: '%s' exceeded %d bytes — rejected.",
                    destination, MAX_UPLOAD_SIZE_BYTES,
                )
                return False
            buf.write(chunk)

    await file.close()
    return True


def _clear_directory(directory: str) -> None:
    """Delete all files inside a directory without removing the directory."""
    for path in glob.glob(os.path.join(directory, "*")):
        try:
            os.remove(path)
        except OSError as exc:
            logger.warning("_clear_directory: could not remove '%s': %s", path, exc)


def _list_pool_images() -> list[str]:
    """
    Return every image path in POOL_DIR matching the configured extensions.

    Shared by _build_pool_index and _sync_pool_index so the glob-pattern
    logic exists in exactly one place.
    """
    patterns = [os.path.join(POOL_DIR, ext) for ext in POOL_IMAGE_EXTENSIONS]
    patterns += [os.path.join(POOL_DIR, ext.upper()) for ext in POOL_IMAGE_EXTENSIONS]
    return sorted(set(f for p in patterns for f in glob.glob(p)))


def _build_pool_index() -> None:
    """
    Build the FAISS index from scratch using every image in POOL_DIR.

    Called on the first upload after a reset (when no index file exists).
    """
    all_files = _list_pool_images()

    if not all_files:
        logger.warning("_build_pool_index: no images in pool — index not built.")
        return

    logger.info("_build_pool_index: embedding %d images.", len(all_files))
    vectors = embed_batch(all_files, MODEL, TRANSFORM, DEVICE)
    hashes  = np.array([compute_hash_hex(p) for p in all_files], dtype=object)
    paths   = np.array(all_files, dtype=object)

    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)

    index_store.save(IndexData(vectors=vectors, paths=paths, hashes=hashes, index=idx))
    logger.info("_build_pool_index: index built with %d images.", len(all_files))


def _sync_pool_index() -> None:
    """
    Incremental sync: add any pool images not yet in the index.

    Called before every /analyze so the index is always consistent with disk.
    """
    all_files = _list_pool_images()

    if not all_files:
        return

    if not index_store.is_initialised():
        _build_pool_index()
        return

    data = index_store.load()
    indexed_set = set(data.paths)
    new_files   = [f for f in all_files if f not in indexed_set]

    if not new_files:
        logger.debug("_sync_pool_index: index is up to date (%d images).", len(all_files))
        return

    logger.info("_sync_pool_index: adding %d new image(s) to index.", len(new_files))
    for path in new_files:
        vector = embed_batch([path], MODEL, TRANSFORM, DEVICE, batch_size=1)
        if vector.size == 0:
            logger.warning("_sync_pool_index: embedding failed for '%s' — skipped.", path)
            continue
        phash = compute_hash_hex(path)
        index_store.append_one(path, vector, phash)
