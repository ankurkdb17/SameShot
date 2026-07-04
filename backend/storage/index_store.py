"""
storage/index_store.py
----------------------
The only place in the codebase that reads from or writes to disk.

Responsibilities
----------------
- Load the FAISS index and numpy metadata arrays.
- Persist them back after any mutation.
- Add a single new vector (fast append, no full rebuild).
- Remove a single vector by filename (rebuild required; FAISS has no delete).
- Expose a lightweight IndexData dataclass so callers never touch raw paths.

Nothing in this module knows about images, embeddings, or hashing.
It only moves arrays and indices in and out of the store directory.

All paths come from config.py — there are no hardcoded strings here.
"""

import logging
import os
from dataclasses import dataclass

import faiss
import numpy as np

from config import (
    DINO_EMBEDDING_DIM,
    HASHES_FILE,
    INDEX_FILE,
    PATHS_FILE,
    STORE_DIR,
    VECTORS_FILE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class IndexData:
    """
    Everything stored on disk, loaded into memory as a single unit.

    Attributes
    ----------
    vectors : np.ndarray, shape (N, DINO_EMBEDDING_DIM), dtype float32
        L2-normalised DINOv2 embeddings for each reference image.
    paths : np.ndarray, shape (N,), dtype object
        Absolute (or relative) filesystem path for each reference image.
    hashes : np.ndarray, shape (N,), dtype object
        Hex-string pHash for each reference image.
    index : faiss.Index
        FAISS IndexFlatIP built from the vectors above.
    """
    vectors: np.ndarray
    paths:   np.ndarray
    hashes:  np.ndarray
    index:   faiss.Index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_initialised() -> bool:
    """Return True if all four store files exist on disk."""
    return all(os.path.exists(p) for p in (INDEX_FILE, VECTORS_FILE, PATHS_FILE, HASHES_FILE))


def load() -> IndexData:
    """
    Load the full index from disk and return it as an IndexData.

    Raises
    ------
    FileNotFoundError
        If any of the four required store files are missing.
    RuntimeError
        If the numpy arrays are inconsistent (mismatched lengths).
    """
    if not is_initialised():
        missing = [p for p in (INDEX_FILE, VECTORS_FILE, PATHS_FILE, HASHES_FILE)
                   if not os.path.exists(p)]
        raise FileNotFoundError(
            f"Index store is not fully initialised. Missing: {missing}"
        )

    logger.debug("Loading index from '%s'...", STORE_DIR)

    vectors = np.load(VECTORS_FILE, allow_pickle=True)
    paths   = np.load(PATHS_FILE,   allow_pickle=True)
    hashes  = np.load(HASHES_FILE,  allow_pickle=True)
    index   = faiss.read_index(INDEX_FILE)

    if not (len(vectors) == len(paths) == len(hashes)):
        raise RuntimeError(
            f"Index store is corrupt: vectors={len(vectors)}, "
            f"paths={len(paths)}, hashes={len(hashes)}"
        )

    logger.debug("Index loaded: %d images.", len(paths))
    return IndexData(vectors=vectors, paths=paths, hashes=hashes, index=index)


def load_empty() -> IndexData:
    """
    Return an empty IndexData (no vectors, no paths, no hashes).

    Used by process_reference_pool when no prior index exists.
    """
    empty_vectors = np.empty((0, DINO_EMBEDDING_DIM), dtype="float32")
    empty_paths   = np.array([], dtype=object)
    empty_hashes  = np.array([], dtype=object)
    empty_index   = _build_faiss_index(empty_vectors)
    return IndexData(
        vectors=empty_vectors,
        paths=empty_paths,
        hashes=empty_hashes,
        index=empty_index,
    )


def save(data: IndexData) -> None:
    """
    Persist an IndexData to disk, creating STORE_DIR if necessary.

    This is the single write path for all persistence in the application.
    """
    os.makedirs(STORE_DIR, exist_ok=True)

    faiss.write_index(data.index, INDEX_FILE)
    np.save(VECTORS_FILE, data.vectors)
    np.save(PATHS_FILE,   data.paths)
    np.save(HASHES_FILE,  data.hashes)

    logger.info("Index saved: %d images in '%s'.", len(data.paths), STORE_DIR)


def append_one(image_path: str, vector: np.ndarray, phash: str) -> bool:
    """
    Add a single image to the on-disk index without a full rebuild.

    Loads the current index, appends the new vector, and saves.
    FAISS IndexFlatIP supports in-place add() so this is fast.

    Parameters
    ----------
    image_path : str
        Path to the image (used as the identifier in the store).
    vector : np.ndarray, shape (1, DINO_EMBEDDING_DIM), dtype float32
        L2-normalised DINOv2 embedding for the image.
    phash : str
        Hex-string pHash of the image.

    Returns
    -------
    bool
        True on success, False if the index doesn't exist yet or the
        image is already indexed.
    """
    if not is_initialised():
        logger.warning("append_one: index not initialised; cannot append.")
        return False

    data = load()

    if image_path in data.paths:
        logger.warning("append_one: '%s' already in index; skipping.", image_path)
        return False

    new_vectors = np.vstack((data.vectors, vector))
    new_paths   = np.append(data.paths,  image_path)
    new_hashes  = np.append(data.hashes, phash)

    # Fast in-place add — no rebuild needed for IndexFlatIP.
    data.index.add(vector)

    updated = IndexData(
        vectors=new_vectors,
        paths=new_paths,
        hashes=new_hashes,
        index=data.index,
    )
    save(updated)
    logger.info("append_one: added '%s' (index size now %d).", image_path, len(new_paths))
    return True


def remove_one(filename: str) -> bool:
    """
    Remove a single image from the on-disk index by its basename.

    FAISS IndexFlatIP has no delete operation, so this rebuilds the index
    from the remaining vectors after removing the target row.

    Parameters
    ----------
    filename : str
        The basename of the file to remove (e.g. "photo.jpg").

    Returns
    -------
    bool
        True if the image was found and removed, False otherwise.
    """
    if not is_initialised():
        logger.warning("remove_one: index not initialised; nothing to remove.")
        return False

    data = load()

    # Find the row whose path basename matches the requested filename.
    target_idx = next(
        (i for i, p in enumerate(data.paths) if os.path.basename(p) == filename),
        -1,
    )

    if target_idx == -1:
        logger.warning("remove_one: '%s' not found in index.", filename)
        return False

    new_vectors = np.delete(data.vectors, target_idx, axis=0)
    new_paths   = np.delete(data.paths,   target_idx)
    new_hashes  = np.delete(data.hashes,  target_idx)

    # Rebuild FAISS index from the remaining vectors.
    new_index = _build_faiss_index(new_vectors)

    updated = IndexData(
        vectors=new_vectors,
        paths=new_paths,
        hashes=new_hashes,
        index=new_index,
    )
    save(updated)
    logger.info(
        "remove_one: removed '%s' at row %d (index size now %d).",
        filename, target_idx, len(new_paths),
    )
    return True


def clear() -> None:
    """
    Delete all four store files from disk.

    Used by the /reset endpoint to wipe the index completely.
    """
    for path in (INDEX_FILE, VECTORS_FILE, PATHS_FILE, HASHES_FILE):
        if os.path.exists(path):
            os.remove(path)
            logger.info("Deleted store file: %s", path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    Build a fresh IndexFlatIP from a (N, D) float32 array.

    IndexFlatIP computes exact inner-product (cosine similarity after
    L2-normalisation). No approximate search — correctness over speed
    because our reference pools are typically small (<10k images).
    """
    dim = vectors.shape[1] if vectors.ndim == 2 and vectors.shape[0] > 0 else DINO_EMBEDDING_DIM
    index = faiss.IndexFlatIP(dim)
    if vectors.shape[0] > 0:
        index.add(vectors)
    return index
