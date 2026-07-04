"""
services/hash_service.py
------------------------
Perceptual hash (pHash) computation for a single image.

Responsibility
--------------
Given an image path, return its pHash as an imagehash.ImageHash object,
or as a hex string for storage.

This module consolidates the two separate compute_phash() implementations
that previously existed in main.py (returning str) and imageHash.py
(returning imagehash.ImageHash). A single implementation removes the
inconsistency and the risk of them drifting apart.

Why pHash?
----------
Perceptual hash captures the visual "fingerprint" of an image in 64 bits.
Two images that are near-identical (different compression, slight crop,
minor colour shift) produce hashes within a small Hamming distance of each
other. We use this as a fast pre-filter to avoid running expensive FAISS
search against the entire index for every query.
"""

import logging

import imagehash
from PIL import Image

from config import PHASH_HAMMING_THRESHOLD

logger = logging.getLogger(__name__)

# Sentinel hex string written to the store when hashing fails so that
# the row is still valid and can be filtered out cleanly during search.
FALLBACK_HASH_HEX = "0000000000000000"


def compute_hash(image_path: str) -> imagehash.ImageHash:
    """
    Compute the perceptual hash of an image.

    The image is converted to grayscale ('L') before hashing so the hash
    is insensitive to colour-mode differences (light/dark themes, etc.),
    matching the grayscale preprocessing applied during embedding.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    imagehash.ImageHash
        The pHash object. Supports subtraction (-) to get Hamming distance.
        Returns a zero-hash sentinel on any read/decode error so callers
        never receive None.
    """
    try:
        img = Image.open(image_path).convert("L")
        return imagehash.phash(img)
    except Exception as exc:
        logger.warning("compute_hash: failed on '%s': %s", image_path, exc)
        return imagehash.hex_to_hash(FALLBACK_HASH_HEX)


def compute_hash_hex(image_path: str) -> str:
    """
    Compute the perceptual hash and return it as a hex string for storage.

    Used when adding an image to the index store, which persists hashes
    as plain strings in a numpy object array.
    """
    return str(compute_hash(image_path))


def filter_candidates(
    query_hash: imagehash.ImageHash,
    stored_hashes,          # np.ndarray of hex strings
    hamming_threshold: int = PHASH_HAMMING_THRESHOLD,
) -> list[int]:
    """
    Return indices of stored images whose pHash is within hamming_threshold
    of the query image's hash.

    Parameters
    ----------
    query_hash : imagehash.ImageHash
        Hash of the query image (from compute_hash).
    stored_hashes : array-like of str
        Hex-string hashes loaded from the index store.
    hamming_threshold : int
        Maximum Hamming distance to be considered a candidate.
        Defaults to PHASH_HAMMING_THRESHOLD from config.

    Returns
    -------
    list[int]
        Indices into stored_hashes that pass the filter.
        Empty list means no hash-similar candidates — caller falls back
        to a full FAISS search.
    """
    candidates = []
    for i, h in enumerate(stored_hashes):
        try:
            stored = imagehash.hex_to_hash(str(h))
            if query_hash - stored <= hamming_threshold:
                candidates.append(i)
        except Exception as exc:
            logger.debug("filter_candidates: skipping index %d — %s", i, exc)
    logger.debug(
        "filter_candidates: %d / %d passed threshold %d.",
        len(candidates), len(stored_hashes), hamming_threshold,
    )
    return candidates
