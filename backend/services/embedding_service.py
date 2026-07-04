"""
services/embedding_service.py
------------------------------
DINOv2 embedding for two distinct use cases:

  1. embed_batch()      — batch-embed a list of image paths for index building.
  2. embed_query()      — embed a single query image at 4 rotations for search.

Both functions receive MODEL, TRANSFORM, and DEVICE as parameters rather than
importing them at module level. This keeps the service pure and testable:
callers (the pipeline) pass in the singletons from models/dino_model.py.

Why rotations?
--------------
A user may upload a screenshot that is rotated relative to the reference pool
(e.g. a portrait crop of a landscape image, or a 90° rotated export). Embedding
at 0°, 90°, 180°, and 270° and taking the best cosine similarity across all four
ensures rotation-invariant matching without modifying the stored index.
"""

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

from config import (
    EMBEDDING_BATCH_SIZE,
    QUERY_ROTATION_ANGLES,
    QUERY_ROTATION_SIZE,
)

logger = logging.getLogger(__name__)


def embed_batch(
    image_paths: list[str],
    model: torch.nn.Module,
    transform,
    device: str,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    """
    Embed a list of images in batches and return a (N, D) float32 array.

    Images that fail to load are silently skipped (logged at WARNING level)
    so a single corrupt file in the reference pool does not abort indexing.

    Parameters
    ----------
    image_paths : list[str]
        Paths to the images to embed.
    model : torch.nn.Module
        The DINOv2 model in eval mode.
    transform : torchvision.transforms.Compose
        Preprocessing pipeline (resize, grayscale, normalise).
    device : str
        "cuda" or "cpu".
    batch_size : int
        Number of images per forward pass.

    Returns
    -------
    np.ndarray, shape (N, D), dtype float32
        L2-normalised embeddings. N <= len(image_paths) because failed
        images are dropped. Returns shape (0,) if every image fails.
    """
    embeddings = []
    total = len(image_paths)

    for batch_start in range(0, total, batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        tensors = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(transform(img))
            except Exception as exc:
                logger.warning("embed_batch: skipping '%s': %s", path, exc)

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            features = model(batch)
            features = torch.nn.functional.normalize(features, dim=-1)
            embeddings.append(features.cpu().numpy())

        encoded_so_far = min(batch_start + batch_size, total)
        logger.info("embed_batch: encoded %d / %d images.", encoded_so_far, total)

    if not embeddings:
        return np.array([], dtype="float32")

    return np.vstack(embeddings).astype("float32")


def embed_query(
    image_path: str,
    model: torch.nn.Module,
    transform,
    device: str,
) -> Optional[np.ndarray]:
    """
    Embed a query image at four rotations and return a (4, D) float32 array.

    Each of the four rows is an L2-normalised embedding for the image at
    0°, 90°, 180°, and 270°.  The retrieval service takes the best cosine
    score across all four so the search is rotation-invariant.

    Parameters
    ----------
    image_path : str
        Path to the query image.
    model, transform, device
        Same DINOv2 singletons used for index building.

    Returns
    -------
    np.ndarray, shape (4, D), dtype float32, or None on failure.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.error("embed_query: cannot open '%s': %s", image_path, exc)
        return None

    vectors = []
    for angle in QUERY_ROTATION_ANGLES:
        try:
            # expand=True preserves corners when rotating non-square images.
            rotated = image.rotate(angle, expand=True)
            resized = rotated.resize((QUERY_ROTATION_SIZE, QUERY_ROTATION_SIZE))
            tensor  = transform(resized).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model(tensor)
                features = torch.nn.functional.normalize(features, dim=-1)

            vectors.append(features.cpu().numpy())
        except Exception as exc:
            logger.warning(
                "embed_query: failed at rotation %d° for '%s': %s",
                angle, image_path, exc,
            )

    if not vectors:
        logger.error("embed_query: all rotations failed for '%s'.", image_path)
        return None

    result = np.vstack(vectors).astype("float32")
    logger.debug(
        "embed_query: produced %d rotation vectors for '%s'.", len(vectors), image_path
    )
    return result
