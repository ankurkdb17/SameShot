"""
tests/conftest.py
------------------
Shared pytest fixtures and import-path setup.

The application modules (config, services.*, storage.*) use absolute
imports like `from config import ...`, which only works if the `backend/`
directory is on sys.path — exactly how uvicorn runs it in production
(`uvicorn app:app` from inside backend/). This conftest replicates that
so tests import the real modules unmodified, not a reorganised copy.
"""

import os
import sys

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def make_image(tmp_path):
    """
    Factory fixture: make_image(name, color) -> path to a solid-color PNG.

    Solid colors give deterministic, easily-reasoned-about pHashes —
    two identical colors always hash identically, two very different
    colors always hash far apart. That determinism is what the hash
    and retrieval tests rely on.
    """
    def _make(name: str, color: tuple[int, int, int], size: int = 64) -> str:
        path = os.path.join(tmp_path, name)
        Image.new("RGB", (size, size), color).save(path)
        return path

    return _make


@pytest.fixture
def make_checker_image(tmp_path):
    """
    Factory fixture: make_checker_image(name) -> path to a checkerboard PNG.

    Solid colors are structurally near-identical to a perceptual hash
    (pHash is DCT-based, not color-based), so two different solid colors
    can land only 1 bit apart. A checkerboard has genuine high-frequency
    structure, giving a hash that's reliably far from a solid-color hash —
    needed for tests that assert two images are perceptually *distant*.
    """
    def _make(name: str, size: int = 64, block: int = 8) -> str:
        path = os.path.join(tmp_path, name)
        img = Image.new("RGB", (size, size), (0, 0, 0))
        pixels = img.load()
        for x in range(size):
            for y in range(size):
                if (x // block + y // block) % 2 == 0:
                    pixels[x, y] = (255, 255, 255)
        img.save(path)
        return path

    return _make


@pytest.fixture
def random_unit_vectors():
    """
    Factory fixture: random_unit_vectors(n, dim, seed) -> (n, dim) float32 array
    of L2-normalised random vectors, mimicking real DINOv2 embeddings closely
    enough to exercise FAISS IndexFlatIP cosine-similarity search.
    """
    def _make(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        vecs = rng.standard_normal((n, dim)).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    return _make
