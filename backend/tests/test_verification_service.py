"""
Tests for services/verification_service.py — SIFT geometric verification.

Note: SIFT needs genuine visual texture/keypoints to match on, which is
hard to construct reliably in a unit test without shipping real photos.
These tests focus on the failure-mode contracts (missing files, blank
images, and the crop-confirms-true happy path with a textured synthetic
image) rather than exhaustively covering match quality — that's what the
threshold constants in config.py, tuned against real images, are for.
"""

import numpy as np
from PIL import Image

from services.verification_service import verify_crop


def _save_noisy_image(path: str, seed: int, size: int = 300) -> None:
    """A high-entropy synthetic image — enough texture for SIFT to find
    real keypoints, unlike a flat solid color."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


class TestVerifyCrop:
    def test_missing_reference_returns_false(self, tmp_path):
        query_path = str(tmp_path / "query.png")
        _save_noisy_image(query_path, seed=1)
        assert verify_crop("/nonexistent/reference.jpg", query_path) is False

    def test_missing_query_returns_false(self, tmp_path):
        ref_path = str(tmp_path / "ref.png")
        _save_noisy_image(ref_path, seed=1)
        assert verify_crop(ref_path, "/nonexistent/query.jpg") is False

    def test_blank_images_have_no_keypoints_returns_false(self, tmp_path):
        # Solid-color images have no SIFT keypoints at all — must fail
        # gracefully (False), not raise.
        ref_path = str(tmp_path / "ref.png")
        query_path = str(tmp_path / "query.png")
        Image.new("RGB", (100, 100), (128, 128, 128)).save(ref_path)
        Image.new("RGB", (100, 100), (128, 128, 128)).save(query_path)
        assert verify_crop(ref_path, query_path) is False

    def test_identical_textured_image_confirms_match(self, tmp_path):
        # A textured image compared against an exact copy of itself has
        # a perfect geometric correspondence — this is the clearest
        # possible "should confirm" case.
        ref_path = str(tmp_path / "ref.png")
        _save_noisy_image(ref_path, seed=42)
        query_path = str(tmp_path / "query.png")
        Image.open(ref_path).save(query_path)

        assert verify_crop(ref_path, query_path) is True

    def test_unrelated_textured_images_do_not_confirm(self, tmp_path):
        ref_path = str(tmp_path / "ref.png")
        query_path = str(tmp_path / "query.png")
        _save_noisy_image(ref_path, seed=1)
        _save_noisy_image(query_path, seed=2)
        assert verify_crop(ref_path, query_path) is False
