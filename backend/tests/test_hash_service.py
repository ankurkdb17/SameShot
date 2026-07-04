"""
Tests for services/hash_service.py — perceptual hash computation and
Hamming-distance candidate filtering.
"""

import numpy as np

from services.hash_service import compute_hash, compute_hash_hex, filter_candidates


class TestComputeHash:
    def test_identical_images_produce_identical_hash(self, make_image):
        path_a = make_image("a.png", (255, 0, 0))
        path_b = make_image("b.png", (255, 0, 0))
        assert compute_hash(path_a) == compute_hash(path_b)

    def test_very_different_images_produce_distant_hashes(self, make_image, make_checker_image):
        red = make_image("red.png", (255, 0, 0))
        checker = make_checker_image("checker.png")
        distance = compute_hash(red) - compute_hash(checker)
        assert distance > 8  # comfortably above PHASH_HAMMING_THRESHOLD

    def test_missing_file_returns_fallback_hash_not_exception(self):
        # Must never raise — the pipeline relies on always getting a hash
        # back, even for a corrupt/missing file, so indexing never aborts.
        result = compute_hash("/nonexistent/path/does_not_exist.jpg")
        assert result is not None

    def test_compute_hash_hex_returns_string(self, make_image):
        path = make_image("a.png", (10, 20, 30))
        result = compute_hash_hex(path)
        assert isinstance(result, str)
        assert len(result) > 0


class TestFilterCandidates:
    def test_finds_exact_hash_match(self, make_image):
        query = compute_hash(make_image("query.png", (100, 150, 200)))
        stored = [str(compute_hash_hex(make_image("stored.png", (100, 150, 200))))]
        assert filter_candidates(query, stored) == [0]

    def test_excludes_distant_hash(self, make_image, make_checker_image):
        query = compute_hash(make_image("query.png", (255, 255, 255)))
        stored_hash = compute_hash_hex(make_checker_image("stored.png"))
        assert filter_candidates(query, [stored_hash], hamming_threshold=8) == []

    def test_respects_custom_threshold(self, make_image):
        query = compute_hash(make_image("query.png", (255, 255, 255)))
        stored_hash = compute_hash_hex(make_image("stored.png", (0, 0, 0)))
        # threshold=64 accepts everything, since max Hamming distance for a
        # 64-bit hash is 64.
        assert filter_candidates(query, [stored_hash], hamming_threshold=64) == [0]

    def test_skips_corrupt_stored_hash_without_raising(self, make_image):
        query = compute_hash(make_image("query.png", (1, 2, 3)))
        # "not-a-hex-hash" can't be parsed — must be skipped, not raised.
        result = filter_candidates(query, ["not-a-hex-hash"])
        assert result == []

    def test_empty_store_returns_empty_list(self, make_image):
        query = compute_hash(make_image("query.png", (1, 2, 3)))
        assert filter_candidates(query, np.array([], dtype=object)) == []
