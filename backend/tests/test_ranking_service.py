"""
Tests for services/ranking_service.py — score classification, filtering,
sorting, and the SIFT score-band boosting logic.
"""

import numpy as np

from services.ranking_service import classify_score, rank_results, rank_sift_result


class TestClassifyScore:
    def test_exact_match_threshold(self):
        assert classify_score(0.98) == "Exactly Same"
        assert classify_score(0.99) == "Exactly Same"

    def test_near_duplicate_band(self):
        assert classify_score(0.89) == "Near Duplicate"
        assert classify_score(0.95) == "Near Duplicate"

    def test_below_near_duplicate_is_different(self):
        assert classify_score(0.88) == "Different"
        assert classify_score(0.0) == "Different"

    def test_boundary_is_inclusive(self):
        # >= threshold, not >, at both boundaries.
        assert classify_score(0.98) == "Exactly Same"
        assert classify_score(0.89) == "Near Duplicate"


class TestRankResults:
    def _paths(self, names):
        return np.array(names, dtype=object)

    def test_sorts_descending_by_score(self):
        score_map = {0: 0.70, 1: 0.95, 2: 0.85}
        paths = self._paths(["low.jpg", "high.jpg", "mid.jpg"])
        results = rank_results(score_map, paths)
        assert [r["name"] for r in results] == ["high.jpg", "mid.jpg", "low.jpg"]

    def test_filters_below_min_score(self):
        score_map = {0: 0.50, 1: 0.95}
        paths = self._paths(["excluded.jpg", "included.jpg"])
        results = rank_results(score_map, paths, min_score=0.60)
        assert len(results) == 1
        assert results[0]["name"] == "included.jpg"

    def test_caps_at_top_k(self):
        score_map = {i: 0.9 - i * 0.01 for i in range(10)}
        paths = self._paths([f"img{i}.jpg" for i in range(10)])
        results = rank_results(score_map, paths, top_k=3)
        assert len(results) == 3

    def test_each_result_has_expected_shape(self):
        score_map = {0: 0.99}
        paths = self._paths(["ref.jpg"])
        results = rank_results(score_map, paths)
        assert results == [{"name": "ref.jpg", "score": 0.99, "status": "Exactly Same"}]

    def test_empty_score_map_returns_empty_list(self):
        assert rank_results({}, self._paths([])) == []

    def test_extracts_basename_from_full_path(self):
        score_map = {0: 0.99}
        paths = self._paths(["/some/deep/path/photo.jpg"])
        results = rank_results(score_map, paths)
        assert results[0]["name"] == "photo.jpg"


class TestRankSiftResult:
    def test_low_band_boosts_to_point_eight(self):
        results = rank_sift_result("ref.jpg", raw_score=0.50)
        assert results[0]["score"] == 0.80
        assert results[0]["status"] == "This might match your image"

    def test_mid_band_boosts_to_point_eight_five(self):
        results = rank_sift_result("ref.jpg", raw_score=0.70)
        assert results[0]["score"] == 0.85
        assert results[0]["status"] == "Near Duplicate"

    def test_high_band_boosts_to_point_nine_six(self):
        results = rank_sift_result("ref.jpg", raw_score=0.85)
        assert results[0]["score"] == 0.96
        assert results[0]["status"] == "Exactly Same"

    def test_returns_single_item_list(self):
        results = rank_sift_result("ref.jpg", raw_score=0.70)
        assert len(results) == 1

    def test_below_all_bands_keeps_raw_score(self):
        results = rank_sift_result("ref.jpg", raw_score=0.20)
        assert results[0]["score"] == 0.20
        assert results[0]["status"] == "This might match your image"
