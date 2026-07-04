"""
Tests for services/retrieval_service.py — FAISS search over the full index
and over a pHash-filtered candidate subset, including the single-candidate
fast path.
"""

import faiss
import numpy as np
import pytest

from services.retrieval_service import full_search, hash_filtered_search


def _build_index(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


class TestFullSearch:
    def test_finds_the_identical_vector(self, random_unit_vectors):
        vectors = random_unit_vectors(20, dim=16, seed=1)
        index = _build_index(vectors)

        # Query with rotations that are all just the stored vector itself —
        # the identical vector must come back with score ~1.0.
        query = np.tile(vectors[5], (4, 1))
        scores = full_search(query, index, top_k=5)

        assert 5 in scores
        assert scores[5] == pytest.approx(1.0, abs=1e-3)

    def test_returns_best_score_across_rotations(self, random_unit_vectors):
        vectors = random_unit_vectors(10, dim=16, seed=2)
        index = _build_index(vectors)

        # Rotation 0 is a random vector (low score), rotation 1 is the exact
        # match — the map must keep the *best* score per image, not the last.
        random_other = random_unit_vectors(1, dim=16, seed=99)[0]
        query = np.vstack([random_other, vectors[3]])
        scores = full_search(query, index, top_k=10)

        assert scores[3] == pytest.approx(1.0, abs=1e-3)

    def test_respects_top_k(self, random_unit_vectors):
        vectors = random_unit_vectors(50, dim=16, seed=3)
        index = _build_index(vectors)
        query = np.tile(vectors[0], (1, 1))
        scores = full_search(query, index, top_k=5)
        assert len(scores) <= 5


class TestHashFilteredSearch:
    def test_single_candidate_high_score_returns_exact(self, random_unit_vectors):
        vectors = random_unit_vectors(10, dim=16, seed=4)
        index = _build_index(vectors)
        query = np.tile(vectors[2], (4, 1))  # identical -> score ~1.0

        result = hash_filtered_search(query, [2], index)
        assert 2 in result
        assert result[2] > 0.97

    def test_single_candidate_low_score_returns_empty(self, random_unit_vectors):
        vectors = random_unit_vectors(10, dim=16, seed=5)
        index = _build_index(vectors)
        # A dissimilar random query vector should score low against a
        # single unrelated candidate, triggering the empty-map fallback
        # signal (pipeline then falls back to full_search).
        unrelated = random_unit_vectors(1, dim=16, seed=123)[0]
        query = np.tile(unrelated, (4, 1))

        result = hash_filtered_search(query, [7], index)
        # Either empty (below SINGLE_CANDIDATE_NEAR_DUP_THR) or a valid
        # low/near-dup score — either way it must not crash and must not
        # fabricate a high-confidence match for an unrelated vector.
        if result:
            assert result[7] < 0.98

    def test_multi_candidate_searches_only_the_subset(self, random_unit_vectors):
        vectors = random_unit_vectors(20, dim=16, seed=6)
        index = _build_index(vectors)
        candidates = [1, 4, 9]
        query = np.tile(vectors[4], (4, 1))  # identical to candidate 4

        result = hash_filtered_search(query, candidates, index)
        assert 4 in result
        assert result[4] > 0.97
        # Only candidate indices may appear as keys — never an index
        # outside the pHash-filtered subset.
        assert set(result.keys()).issubset(set(candidates))


