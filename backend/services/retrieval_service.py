"""
services/retrieval_service.py
------------------------------
FAISS vector search — the retrieval stage of the pipeline.

Responsibility
--------------
Given query embeddings (4 rotation vectors) and a loaded FAISS index,
return a dict mapping {image_index -> best_cosine_score} across all rotations.

Two search paths
----------------
hash_filtered_search()
    Called when pHash pre-filtering produced at least one candidate.
    Builds a temporary in-memory FAISS index from the candidate vectors
    only, so we search a small subset rather than the full index.
    This is the fast path for the common case where query and reference
    are genuinely similar.

full_search()
    Called when no hash candidates were found (the fallback path).
    Searches the full FAISS index with top-K retrieval.

Both functions return the same type: dict[int, float] mapping the
*original* index position (into the full vectors/paths arrays) to the
best cosine similarity found across all query rotations.

Single-candidate fast path
--------------------------
When hash filtering yields exactly one candidate, we skip building a
temporary index and compute the dot product directly, returning early if
the score is high enough. This handles the "exact duplicate in pool"
case with minimum overhead.
"""

import logging

import faiss
import numpy as np

from config import (
    EXACT_MATCH_THRESHOLD,
    FAISS_TOP_K,
    NEAR_DUP_THRESHOLD,
    SINGLE_CANDIDATE_EXACT_THR,
    SINGLE_CANDIDATE_NEAR_DUP_OUT,
    SINGLE_CANDIDATE_NEAR_DUP_THR,
)

logger = logging.getLogger(__name__)


# Return type alias for clarity in the pipeline.
# Maps original-index-position -> best cosine score across all rotations.
ScoreMap = dict[int, float]


def hash_filtered_search(
    query_vectors: np.ndarray,          # shape (4, D)
    candidate_indices: list[int],       # positions into the full index
    full_index: faiss.Index,            # the full FAISS index (for reconstruct)
    top_k: int = FAISS_TOP_K,
) -> ScoreMap:
    """
    Search only within the pHash-filtered candidate subset.

    Parameters
    ----------
    query_vectors : np.ndarray, shape (R, D)
        R rotation embeddings for the query image.
    candidate_indices : list[int]
        Row indices (into the full index) that passed pHash filtering.
    full_index : faiss.Index
        The loaded FAISS index used to reconstruct candidate vectors.
    top_k : int
        Number of nearest neighbours to retrieve per rotation.

    Returns
    -------
    ScoreMap
        {original_row_index -> best cosine score across all rotations}
    """
    n_candidates = len(candidate_indices)

    # --- Single-candidate fast path ---
    if n_candidates == 1:
        return _single_candidate_search(query_vectors, candidate_indices[0], full_index)

    # --- Multi-candidate path: build temporary sub-index ---
    logger.debug(
        "hash_filtered_search: building sub-index from %d candidates.", n_candidates
    )
    candidate_vectors = np.array(
        [full_index.reconstruct(int(i)) for i in candidate_indices],
        dtype="float32",
    )

    sub_index = faiss.IndexFlatIP(candidate_vectors.shape[1])
    sub_index.add(candidate_vectors)

    k = min(top_k, n_candidates)
    score_map: ScoreMap = {}

    for qv in query_vectors:
        distances, indices = sub_index.search(qv.reshape(1, -1), k)
        for rank in range(indices.shape[1]):
            sub_idx = int(indices[0][rank])
            if sub_idx == -1:
                continue
            original_idx = candidate_indices[sub_idx]
            score = float(distances[0][rank])
            # Keep only the best score per image across all rotations.
            if score > score_map.get(original_idx, -1.0):
                score_map[original_idx] = score

    logger.debug(
        "hash_filtered_search: %d results from %d candidates.", len(score_map), n_candidates
    )
    return score_map


def full_search(
    query_vectors: np.ndarray,          # shape (R, D)
    index: faiss.Index,
    top_k: int = FAISS_TOP_K,
) -> ScoreMap:
    """
    Full FAISS search with no pre-filtering (fallback path).

    Called when pHash filtering produced zero candidates.

    Parameters
    ----------
    query_vectors : np.ndarray, shape (R, D)
        R rotation embeddings for the query image.
    index : faiss.Index
        The full loaded FAISS index.
    top_k : int
        Number of nearest neighbours to retrieve per rotation.

    Returns
    -------
    ScoreMap
        {original_row_index -> best cosine score across all rotations}
    """
    logger.info("full_search: no hash candidates — searching full index (k=%d).", top_k)
    k = min(top_k, index.ntotal)
    score_map: ScoreMap = {}

    for qv in query_vectors:
        distances, indices = index.search(qv.reshape(1, -1), k)
        for rank in range(indices.shape[1]):
            idx = int(indices[0][rank])
            if idx == -1:
                continue
            score = float(distances[0][rank])
            if score > score_map.get(idx, -1.0):
                score_map[idx] = score

    logger.debug("full_search: %d results.", len(score_map))
    return score_map


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _single_candidate_search(
    query_vectors: np.ndarray,
    candidate_idx: int,
    full_index: faiss.Index,
) -> ScoreMap:
    """
    Fast path when pHash filtering returns exactly one candidate.

    Reconstructs the stored vector directly and computes dot products
    against all rotation embeddings without building a temporary index.
    Returns early with a pre-determined result so the pipeline can
    skip SIFT verification entirely.
    """
    # stored: shape (D,) — reconstruct returns a flat vector.
    stored = full_index.reconstruct(int(candidate_idx))          # (D,)
    best_score = max(float(np.dot(qv.flatten(), stored)) for qv in query_vectors)

    logger.debug(
        "_single_candidate_search: candidate %d, best score %.4f.",
        candidate_idx, best_score,
    )

    if best_score >= SINGLE_CANDIDATE_EXACT_THR:
        return {candidate_idx: best_score}

    if best_score >= SINGLE_CANDIDATE_NEAR_DUP_THR:
        # Return a synthetic near-duplicate score so downstream ranking
        # labels it correctly without further verification.
        return {candidate_idx: SINGLE_CANDIDATE_NEAR_DUP_OUT}

    # Score too low — return empty. The pipeline (pipeline.py Stage 6)
    # detects this empty map and falls back to full_search.
    return {}
