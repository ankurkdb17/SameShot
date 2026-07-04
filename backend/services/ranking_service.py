"""
services/ranking_service.py
----------------------------
Score classification and result ranking — the final stage before the response.

Responsibility
--------------
Given a ScoreMap {index -> cosine_score}, return a sorted list of result dicts
ready to be returned as JSON by the route handler.

Two ranking paths
-----------------
rank_results()
    Standard path: filter by MIN_SCORE_THRESHOLD, classify each score,
    sort descending, return top FAISS_TOP_K results.

rank_sift_result()
    Called when SIFT verification confirms a crop match. Overrides the raw
    FAISS score with a boosted value based on the score band, and forces the
    status label to reflect the strength of the match.

Score bands (SIFT path only)
-----------------------------
The raw cosine score for a crop-match can be misleadingly low because
the query covers only a small region of the reference image, reducing
global embedding similarity. The bands compensate for this:

    [0.40, 0.60) -> 0.80  (plausible match, SIFT confirmed)
    [0.60, 0.80) -> 0.85  (likely match, SIFT confirmed)
    [0.80, inf)  -> 0.96  (strong match, SIFT confirmed)
"""

import logging
import os

from config import (
    EXACT_MATCH_THRESHOLD,
    FAISS_TOP_K,
    MIN_SCORE_THRESHOLD,
    NEAR_DUP_THRESHOLD,
    SIFT_SCORE_BAND_HIGH_MIN,
    SIFT_SCORE_BAND_HIGH_OUT,
    SIFT_SCORE_BAND_LOW_MAX,
    SIFT_SCORE_BAND_LOW_MIN,
    SIFT_SCORE_BAND_LOW_OUT,
    SIFT_SCORE_BAND_MID_MAX,
    SIFT_SCORE_BAND_MID_MIN,
    SIFT_SCORE_BAND_MID_OUT,
)

logger = logging.getLogger(__name__)

# Type alias: one result dict as returned to the API caller.
ResultDict = dict[str, object]


def classify_score(score: float) -> str:
    """
    Map a cosine similarity score to a human-readable status label.

    Parameters
    ----------
    score : float
        Cosine similarity in [0, 1] (inner product of L2-normalised vectors).

    Returns
    -------
    str
        "Exactly Same", "Near Duplicate", or "Different".
    """
    if score >= EXACT_MATCH_THRESHOLD:
        return "Exactly Same"
    if score >= NEAR_DUP_THRESHOLD:
        return "Near Duplicate"
    return "Different"


def rank_results(
    score_map: dict[int, float],
    image_paths,                    # np.ndarray of path strings
    min_score: float = MIN_SCORE_THRESHOLD,
    top_k: int = FAISS_TOP_K,
) -> list[ResultDict]:
    """
    Convert a ScoreMap into a sorted, filtered list of result dicts.

    Parameters
    ----------
    score_map : dict[int, float]
        {original_index -> best cosine score} from the retrieval stage.
    image_paths : array-like of str
        Full path for each image in the index (same ordering as vectors).
    min_score : float
        Results below this threshold are discarded entirely.
    top_k : int
        Maximum number of results to return.

    Returns
    -------
    list[ResultDict]
        Each dict: {"name": filename, "score": float, "status": str}
        Sorted by score descending.
    """
    sorted_matches = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    results = []

    for idx, score in sorted_matches:
        if score < min_score:
            continue
        results.append({
            "name":   os.path.basename(image_paths[idx]),
            "score":  round(score, 4),
            "status": classify_score(score),
        })

    capped = results[:top_k]
    logger.debug("rank_results: %d results after filtering (top %d).", len(results), top_k)
    return capped


def rank_sift_result(
    reference_path: str,
    raw_score: float,
) -> list[ResultDict]:
    """
    Build a single-item result list for a SIFT-confirmed crop match.

    Applies score-band boosting to compensate for the low cosine similarity
    that is typical when the query is a cropped sub-region of the reference.

    Parameters
    ----------
    reference_path : str
        Full path to the reference (pool) image that SIFT confirmed.
    raw_score : float
        The original cosine similarity score from FAISS for this image.

    Returns
    -------
    list[ResultDict]
        A list containing exactly one result dict.
    """
    boosted_score, status = _apply_sift_band(raw_score)

    result: ResultDict = {
        "name":   os.path.basename(reference_path),
        "score":  round(boosted_score, 4),
        "status": status,
    }
    logger.debug(
        "rank_sift_result: '%s' raw=%.4f boosted=%.4f status='%s'.",
        result["name"], raw_score, boosted_score, status,
    )
    return [result]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_sift_band(score: float) -> tuple[float, str]:
    """
    Return (boosted_score, status_label) for a SIFT-confirmed match.

    The status label reflects the strength of the boosted score rather than
    the raw FAISS score, which can be misleadingly low for crop-matches.
    """
    if SIFT_SCORE_BAND_LOW_MIN <= score < SIFT_SCORE_BAND_LOW_MAX:
        return SIFT_SCORE_BAND_LOW_OUT, "This might match your image"

    if SIFT_SCORE_BAND_MID_MIN <= score < SIFT_SCORE_BAND_MID_MAX:
        return SIFT_SCORE_BAND_MID_OUT, "Near Duplicate"

    if score >= SIFT_SCORE_BAND_HIGH_MIN:
        return SIFT_SCORE_BAND_HIGH_OUT, "Exactly Same"

    # Score below all bands — still confirmed by SIFT but very low confidence.
    return score, "This might match your image"
