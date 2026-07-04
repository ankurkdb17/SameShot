"""
services/verification_service.py
---------------------------------
SIFT-based geometric verification — the final filter before ranking.

Responsibility
--------------
Given two image paths, determine whether one is a cropped or transformed
region of the other using SIFT keypoint matching and RANSAC homography.

When is this called?
--------------------
Only when the best FAISS cosine score across all results is below
SIFT_FALLBACK_THRESHOLD (0.90). In that case the pipeline cannot be
confident from embedding similarity alone and asks SIFT to confirm or
reject any candidate scoring above SIFT_MIN_CANDIDATE_SCORE (0.40).

This handles the specific case of a tightly cropped sub-image that shares
few global visual features with the full reference — embedding similarity
drops but SIFT keypoints still find the geometric correspondence.

Algorithm
---------
1. Detect SIFT keypoints and descriptors in both images.
2. BFMatcher with Lowe's ratio test to find reliable matches.
3. RANSAC homography to find the geometric transformation.
4. Accept if inlier count >= SIFT_MIN_INLIERS.
"""

import logging

import cv2
import numpy as np

from config import (
    SIFT_LOWE_RATIO,
    SIFT_MIN_GOOD_MATCHES,
    SIFT_MIN_INLIERS,
    SIFT_RANSAC_REPROJ_THRESH,
)

logger = logging.getLogger(__name__)


def verify_crop(reference_path: str, query_path: str) -> bool:
    """
    Return True if query_path appears to be a crop or sub-region of reference_path.

    Parameters
    ----------
    reference_path : str
        Path to the reference (pool) image.
    query_path : str
        Path to the query image.

    Returns
    -------
    bool
        True  → geometric correspondence confirmed (query is a crop of reference).
        False → not enough matches, homography failed, or image load error.
    """
    ref = cv2.imread(reference_path)
    qry = cv2.imread(query_path)

    if ref is None or qry is None:
        logger.warning(
            "verify_crop: could not load images — ref='%s', query='%s'.",
            reference_path, query_path,
        )
        return False

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    qry_gray = cv2.cvtColor(qry, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
    kp_qry, des_qry = sift.detectAndCompute(qry_gray, None)

    if des_ref is None or des_qry is None:
        logger.debug("verify_crop: no descriptors — too few keypoints.")
        return False

    des_ref = des_ref.astype(np.float32)
    des_qry = des_qry.astype(np.float32)

    # BFMatcher with L2 distance (correct for SIFT; NORM_HAMMING is for ORB).
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(des_ref, des_qry, k=2)

    # Lowe's ratio test: keep only matches where the best match is clearly
    # better than the second-best.
    good = [m for m, n in raw_matches if m.distance < SIFT_LOWE_RATIO * n.distance]

    if len(good) < SIFT_MIN_GOOD_MATCHES:
        logger.debug(
            "verify_crop: only %d good matches (need %d).", len(good), SIFT_MIN_GOOD_MATCHES
        )
        return False

    src_pts = np.float32([kp_qry[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, SIFT_RANSAC_REPROJ_THRESH)

    if H is None:
        logger.debug("verify_crop: homography could not be computed.")
        return False

    inliers = int(np.sum(mask))
    if inliers < SIFT_MIN_INLIERS:
        logger.debug(
            "verify_crop: only %d RANSAC inliers (need %d).", inliers, SIFT_MIN_INLIERS
        )
        return False

    logger.debug("verify_crop: confirmed with %d inliers.", inliers)
    return True
