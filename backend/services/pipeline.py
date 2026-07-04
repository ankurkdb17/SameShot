"""
services/pipeline.py
--------------------
DuplicateDetectionPipeline — the single entry point for image search.

This class is the only place that knows how the detection stages connect.
Reading it top-to-bottom gives a complete picture of the algorithm.

Pipeline
--------

  DuplicateDetectionPipeline.run(query_image_path)
      │
      ├── Stage 1: Validate          — image exists, index is ready
      │
      ├── Stage 2: Preprocess        — load image into PIL (shared by stages 3 & 4)
      │
      ├── Stage 3 ║ Stage 4          ← PARALLEL (ThreadPoolExecutor, 2 workers)
      │   ├── [Thread 1] Hash        — compute pHash of query image
      │   └── [Thread 2] Embed       — generate 4 rotation embeddings via DINOv2
      │
      ├── Stage 5: Generate Candidates — pHash filter against stored hashes
      │
      ├── Stage 6: FAISS Retrieval   — search hash-filtered subset (or full index)
      │
      ├── Stage 7: SIFT Verification — geometric check if FAISS scores are low
      │
      ├── Stage 8: Ranking           — sort, filter, build response dicts
      │
      └── Stage 9: Return JSON       — list[dict] ready for the route handler

Parallel stage justification
-----------------------------
pHash computation (CPU-bound, PIL + imagehash) and DINOv2 embedding
(torch.no_grad forward pass, GPU or CPU) share only the raw image file
as input and produce completely independent outputs. They are the only
genuinely parallel tasks in the pipeline. Everything downstream of stage
4 depends on both outputs, so parallelism stops there.

ThreadPoolExecutor with max_workers=2 is the right primitive:
- Both tasks release the GIL (PIL I/O + numpy/torch ops).
- No async complexity, no process overhead.
- Latency drops by ~30-50% on CPU; more on GPU where embedding dominates.

Usage
-----
    from models.dino_model import MODEL, TRANSFORM, DEVICE
    from services.pipeline import DuplicateDetectionPipeline

    pipeline = DuplicateDetectionPipeline(MODEL, TRANSFORM, DEVICE)
    results  = pipeline.run("uploads/query/photo.jpg")
    # results: list[dict] e.g. [{"name": "ref.jpg", "score": 0.97, "status": "Exactly Same"}]
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from config import SIFT_FALLBACK_THRESHOLD, SIFT_MIN_CANDIDATE_SCORE
from services.embedding_service import embed_query
from services.hash_service import compute_hash, filter_candidates
from services.ranking_service import rank_results, rank_sift_result
from services.retrieval_service import full_search, hash_filtered_search
from services.verification_service import verify_crop
from storage import index_store

logger = logging.getLogger(__name__)


class DuplicateDetectionPipeline:
    """
    Orchestrates the near-duplicate image detection pipeline.

    Parameters
    ----------
    model : torch.nn.Module
        DINOv2 model in eval mode (from models/dino_model.py).
    transform : torchvision.transforms.Compose
        Preprocessing pipeline (from models/dino_model.py).
    device : str
        "cuda" or "cpu" (from models/dino_model.py).
    """

    def __init__(self, model: torch.nn.Module, transform, device: str) -> None:
        self._model     = model
        self._transform = transform
        self._device    = device

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, query_image_path: str) -> list[dict]:
        """
        Run the full detection pipeline on a query image.

        Parameters
        ----------
        query_image_path : str
            Path to the query image on disk.

        Returns
        -------
        list[dict]
            Up to FAISS_TOP_K results, each:
            {"name": str, "score": float, "status": str}
            Sorted by score descending.

        Raises
        ------
        FileNotFoundError
            If the query image or the FAISS index does not exist.
        RuntimeError
            If embedding fails entirely (all rotations raise exceptions).
        """
        total_start = time.perf_counter()
        logger.info("Pipeline started for '%s'.", query_image_path)

        # ----------------------------------------------------------
        # Stage 1: Validate
        # ----------------------------------------------------------
        self._validate(query_image_path)

        # ----------------------------------------------------------
        # Stage 2: Load index data
        # ----------------------------------------------------------
        t = time.perf_counter()
        data = index_store.load()
        logger.debug("Stage 2 (load index): %.3fs — %d images.", time.perf_counter() - t, len(data.paths))

        # ----------------------------------------------------------
        # Stage 3 ║ Stage 4 — PARALLEL
        #   Thread 1: compute pHash
        #   Thread 2: generate 4-rotation DINOv2 embeddings
        # ----------------------------------------------------------
        t = time.perf_counter()
        query_hash, query_vectors = self._parallel_hash_and_embed(query_image_path)
        parallel_time = time.perf_counter() - t
        logger.debug(
            "Stage 3+4 (parallel hash+embed): %.3fs.", parallel_time
        )

        if query_vectors is None:
            raise RuntimeError(
                f"Embedding failed for '{query_image_path}' — "
                "all rotation attempts raised exceptions."
            )

        # ----------------------------------------------------------
        # Stage 5: Generate candidates via pHash filter
        # ----------------------------------------------------------
        t = time.perf_counter()
        candidate_indices = filter_candidates(query_hash, data.hashes)
        logger.debug(
            "Stage 5 (hash filter): %.3fs — %d / %d candidates.",
            time.perf_counter() - t, len(candidate_indices), len(data.hashes),
        )

        # ----------------------------------------------------------
        # Stage 6: FAISS retrieval
        #
        # Two paths:
        #   a) Hash candidates exist → search the filtered subset only.
        #      If the single-candidate fast path scores too low it returns
        #      an empty map — we fall through to full search so a bad hash
        #      hit never silently produces zero results.
        #   b) No hash candidates → search the full index directly.
        # ----------------------------------------------------------
        t = time.perf_counter()
        if candidate_indices:
            logger.info(
                "Stage 6: hash-filtered search (%d candidates).", len(candidate_indices)
            )
            score_map = hash_filtered_search(
                query_vectors, candidate_indices, data.index
            )
            # FIX BUG-2: if hash-filtered search returned nothing (single
            # low-scoring candidate fast path), fall back to full search
            # rather than silently returning empty results.
            if not score_map:
                logger.info(
                    "Stage 6: hash-filtered search yielded no results — "
                    "falling back to full FAISS search."
                )
                score_map = full_search(query_vectors, data.index)
        else:
            logger.info("Stage 6: no hash candidates — full FAISS search.")
            score_map = full_search(query_vectors, data.index)
        logger.debug("Stage 6 (FAISS retrieval): %.3fs.", time.perf_counter() - t)

        # ----------------------------------------------------------
        # Stage 7: SIFT verification (conditional)
        #
        # Only runs when the best score across all FAISS results is below
        # SIFT_FALLBACK_THRESHOLD. This handles crop-match cases where
        # embedding similarity is low but geometric correspondence exists.
        # ----------------------------------------------------------
        t = time.perf_counter()
        sift_result = self._maybe_run_sift(score_map, data.paths, query_image_path)
        logger.debug("Stage 7 (SIFT verification): %.3fs.", time.perf_counter() - t)

        if sift_result is not None:
            total_time = time.perf_counter() - total_start
            logger.info(
                "Pipeline complete (SIFT path): %.3fs — 1 result.", total_time
            )
            return sift_result

        # ----------------------------------------------------------
        # Stage 8: Ranking
        # ----------------------------------------------------------
        t = time.perf_counter()
        results = rank_results(score_map, data.paths)
        logger.debug("Stage 8 (ranking): %.3fs — %d results.", time.perf_counter() - t, len(results))

        # ----------------------------------------------------------
        # Stage 9: Return
        # ----------------------------------------------------------
        total_time = time.perf_counter() - total_start
        logger.info(
            "Pipeline complete (FAISS path): %.3fs — %d result(s).",
            total_time, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Private stage helpers
    # ------------------------------------------------------------------

    def _validate(self, query_image_path: str) -> None:
        """Stage 1: Raise early if pre-conditions are not met."""
        if not os.path.exists(query_image_path):
            raise FileNotFoundError(f"Query image not found: '{query_image_path}'")

        if not index_store.is_initialised():
            raise FileNotFoundError(
                "FAISS index is not initialised. "
                "Upload reference images to the pool first."
            )
        logger.debug("Stage 1 (validate): OK.")

    def _parallel_hash_and_embed(self, query_image_path: str):
        """
        Stage 3 ║ Stage 4 — run pHash and DINOv2 embedding concurrently.

        Returns
        -------
        tuple[imagehash.ImageHash, np.ndarray | None]
            (query_hash, query_vectors)
            query_vectors is None if embedding failed entirely.
        """
        query_hash    = None
        query_vectors = None

        def compute_hash_task():
            t = time.perf_counter()
            h = compute_hash(query_image_path)
            logger.debug("  [Thread-Hash]  pHash done in %.3fs.", time.perf_counter() - t)
            return h

        def embed_task():
            t = time.perf_counter()
            v = embed_query(
                query_image_path,
                self._model,
                self._transform,
                self._device,
            )
            logger.debug("  [Thread-Embed] embedding done in %.3fs.", time.perf_counter() - t)
            return v

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_hash  = executor.submit(compute_hash_task)
            future_embed = executor.submit(embed_task)

            # Collect both results; raise immediately if either future failed.
            for future in as_completed([future_hash, future_embed]):
                exc = future.exception()
                if exc:
                    logger.error("Parallel stage raised an exception: %s", exc)
                    raise exc

        query_hash    = future_hash.result()
        query_vectors = future_embed.result()

        return query_hash, query_vectors

    def _maybe_run_sift(
        self,
        score_map: dict,
        image_paths,
        query_image_path: str,
    ):
        """
        Stage 7: Conditionally run SIFT verification.

        Returns a ranked result list if SIFT confirms a match, or None
        if SIFT was not needed / did not find a match (normal FAISS path
        continues).

        SIFT is triggered when:
          - score_map is non-empty, AND
          - the best FAISS score is below SIFT_FALLBACK_THRESHOLD

        SIFT is attempted on candidates scoring above SIFT_MIN_CANDIDATE_SCORE,
        in descending score order. The first confirmed match is returned
        immediately (no need to check weaker candidates once one is confirmed).
        """
        if not score_map:
            return None

        best_score = max(score_map.values())

        if best_score >= SIFT_FALLBACK_THRESHOLD:
            logger.debug(
                "Stage 7 (SIFT): skipped — best score %.4f >= threshold %.4f.",
                best_score, SIFT_FALLBACK_THRESHOLD,
            )
            return None

        logger.info(
            "Stage 7 (SIFT): best FAISS score %.4f < %.4f — attempting SIFT verification.",
            best_score, SIFT_FALLBACK_THRESHOLD,
        )

        sorted_candidates = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

        for idx, score in sorted_candidates:
            if score < SIFT_MIN_CANDIDATE_SCORE:
                break   # remaining candidates are below the minimum; stop

            reference_path = str(image_paths[idx])
            logger.debug(
                "  SIFT: checking '%s' (score=%.4f).", reference_path, score
            )

            if verify_crop(reference_path, query_image_path):
                logger.info(
                    "  SIFT: confirmed match — '%s' (raw score=%.4f).",
                    reference_path, score,
                )
                return rank_sift_result(reference_path, score)

        logger.debug("Stage 7 (SIFT): no crop match confirmed.")
        return None
