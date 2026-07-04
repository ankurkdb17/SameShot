"""
config.py
---------
Single source of truth for all configuration in this application.

Every magic number, path, threshold, and model name lives here.
No other module should define these values — only import them.
"""

import os

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

DINO_MODEL_NAME = "dinov2_vitb14"

# Output dimension of dinov2_vitb14's CLS token.
# Used when initialising an empty numpy array before the index exists.
DINO_EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# Images are resized to this square before being passed to DINOv2.
DINO_INPUT_SIZE = 224

# ImageNet mean/std used by all torchvision DINOv2 checkpoints.
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD  = (0.229, 0.224, 0.225)

# Number of images sent to the GPU in one forward pass during index building.
EMBEDDING_BATCH_SIZE = 32

# Rotations applied to the query image so we catch rotated duplicates.
QUERY_ROTATION_ANGLES = [0, 90, 180, 270]

# Side length the query image is resized to before each rotation-embedding.
QUERY_ROTATION_SIZE = 500


# ---------------------------------------------------------------------------
# Similarity thresholds
# ---------------------------------------------------------------------------

# FAISS inner-product scores (cosine similarity after L2-normalisation).
EXACT_MATCH_THRESHOLD  = 0.98   # >= this -> "Exactly Same"
NEAR_DUP_THRESHOLD     = 0.89   # >= this -> "Near Duplicate"
MIN_SCORE_THRESHOLD    = 0.60   # below this -> result discarded entirely

# Threshold used to decide whether to run the SIFT crop-finding fallback.
# If the best FAISS score across all results is below this, SIFT is attempted.
SIFT_FALLBACK_THRESHOLD = 0.90

# Minimum FAISS score a candidate must have before SIFT is even attempted.
SIFT_MIN_CANDIDATE_SCORE = 0.40

# Score bands used when SIFT confirms a match (overrides the raw FAISS score).
SIFT_SCORE_BAND_LOW_MIN  = 0.40   # [0.40, 0.60) -> boosted to 0.80
SIFT_SCORE_BAND_LOW_MAX  = 0.60
SIFT_SCORE_BAND_LOW_OUT  = 0.80

SIFT_SCORE_BAND_MID_MIN  = 0.60   # [0.60, 0.80) -> boosted to 0.85
SIFT_SCORE_BAND_MID_MAX  = 0.80
SIFT_SCORE_BAND_MID_OUT  = 0.85

SIFT_SCORE_BAND_HIGH_MIN = 0.80   # [0.80, inf) -> boosted to 0.96
SIFT_SCORE_BAND_HIGH_OUT = 0.96


# ---------------------------------------------------------------------------
# Perceptual hash
# ---------------------------------------------------------------------------

# Maximum Hamming distance between two pHashes to be considered a candidate.
PHASH_HAMMING_THRESHOLD = 8


# ---------------------------------------------------------------------------
# FAISS retrieval
# ---------------------------------------------------------------------------

# Number of nearest neighbours returned by FAISS per query vector.
FAISS_TOP_K = 5

# Single-candidate fast-path thresholds (used when hash filtering yields exactly 1 result).
SINGLE_CANDIDATE_EXACT_THR    = 0.97
SINGLE_CANDIDATE_NEAR_DUP_THR = 0.70
SINGLE_CANDIDATE_NEAR_DUP_OUT = 0.90   # synthetic score returned in that case


# ---------------------------------------------------------------------------
# SIFT verification
# ---------------------------------------------------------------------------

SIFT_LOWE_RATIO           = 0.75   # Lowe's ratio test threshold
SIFT_MIN_GOOD_MATCHES     = 10     # minimum matches after ratio test
SIFT_MIN_INLIERS          = 8      # minimum RANSAC inliers
SIFT_RANSAC_REPROJ_THRESH = 5.0    # reprojection error tolerance (pixels)


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

# File extensions accepted by /upload/pool and /upload/query.
# Enforced at upload time (not just at indexing time) so arbitrary file
# types can never be written to disk via the API.
ALLOWED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Maximum accepted upload size per file, in bytes. Rejects oversized
# uploads before they are fully written to disk.
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

# Root directory for the FAISS index and numpy metadata files.
STORE_DIR = "dinov2_faiss_store"

INDEX_FILE   = os.path.join(STORE_DIR, "dinov2.index")
VECTORS_FILE = os.path.join(STORE_DIR, "image_vectors.npy")
PATHS_FILE   = os.path.join(STORE_DIR, "image_paths.npy")
HASHES_FILE  = os.path.join(STORE_DIR, "image_hashes.npy")

# Upload directories served by FastAPI's StaticFiles mount.
UPLOAD_DIR = "uploads"
POOL_DIR   = os.path.join(UPLOAD_DIR, "pool")
QUERY_DIR  = os.path.join(UPLOAD_DIR, "query")

# Extensions scanned when building the reference pool index.
POOL_IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

# Comma-separated list of allowed origins.
# Override at runtime via the CORS_ORIGINS environment variable, e.g.:
#   CORS_ORIGINS=http://localhost:5173,https://yourdomain.com
_cors_env = os.environ.get("CORS_ORIGINS", "http://localhost:5173")
CORS_ORIGINS: list[str] = [o.strip() for o in _cors_env.split(",") if o.strip()]
