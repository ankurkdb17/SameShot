# SameShot

Near-duplicate and exact-duplicate image detection. Upload a pool of reference
images, then check any query image against that pool — SameShot tells you if
it's an exact match, a near-duplicate (recompressed, resized, colour-shifted,
rotated), or a crop of something already in the pool.

```
Query image  ──►  [ pHash pre-filter ]  ──►  [ FAISS vector search ]  ──►  [ SIFT crop check ]  ──►  Ranked results
                    (fast, cheap)              (DINOv2 embeddings)          (only if score is low)
```

## Why this exists

Reverse image search is usually framed as "is this the same image." In
practice the interesting cases are the ambiguous ones: a screenshot rotated
90°, a JPEG re-saved at a different quality, or a tight crop of a much
larger photo. SameShot's pipeline is built specifically to handle those
three cases without needing a second model per case:

- **Rotation invariance** — the query image is embedded at 0°/90°/180°/270°
  and the best score across all four is used.
- **Compression/colour invariance** — DINOv2 embeddings are computed on a
  grayscale version of the image, so hue/saturation shifts and colour-mode
  differences (dark-mode screenshots, filters) don't move the embedding.
- **Crop detection** — a tight crop of a reference image can have low global
  embedding similarity (most of the reference frame is gone) but still share
  every local keypoint with the region it was cropped from. When FAISS scores
  come back low, SIFT + RANSAC geometric verification runs as a fallback to
  catch exactly this case.

## Architecture

```
backend/
├── app.py                     FastAPI app factory — the only entry point
├── config.py                  every threshold, path, and magic number lives here
├── security.py                filename sanitisation for client-supplied input
├── models/
│   └── dino_model.py          loads DINOv2 once; every service imports the singleton
├── routes/
│   └── images.py              thin HTTP handlers — no business logic
├── services/
│   ├── hash_service.py        perceptual hash (pHash) — fast pre-filter
│   ├── embedding_service.py   DINOv2 embeddings (batch + 4-rotation query)
│   ├── retrieval_service.py   FAISS search (hash-filtered subset or full index)
│   ├── verification_service.py  SIFT + RANSAC crop verification
│   ├── ranking_service.py     score classification, sorting, SIFT score-band boosting
│   └── pipeline.py            orchestrates all of the above — read this file first
├── storage/
│   └── index_store.py         the only module that touches disk (FAISS index + numpy arrays)
└── tests/                     pytest suite for the services/storage layer

frontend/
└── src/
    ├── pages/Home.tsx         landing page
    └── pages/DuplicateImg.tsx image pool UI, upload, analyze, results
```

**Design rule this codebase follows:** every module has exactly one reason to
change. `routes/` only knows about HTTP. `services/` only knows about the
detection algorithm. `storage/` only knows about reading and writing files.
`pipeline.py` is the one place that knows how the stages connect — reading it
top to bottom gives you the whole algorithm without touching any other file.

### Request pipeline (`services/pipeline.py`)

```
1. Validate         query image exists, FAISS index is initialised
2. Load index       read vectors/paths/hashes/FAISS index from disk
3 ║ 4. Parallel      pHash (thread 1)  ║  DINOv2 embedding × 4 rotations (thread 2)
5. Hash filter       keep only stored images within a Hamming-distance threshold
6. FAISS retrieval   search the filtered subset (fast path) or the full index (fallback)
7. SIFT verification only runs if the best FAISS score is below a threshold —
                     catches crop-matches that pass geometric but not embedding similarity
8. Rank              filter by minimum score, classify, sort, cap at top-K
9. Return            list[{name, score, status}]
```

Stages 3 and 4 run concurrently in a `ThreadPoolExecutor` — pHash (PIL, CPU)
and DINOv2 inference (torch) both release the GIL and depend only on the raw
query image, so there's no reason to serialise them.

## Getting started

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

The first request that needs the DINOv2 model will trigger a one-time
download from `facebookresearch/dinov2` via `torch.hub` (cached afterwards
in `~/.cache/torch/hub`).

API docs (Swagger UI): `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local      # adjust VITE_API_BASE_URL if the backend isn't on localhost:8000
npm run dev
```

Frontend: `http://localhost:5173`

### Docker (full stack)

```bash
docker compose up --build
```

Runs both services with persistent volumes for the reference pool and FAISS
index (`docker compose down -v` to wipe them). See `docker-compose.yml` for
environment variable overrides (CORS origins, API base URL).

### Running the tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

The suite covers the service and storage layers in isolation — hashing,
FAISS retrieval (including the single-candidate fast path), score
classification/ranking, SIFT verification failure modes, filename
sanitisation, and the index persistence layer — without needing a GPU or
downloading the DINOv2 model. `embedding_service.py` and `pipeline.py`
depend on the real model and are exercised via manual/integration testing
against a running server instead.

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload/pool` | Upload one or more reference images |
| `DELETE` | `/delete/pool/{filename}` | Remove a reference image from pool + index |
| `POST` | `/upload/query` | Replace the current query image |
| `DELETE` | `/delete/query` | Clear the query image |
| `GET` | `/analyze` | Run detection on the current query image |
| `POST` | `/reset` | Wipe all images and the FAISS index |

Uploads are validated against an allow-list of image extensions and a
20 MB size limit (`config.ALLOWED_UPLOAD_EXTENSIONS`,
`config.MAX_UPLOAD_SIZE_BYTES`), and every client-supplied filename is
sanitised (`security.safe_filename`) before touching the filesystem.

## Design decisions & tradeoffs

- **`IndexFlatIP` over an approximate index (HNSW/IVF):** exact search is
  intentionally chosen over approximate nearest-neighbour search. Reference
  pools here are expected to be small (thousands, not millions, of images),
  so exactness costs little and removes an entire class of recall bugs.
- **pHash as a pre-filter, not the primary signal:** perceptual hashing alone
  is too coarse to reliably distinguish "near duplicate" from "different but
  similar," but it's extremely cheap, so it's used to shrink the FAISS search
  space rather than to make the final decision.
- **SIFT only runs conditionally:** SIFT + RANSAC is the most expensive stage
  in the pipeline. It only runs when FAISS confidence is already low, so the
  common case (clear match or clear non-match) never pays for it.
- **Single-writer FAISS index, rebuilt on delete:** `IndexFlatIP` has no
  native delete operation, so removing an image rebuilds the index from the
  remaining vectors. This is O(n) but simple and correct; a production
  system with a much larger pool would use `IndexIDMap` with soft-deletes
  instead.

## Known limitations

- Single-process deployment: the DINOv2 model is loaded once per process, so
  the backend runs with `--workers 1`. Horizontal scaling would need the
  model + index behind a shared service rather than loaded per-worker.
- No authentication — this is a single-tenant tool as built. Anyone who can
  reach the API can upload to and query the pool.
- The reference pool lives on local disk / a Docker volume, not object
  storage — fine for a demo or single-machine deployment, not for multi-node.
