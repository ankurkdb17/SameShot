import os
import shutil
import glob
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# --- IMPORT YOUR ML MODULES ---
# We import the smart incremental function and the fast delete function
from main import process_reference_pool, remove_image_from_index,add_image_to_index
from imageHash import find_similar_images

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SETUP DIRECTORIES ---
UPLOAD_DIR = "uploads"
POOL_DIR = os.path.join(UPLOAD_DIR, "pool")
QUERY_DIR = os.path.join(UPLOAD_DIR, "query")

os.makedirs(POOL_DIR, exist_ok=True)
os.makedirs(QUERY_DIR, exist_ok=True)

# Mount images for frontend viewing
app.mount("/images", StaticFiles(directory="uploads"), name="images")

# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------
@app.post("/reset")
def reset_backend():
    """Clears all images AND FAISS index (full reset)"""

    # -----------------------------
    # 1. Clear pool images
    # -----------------------------
    for f in glob.glob(os.path.join(POOL_DIR, "*")):
        try:
            os.remove(f)
        except:
            pass

    # -----------------------------
    # 2. Clear query images
    # -----------------------------
    for f in glob.glob(os.path.join(QUERY_DIR, "*")):
        try:
            os.remove(f)
        except:
            pass

    # -----------------------------
    # 3. Clear FAISS index + metadata
    # -----------------------------
    FAISS_DIR = "dinov2_faiss_store"
    faiss_files = [
        "dinov2.index",
        "image_vectors.npy",
        "image_paths.npy",
        "image_hashes.npy",
    ]

    for fname in faiss_files:
        path = os.path.join(FAISS_DIR, fname)
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

    return {
        "status": "fully_reset",
        "pool_cleared": True,
        "query_cleared": True,
        "faiss_cleared": True
    }
@app.post("/upload/pool")
async def upload_pool(files: List[UploadFile] = File(...)):
    """Appends new images to the pool folder AND index instantly"""
    saved_files = []
    indexed_files = []

    for file in files:
        file_path = os.path.join(POOL_DIR, file.filename)

        # Save only if not already present
        if not os.path.exists(file_path):
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_files.append(file.filename)
            if not os.path.exists("dinov2_faiss_store/dinov2.index"):
                process_reference_pool(POOL_DIR)


            # 🚀 INSTANT INDEX UPDATE
            added = add_image_to_index(file_path)
            if added:
                indexed_files.append(file.filename)

    return {
        "added_to_disk": saved_files,
        "added_to_index": indexed_files,
        "count": len(saved_files)
    }


@app.delete("/delete/pool/{filename}")
def delete_pool_image(filename: str):
    """
    Deletes a specific image from the pool AND the index instantly.
    Uses 'remove_image_from_index' to avoid re-training the model.
    """
    file_path = os.path.join(POOL_DIR, filename)
    
    # 1. Remove the actual file from disk
    file_deleted = False
    if os.path.exists(file_path):
        os.remove(file_path)
        file_deleted = True

    # 2. Remove from Index (Fast, no retraining)
    index_updated = remove_image_from_index(filename)
    
    if file_deleted or index_updated:
        return {
            "status": "deleted", 
            "file": filename, 
            "disk_deleted": file_deleted,
            "index_updated": index_updated
        }
    
    return {"status": "not_found", "detail": "File not found on disk or index"}

@app.post("/upload/query")
async def upload_query(file: UploadFile = File(...)):
    """Replaces the query image (Delete old -> Save new)"""
    # 1. Delete existing query images
    for f in glob.glob(os.path.join(QUERY_DIR, "*")):
        try: os.remove(f)
        except: pass
    
    # 2. Save new one
    file_path = os.path.join(QUERY_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"status": "updated", "file": file.filename}

@app.delete("/delete/query")
def delete_query_image():
    """Clears the query image folder"""
    for f in glob.glob(os.path.join(QUERY_DIR, "*")):
        try: os.remove(f)
        except: pass
    return {"status": "cleared"}

# ---------------------------------------------------------
# ANALYZE ENDPOINT (SMART INCREMENTAL)
# ---------------------------------------------------------
@app.get("/analyze")
def trigger_analysis():
    """Runs the ML logic on whatever is currently in the folders"""
    
    # 1. Check if we have a query image
    query_files = glob.glob(os.path.join(QUERY_DIR, "*"))
    if not query_files:
        raise HTTPException(status_code=400, detail="No query image found on server.")
    
    query_path = query_files[0] 

    # 2. Run Smart Incremental Training
    # This will now return details about how many NEW images were added
    print(f"DEBUG: Checking pool status...")
    train_status = process_reference_pool(POOL_DIR)
    
    if train_status.get("status") == "error":
        print(f"ERROR in Training: {train_status.get('message')}")
        raise HTTPException(status_code=500, detail=train_status.get("message"))

    # Log the smart update details
    new_count = train_status.get("newly_added", 0)
    total_count = train_status.get("count", 0)
    if new_count > 0:
        print(f"⚡ INCREMENTAL UPDATE: Added {new_count} new images. Total Pool: {total_count}")
    else:
        print(f"✅ Index up-to-date. Using existing {total_count} images.")

    # 3. Search
    print(f"DEBUG: Searching for matches...")
    matches = find_similar_images(query_path, store_dir="dinov2_faiss_store")
    
    # --- SAFER PRINT LOGIC ---
    if matches is None:
        print("ERROR: imageHash.py returned None!")
        return {"results": []}
        
    if isinstance(matches, dict) and "error" in matches:
        print(f"ERROR from ML: {matches['error']}")
        return {"results": [], "error": matches["error"]}

    print(f"DEBUG: Found {len(matches)} matches.")
    # -------------------------
    
    return {"results": matches}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)