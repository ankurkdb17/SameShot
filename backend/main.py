import torch
import numpy as np
import faiss
import glob
import os
import concurrent.futures
from PIL import Image
import torchvision.transforms as T
import imagehash

# ----------------------------
# DINOv2 SETUP
# ----------------------------

def get_dino_tools(model_name="dinov2_vitb14"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device)
    model.eval()

    # TRANSFORM: GRAYSCALE (Color Blind Mode)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    return model, transform, device

# --- LOAD MODEL GLOBALLY ---
print("⏳ Loading DINOv2 Model... (This may take a moment)")
MODEL, TRANSFORM, DEVICE = get_dino_tools()
print("✅ Model Loaded Successfully")


# ----------------------------
# OPTIMIZED EMBEDDING FUNCTION (BATCH)
# ----------------------------

def embed_images_batch(image_paths, model, transform, device, batch_size=32):
    embeddings = []
    total = len(image_paths)

    for i in range(0, total, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []

        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                img_tensor = transform(image)
                batch_tensors.append(img_tensor)
            except Exception as e:
                print(f"⚠️ Error loading {path}: {e}")

        if not batch_tensors:
            continue

        batch_stack = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            features = model(batch_stack)
            features = torch.nn.functional.normalize(features, dim=-1)
            embeddings.append(features.cpu().numpy())

        print(f"   Encoded {min(i + batch_size, total)}/{total} new images...")

    if not embeddings:
        return np.array([])
        
    return np.vstack(embeddings).astype("float32")


# ----------------------------
# PERCEPTUAL HASH FUNCTION
# ----------------------------

def compute_phash(image_path):
    try:
        img = Image.open(image_path).convert("L")
        return str(imagehash.phash(img))
    except:
        return "0000000000000000"


# ----------------------------
# SMART INCREMENTAL PROCESSING
# ----------------------------

def process_reference_pool(folder_path):
    print(f"📂 Scanning folder: {folder_path}")
    
    # 1. Get all files currently on disk
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    current_files = []
    for ext in extensions:
        current_files.extend(glob.glob(os.path.join(folder_path, ext)))
        current_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    current_files = sorted(list(set(current_files)))

    if not current_files:
        print(f"❌ No images found in '{folder_path}'")
        return {"status": "error", "message": "No images found"}

    # 2. Setup DB Paths
    save_dir = "dinov2_faiss_store"
    os.makedirs(save_dir, exist_ok=True)
    
    path_vectors = os.path.join(save_dir, "image_vectors.npy")
    path_paths = os.path.join(save_dir, "image_paths.npy")
    path_hashes = os.path.join(save_dir, "image_hashes.npy")
    path_index = os.path.join(save_dir, "dinov2.index")

    # 3. Load Existing Data (if available)
    old_vectors = np.empty((0, 768), dtype="float32") # 768 is DINOv2-ViT-B dimension
    old_paths = np.array([])
    old_hashes = np.array([])
    
    if os.path.exists(path_vectors) and os.path.exists(path_paths) and os.path.exists(path_hashes):
        try:
            old_vectors = np.load(path_vectors, allow_pickle=True)
            old_paths = np.load(path_paths, allow_pickle=True)
            old_hashes = np.load(path_hashes, allow_pickle=True)
            print(f"📚 Loaded existing index with {len(old_paths)} images.")
        except Exception as e:
            print(f"⚠️ Could not load existing index ({e}). Starting fresh.")

    # 4. Filter: Find which files are NEW
    existing_files_set = set(old_paths)
    files_to_process = [f for f in current_files if f not in existing_files_set]

    if not files_to_process:
        print("✅ Index is already up to date. No new images to add.")
        return {"status": "success", "count": len(current_files), "newly_added": 0}

    print(f"🚀 Found {len(files_to_process)} NEW images to process...")

    # 5. Process ONLY the new files
    new_vectors = embed_images_batch(files_to_process, MODEL, TRANSFORM, DEVICE, batch_size=32)

    if new_vectors.size == 0 and len(files_to_process) > 0:
         return {"status": "error", "message": "Failed to embed new images"}

    print("⚡ Generating hashes for new images...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        new_hashes = list(executor.map(compute_phash, files_to_process))

    # 6. Merge Old + New Data
    # If old data existed, stack new on top. If empty, just use new.
    if old_vectors.shape[0] > 0:
        final_vectors = np.vstack((old_vectors, new_vectors))
        final_paths = np.concatenate((old_paths, np.array(files_to_process)))
        final_hashes = np.concatenate((old_hashes, np.array(new_hashes)))
    else:
        final_vectors = new_vectors
        final_paths = np.array(files_to_process)
        final_hashes = np.array(new_hashes)

    # 7. Rebuild and Save FAISS Index
    print(f"💾 Saving updated index with {len(final_paths)} total images...")
    
    d = final_vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(final_vectors)

    faiss.write_index(index, path_index)
    np.save(path_vectors, final_vectors)
    np.save(path_paths, final_paths)
    np.save(path_hashes, final_hashes)

    print("✅ Index updated successfully.")
    
    return {
        "status": "success", 
        "count": len(final_paths),
        "newly_added": len(files_to_process)
    }


# ----------------------------
# DELETE WITHOUT RETRAINING
# ----------------------------

def remove_image_from_index(filename, store_dir="dinov2_faiss_store"):
    print(f"🗑️ Attempting to remove '{filename}' from index...")
    
    path_vectors = os.path.join(store_dir, "image_vectors.npy")
    path_paths = os.path.join(store_dir, "image_paths.npy")
    path_hashes = os.path.join(store_dir, "image_hashes.npy")
    path_index = os.path.join(store_dir, "dinov2.index")

    if not (os.path.exists(path_vectors) and os.path.exists(path_paths)):
        return False

    try:
        vectors = np.load(path_vectors, allow_pickle=True)
        paths = np.load(path_paths, allow_pickle=True)
        hashes = np.load(path_hashes, allow_pickle=True)

        idx_to_remove = -1
        for i, p in enumerate(paths):
            if os.path.basename(p) == filename:
                idx_to_remove = i
                break
        
        if idx_to_remove == -1:
            return False

        print(f"Found at index {idx_to_remove}. Removing...")
        new_vectors = np.delete(vectors, idx_to_remove, axis=0)
        new_paths = np.delete(paths, idx_to_remove, axis=0)
        new_hashes = np.delete(hashes, idx_to_remove, axis=0)

        d = new_vectors.shape[1]
        new_index = faiss.IndexFlatIP(d)
        new_index.add(new_vectors)

        faiss.write_index(new_index, path_index)
        np.save(path_vectors, new_vectors)
        np.save(path_paths, new_paths)
        np.save(path_hashes, new_hashes)

        print(f"✅ Successfully removed '{filename}'.")
        return True

    except Exception as e:
        print(f"❌ Error updating index: {e}")
        return False

def add_image_to_index(image_path, store_dir="dinov2_faiss_store"):
    print(f"➕ Adding new image to index: {image_path}")

    path_vectors = os.path.join(store_dir, "image_vectors.npy")
    path_paths   = os.path.join(store_dir, "image_paths.npy")
    path_hashes  = os.path.join(store_dir, "image_hashes.npy")
    path_index   = os.path.join(store_dir, "dinov2.index")

    # Safety check
    if not os.path.exists(image_path):
        print("❌ Image does not exist.")
        return False

    # Load existing data
    if not (os.path.exists(path_vectors) and os.path.exists(path_paths) and os.path.exists(path_hashes)):
        print("❌ Index not found. Run process_reference_pool() first.")
        return False

    vectors = np.load(path_vectors, allow_pickle=True)
    paths   = np.load(path_paths, allow_pickle=True)
    hashes  = np.load(path_hashes, allow_pickle=True)

    # Avoid duplicates
    if image_path in paths:
        print("⚠️ Image already exists in index.")
        return False

    # Embed image
    new_vector = embed_images_batch(
        [image_path],
        MODEL,
        TRANSFORM,
        DEVICE,
        batch_size=1
    )

    if new_vector.size == 0:
        print("❌ Failed to embed image.")
        return False

    # Compute pHash
    new_hash = compute_phash(image_path)

    # Append to numpy stores
    vectors = np.vstack((vectors, new_vector))
    paths   = np.append(paths, image_path)
    hashes  = np.append(hashes, new_hash)

    # Update FAISS index (fast append)
    d = vectors.shape[1]
    index = faiss.read_index(path_index)
    index.add(new_vector)

    # Save everything
    faiss.write_index(index, path_index)
    np.save(path_vectors, vectors)
    np.save(path_paths, paths)
    np.save(path_hashes, hashes)

    print("✅ Image added successfully.")
    return True


if __name__ == "__main__":
    process_reference_pool("uploads/pool")