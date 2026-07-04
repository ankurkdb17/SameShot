import torch
import numpy as np
import faiss
import os
from PIL import Image
import torchvision.transforms as T
import imagehash
import cv2

def cropfinds(path1,path2):

    img_orig = cv2.imread(path1)
    img_crop = cv2.imread(path2)

    if img_orig is None or img_crop is None:
      return False

    gray1 = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)


    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

# Check keypoints / descriptors
    if des1 is None or des2 is None:
        return False
      

# Ensure correct dtype (VERY IMPORTANT)
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

# BFMatcher for SIFT (L2 distance)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

# Lowe ratio test
    good = []
    for pair in matches:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

# Minimum matches check
    if len(good) < 10:
        return False

# Extract matched points
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)

# Homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        return False

# Inlier check
    inliers = int(np.sum(mask))
    if inliers < 8:
        return False

    return True
def get_dino_tools(model_name="dinov2_vitb14"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device)
    model.eval()

    # UPDATED TRANSFORM: GRAYSCALE (Color Blind Mode)
    # This matches the logic in your main.py
    transform = T.Compose([
        T.Resize((224, 224)),
        
        # This makes the AI "color blind" so it sees Dark/Light mode as the same.
        T.Grayscale(num_output_channels=3),
        
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    return model, transform, device

print("⏳ Loading DINOv2 Model for Search...")
MODEL, TRANSFORM, DEVICE = get_dino_tools()
print("✅ Search Model Loaded")


def embed_query_with_rotations(image_path, model, transform, device):
    """
    Generates 4 embeddings for the query image (0, 90, 180, 270 degrees).
    The transform will convert them to Grayscale automatically.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        angles = [0, 90, 180, 270]

        vectors = []
        for angle in angles:
            # expand=True ensures we don't crop corners when rotating
            rotated = image.rotate(angle, expand=True)
            
            # Resize BEFORE transform to keep consistent logic
            rotated_resized = rotated.resize((500, 500)) 
            
            # Transform (includes Grayscale conversion)
            img_tensor = transform(rotated_resized).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model(img_tensor)
                features = torch.nn.functional.normalize(features, dim=-1)

            vectors.append(features.cpu().numpy())

        return np.vstack(vectors).astype("float32")  # Returns shape (4, 768)
    except Exception as e:
        print(f"Error rotating/embedding query: {e}")
        return np.array([])


def compute_phash(image_path):
    try:
        # 👇 UPDATED: Convert to Grayscale ('L') before hashing
        # This ensures the hash ignores color information completely
        img = Image.open(image_path).convert("L")
        return imagehash.phash(img)
    except:
        return None


def find_similar_images(query_image_path, store_dir="dinov2_faiss_store"):
    index_path = os.path.join(store_dir, "dinov2.index")

    if not os.path.exists(index_path):
        return {"error": "Index not found. Upload reference images first."}

    # Load stored data
    try:
        image_files = np.load(os.path.join(store_dir, "image_paths.npy"), allow_pickle=True)
        image_hashes = np.load(os.path.join(store_dir, "image_hashes.npy"), allow_pickle=True)
        index = faiss.read_index(index_path)
    except Exception as e:
        return {"error": f"Failed to load index: {e}"}

    if not os.path.exists(query_image_path):
        return {"error": "Query image not found."}


    query_hash = compute_phash(query_image_path)
    if query_hash is None:
        return {"error": "Failed to compute pHash."}

    # 🔥 GENERATE 4 VECTORS (Original + 3 Rotations)
    # These will be Grayscale thanks to the transform
    query_vectors = embed_query_with_rotations(
        query_image_path, MODEL, TRANSFORM, DEVICE
    )
    
    if query_vectors.size == 0:
        return {"error": "Failed to process query image embeddings."}

  

    HASH_THRESHOLD = 8
    candidate_indices = []

    for i, h in enumerate(image_hashes):
        try:
            stored_hash = imagehash.hex_to_hash(str(h))
            if query_hash - stored_hash <= HASH_THRESHOLD:
                candidate_indices.append(i)
        except:
            continue

    print(f"⚡ Hash candidates found: {len(candidate_indices)}")


    # Thresholds (Grayscale should give high similarity for color-shifted duplicates)
    EXACT_MATCH_THR = 0.98
    NEAR_DUP_THR = 0.89

    def classify_score(s):
        if s >= EXACT_MATCH_THR: return "Exactly Same"
        if s >= NEAR_DUP_THR: return "Near Duplicate"
        return "Different"

    final_results_map = {} 

    def search_vectors(search_index, search_k, id_map_func):
        # Check all 4 query rotations against the index
        for qv in query_vectors:
            dists, idxs = search_index.search(qv.reshape(1, -1), search_k)
            
            for rank in range(idxs.shape[1]):
                found_idx = idxs[0][rank]
                if found_idx == -1: continue
                
                real_idx = id_map_func(found_idx)
                score = float(dists[0][rank])
                
                # Update if this rotation gave a better score for this image
                if real_idx in final_results_map:
                    if score > final_results_map[real_idx]:
                        final_results_map[real_idx] = score
                else:
                    final_results_map[real_idx] = score

    if candidate_indices:
        # SEARCH SUBSET (Hash Candidates)
        if(len(candidate_indices)==1):
            idx = candidate_indices[0]
            stored_vector = index.reconstruct(int(idx)).reshape(1, -1)
            best_score = 0.0
            for qv in query_vectors:
                score =float(np.dot(qv, stored_vector.T))
                if score > best_score:
                    best_score = score
            if(best_score>=.97):
                return [{
                "name": os.path.basename(image_files[idx]),
                "score": round(best_score, 4),
                "status": "Exactly Same"
                       }]
            elif(best_score>=.7):
                return [{
                "name": os.path.basename(image_files[idx]),
                "score": round(.9, 4),
                "status": "Near Duplicate"
                       }]

        
        print("🔎 Searching within hash-filtered candidates")
        candidate_vectors = np.array([index.reconstruct(int(i)) for i in candidate_indices])
        
        temp_index = faiss.IndexFlatIP(candidate_vectors.shape[1])
        temp_index.add(candidate_vectors)

        k = min(5, len(candidate_indices))
        id_mapper = lambda temp_id: candidate_indices[temp_id]
        search_vectors(temp_index, k, id_mapper)

    else:
        # SEARCH ALL (Fallback)
        print("⚠ No hash matches found, doing full FAISS search")
        k = 5
        id_mapper = lambda real_id: real_id
        search_vectors(index, k, id_mapper)

  
    results = []
    sorted_matches = sorted(final_results_map.items(), key=lambda x: -x[1])
    maxi=0
    for idx, score in sorted_matches:
        if score < 0.60: 
            continue
        maxi=max(maxi,score)

        results.append({
            "name": os.path.basename(image_files[idx]),
            "score": round(score, 4),
            "status": classify_score(score)
        })
    print(maxi)    
    if(maxi<0.9):
        for idx,score in sorted_matches:
            path1=image_files[idx]
            path2=query_image_path
            if(score<0.4):
                continue
            res=cropfinds(path1,path2)
            print(res)
            if 0.6 <= score < 0.96:
               status = "Near Duplicate"
            elif score < 0.6:
               status = "This might match your image"
            else:
               status = "Exactly Same"
            print(res)   

            if(score>=0.4 and score<0.6):
                score=0.8
            elif(score>=0.6 and score<0.8):
                score=0.85
            elif(score>=0.8):
                score=0.96        
            if(res):
                return[{
                "name": os.path.basename(image_files[idx]),
                "score": round(score, 4),
                "status": status
                }]
            
    
    return results[:5]

if __name__ == "__main__":
    print(find_similar_images("uploads/query/test_image.jpg"))
