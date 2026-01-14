import cv2
import numpy as np
import requests
from insightface.app import FaceAnalysis


class FaceProcesser:
    def __init__(self):
        """Initialize the FaceAnalysis model (ArcFace)"""
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
    
    def get_embedding_from_url(self, image_url: str) -> np.ndarray:
        """
        Download an image from URL and extract face embedding
        
        Args:
            image_url: URL of the image
            
        Returns:
            np.ndarray: Face embedding (512-dimensional vector)
            
        Raises:
            ValueError: If no face detected in image
            requests.exceptions.RequestException: If URL fetch fails
        """
        # Download image from URL
        response = requests.get(image_url)
        response.raise_for_status()
        
        # Load image directly into memory
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError(f"Could not decode image from URL: {image_url}")
        
        # Detect faces + get embedding
        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("No face detected in the image.")
        
        # Pick the largest face (by bbox area)
        if len(faces) > 1:
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True
            )
        
        face = faces[0]
        
        # Get L2-normalized ArcFace embedding (512 dims)
        emb = face.normed_embedding.astype(np.float32)
        
        return emb



# def main():
#     image_path = "brad.png"

#     # 1) Load image (OpenCV loads as BGR)
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Could not read {image_path}. Make sure it's in this folder.")

#     # 2) Load InsightFace (ArcFace embedding model)
#     # ctx_id = -1 => CPU (Mac-friendly)
#     app = FaceAnalysis(name="buffalo_l")
#     app.prepare(ctx_id=-1, det_size=(640, 640))

#     # 3) Detect faces + get embedding
#     faces = app.get(img)
#     if len(faces) == 0:
#         raise ValueError("No face detected in the image.")
#     if len(faces) > 1:
#         # pick the largest face (by bbox area)
#         faces = sorted(
#             faces,
#             key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
#             reverse=True
#         )

#     face = faces[0]

#     # normed_embedding is L2-normalized ArcFace vector (usually 512 dims)
#     emb = face.normed_embedding.astype(np.float32)  # shape: (512,)

#     print("Embedding shape:", emb.shape)
#     print("First 10 values:", emb[:10])

#     # 4) Save to processed-face-data folder
#     output_dir = "processed-face-data"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Use input filename (without extension) for output
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
#     output_path = os.path.join(output_dir, f"{base_name}_arcface_512.npy")
    
#     np.save(output_path, emb)
#     print(f"Saved: {output_path}")
