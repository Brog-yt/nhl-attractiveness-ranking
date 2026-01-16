import cv2
import numpy as np
import requests
from pathlib import Path
from typing import Union
from insightface.app import FaceAnalysis


class FaceProcesser:
    def __init__(self):
        """Initialize the FaceAnalysis model (ArcFace)"""
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.1)
    
    def _get_embedding_from_bgr_image(self, img: np.ndarray) -> np.ndarray:
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            raise ValueError("Invalid image array provided (img is None/empty).")
        
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
    
    def get_embedding_from_url(self, image_url: str, timeout: float = 15.0) -> np.ndarray:
        """
        Download an image from URL and extract face embedding
        
        Args:
            image_url: URL of the image
            timeout: requests timeout in seconds
            
        Returns:
            np.ndarray: Face embedding (512-dimensional vector)
            
        Raises:
            ValueError: If no face detected in image
            requests.exceptions.RequestException: If URL fetch fails
        """
        # Download image from URL
        response = requests.get(image_url, timeout=timeout)
        response.raise_for_status()
        
        # Load image directly into memory
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError(f"Could not decode image from URL: {image_url}")
        
        return self._get_embedding_from_bgr_image(img)
    
    def get_embedding_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load a local image file from disk and extract face embedding.
        Perfect for Kaggle datasets (Images/*.jpg).
        
        Args:
            image_path: Path to image file
            
        Returns:
            np.ndarray: Face embedding (512-dimensional vector)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If OpenCV can't read it or no face detected
        """
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {p}")
        
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read/decode image from path: {p}")
        
        return self._get_embedding_from_bgr_image(img)
