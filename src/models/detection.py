import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Optional, Tuple

class FeatureDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        self.model = YOLO(model_path)
    
    def detect_features(self, image: np.ndarray, confidence: float = 0.7) -> List[Tuple[float, float, float, float]]:
        if self.model is None:
            raise ValueError("Model not loaded")
        
        results = self.model(image, conf=confidence)
        
        features = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                features.append((float(x1), float(y1), float(x2), float(y2)))
        
        return features
    
    def extract_feature_patches(self, image: np.ndarray, features: List[Tuple]) -> List[np.ndarray]:
        patches = []
        for x1, y1, x2, y2 in features:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            patch = image[y1:y2, x1:x2]
            if patch.size > 0:
                patches.append(patch)
        
        return patches
    
    def compute_feature_descriptors(self, patches: List[np.ndarray]) -> np.ndarray:
        descriptors = []
        
        for patch in patches:
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
            
            patch_resized = cv2.resize(patch_gray, (32, 32))
            
            hist = cv2.calcHist([patch_resized], [0], None, [32], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            descriptors.append(hist)
        
        return np.array(descriptors) if descriptors else np.array([])