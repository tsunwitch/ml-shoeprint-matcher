import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Optional, Tuple

class FeatureDetector:
    def __init__(self, model_path: Optional[str] = None, use_sahi: bool = False):
        self.model = None
        self.use_sahi = use_sahi
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        if self.use_sahi:
            from sahi.model import Yolov8DetectionModel
            self.model = Yolov8DetectionModel(model_path=model_path, confidence_threshold=0.3, device="cpu")
        else:
            self.model = YOLO(model_path)
    
    def merge_overlapping_features(self, features: List[Tuple[float, float, float, float]], iou_threshold: float = 0.5) -> List[Tuple[float, float, float, float]]:
        from src.utils.image_ops import calculate_iou
        merged = []
        used = set()
        for i, box1 in enumerate(features):
            if i in used:
                continue
            group = [box1]
            for j, box2 in enumerate(features):
                if i != j and j not in used:
                    if calculate_iou(box1, box2) > iou_threshold:
                        group.append(box2)
                        used.add(j)
            # Merge group into one box (average coordinates)
            if len(group) == 1:
                merged.append(group[0])
            else:
                x1 = np.mean([b[0] for b in group])
                y1 = np.mean([b[1] for b in group])
                x2 = np.mean([b[2] for b in group])
                y2 = np.mean([b[3] for b in group])
                merged.append((x1, y1, x2, y2))
            used.add(i)
        return merged

    def detect_features(self, image: np.ndarray, confidence: float = 0.7) -> List[Tuple[float, float, float, float]]:
        if self.model is None:
            raise ValueError("Model not loaded")
        
        features = []
        if self.use_sahi:
            from sahi.utils.cv import read_image_as_pil
            from sahi.predict import get_sliced_prediction
            pil_img = read_image_as_pil(image)
            result = get_sliced_prediction(
                pil_img,
                self.model,
                slice_height=512,
                slice_width=512,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
            for det in result.object_prediction_list:
                x1, y1, x2, y2 = det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy
                features.append((float(x1), float(y1), float(x2), float(y2)))
        else:
            results = self.model(image, conf=confidence)
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    features.append((float(x1), float(y1), float(x2), float(y2)))
        # Merge overlapping features
        features = self.merge_overlapping_features(features, iou_threshold=0.5)
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