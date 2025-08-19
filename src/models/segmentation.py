import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, Optional

class ShoeSegmenter:
    def __init__(self, model_path: str = None):
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        self.model = YOLO(model_path)
    
    def segment_shoe(self, image: np.ndarray, confidence: float = 0.5) -> Tuple[np.ndarray, Optional[Tuple]]:
        if self.model is None:
            raise ValueError("Model not loaded")
        
        results = self.model(image, conf=confidence)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return image, None
        
        result = results[0]
        
        if result.masks is not None and len(result.masks) > 0:
            mask = result.masks[0].data.cpu().numpy()[0]
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                # Increased horizontal margin to fix unnecessary cropping
                margin = int(0.90 * w)
                x_new = max(x - margin, 0)
                w_new = min(w + 2 * margin, image.shape[1] - x_new)
                cropped = image[y:y+h, x_new:x_new+w_new]
                bbox = (x_new, y, w_new, h)
                return cropped, bbox
        
        elif result.boxes is not None and len(result.boxes) > 0:
            box = result.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            cropped = image[y1:y2, x1:x2]
            bbox = (x1, y1, x2-x1, y2-y1)
            
            return cropped, bbox
        
        return image, None
    
    def get_shoe_mask(self, image: np.ndarray, confidence: float = 0.5, horizontal_margin_ratio: float = 0.1) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded")

        results = self.model(image, conf=confidence)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if len(results) > 0 and results[0].masks is not None:
            result_mask = results[0].masks[0].data.cpu().numpy()[0]
            mask = cv2.resize(result_mask, (image.shape[1], image.shape[0]))
            mask = (mask > 0.5).astype(np.uint8) * 255

            # Dilate horizontally to match margin
            h, w = mask.shape
            margin = int(horizontal_margin_ratio * w)
            kernel = np.ones((1, max(1, margin)), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask