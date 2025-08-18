import cv2
import numpy as np
from typing import Tuple, List, Optional

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, target_size)

def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def extract_axis_profile(image: np.ndarray, axis_line: Tuple[Tuple[float, float], Tuple[float, float]], 
                         num_samples: int = 100, mask: Optional[np.ndarray] = None) -> np.ndarray:
    start_point, end_point = axis_line
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    profile = []
    for i in range(num_samples):
        t = i / (num_samples - 1)
        x = int(start_point[0] + t * (end_point[0] - start_point[0]))
        y = int(start_point[1] + t * (end_point[1] - start_point[1]))
        window_size = 10
        x_min = max(0, x - window_size)
        x_max = min(image.shape[1], x + window_size)
        y_min = max(0, y - window_size)
        y_max = min(image.shape[0], y + window_size)
        window = gray[y_min:y_max, x_min:x_max]
        if mask is not None:
            mask_window = mask[y_min:y_max, x_min:x_max]
            valid_pixels = window[mask_window > 0].astype(float)
            if valid_pixels.size > 0:
                profile.append(float(np.mean(valid_pixels)))
            else:
                profile.append(0.0)
        else:
            window_float = window.astype(float)
            if window_float.size > 0:
                profile.append(float(np.mean(window_float)))
            else:
                profile.append(0.0)
    return np.array(profile)

def calculate_iou(box1: Tuple[float, float, float, float], 
                  box2: Tuple[float, float, float, float]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    x_intersection = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_intersection = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    intersection_area = x_intersection * y_intersection
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0
    
    return intersection_area / union_area