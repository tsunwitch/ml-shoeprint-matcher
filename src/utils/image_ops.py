import cv2
import numpy as np
from typing import Tuple, List

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
                         num_samples: int = 100) -> np.ndarray:
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
        if window.size > 0:
            profile.append(np.mean(window))
        else:
            profile.append(0)
    
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