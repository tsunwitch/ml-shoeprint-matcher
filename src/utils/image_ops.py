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
                         num_samples: int = 100, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    start_point, end_point = axis_line
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    left_profile = []
    right_profile = []
    # Compute axis direction
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    axis_length = np.hypot(dx, dy)
    if axis_length == 0:
        axis_vec = np.array([1, 0])
    else:
        axis_vec = np.array([dx, dy]) / axis_length
    # Perpendicular direction
    perp_vec = np.array([-axis_vec[1], axis_vec[0]])
    window_size = 10
    for i in range(num_samples):
        t = i / (num_samples - 1)
        x = int(start_point[0] + t * dx)
        y = int(start_point[1] + t * dy)
        # Sample left and right windows
        left_cx = int(x - perp_vec[0] * window_size)
        left_cy = int(y - perp_vec[1] * window_size)
        right_cx = int(x + perp_vec[0] * window_size)
        right_cy = int(y + perp_vec[1] * window_size)
        # Extract left window
        lx_min = max(0, left_cx - window_size)
        lx_max = min(image.shape[1], left_cx + window_size)
        ly_min = max(0, left_cy - window_size)
        ly_max = min(image.shape[0], left_cy + window_size)
        left_window = gray[ly_min:ly_max, lx_min:lx_max]
        # Extract right window
        rx_min = max(0, right_cx - window_size)
        rx_max = min(image.shape[1], right_cx + window_size)
        ry_min = max(0, right_cy - window_size)
        ry_max = min(image.shape[0], right_cy + window_size)
        right_window = gray[ry_min:ry_max, rx_min:rx_max]
        # Masking
        if mask is not None:
            left_mask = mask[ly_min:ly_max, lx_min:lx_max]
            right_mask = mask[ry_min:ry_max, rx_min:rx_max]
            left_valid = left_window[left_mask > 0].astype(float)
            right_valid = right_window[right_mask > 0].astype(float)
            left_profile.append(float(np.mean(left_valid)) if left_valid.size > 0 else 0.0)
            right_profile.append(float(np.mean(right_valid)) if right_valid.size > 0 else 0.0)
        else:
            left_window_float = left_window.astype(float)
            right_window_float = right_window.astype(float)
            left_profile.append(float(np.mean(left_window_float)) if left_window_float.size > 0 else 0.0)
            right_profile.append(float(np.mean(right_window_float)) if right_window_float.size > 0 else 0.0)
    return np.array(left_profile), np.array(right_profile)

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