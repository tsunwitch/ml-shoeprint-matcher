import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def draw_bbox(image: np.ndarray, bbox: Tuple[float, float, float, float], 
              color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    img_copy = image.copy()
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    return img_copy

def draw_features(image: np.ndarray, features: List[Tuple[float, float, float, float]], 
                 color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    img_copy = image.copy()
    for i, feature in enumerate(features):
        x1, y1, x2, y2 = [int(coord) for coord in feature]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_copy, str(i+1), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_copy

def draw_axis(image: np.ndarray, axis_line: Tuple[Tuple[float, float], Tuple[float, float]], 
             color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 3) -> np.ndarray:
    img_copy = image.copy()
    start_point = tuple([int(coord) for coord in axis_line[0]])
    end_point = tuple([int(coord) for coord in axis_line[1]])
    cv2.line(img_copy, start_point, end_point, color, thickness)
    return img_copy

def draw_mask_overlay(image: np.ndarray, mask: np.ndarray, 
                      color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.3) -> np.ndarray:
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = color
    
    return cv2.addWeighted(overlay, 1-alpha, mask_colored, alpha, 0)

def plot_dtw_profiles(profile1: np.ndarray, profile2: np.ndarray, 
                     title: str = "DTW Profile Comparison") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    
    x = np.arange(len(profile1))
    ax.plot(x, profile1, label='Profile 1', linewidth=2)
    ax.plot(x, profile2, label='Profile 2', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Position along axis')
    ax.set_ylabel('Intensity')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_comparison_image(img1: np.ndarray, img2: np.ndarray, 
                           title1: str = "Image 1", title2: str = "Image 2") -> np.ndarray:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    max_height = max(h1, h2)
    
    if h1 < max_height:
        pad = max_height - h1
        img1 = np.pad(img1, ((0, pad), (0, 0), (0, 0)), mode='constant')
    if h2 < max_height:
        pad = max_height - h2
        img2 = np.pad(img2, ((0, pad), (0, 0), (0, 0)), mode='constant')
    
    combined = np.hstack([img1, np.ones((max_height, 20, 3), dtype=np.uint8)*255, img2])
    
    cv2.putText(combined, title1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(combined, title2, (w1 + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return combined