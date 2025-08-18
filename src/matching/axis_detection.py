import cv2
import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA

def detect_shoe_axis(image: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Detects the main axis of the shoeprint using Canny, contours, moments, and PCA.
    Returns: ((x1, y1), (x2, y2)) endpoints of the axis line
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    # Denoise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Edge detection
    edges = cv2.Canny(blur, 50, 150)
    # Morphological closing to connect ridges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image.shape[:2]
        return ((w//2, 0), (w//2, h))
    # Filter contours by area (remove small noise)
    min_area = 0.01 * (image.shape[0] * image.shape[1])
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if not large_contours:
        h, w = image.shape[:2]
        return ((w//2, 0), (w//2, h))
    # Combine all large contours into one set of points
    all_pts = np.vstack([cnt.reshape(-1, 2) for cnt in large_contours])
    # Moments for centroid
    M = cv2.moments(all_pts)
    if M["m00"] == 0:
        cx, cy = np.mean(all_pts, axis=0).tolist()
    else:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    # PCA for orientation
    pca = PCA(n_components=2)
    pca.fit(all_pts)
    axis_vec = pca.components_[0]
    # Axis endpoints
    length = int(max(image.shape[:2]) * 0.4)
    x1 = int(cx - axis_vec[0] * length)
    y1 = int(cy - axis_vec[1] * length)
    x2 = int(cx + axis_vec[0] * length)
    y2 = int(cy + axis_vec[1] * length)
    return ((x1, y1), (x2, y2))
