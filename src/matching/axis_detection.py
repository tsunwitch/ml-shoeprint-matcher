import cv2
import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA

def axis_from_mask(mask: np.ndarray, image: np.ndarray = None):
    """
    Given a binary mask and shoeprint image, compute center and rotation using image moments on shoeprint pixels within the mask.
    Returns axis endpoints ((x1, y1), (x2, y2)).
    """
    if image is not None:
        if image.ndim == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image.copy()
        shoeprint = np.where(mask > 0, image_gray, 0)
    else:
        shoeprint = mask.copy()
    M = cv2.moments(shoeprint)
    if M["m00"] == 0:
        h, w = mask.shape
        return ((w//2, 0), (w//2, h))
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    axis_vec = np.array([np.cos(angle), np.sin(angle)])
    ys, xs = np.nonzero(mask)
    coords = np.column_stack((xs, ys))
    rel_coords = coords - np.array([cx, cy])
    projections = np.dot(rel_coords, axis_vec)
    min_proj = projections.min()
    max_proj = projections.max()
    pt1 = np.array([cx, cy]) + min_proj * axis_vec
    pt2 = np.array([cx, cy]) + max_proj * axis_vec
    pt1 = tuple(map(int, pt1))
    pt2 = tuple(map(int, pt2))
    return (pt1, pt2)


def get_axis_preprocessing_steps(image: np.ndarray):
    """
    Returns a dict of intermediate images for axis detection: grayscale, blur, edges, closing.
    """
    steps = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    steps['Grayscale'] = gray
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    steps['Gaussian Blur'] = blur
    edges = cv2.Canny(blur, 50, 150)
    steps['Canny Edges'] = edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    steps['Morphological Closing'] = closed
    return steps


def axis_detection_pipeline(image: np.ndarray):
    """
    Processes image for axis detection, returns dict of steps and axis endpoints.
    Returns: steps (dict), axis_line ((x1, y1), (x2, y2))
    """
    steps = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    steps['Grayscale'] = gray
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    steps['Gaussian Blur'] = blur
    edges = cv2.Canny(blur, 50, 150)
    steps['Canny Edges'] = edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    steps['Morphological Closing'] = closed

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image.shape[:2]
        axis_line = ((w//2, 0), (w//2, h))
        return steps, axis_line
    min_area = 0.01 * (image.shape[0] * image.shape[1])
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if not large_contours:
        h, w = image.shape[:2]
        axis_line = ((w//2, 0), (w//2, h))
        return steps, axis_line
    all_pts = np.vstack([cnt.reshape(-1, 2) for cnt in large_contours])
    M = cv2.moments(all_pts)
    if M["m00"] == 0:
        mean_pt = np.mean(all_pts.astype(np.float32), axis=0)
        cx, cy = float(mean_pt[0]), float(mean_pt[1])
    else:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    pca = PCA(n_components=2)
    pca.fit(all_pts)
    axis_vec = pca.components_[0]
    x, y, w, h = cv2.boundingRect(all_pts)
    t_values = []
    if axis_vec[0] != 0:
        t_values.append((x - cx) / axis_vec[0])
        t_values.append((x + w - cx) / axis_vec[0])
    if axis_vec[1] != 0:
        t_values.append((y - cy) / axis_vec[1])
        t_values.append((y + h - cy) / axis_vec[1])
    points = []
    for t in t_values:
        px = int(cx + t * axis_vec[0])
        py = int(cy + t * axis_vec[1])
        if x <= px <= x + w and y <= py <= y + h:
            points.append((px, py))
    if len(points) >= 2:
        points = sorted(list(set(points)))
        axis_line = (points[0], points[-1])
    else:
        axis_line = ((x, y), (x + w, y + h))
    return steps, axis_line


def detect_shoe_axis(image: np.ndarray, mask: 'np.ndarray | None' = None):
    """
    Returns axis endpoints. If a mask is provided, use it for PCA; else fallback to old method.
    """
    if mask is not None:
        return axis_from_mask(mask, image=image)
    if image.ndim == 2 and np.unique(image).size <= 2:
        return axis_from_mask(image)
    _, axis_line = axis_detection_pipeline(image)
    pt1, pt2 = axis_line
    pt1 = tuple(map(int, pt1[:2]))
    pt2 = tuple(map(int, pt2[:2]))
    return (pt1, pt2)
