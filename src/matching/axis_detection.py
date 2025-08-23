import cv2
import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA

def detect_shoe_axis(image: np.ndarray, mask: 'np.ndarray | None' = None):
    """
    Returns axis endpoints using image moments within the mask.
    """
    if mask is not None:
        return axis_from_mask(mask, image=image)
    if image.ndim == 2 and np.unique(image).size <= 2:
        return axis_from_mask(image)
    h, w = image.shape[:2]
    return ((w//2, 0), (w//2, h))


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


