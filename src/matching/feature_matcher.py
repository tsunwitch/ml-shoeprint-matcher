import numpy as np
from typing import List, Tuple
from ..utils.image_ops import calculate_iou

class FeatureMatcher:
    def __init__(self, distance_threshold: float = 30.0, min_matches: int = 3):
        self.distance_threshold = distance_threshold
        self.min_matches = min_matches
    
    def match_features(self, features1: List[Tuple], features2: List[Tuple]) -> int:
        if not features1 or not features2:
            return 0

        def center(box):
            x1, y1, x2, y2 = box
            return ((x1 + x2) / 2, (y1 + y2) / 2)

        matches = 0
        matched_indices = set()

        for f1 in features1:
            c1 = center(f1)
            best_dist = float('inf')
            best_idx = -1
            for idx, f2 in enumerate(features2):
                if idx in matched_indices:
                    continue
                c2 = center(f2)
                dist = np.linalg.norm(np.array(c1) - np.array(c2))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_dist <= self.distance_threshold:
                matches += 1
                matched_indices.add(best_idx)
        return matches
    
    def compute_match_score(self, features1: List[Tuple], features2: List[Tuple]) -> float:
        if not features1 and not features2:
            return 1.0
        if not features1 or not features2:
            return 0.0
        
        matches = self.match_features(features1, features2)
        
        score = matches / max(len(features1), len(features2))
        
        return score
    
    def is_match(self, features1: List[Tuple], features2: List[Tuple]) -> bool:
        matches = self.match_features(features1, features2)
        return matches >= self.min_matches
    
    def find_best_match(self, query_features: List[Tuple], 
                       database_features: List[List[Tuple]]) -> Tuple[int, float]:
        best_idx = -1
        best_score = 0
        
        for idx, db_features in enumerate(database_features):
            score = self.compute_match_score(query_features, db_features)
            if score > best_score:
                best_score = score
                best_idx = idx
        
        return best_idx, best_score