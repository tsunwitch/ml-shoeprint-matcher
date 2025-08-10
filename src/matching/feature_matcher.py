import numpy as np
from typing import List, Tuple
from ..utils.image_ops import calculate_iou

class FeatureMatcher:
    def __init__(self, iou_threshold: float = 0.5, min_matches: int = 3):
        self.iou_threshold = iou_threshold
        self.min_matches = min_matches
    
    def match_features(self, features1: List[Tuple], features2: List[Tuple]) -> int:
        if not features1 or not features2:
            return 0
        
        matches = 0
        matched_indices = set()
        
        for f1 in features1:
            best_iou = 0
            best_idx = -1
            
            for idx, f2 in enumerate(features2):
                if idx in matched_indices:
                    continue
                
                iou = calculate_iou(f1, f2)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_iou >= self.iou_threshold:
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