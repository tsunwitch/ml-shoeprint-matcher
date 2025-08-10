import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

class DTWMatcher:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
    
    def compute_distance(self, profile1: np.ndarray, profile2: np.ndarray) -> float:
        # Ensure profiles are 1-D
        profile1 = np.array(profile1).flatten()
        profile2 = np.array(profile2).flatten()
        
        profile1_norm = self._normalize_profile(profile1)
        profile2_norm = self._normalize_profile(profile2)
        
        # Reshape for fastdtw (needs sequence of scalars)
        profile1_norm = profile1_norm.reshape(-1, 1)
        profile2_norm = profile2_norm.reshape(-1, 1)
        
        distance, path = fastdtw(profile1_norm, profile2_norm, dist=euclidean)
        
        normalized_distance = distance / len(profile1_norm)
        
        return normalized_distance
    
    def _normalize_profile(self, profile: np.ndarray) -> np.ndarray:
        if profile.std() == 0:
            return profile
        
        normalized = (profile - profile.mean()) / profile.std()
        
        return normalized
    
    def compute_similarity(self, profile1: np.ndarray, profile2: np.ndarray) -> float:
        distance = self.compute_distance(profile1, profile2)
        
        similarity = 1.0 / (1.0 + distance)
        
        return similarity
    
    def batch_compare(self, query_profile: np.ndarray, database_profiles: list) -> list:
        distances = []
        
        for db_profile in database_profiles:
            distance = self.compute_distance(query_profile, db_profile)
            distances.append(distance)
        
        return distances