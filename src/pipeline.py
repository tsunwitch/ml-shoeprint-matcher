import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml

from .models.segmentation import ShoeSegmenter
from .models.detection import FeatureDetector
from .utils.image_ops import extract_axis_profile, calculate_iou
from .matching.dtw_matcher import DTWMatcher
from .matching.feature_matcher import FeatureMatcher

class ShoeprintPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.segmenter = None
        self.feature_detector = None
        self.dtw_matcher = DTWMatcher()
        self.feature_matcher = FeatureMatcher()
        
        self.database = {
            'profiles': {},
            'features': {},
            'metadata': {}
        }
    
    def load_models(self, segmentation_path: Optional[str] = None, feature_path: Optional[str] = None):
        if segmentation_path:
            self.segmenter = ShoeSegmenter(segmentation_path)
        
        if feature_path:
            self.feature_detector = FeatureDetector(feature_path)
    
    def process_image(self, image_path: str) -> Dict:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        results = {
            'original_image': image,
            'image_path': image_path
        }
        
        results['cropped_shoe'] = image
        results['shoe_bbox'] = None
        
        if self.feature_detector:
            confidence = self.config['models']['feature_detection']['confidence']
            features = self.feature_detector.detect_features(image, confidence=confidence)
            results['features'] = features
            
            patches = self.feature_detector.extract_feature_patches(image, features)
            descriptors = self.feature_detector.compute_feature_descriptors(patches)
            results['feature_descriptors'] = descriptors
        
        return results
    
    def add_to_database(self, shoe_id: str, image_results: Dict):
        
        image_id = f"{shoe_id}_{Path(image_results['image_path']).stem}"
        
        profile = self._extract_profile_from_image(image_results['original_image'])
        
        self.database['profiles'][image_id] = profile
        self.database['features'][image_id] = {
            'boxes': image_results.get('features', []),
            'descriptors': image_results.get('feature_descriptors', np.array([]))
        }
        self.database['metadata'][image_id] = {
            'shoe_id': shoe_id,
            'path': image_results['image_path'],
            'cropped_shoe': image_results.get('original_image'),
            'original_image': image_results.get('original_image'),
            'features': image_results.get('features', [])
        }
    
    def search(self, query_image_path: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        query_results = self.process_image(query_image_path)
        
        query_profile = self._extract_profile_from_image(query_results['original_image'])
        
        
        dtw_scores = {}
        for image_id, profile in self.database['profiles'].items():
            distance = self.dtw_matcher.compute_distance(query_profile, profile)
            dtw_scores[image_id] = distance
        
        
        sorted_by_dtw = sorted(dtw_scores.items(), key=lambda x: x[1])
        
        
        if sorted_by_dtw:
            best_dtw_score = sorted_by_dtw[0][1]
            
            dtw_threshold = best_dtw_score * 1.5
            filtered_candidates = [img_id for img_id, score in sorted_by_dtw 
                                  if score <= dtw_threshold]
            
            
            if len(filtered_candidates) < top_k * 2:
                filtered_candidates = [img_id for img_id, _ in sorted_by_dtw[:top_k * 2]]
        else:
            filtered_candidates = list(self.database['profiles'].keys())
        
        print(f"Stage 1: DTW filtering reduced {len(self.database['profiles'])} images to {len(filtered_candidates)} candidates")
        
        
        feature_scores = {}
        query_features = query_results.get('features', [])
        
        for image_id in filtered_candidates:
            db_features = self.database['features'][image_id]['boxes']
            matches = self.feature_matcher.match_features(query_features, db_features)
            feature_scores[image_id] = matches
        
        
        final_scores = {}
        for image_id in filtered_candidates:
            
            feature_weight = self.config['matching']['weights'].get('feature_weight', 0.8)
            dtw_weight = self.config['matching']['weights'].get('dtw_weight', 0.2)
            
            
            dtw_norm = 1.0 / (1.0 + dtw_scores[image_id])
            feature_norm = feature_scores.get(image_id, 0) / max(1, len(query_features))
            
            final_scores[image_id] = feature_weight * feature_norm + dtw_weight * dtw_norm
        
        
        results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        print(f"Stage 2: Returning top {len(results)} matches based on feature similarity")
        
        
        return [(image_id, score, self.database['metadata'][image_id]) for image_id, score in results]
    
    def _extract_profile_from_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        h, w = gray.shape
        axis_line = ((w//2, 0), (w//2, h))
        
        profile = extract_axis_profile(gray, axis_line, num_samples=100)
        
        return profile