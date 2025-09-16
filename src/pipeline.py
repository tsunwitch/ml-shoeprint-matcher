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
        self.dtw_matcher = DTWMatcher(window_size=self.config['matching']['dtw']['window_size'])
        self.feature_matcher = FeatureMatcher(
            distance_threshold=self.config['matching']['features'].get('distance_threshold', 30.0),
            min_matches=self.config['matching']['features'].get('min_matches', 3)
        )
        
        self.database = {
            'profiles': {},
            'features': {},
            'metadata': {}
        }
    
    def load_models(self, detection_path: Optional[str] = None, feature_path: Optional[str] = None):
        if detection_path:
            self.detector = FeatureDetector(detection_path)
        if feature_path:
            use_sahi = self.config['models']['feature_detection'].get('use_sahi', False)
            self.feature_detector = FeatureDetector(feature_path, use_sahi=use_sahi)
    
    def process_image(self, image_path: str) -> Dict:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        results = {
            'original_image': image,
            'image_path': image_path
        }

        # Use detection model to get shoe bounding box and crop
        cropped_shoe = image
        shoe_bbox = None
        if hasattr(self, 'detector') and self.detector is not None:
            confidence = self.config['models']['shoe_detection']['confidence']
            boxes = self.detector.detect_features(image, confidence=confidence)
            if boxes:
                x1, y1, x2, y2 = boxes[0]
                cropped_shoe = image[int(y1):int(y2), int(x1):int(x2)]
                shoe_bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
        results['cropped_shoe'] = cropped_shoe
        results['shoe_bbox'] = shoe_bbox

        # Axis detection using cropped shoe
        try:
            from .matching.axis_detection import detect_shoe_axis
            axis_line_cropped = detect_shoe_axis(cropped_shoe)
        except Exception as e:
            h, w = cropped_shoe.shape[:2]
            axis_line_cropped = ((w//2, 0), (w//2, h))
        
        # Translate axis coordinates to original image space if cropping was applied
        if shoe_bbox is not None:
            crop_x, crop_y = shoe_bbox[0], shoe_bbox[1]
            (x1_crop, y1_crop), (x2_crop, y2_crop) = axis_line_cropped
            axis_line = ((x1_crop + crop_x, y1_crop + crop_y), (x2_crop + crop_x, y2_crop + crop_y))
        else:
            axis_line = axis_line_cropped
            
        results['axis_line'] = axis_line         # axis in original image coordinates
        results['axis_line_cropped'] = axis_line_cropped  # axis in cropped image coordinates

        if self.feature_detector:
            confidence = self.config['models']['feature_detection']['confidence']
            features_cropped = self.feature_detector.detect_features(cropped_shoe, confidence=confidence)

            # Translate features from cropped image coordinates to original image coordinates
            features_original = []
            if shoe_bbox is not None and features_cropped:
                crop_x, crop_y = shoe_bbox[0], shoe_bbox[1]  # offset of crop in original image
                for x1, y1, x2, y2 in features_cropped:
                    # Add crop offset to translate coordinates back to original image space
                    features_original.append((x1 + crop_x, y1 + crop_y, x2 + crop_x, y2 + crop_y))
            elif features_cropped:
                # No cropping was done, features are already in original image coordinates
                features_original = features_cropped
                
            results['features_original'] = features_original  # features in original image coordinates
            results['features_cropped'] = features_cropped      # features in cropped image coordinates

            norm_features = self._normalize_features(features_cropped, axis_line_cropped, cropped_shoe.shape)
            results['features'] = norm_features

            patches = self.feature_detector.extract_feature_patches(cropped_shoe, features_cropped)
            descriptors = self.feature_detector.compute_feature_descriptors(patches)
            results['feature_descriptors'] = descriptors

        return results

    def _normalize_features(self, features, axis_line, image_shape):
        if not features:
            return []
        (x1, y1), (x2, y2) = axis_line
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        axis_angle = np.arctan2(dy, dx)
        cos_a = np.cos(-axis_angle)
        sin_a = np.sin(-axis_angle)
        h, w = image_shape[:2]
        scale = max(h, w)
        norm_features = []
        for box in features:
            x1b, y1b, x2b, y2b = box
            x1b -= cx
            y1b -= cy
            x2b -= cx
            y2b -= cy
            x1r = x1b * cos_a - y1b * sin_a
            y1r = x1b * sin_a + y1b * cos_a
            x2r = x2b * cos_a - y2b * sin_a
            y2r = x2b * sin_a + y2b * cos_a
            x1r /= scale
            y1r /= scale
            x2r /= scale
            y2r /= scale
            norm_features.append((x1r, y1r, x2r, y2r))
        return norm_features
    
    def add_to_database(self, shoe_id: str, image_results: Dict):
        image_id = f"{shoe_id}_{Path(image_results['image_path']).stem}"
        left_profile, right_profile = self._extract_profile_from_image(image_results['original_image'])
        self.database['profiles'][image_id] = {
            'left': left_profile,
            'right': right_profile
        }
        self.database['features'][image_id] = {
            'boxes': image_results.get('features', []),
            'descriptors': image_results.get('feature_descriptors', np.array([]))
        }
        self.database['metadata'][image_id] = {
            'shoe_id': shoe_id,
            'path': image_results['image_path'],
            'cropped_shoe': image_results.get('original_image'),
            'original_image': image_results.get('original_image'),
            'features': image_results.get('features', []),  # normalized features for matching
            'features_original': image_results.get('features_original', []),  # features in original image coordinates
            'features_cropped': image_results.get('features_cropped', []),   # features in cropped image coordinates
            'axis_line': image_results.get('axis_line', None),         # axis in original image coordinates
            'axis_line_cropped': image_results.get('axis_line_cropped', None)  # axis in cropped coordinates
        }
    
    def search(self, query_image_path: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        query_results = self.process_image(query_image_path)
        query_left, query_right = self._extract_profile_from_image(query_results['original_image'])

        dtw_scores = {}
        for image_id, profiles in self.database['profiles'].items():
            left_dist = self.dtw_matcher.compute_distance(query_left, profiles['left'])
            right_dist = self.dtw_matcher.compute_distance(query_right, profiles['right'])
            avg_dist = (left_dist + right_dist) / 2.0
            dtw_scores[image_id] = avg_dist

        sorted_by_dtw = sorted(dtw_scores.items(), key=lambda x: x[1])

        if sorted_by_dtw:
            dtw_threshold = self.config['matching']['dtw']['distance_threshold']
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
    
    def _extract_profile_from_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        mask = None
        if (self.config['matching']['dtw']['use_segmentation'] and self.segmenter is not None):
            try:
                mask = self.segmenter.get_shoe_mask(image)
            except Exception as e:
                print(f"Segmentation mask failed, using full image: {e}")

        # Detect axis using mask-based axis detection
        try:
            from .matching.axis_detection import detect_shoe_axis
            axis_line = detect_shoe_axis(gray, mask=mask)
        except Exception as e:
            h, w = gray.shape
            axis_line = ((w//2, 0), (w//2, h))

        window_size = self.config['matching'].get('profile_window_size', 10)
        left_profile, right_profile = extract_axis_profile(gray, axis_line, num_samples=100, mask=mask, window_size=window_size)
        return left_profile, right_profile
    
    def get_segmentation_bbox(self, image: np.ndarray) -> Optional[Tuple]:
        if (self.config['matching']['dtw']['use_segmentation'] and 
            self.segmenter is not None):
            try:
                _, bbox = self.segmenter.segment_shoe(image)
                return bbox
            except Exception as e:
                print(f"Segmentation failed: {e}")
        return None