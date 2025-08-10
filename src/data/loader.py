import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

class ShoeDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.annotations = {}
        self.images = {}
        
    def load_all_data(self) -> Dict:
        shoe_folders = sorted([f for f in self.data_dir.iterdir() if f.is_dir()])
        
        for shoe_folder in shoe_folders:
            shoe_id = shoe_folder.name
            self.annotations[shoe_id] = {}
            self.images[shoe_id] = {}
            
            annotator_folders = [f for f in shoe_folder.iterdir() if f.is_dir()]
            
            for annotator_folder in annotator_folders:
                annotator_name = annotator_folder.name.rsplit('_', 1)[0]
                self.annotations[shoe_id][annotator_name] = {}
                self.images[shoe_id][annotator_name] = {}
                
                json_files = list(annotator_folder.glob("*.json"))
                
                for json_file in json_files:
                    image_name = json_file.name.replace('.json', '')
                    image_path = annotator_folder / image_name
                    
                    if image_path.exists():
                        with open(json_file, 'r') as f:
                            annotation = json.load(f)
                        
                        shoe_print_id = image_name.replace('.jpg', '')
                        self.annotations[shoe_id][annotator_name][shoe_print_id] = annotation
                        self.images[shoe_id][annotator_name][shoe_print_id] = str(image_path)
        
        return {
            'annotations': self.annotations,
            'images': self.images
        }
    
    def get_marking_by_type(self, annotation: Dict, type_name: str) -> List[Dict]:
        type_id = None
        for type_info in annotation['metadata']['types']:
            if type_info['name'] == type_name:
                type_id = type_info['id']
                break
        
        if not type_id:
            return []
        
        markings = []
        for marking in annotation['data']['markings']:
            if marking.get('typeId') == type_id:
                markings.append(marking)
        
        return markings
    
    def extract_bbox(self, marking: Dict) -> Tuple[float, float, float, float]:
        x1 = marking['origin']['x']
        y1 = marking['origin']['y']
        x2 = marking['endpoint']['x']
        y2 = marking['endpoint']['y']
        
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)
        
        return x_min, y_min, x_max, y_max
    
    def load_image(self, image_path: str) -> np.ndarray:
        return cv2.imread(image_path)
    
    def get_image_shape(self, image_path: str) -> Tuple[int, int]:
        img = cv2.imread(image_path)
        return img.shape[:2] if img is not None else (0, 0)