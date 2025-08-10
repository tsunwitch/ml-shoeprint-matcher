import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import yaml

class YOLOConverter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        
    def prepare_directories(self, dataset_name: str):
        dataset_path = self.output_dir / dataset_name
        
        for split in ['train', 'val', 'test']:
            (dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        return dataset_path
    
    def convert_bbox_to_yolo(self, bbox: Tuple[float, float, float, float], 
                             img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        x_min, y_min, x_max, y_max = bbox
        
        x_center = (x_min + x_max) / 2.0 / img_width
        y_center = (y_min + y_max) / 2.0 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        return x_center, y_center, width, height
    
    def convert_segmentation_to_yolo(self, bbox: Tuple[float, float, float, float],
                                    img_width: int, img_height: int) -> List[float]:
        x_min, y_min, x_max, y_max = bbox
        
        points = [
            x_min / img_width, y_min / img_height,
            x_max / img_width, y_min / img_height,
            x_max / img_width, y_max / img_height,
            x_min / img_width, y_max / img_height
        ]
        
        return points
    
    def create_segmentation_dataset(self, data: Dict, loader, split_info: Dict):
        dataset_path = self.prepare_directories('shoe_segmentation')
        
        for split_name, shoe_ids in split_info.items():
            for shoe_id in shoe_ids:
                if shoe_id not in data['annotations']:
                    continue
                    
                for annotator, annotations in data['annotations'][shoe_id].items():
                    for print_id, annotation in annotations.items():
                        image_path = data['images'][shoe_id][annotator][print_id]
                        
                        mark_annotations = loader.get_marking_by_type(annotation, 'Mark')
                        if not mark_annotations:
                            continue
                        
                        img_height, img_width = loader.get_image_shape(image_path)
                        if img_height == 0:
                            continue
                        
                        output_name = f"{shoe_id}_{annotator}_{print_id}"
                        
                        shutil.copy(
                            image_path,
                            dataset_path / 'images' / split_name / f"{output_name}.jpg"
                        )
                        
                        label_path = dataset_path / 'labels' / split_name / f"{output_name}.txt"
                        with open(label_path, 'w') as f:
                            for mark in mark_annotations:
                                bbox = loader.extract_bbox(mark)
                                points = self.convert_segmentation_to_yolo(bbox, img_width, img_height)
                                points_str = ' '.join([str(p) for p in points])
                                f.write(f"0 {points_str}\n")
        
        yaml_content = {
            'path': str(dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {0: 'shoe'},
            'nc': 1
        }
        
        with open(dataset_path / 'data.yaml', 'w') as f:
            yaml.dump(yaml_content, f)
        
        return dataset_path
    
    def create_feature_dataset(self, data: Dict, loader, split_info: Dict):
        dataset_path = self.prepare_directories('feature_detection')
        
        for split_name, shoe_ids in split_info.items():
            for shoe_id in shoe_ids:
                if shoe_id not in data['annotations']:
                    continue
                    
                for annotator, annotations in data['annotations'][shoe_id].items():
                    for print_id, annotation in annotations.items():
                        image_path = data['images'][shoe_id][annotator][print_id]
                        
                        feature_annotations = loader.get_marking_by_type(annotation, 'UniqueFeature')
                        if not feature_annotations:
                            continue
                        
                        img_height, img_width = loader.get_image_shape(image_path)
                        if img_height == 0:
                            continue
                        
                        output_name = f"{shoe_id}_{annotator}_{print_id}"
                        
                        shutil.copy(
                            image_path,
                            dataset_path / 'images' / split_name / f"{output_name}.jpg"
                        )
                        
                        label_path = dataset_path / 'labels' / split_name / f"{output_name}.txt"
                        with open(label_path, 'w') as f:
                            for feature in feature_annotations:
                                bbox = loader.extract_bbox(feature)
                                x_center, y_center, width, height = self.convert_bbox_to_yolo(
                                    bbox, img_width, img_height
                                )
                                f.write(f"0 {x_center} {y_center} {width} {height}\n")
        
        yaml_content = {
            'path': str(dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {0: 'feature'},
            'nc': 1
        }
        
        with open(dataset_path / 'data.yaml', 'w') as f:
            yaml.dump(yaml_content, f)
        
        return dataset_path