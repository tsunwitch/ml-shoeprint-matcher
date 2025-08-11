from ultralytics import YOLO
from pathlib import Path
import yaml

class ModelTrainer:
    def __init__(self, model_dir: str = "trained_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
    def train_segmentation_model(self, data_yaml: str, config: dict):
        model = YOLO(config['model_size'])
        
        results = model.train(
            data=data_yaml,
            epochs=config['epochs'],
            batch=config['batch_size'],
            imgsz=config['imgsz'],
            project=str(self.model_dir),
            name='shoe_segmentation',
            exist_ok=True,
            device=0,
            patience=20,
            save=True,
            pretrained=True
        )
        
        return self.model_dir / 'shoe_segmentation' / 'weights' / 'best.pt'
    
    def train_feature_model(self, data_yaml: str, config: dict):
        model = YOLO(config['model_size'])
        
        results = model.train(
            data=data_yaml,
            epochs=config['epochs'],
            batch=config['batch_size'],
            imgsz=config['imgsz'],
            project=str(self.model_dir),
            name='feature_detection',
            exist_ok=True,
            device=0,
            patience=20,
            save=True,
            pretrained=True
        )
        
        return self.model_dir / 'feature_detection' / 'weights' / 'best.pt'
    
    def load_model(self, model_path: str):
        return YOLO(model_path)