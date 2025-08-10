import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from pathlib import Path
from src.models.trainer import ModelTrainer

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = ModelTrainer(config['paths']['models'])
    
    seg_data_yaml = Path(config['paths']['yolo_datasets']) / 'shoe_segmentation' / 'data.yaml'
    feat_data_yaml = Path(config['paths']['yolo_datasets']) / 'feature_detection' / 'data.yaml'
    
    if seg_data_yaml.exists():
        print("Training shoe segmentation model...")
        seg_model_path = trainer.train_segmentation_model(
            str(seg_data_yaml),
            config['models']['shoe_segmentation']
        )
        print(f"Segmentation model saved to: {seg_model_path}")
    else:
        print(f"Segmentation dataset not found at {seg_data_yaml}")
        print("Please run prepare_data.py first")
        return
    
    if feat_data_yaml.exists():
        print("\nTraining feature detection model...")
        feat_model_path = trainer.train_feature_model(
            str(feat_data_yaml),
            config['models']['feature_detection']
        )
        print(f"Feature model saved to: {feat_model_path}")
    else:
        print(f"Feature dataset not found at {feat_data_yaml}")
    
    print("\nTraining complete!")
    print("You can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()