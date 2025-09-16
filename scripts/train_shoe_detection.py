#!/usr/bin/env python3
"""
Training script for shoe detection (bounding box) model only.
Usage: python scripts/train_shoe_detection.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from pathlib import Path
from src.models.trainer import ModelTrainer

def main():
    """Train only the shoe detection model."""
    print("=== Shoe Detection Model Training ===")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = ModelTrainer(config['paths']['models'])
    
    # Check if detection dataset exists
    det_data_yaml = Path(config['paths']['yolo_datasets']) / 'shoe_detection' / 'data.yaml'
    
    if not det_data_yaml.exists():
        print(f"❌ Detection dataset not found at {det_data_yaml}")
        print("Please run prepare_data.py first to create the dataset")
        return
    
    print(f"📁 Using dataset: {det_data_yaml}")
    print(f"🏋️  Model size: {config['models']['shoe_detection']['model_size']}")
    print(f"📈 Epochs: {config['models']['shoe_detection']['epochs']}")
    print(f"📦 Batch size: {config['models']['shoe_detection']['batch_size']}")
    print(f"🖼️  Image size: {config['models']['shoe_detection']['imgsz']}")
    
    try:
        print("\n🚀 Starting detection model training...")
        det_model_path = trainer.train_detection_model(
            str(det_data_yaml),
            config['models']['shoe_detection']
        )
        print(f"✅ Detection model training completed!")
        print(f"💾 Model saved to: {det_model_path}")
        print(f"\n🏆 You can now use this model in the pipeline by updating your config")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
