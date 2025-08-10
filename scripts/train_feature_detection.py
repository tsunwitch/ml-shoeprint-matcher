#!/usr/bin/env python3
"""
Training script for feature detection model only.
Usage: python scripts/train_feature_detection.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from pathlib import Path
from src.models.trainer import ModelTrainer

def main():
    """Train only the feature detection model."""
    print("=== Feature Detection Model Training ===")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = ModelTrainer(config['paths']['models'])
    
    # Check if feature detection dataset exists
    feat_data_yaml = Path(config['paths']['yolo_datasets']) / 'feature_detection' / 'data.yaml'
    
    if not feat_data_yaml.exists():
        print(f"❌ Feature detection dataset not found at {feat_data_yaml}")
        print("Please run prepare_data.py first to create the dataset")
        return
    
    print(f"📁 Using dataset: {feat_data_yaml}")
    print(f"🏗️  Model size: {config['models']['feature_detection']['model_size']}")
    print(f"📊 Epochs: {config['models']['feature_detection']['epochs']}")
    print(f"📦 Batch size: {config['models']['feature_detection']['batch_size']}")
    print(f"🖼️  Image size: {config['models']['feature_detection']['imgsz']}")
    
    try:
        print("\n🚀 Starting feature detection model training...")
        feat_model_path = trainer.train_feature_model(
            str(feat_data_yaml),
            config['models']['feature_detection']
        )
        print(f"✅ Feature detection model training completed!")
        print(f"💾 Model saved to: {feat_model_path}")
        print(f"\n🎯 You can now use this model in the pipeline by updating your config")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
