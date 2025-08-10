#!/usr/bin/env python3
"""
Training script for shoe segmentation model only.
Usage: python scripts/train_segmentation.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from pathlib import Path
from src.models.trainer import ModelTrainer

def main():
    """Train only the shoe segmentation model."""
    print("=== Shoe Segmentation Model Training ===")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = ModelTrainer(config['paths']['models'])
    
    # Check if segmentation dataset exists
    seg_data_yaml = Path(config['paths']['yolo_datasets']) / 'shoe_segmentation' / 'data.yaml'
    
    if not seg_data_yaml.exists():
        print(f"❌ Segmentation dataset not found at {seg_data_yaml}")
        print("Please run prepare_data.py first to create the dataset")
        return
    
    print(f"📁 Using dataset: {seg_data_yaml}")
    print(f"🏗️  Model size: {config['models']['shoe_segmentation']['model_size']}")
    print(f"📊 Epochs: {config['models']['shoe_segmentation']['epochs']}")
    print(f"📦 Batch size: {config['models']['shoe_segmentation']['batch_size']}")
    print(f"🖼️  Image size: {config['models']['shoe_segmentation']['imgsz']}")
    
    try:
        print("\n🚀 Starting segmentation model training...")
        seg_model_path = trainer.train_segmentation_model(
            str(seg_data_yaml),
            config['models']['shoe_segmentation']
        )
        print(f"✅ Segmentation model training completed!")
        print(f"💾 Model saved to: {seg_model_path}")
        print(f"\n🎯 You can now use this model in the pipeline by updating your config")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
