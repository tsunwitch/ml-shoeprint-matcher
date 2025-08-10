#!/usr/bin/env python3
"""
Training script for axis detection model only.
This model will detect the longitudinal axis of the shoeprint.
Usage: python scripts/train_axis_detection.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from pathlib import Path
from src.models.trainer import ModelTrainer

def main():
    """Train only the axis detection model."""
    print("=== Axis Detection Model Training ===")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = ModelTrainer(config['paths']['models'])
    
    # Check if axis detection dataset exists
    axis_data_yaml = Path(config['paths']['yolo_datasets']) / 'axis_detection' / 'data.yaml'
    
    if not axis_data_yaml.exists():
        print(f"❌ Axis detection dataset not found at {axis_data_yaml}")
        print("Please create the axis detection dataset first")
        print("This dataset should contain images with axis annotations (line from toe to heel)")
        return
    
    print(f"📁 Using dataset: {axis_data_yaml}")
    print(f"🏗️  Model size: {config['models']['axis_detection']['model_size']}")
    print(f"📊 Epochs: {config['models']['axis_detection']['epochs']}")
    print(f"📦 Batch size: {config['models']['axis_detection']['batch_size']}")
    print(f"🖼️  Image size: {config['models']['axis_detection']['imgsz']}")
    
    print("\n📝 This model will learn to detect:")
    print("   - Longitudinal axis of the shoe (toe to heel)")
    print("   - Handle rotated shoes at any angle")
    print("   - Provide rotation-invariant axis detection")
    
    try:
        print("\n🚀 Starting axis detection model training...")
        axis_model_path = trainer.train_axis_detection_model(
            str(axis_data_yaml),
            config['models']['axis_detection']
        )
        print(f"✅ Axis detection model training completed!")
        print(f"💾 Model saved to: {axis_model_path}")
        print(f"\n🎯 You can now use this model for rotation-invariant feature matching")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
