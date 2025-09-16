#!/usr/bin/env python3
"""
Master training script - trains all models sequentially.
Usage: python scripts/train_all_models.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from pathlib import Path
from src.models.trainer import ModelTrainer

def main():
    """Train all models in sequence."""
    print("=== Training All Models ===")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = ModelTrainer(config['paths']['models'])
    
    # Check which datasets exist
    datasets = {
        'segmentation': Path(config['paths']['yolo_datasets']) / 'shoe_segmentation' / 'data.yaml',
        'detection': Path(config['paths']['yolo_datasets']) / 'shoe_detection' / 'data.yaml',
        'feature_detection': Path(config['paths']['yolo_datasets']) / 'feature_detection' / 'data.yaml',
        'axis_detection': Path(config['paths']['yolo_datasets']) / 'axis_detection' / 'data.yaml'
    }
    
    available_datasets = {name: path for name, path in datasets.items() if path.exists()}
    missing_datasets = {name: path for name, path in datasets.items() if not path.exists()}
    
    if missing_datasets:
        print("⚠️  Missing datasets:")
        for name, path in missing_datasets.items():
            print(f"   - {name}: {path}")
        print("Please create missing datasets first.")
        print()
    
    if not available_datasets:
        print("❌ No datasets found. Please run data preparation first.")
        return
    
    print(f"✅ Found {len(available_datasets)} datasets:")
    for name in available_datasets:
        print(f"   - {name}")
    print()
    
    trained_models = []
    
    # 1. Train segmentation model
    if 'segmentation' in available_datasets:
        print("🚀 Training shoe segmentation model...")
        try:
            seg_model_path = trainer.train_segmentation_model(
                str(available_datasets['segmentation']),
                config['models']['shoe_segmentation']
            )
            print(f"✅ Segmentation model saved to: {seg_model_path}")
            trained_models.append(('Segmentation', seg_model_path))
        except Exception as e:
            print(f"❌ Segmentation training failed: {e}")
        print()

    # 2. Train shoe detection model
    if 'detection' in available_datasets:
        print("🚀 Training shoe detection model (bounding boxes)...")
        try:
            det_model_path = trainer.train_detection_model(
                str(available_datasets['detection']),
                config['models']['shoe_detection']
            )
            print(f"✅ Detection model saved to: {det_model_path}")
            trained_models.append(('Detection', det_model_path))
        except Exception as e:
            print(f"❌ Detection training failed: {e}")
        print()

    # 3. Train feature detection model
    if 'feature_detection' in available_datasets:
        print("🚀 Training feature detection model...")
        try:
            feat_model_path = trainer.train_feature_model(
                str(available_datasets['feature_detection']),
                config['models']['feature_detection']
            )
            print(f"✅ Feature detection model saved to: {feat_model_path}")
            trained_models.append(('Feature Detection', feat_model_path))
        except Exception as e:
            print(f"❌ Feature detection training failed: {e}")
        print()

    # 4. Train axis detection model
    if 'axis_detection' in available_datasets:
        print("🚀 Training axis detection model...")
        try:
            axis_model_path = trainer.train_axis_detection_model(
                str(available_datasets['axis_detection']),
                config['models']['axis_detection']
            )
            print(f"✅ Axis detection model saved to: {axis_model_path}")
            trained_models.append(('Axis Detection', axis_model_path))
        except Exception as e:
            print(f"❌ Axis detection training failed: {e}")
        print()
    
    # Summary
    print("="*50)
    print("🎉 Training Summary:")
    if trained_models:
        for model_name, model_path in trained_models:
            print(f"✅ {model_name}: {model_path}")
    else:
        print("❌ No models were trained successfully")
    
    print("\n🚀 You can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()
