#!/usr/bin/env python3
"""
Legacy training script - now redirects to train_all_models.py
For individual model training, use:
- scripts/train_segmentation.py
- scripts/train_feature_detection.py  
- scripts/train_axis_detection.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("⚠️  This script is deprecated.")
    print("Please use the new individual training scripts:")
    print("  🔧 Train segmentation:     python scripts/train_segmentation.py")
    print("  🔍 Train feature detection: python scripts/train_feature_detection.py")
    print("  📐 Train axis detection:   python scripts/train_axis_detection.py")
    print("  🚀 Train all models:       python scripts/train_all_models.py")
    print()
    
    choice = input("Do you want to train all models now? [y/N]: ").lower().strip()
    if choice in ['y', 'yes']:
        print("Redirecting to train_all_models.py...")
        import subprocess
        subprocess.run([sys.executable, "scripts/train_all_models.py"])
    else:
        print("Exiting. Use the individual scripts above for specific model training.")

if __name__ == "__main__":
    main()