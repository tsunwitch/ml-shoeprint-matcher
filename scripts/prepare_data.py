import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from pathlib import Path
from src.data.loader import ShoeDataLoader
from src.data.converter import YOLOConverter
from src.data.splitter import DataSplitter

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Loading shoe data...")
    loader = ShoeDataLoader(config['paths']['raw_data'])
    data = loader.load_all_data()
    
    print(f"Found {len(data['annotations'])} shoes")
    
    shoe_ids = list(data['annotations'].keys())
    splitter = DataSplitter(
        train_ratio=config['data']['train_split'],
        val_ratio=config['data']['val_split'],
        test_ratio=config['data']['test_split'],
        seed=config['data']['random_seed']
    )
    
    split_info = splitter.split_by_shoe_id(shoe_ids)
    
    print("\nData split statistics:")
    stats = splitter.get_split_stats(split_info)
    for key, value in stats.items():
        if 'percentage' in key:
            print(f"  {key}: {value:.1f}%")
        else:
            print(f"  {key}: {value}")
    
    converter = YOLOConverter(config['paths']['yolo_datasets'])
    
    print("\nCreating shoe segmentation dataset...")
    seg_path = converter.create_segmentation_dataset(data, loader, split_info)
    print(f"  Saved to: {seg_path}")
    
    print("\nCreating feature detection dataset...")
    feat_path = converter.create_feature_dataset(data, loader, split_info)
    print(f"  Saved to: {feat_path}")
    
    print("\nData preparation complete!")
    print(f"Next step: Run train_models.py to train the YOLO models")

if __name__ == "__main__":
    main()