# Dead simple CLI for shoeprint matching
import sys
import json
from pathlib import Path
from src.pipeline import ShoeprintPipeline
import argparse

def index(folder, output_json):
    pipeline = ShoeprintPipeline("config.yaml")
    seg_model = Path(pipeline.config['paths']['models']) / 'shoe_segmentation' / 'weights' / 'best.pt'
    feat_model = Path(pipeline.config['paths']['models']) / 'feature_detection' / 'weights' / 'best.pt'
    if seg_model.exists():
        pipeline.load_models(segmentation_path=str(seg_model))
    if feat_model.exists():
        pipeline.load_models(feature_path=str(feat_model))

    folder = Path(folder)
    exts = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    image_paths = []
    for ext in exts:
        image_paths.extend(folder.rglob(ext))
    def make_json_serializable(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(make_json_serializable(v) for v in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    for img_path in image_paths:
        print(f"DEBUG: Processing image: {img_path}")
        try:
            results = pipeline.process_image(str(img_path))
            # Debug: print if original_image is present and its type
            if 'original_image' in results:
                print(f"DEBUG: 'original_image' type: {type(results['original_image'])}")
                if results['original_image'] is not None:
                    print(f"DEBUG: 'original_image' shape: {getattr(results['original_image'], 'shape', None)}")
                else:
                    print(f"DEBUG: 'original_image' is None")
            else:
                print(f"DEBUG: 'original_image' not in results")
            if 'original_image' not in results or results['original_image'] is None:
                print(f"Skipped: {img_path} (image not loaded or processed)")
                continue
            shoe_id = img_path.parent.name
            pipeline.add_to_database(shoe_id, results)
            # Remove image arrays from metadata, keep only the path and stats
            for meta in pipeline.database.get('metadata', {}).values():
                if 'original_image' in meta:
                    meta['original_image'] = None
                if 'cropped_shoe' in meta:
                    meta['cropped_shoe'] = None
            serializable_db = make_json_serializable(pipeline.database)
            with open(output_json, 'w') as f:
                json.dump(serializable_db, f)
            print(f"Indexed and saved: {img_path}")
        except Exception as e:
            print(f"Failed: {img_path} ({e})")
    print(f"Done. Saved to {output_json}")

def search(db_json, query_image, top_k=10):
    pipeline = ShoeprintPipeline("config.yaml")
    seg_model = Path(pipeline.config['paths']['models']) / 'shoe_segmentation' / 'weights' / 'best.pt'
    feat_model = Path(pipeline.config['paths']['models']) / 'feature_detection' / 'weights' / 'best.pt'
    if seg_model.exists():
        pipeline.load_models(segmentation_path=str(seg_model))
    if feat_model.exists():
        pipeline.load_models(feature_path=str(feat_model))
    with open(db_json, 'r') as f:
        pipeline.database = json.load(f)
    results = pipeline.search(query_image, top_k=top_k)
    print("Rank | Image ID         | Score    | Image Path")
    print("----------------------------------------------------------")
    for idx, (image_id, score, metadata) in enumerate(results, 1):
        print(f"{idx:4} | {image_id:15} | {score:8.3f} | {metadata.get('path', '')}")

def main():
    parser = argparse.ArgumentParser(description='Shoeprint Matcher CLI')
    subparsers = parser.add_subparsers(dest='command')

    index_parser = subparsers.add_parser('index', help='Index shoeprint images in a folder')
    index_parser.add_argument('folder', help='Path to folder with shoeprint images')
    index_parser.add_argument('db', help='Path to output JSON database')

    search_parser = subparsers.add_parser('search', help='Search for best matches to a query image')
    search_parser.add_argument('db', help='Path to JSON database')
    search_parser.add_argument('query', help='Path to query shoeprint image')
    search_parser.add_argument('--top', type=int, default=10, help='Number of top matches to display')

    args = parser.parse_args()
    if args.command == 'index':
        index(args.folder, args.db)
    elif args.command == 'search':
        search(args.db, args.query, args.top)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
