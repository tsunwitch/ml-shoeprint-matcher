# Dead simple CLI for shoeprint matching
import sys
import json
import csv
from pathlib import Path
from src.pipeline import ShoeprintPipeline
import argparse

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

    # Save results to file if requested
    if hasattr(search, 'output_file') and search.output_file:
        output_file = search.output_file
        if output_file.lower().endswith('.json'):
            out = [
                {'rank': idx, 'image_id': image_id, 'score': score, 'image_path': metadata.get('path', '')}
                for idx, (image_id, score, metadata) in enumerate(results, 1)
            ]
            with open(output_file, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"Results saved to {output_file}")
        else:
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['rank', 'image_id', 'score', 'image_path'])
                for idx, (image_id, score, metadata) in enumerate(results, 1):
                    writer.writerow([idx, image_id, score, metadata.get('path', '')])
            print(f"Results saved to {output_file}")

    # Add query image to database if requested
    if hasattr(search, 'add_query') and search.add_query:
        already_present = any(meta.get('path', '') == query_image for meta in pipeline.database.get('metadata', {}).values())
        if already_present:
            print(f"Query image already in database, not adding: {query_image}")
        else:
            results_query = pipeline.process_image(query_image)
            shoe_id = Path(query_image).parent.name
            pipeline.add_to_database(shoe_id, results_query)
            for meta in pipeline.database.get('metadata', {}).values():
                if 'original_image' in meta:
                    meta['original_image'] = None
                if 'cropped_shoe' in meta:
                    meta['cropped_shoe'] = None
            serializable_db = make_json_serializable(pipeline.database)
            with open(db_json, 'w') as f:
                json.dump(serializable_db, f)
            print(f"Query image added to database: {query_image}")

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
    search_parser.add_argument('--output', help='Path to output CSV or JSON file with matches')
    search_parser.add_argument('--add-query', action='store_true', help='Add query image to database after searching')

    args = parser.parse_args()
    if args.command == 'index':
        index(args.folder, args.db)
    elif args.command == 'search':
        # Pass output file and add_query to search function using attributes
        search.output_file = args.output if args.output else None
        search.add_query = args.add_query if args.add_query else False
        search(args.db, args.query, args.top)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
