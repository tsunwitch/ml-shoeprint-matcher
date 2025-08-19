import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import yaml
from PIL import Image

from src.pipeline import ShoeprintPipeline
from src.data.loader import ShoeDataLoader
from src.utils.visualization import draw_features, draw_bbox, create_comparison_image

st.set_page_config(page_title="Shoeprint Forensics", layout="wide")

@st.cache_resource
def load_pipeline():
    pipeline = ShoeprintPipeline("config.yaml")
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    seg_model = Path(config['paths']['models']) / 'shoe_segmentation' / 'weights' / 'best.pt'
    feat_model = Path(config['paths']['models']) / 'feature_detection' / 'weights' / 'best.pt'
    
    if seg_model.exists():
        pipeline.load_models(segmentation_path=str(seg_model))
    
    if feat_model.exists():
        pipeline.load_models(feature_path=str(feat_model))
    
    return pipeline

@st.cache_resource
def load_database():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    loader = ShoeDataLoader(config['paths']['raw_data'])
    data = loader.load_all_data()
    
    pipeline = load_pipeline()
    
    for shoe_id in data['images'].keys():
        for annotator in data['images'][shoe_id]:
            left_added = False
            right_added = False
            for print_id, image_path in data['images'][shoe_id][annotator].items():
                if 'L1' in print_id and not left_added:
                    try:
                        results = pipeline.process_image(image_path)
                        pipeline.add_to_database(shoe_id, results)
                        left_added = True
                    except Exception as e:
                        st.warning(f"Failed to process {image_path}: {e}")
                elif 'P1' in print_id and not right_added:
                    try:
                        results = pipeline.process_image(image_path)
                        pipeline.add_to_database(shoe_id, results)
                        right_added = True
                    except Exception as e:
                        st.warning(f"Failed to process {image_path}: {e}")
                
                if left_added and right_added:
                    break
            break
    
    return pipeline

def main():
    st.title("🦶 Shoeprint Forensics System")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Detection", "Feature Detection", "Search", "DTW Profile", "Config"])
    
    with tab1:
        st.header("Shoe Detection")
        
        uploaded_file = st.file_uploader("Upload a shoeprint image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            pipeline = load_pipeline()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image_np, use_container_width=True)
            
            with col2:
                st.subheader("Detection Results")
                
                if pipeline.segmenter:
                    from src.utils.visualization import draw_mask_overlay
                    with open('config.yaml', 'r') as f:
                        config = yaml.safe_load(f)
                    margin_ratio = config['models']['shoe_segmentation'].get('horizontal_margin_ratio', 0.1)
                    mask = pipeline.segmenter.get_shoe_mask(image_np, horizontal_margin_ratio=margin_ratio)
                    img_with_mask = draw_mask_overlay(image_np, mask, color=(0,255,0), alpha=0.3)
                    st.image(img_with_mask, use_container_width=True)
                    st.caption("Segmentation mask (green overlay)")
                else:
                    st.caption("No segmentation model loaded")
                
                if pipeline.feature_detector:
                    st.caption("Feature detection will work on the full image")
                else:
                    st.caption("Feature detection model not loaded")
    
    with tab2:
        st.header("Feature Detection")
        
        uploaded_file2 = st.file_uploader("Upload image for feature detection", type=['jpg', 'jpeg', 'png'], key="feat")
        
        if uploaded_file2 is not None:
            image = Image.open(uploaded_file2)
            image_np = np.array(image)
            
            pipeline = load_pipeline()
            
            if pipeline.feature_detector:
                
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file2.getbuffer())
                    tmp_path = tmp_file.name
                
                
                confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
                
                results = pipeline.process_image(tmp_path)
                
                
                if pipeline.feature_detector:
                    features = pipeline.feature_detector.detect_features(
                        results['cropped_shoe'], 
                        confidence=confidence
                    )
                    results['features'] = features
                
                
                import os
                os.unlink(tmp_path)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image_np, use_container_width=True)
                
                with col2:
                    st.subheader("Detected Features")
                    if 'features' in results and results['features']:
                        from src.utils.visualization import draw_features
                        img_with_features = draw_features(image_np, results['features'])
                        st.image(img_with_features, use_container_width=True)
                        st.info(f"Found {len(results['features'])} features")
                    else:
                        st.warning("No features detected")
                        
                        st.caption(f"Image shape: {image_np.shape}")
                        st.caption(f"Using confidence: {confidence}")
                        if pipeline.segmenter:
                            st.caption("Segmentation model loaded but not used for cropping ✓")
                        if pipeline.feature_detector:
                            st.caption("Feature model loaded ✓")
            else:
                st.warning("Feature detection model not loaded. Run train_models.py first.")
    
    with tab3:
        st.header("Shoe Search")
        
        if st.button("Load Database"):
            with st.spinner("Loading database..."):
                pipeline = load_database()
                st.success(f"Loaded {len(pipeline.database['profiles'])} shoe images into database")
        
        uploaded_query = st.file_uploader("Upload query shoeprint", type=['jpg', 'jpeg', 'png'], key="search")
        
        if uploaded_query is not None:
            image = Image.open(uploaded_query)
            image_np = np.array(image)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Query Image")
                st.image(image_np, use_container_width=True)
            
            if st.button("Search"):
                pipeline = load_database()
                
                with st.spinner("Searching..."):
                    
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_query.getbuffer())
                        temp_path = tmp_file.name
                    
                    
                    query_results = pipeline.process_image(temp_path)
                    query_features = query_results.get('features', [])
                    query_features_vis = query_results.get('features_original', [])
                    
                    results = pipeline.search(temp_path, top_k=10)
                    
                    import os
                    os.unlink(temp_path)
                    
                    with col2:
                        st.subheader("Query Features & DTW Profile")
                        qcol1, qcol2, qcol3 = st.columns(3)
                        with qcol1:
                            # Unified overlay: segmentation mask, axis, features (blue)
                            from src.utils.visualization import draw_mask_overlay, draw_axis, draw_features
                            mask = None
                            axis_line = query_results.get('axis_line', None)
                            if pipeline.segmenter:
                                with open('config.yaml', 'r') as f:
                                    config = yaml.safe_load(f)
                                margin_ratio = config['models']['shoe_segmentation'].get('horizontal_margin_ratio', 0.1)
                                mask = pipeline.segmenter.get_shoe_mask(image_np, horizontal_margin_ratio=margin_ratio)
                            img_with_overlay = image_np.copy()
                            if mask is not None:
                                img_with_overlay = draw_mask_overlay(img_with_overlay, mask, color=(0,255,0), alpha=0.3)
                            if axis_line is not None:
                                img_with_overlay = draw_axis(img_with_overlay, axis_line, color=(255,0,0), thickness=4)
                            if query_features_vis:
                                img_with_overlay = draw_features(img_with_overlay, query_features_vis, color=(0, 0, 255))
                            st.image(img_with_overlay, use_container_width=True)
                            st.caption("Segmentation mask (green) is exactly the region used for DTW profile extraction. Axis (red), features (blue boxes)")
                        with qcol2:
                            from src.utils.image_ops import extract_axis_profile
                            axis_line = query_results.get('axis_line', None)
                            mask = None
                            pipeline = load_pipeline()
                            if pipeline.segmenter:
                                with open('config.yaml', 'r') as f:
                                    config = yaml.safe_load(f)
                                margin_ratio = config['models']['shoe_segmentation'].get('horizontal_margin_ratio', 0.1)
                                mask = pipeline.segmenter.get_shoe_mask(image_np, horizontal_margin_ratio=margin_ratio)
                            if axis_line is not None:
                                left_profile, right_profile = extract_axis_profile(image_np, axis_line, num_samples=100, mask=mask)
                                import matplotlib.pyplot as plt
                                fig, ax = plt.subplots(figsize=(4, 6))
                                ax.plot(left_profile, range(len(left_profile)), 'g-', linewidth=2, label='Left')
                                ax.plot(right_profile, range(len(right_profile)), 'b-', linewidth=2, label='Right')
                                ax.set_ylabel('Position along axis')
                                ax.set_xlabel('Average intensity')
                                ax.set_title('Query DTW Profiles (Left/Right)')
                                ax.grid(True, alpha=0.3)
                                ax.invert_yaxis()
                                ax.legend()
                                st.pyplot(fig)
                        st.divider()
                        st.subheader("Top 10 Matches")
                        for i in range(0, len(results), 2):
                            cols = st.columns(2)
                            for j in range(2):
                                if i + j < len(results):
                                    image_id, score, metadata = results[i + j]
                                    mcol1, mcol2 = cols[j].columns(2)
                                    with mcol1:
                                        # Show segmentation mask and axis for match
                                        from src.utils.visualization import draw_mask_overlay, draw_axis
                                        if 'original_image' in metadata:
                                            img = metadata['original_image']
                                        else:
                                            img = cv2.imread(metadata['path'])
                                            if img is not None:
                                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        axis_line = metadata.get('axis_line', None)
                                        mask = None
                                        if pipeline.segmenter and img is not None:
                                            mask = pipeline.segmenter.get_shoe_mask(img)
                                        img_with_overlay = img.copy() if img is not None else None
                                        features_vis = metadata.get('features_original', [])
                                        if img_with_overlay is not None:
                                            if mask is not None:
                                                img_with_overlay = draw_mask_overlay(img_with_overlay, mask, color=(0,255,0), alpha=0.3)
                                            if axis_line is not None:
                                                img_with_overlay = draw_axis(img_with_overlay, axis_line, color=(255,0,0), thickness=4)
                                            if features_vis:
                                                img_with_overlay = draw_features(img_with_overlay, features_vis, color=(0, 0, 255))
                                                feature_count = len(features_vis)
                                            else:
                                                feature_count = 0
                                            st.image(img_with_overlay, use_container_width=True)
                                            st.caption("Segmentation mask (green), detected axis (red), and features (blue boxes)")
                                        else:
                                            feature_count = 0
                                        st.caption(f"#{i+j+1}: {image_id}")
                                        st.caption(f"Score: {score:.3f}")
                                        st.caption(f"Features: {feature_count}")
                                        if query_features and metadata.get('features'):
                                            matches = pipeline.feature_matcher.match_features(
                                                query_features, 
                                                metadata['features']
                                            )
                                            st.caption(f"Matching features: {matches}/{len(query_features)}")
                                    with mcol2:
                                        from src.utils.image_ops import extract_axis_profile
                                        axis_line = metadata.get('axis_line', None)
                                        match_img = metadata.get('original_image', None)
                                        mask = None
                                        pipeline = load_pipeline()
                                        if pipeline.segmenter and match_img is not None:
                                            mask = pipeline.segmenter.get_shoe_mask(match_img)
                                        if match_img is not None and axis_line is not None:
                                            left_profile, right_profile = extract_axis_profile(match_img, axis_line, num_samples=100, mask=mask)
                                            import matplotlib.pyplot as plt
                                            fig, ax = plt.subplots(figsize=(4, 6))
                                            ax.plot(left_profile, range(len(left_profile)), 'r-', linewidth=2, label='Left')
                                            ax.plot(right_profile, range(len(right_profile)), 'm-', linewidth=2, label='Right')
                                            ax.set_ylabel('Position along axis')
                                            ax.set_xlabel('Average intensity')
                                            ax.set_title('Match DTW Profiles (Left/Right)')
                                            ax.grid(True, alpha=0.3)
                                            ax.invert_yaxis()
                                            ax.legend()
                                            st.pyplot(fig)
    
    with tab4:
        st.header("Axis Detection & DTW Profile")
        from src.matching.axis_detection import detect_shoe_axis
        from src.utils.visualization import draw_axis
        uploaded_file4 = st.file_uploader("Upload image for axis visualization", type=['jpg', 'jpeg', 'png'], key="axis")
        if uploaded_file4 is not None:
            image = Image.open(uploaded_file4)
            image_np = np.array(image)
            axis_line = detect_shoe_axis(image_np)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Original Image")
                st.image(image_np, use_container_width=True)
            with col2:
                st.subheader("Detected Shoe Axis")
                img_with_axis = draw_axis(image_np, axis_line, color=(255, 0, 0), thickness=4)
                st.image(img_with_axis, use_container_width=True)
                st.caption(f"🔵 Blue: Detected axis from Canny/contour/PCA")
            with col3:
                st.subheader("DTW Profile")
                from src.utils.image_ops import extract_axis_profile
                profile = extract_axis_profile(image_np, axis_line, num_samples=100)
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 8))
                ax.plot(profile, range(len(profile)), 'b-', linewidth=2)
                ax.set_ylabel('Position along axis')
                ax.set_xlabel('Average intensity')
                ax.set_title('DTW Profile (along detected axis)')
                ax.grid(True, alpha=0.3)
                ax.invert_yaxis()
                st.pyplot(fig)
                st.caption(f"Profile has {len(profile)} sample points")
    
    with tab5:
        st.header("Configuration")
        
        with open('config.yaml', 'r') as f:
            config_content = f.read()
        
        st.subheader("Current config.yaml:")
        st.code(config_content, language='yaml')
        
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Feature Detection Confidence:** {config['models']['feature_detection']['confidence']}")
            st.info(f"**DTW Distance Threshold:** {config['matching']['dtw']['distance_threshold']}")
        with col2:
            st.info(f"**Min Feature Matches:** {config['matching']['features']['min_matches']}")
            st.info(f"**Distance Threshold:** {config['matching']['features']['distance_threshold']}")
        
        st.caption("To change these values, edit config.yaml and restart the app")

if __name__ == "__main__":
    main()