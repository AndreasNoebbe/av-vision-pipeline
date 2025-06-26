"""
frontend_app.py

Streamlit frontend for AV Vision Pipeline
Connected to live HuggingFace Spaces backend
"""

import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import json

# Page configuration
st.set_page_config(
    page_title="AV Vision Pipeline",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 AV Vision Pipeline")
st.markdown("**Autonomous Vehicle Object Detection System**")
st.markdown("Upload an image to detect vehicles, pedestrians, and road infrastructure in real-time!")

# Live API endpoint
API_URL = "https://andedam-av-vision-pipeline.hf.space"

def check_api_health():
    """Check if the HuggingFace API backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def detect_objects(image_file):
    """Send image to HuggingFace API backend for object detection"""
    try:
        files = {"file": image_file}
        response = requests.post(f"{API_URL}/detect", files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def draw_detections(image, detections):
    """Draw bounding boxes on the image"""
    img_array = np.array(image)
    
    # Color mapping for different categories
    colors = {
        'vehicle': (0, 255, 0),      # Green
        'pedestrian': (255, 0, 0),   # Red  
        'infrastructure': (0, 0, 255) # Blue
    }
    
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        category = detection['av_category']
        color = colors.get(category, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{detection['class_name']}: {detection['confidence']:.2f}"
        cv2.putText(img_array, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return Image.fromarray(img_array)

# Sidebar for API status
with st.sidebar:
    st.header("System Status")
    
    api_status, api_info = check_api_health()
    if api_status:
        st.success("✅ Live API Online")
        if api_info and 'model' in api_info:
            model_info = api_info['model']
            st.info(f"🤖 Model: {model_info['model_name']}")
            st.info(f"🎯 Confidence: {model_info['confidence_threshold']}")
    else:
        st.error("❌ API Offline")
    
    st.header("Detection Categories")
    st.markdown("🟢 **Vehicles**: Cars, trucks, buses, motorcycles")
    st.markdown("🔴 **Pedestrians**: People, cyclists") 
    st.markdown("🔵 **Infrastructure**: Traffic lights, stop signs")
    
    st.header("Live System")
    st.markdown("✅ **Backend**: HuggingFace Spaces")
    st.markdown("✅ **Model**: YOLOv8 + PyTorch")
    st.markdown("✅ **Frontend**: Streamlit Cloud")

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image with vehicles, pedestrians, or traffic infrastructure"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Detection button
        if st.button("🔍 Detect Objects", type="primary"):
            if not api_status:
                st.error("Cannot connect to API. Please try again later.")
            else:
                with st.spinner("Running real-time object detection..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Get detections from live API
                    detections = detect_objects(uploaded_file)
                    
                    if detections is not None:
                        # Store results in session state
                        st.session_state.detections = detections
                        st.session_state.image = image
                        st.success(f"✅ Detected {len(detections)} objects!")

with col2:
    st.header("Detection Results")
    
    if 'detections' in st.session_state and 'image' in st.session_state:
        detections = st.session_state.detections
        image = st.session_state.image
        
        if detections:
            # Draw detections on image
            result_image = draw_detections(image, detections)
            st.image(result_image, caption="Live Detection Results", use_container_width=True)
            
            # Detection statistics
            st.subheader("Detection Summary")
            
            # Count by category
            category_counts = {}
            for detection in detections:
                category = detection['av_category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Display counts
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("🟢 Vehicles", category_counts.get('vehicle', 0))
            with col_b:
                st.metric("🔴 Pedestrians", category_counts.get('pedestrian', 0))
            with col_c:
                st.metric("🔵 Infrastructure", category_counts.get('infrastructure', 0))
            
            # Detailed results
            st.subheader("Detailed Results")
            
            for i, detection in enumerate(detections, 1):
                with st.expander(f"Detection {i}: {detection['class_name']} ({detection['confidence']:.1%})"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write(f"**Class:** {detection['class_name']}")
                        st.write(f"**Category:** {detection['av_category']}")
                        st.write(f"**Confidence:** {detection['confidence']:.1%}")
                    
                    with col_b:
                        st.write(f"**Center:** ({detection['center'][0]:.0f}, {detection['center'][1]:.0f})")
                        bbox = detection['bbox']
                        st.write(f"**Box:** ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")
            
            # Export results
            st.subheader("Export Results")
            
            # JSON download
            json_data = json.dumps(detections, indent=2)
            st.download_button(
                label="📥 Download Detection Data",
                data=json_data,
                file_name="detection_results.json",
                mime="application/json"
            )
        else:
            st.info("No objects detected in the uploaded image.")
    else:
        st.info("Upload an image and click 'Detect Objects' to see live detection results.")

# Footer
st.markdown("---")
st.markdown("**AV Vision Pipeline** - Built by Andreas Nøbbe | [GitHub Repository](https://github.com/AndreasNoebbe/av-vision-pipeline) | [Live API](https://andedam-av-vision-pipeline.hf.space/docs)")
