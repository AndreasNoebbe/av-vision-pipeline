"""
frontend_app.py

Streamlit frontend for AV Vision Pipeline
This creates a web interface for uploading images and viewing object detection results.
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
st.markdown("Upload an image to detect vehicles, pedestrians, and road infrastructure.")

# API endpoint
API_URL = "http://localhost:8000"

def check_api_health():
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def detect_objects(image_file):
    """Send image to FastAPI backend for object detection"""
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
    
    if check_api_health():
        st.success("✅ API Online")
    else:
        st.error("❌ API Offline")
        st.markdown("Make sure your FastAPI server is running:")
        st.code("uvicorn src.api:app --reload --host 0.0.0.0 --port 8000")
    
    st.header("Detection Categories")
    st.markdown("🟢 **Vehicles**: Cars, trucks, buses, motorcycles")
    st.markdown("🔴 **Pedestrians**: People, cyclists") 
    st.markdown("🔵 **Infrastructure**: Traffic lights, stop signs")

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
            if not check_api_health():
                st.error("Cannot connect to API. Please start the FastAPI server.")
            else:
                with st.spinner("Analyzing image..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Get detections
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
            st.image(result_image, caption="Detection Results", use_container_width=True)
            
            # Detection statistics
            st.subheader("Detection Summary")
            
            # Count by category
            category_counts = {}
            for detection in detections:
                category = detection['av_category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Display counts
            for category, count in category_counts.items():
                emoji = "🟢" if category == "vehicle" else "🔴" if category == "pedestrian" else "🔵"
                st.metric(f"{emoji} {category.title()}s", count)
            
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
                        st.write(f"**Bounding Box:** ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")
            
            # Export results
            st.subheader("Export Results")
            
            # JSON download
            json_data = json.dumps(detections, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="detection_results.json",
                mime="application/json"
            )
        else:
            st.info("No objects detected in the image.")
    else:
        st.info("Upload an image and click 'Detect Objects' to see results here.")

# Footer
st.markdown("---")
st.markdown("**AV Vision Pipeline** - Built with FastAPI + Streamlit for autonomous vehicle object detection")