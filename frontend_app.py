"""
frontend_app.py

Streamlit frontend for AV Vision Pipeline (Demo Version)
This creates a web interface demonstrating the object detection system.
"""

import streamlit as st
import requests
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
st.markdown("Upload an image to see the object detection interface (Demo Version)")

# Demo mode warning
st.warning("⚠️ **Demo Mode**: This is a frontend demonstration. The actual object detection runs on a local FastAPI backend. [View full technical implementation on GitHub](https://github.com/AndreasNoebbe/av-vision-pipeline)")

# API endpoint (will fail in demo mode)
API_URL = "https://andedam-av-vision-pipeline.hf.space"

def check_api_health():
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def create_demo_detections():
    """Create demo detection results for showcase"""
    return [
        {
            "class_name": "car",
            "av_category": "vehicle",
            "confidence": 0.92,
            "bbox": [150, 200, 350, 350],
            "center": [250, 275]
        },
        {
            "class_name": "person", 
            "av_category": "pedestrian",
            "confidence": 0.87,
            "bbox": [400, 150, 480, 320],
            "center": [440, 235]
        },
        {
            "class_name": "traffic light",
            "av_category": "infrastructure", 
            "confidence": 0.94,
            "bbox": [80, 50, 120, 150],
            "center": [100, 100]
        }
    ]

def draw_demo_detections(image):
    """Draw demo bounding boxes on the image"""
    import cv2
    
    img_array = np.array(image)
    
    # Color mapping for different categories
    colors = {
        'vehicle': (0, 255, 0),      # Green
        'pedestrian': (255, 0, 0),   # Red  
        'infrastructure': (0, 0, 255) # Blue
    }
    
    detections = create_demo_detections()
    
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Scale to image size
        h, w = img_array.shape[:2]
        x1 = int(x1 * w / 640)
        y1 = int(y1 * h / 480) 
        x2 = int(x2 * w / 640)
        y2 = int(y2 * h / 480)
        
        category = detection['av_category']
        color = colors.get(category, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 3)
        
        # Draw label
        label = f"{detection['class_name']}: {detection['confidence']:.2f}"
        cv2.putText(img_array, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return Image.fromarray(img_array)

# Sidebar for system info
with st.sidebar:
    st.header("System Status")
    
    if check_api_health():
        st.success("✅ API Online")
    else:
        st.error("❌ API Offline (Demo Mode)")
        st.info("💡 This demo shows the interface design. The full system with FastAPI backend runs locally.")
    
    st.header("Detection Categories")
    st.markdown("🟢 **Vehicles**: Cars, trucks, buses, motorcycles")
    st.markdown("🔴 **Pedestrians**: People, cyclists") 
    st.markdown("🔵 **Infrastructure**: Traffic lights, stop signs")
    
    st.header("Technical Stack")
    st.markdown("• **Backend**: FastAPI + YOLO")
    st.markdown("• **Frontend**: Streamlit")
    st.markdown("• **ML**: PyTorch + OpenCV")
    st.markdown("• **Testing**: Pytest")

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to see the detection interface demo"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Detection button
        if st.button("🔍 Detect Objects (Demo)", type="primary"):
            with st.spinner("Running demo detection..."):
                # Create demo results
                st.session_state.demo_detections = create_demo_detections()
                st.session_state.demo_image = image
                st.success("✅ Demo detection completed!")

with col2:
    st.header("Detection Results")
    
    if 'demo_detections' in st.session_state and 'demo_image' in st.session_state:
        detections = st.session_state.demo_detections
        image = st.session_state.demo_image
        
        # Draw demo detections on image
        result_image = draw_demo_detections(image)
        st.image(result_image, caption="Demo Detection Results", use_container_width=True)
        
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
                    st.write(f"**Center:** ({detection['center'][0]}, {detection['center'][1]})")
                    bbox = detection['bbox']
                    st.write(f"**Bounding Box:** ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")
        
        # Export results
        st.subheader("Export Demo Results")
        
        # JSON download
        json_data = json.dumps(detections, indent=2)
        st.download_button(
            label="📥 Download Demo JSON",
            data=json_data,
            file_name="demo_detection_results.json",
            mime="application/json"
        )
    else:
        st.info("Upload an image and click 'Detect Objects (Demo)' to see the interface demonstration.")

# Technical implementation section
st.markdown("---")

with st.expander("🔧 Technical Implementation Details"):
    st.markdown("""
    ### Architecture Overview
    
    **Backend (FastAPI)**:
    - RESTful API with automatic OpenAPI documentation
    - YOLO object detection model integration
    - Image upload and processing endpoints
    - Error handling and validation
    
    **Frontend (Streamlit)**:
    - Interactive web interface
    - Real-time image upload and display
    - Results visualization with bounding boxes
    - Export functionality for detection data
    
    **Object Detection**:
    - YOLOv8 model powered by PyTorch
    - Detects vehicles, pedestrians, and infrastructure
    - Returns confidence scores and bounding box coordinates
    - Optimized for autonomous vehicle applications
    
    **Development Practices**:
    - Unit testing with pytest (6 tests passing)
    - Virtual environment isolation
    - Modular code structure
    - Comprehensive error handling
    - Version control with Git
    """)

# Footer
st.markdown("---")
st.markdown("**AV Vision Pipeline** - Built by Andreas Nøbbe | [GitHub Repository](https://github.com/AndreasNoebbe/av-vision-pipeline)")