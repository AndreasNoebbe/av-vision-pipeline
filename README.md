# AV Vision Pipeline

Real-time object detection system for autonomous vehicles using FastAPI and YOLO.

## Live Demo

**[Try the Live System](https://av-vision-pipeline-8tecatavznrsstmabcuu8c.streamlit.app)** • **[API Documentation](https://andedam-av-vision-pipeline.hf.space/docs)**

## Overview

Detects vehicles, pedestrians, and traffic infrastructure in uploaded images. Returns structured JSON data with confidence scores and bounding box coordinates.

**Detection Categories:**
-  **Vehicles:** Cars, trucks, buses, motorcycles
-  **Pedestrians:** People, cyclists  
-  **Infrastructure:** Traffic lights, stop signs

## Architecture

| Component | Technology | Deployment |
|-----------|------------|------------|
| **Backend** | FastAPI + YOLOv8 | HuggingFace Spaces |
| **Frontend** | Streamlit | Streamlit Cloud |
| **ML Model** | PyTorch + OpenCV | Cloud-hosted |

## Project Structure
├── src/
│   ├── api.py          # FastAPI application
│   ├── detector.py     # YOLO detection logic
│   └── __init__.py
├── tests/
│   └── test_detector.py
├── frontend_app.py     # Streamlit interface
├── requirements.txt    # Streamlit deployment
└── requirements-full.txt # Local development
