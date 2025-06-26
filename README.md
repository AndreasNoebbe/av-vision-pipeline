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

## Example:
![image](https://github.com/user-attachments/assets/38be8506-b038-4693-8359-aec541fddbab)

![image](https://github.com/user-attachments/assets/5149d78c-8c23-4554-9a4a-6221f6091e3d)
