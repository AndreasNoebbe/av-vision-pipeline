AV Vision Pipeline
Real-time object detection system for autonomous vehicles using FastAPI and YOLO.
Live Demo: https://av-vision-pipeline-8tecatavznrsstmabcuu8c.streamlit.app
API Docs: https://andedam-av-vision-pipeline.hf.space/docs
What it does
Detects vehicles, pedestrians, and traffic infrastructure in images. Returns JSON with confidence scores and bounding box coordinates.
Detects: Cars, trucks, buses, motorcycles, people, cyclists, traffic lights, stop signs
Architecture

Backend: FastAPI + YOLOv8 (HuggingFace Spaces)
Frontend: Streamlit interface (Streamlit Cloud)
Model: PyTorch + OpenCV
