"""
src/detector.py

Object Detection Module for Autonomous Vehicle Vision Pipeline

This module contains our core object detection logic.
Why we structure it this way:
- Single Responsibility: This module only handles object detection
- Testable: Easy to write unit tests for this class
- Reusable: Can be imported by API, tests, or other modules
- Maintainable: Clear interface and documentation
"""

import logging
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Set up logging for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectDetectionError(Exception):
    """Custom exception for detection-related errors"""
    pass


class AVObjectDetector:
    """
    Autonomous Vehicle Object Detector
    
    This class handles object detection for autonomous vehicle applications.
    It's designed to detect vehicles, pedestrians, and other road objects.
    
    Why we use a class instead of just functions?
    - State management: Model stays loaded in memory
    - Configuration: Easy to set detection parameters
    - Extensibility: Easy to add new detection methods
    """
    
    # Classes relevant to autonomous vehicles
    AV_RELEVANT_CLASSES = {
        'person': 'pedestrian',
        'bicycle': 'vehicle', 
        'car': 'vehicle',
        'motorcycle': 'vehicle',
        'bus': 'vehicle',
        'truck': 'vehicle',
        'traffic light': 'infrastructure',
        'stop sign': 'infrastructure'
    }
    
    def __init__(self, 
                 model_name: str = "yolov8n.pt", 
                 confidence_threshold: float = 0.5,
                 device: str = "cpu"):
        """
        Initialize the detector
        
        Args:
            model_name: YOLO model to use (n=nano, s=small, m=medium, l=large, x=extra-large)
            confidence_threshold: Minimum confidence for detections
            device: 'cpu' or 'cuda' for GPU acceleration
            
        Why these parameters?
        - model_name: Balance between speed and accuracy (nano is fastest)
        - confidence_threshold: Filter out low-confidence false positives  
        - device: GPU acceleration when available
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        
        # Initialize the model
        self._load_model()
        
        logger.info(f"AVObjectDetector initialized with {model_name} on {device}")
    
    def _load_model(self) -> None:
        """
        Load the YOLO model
        
        Why this is a separate method:
        - Error handling: Isolate model loading errors
        - Reusability: Can reload model if needed
        - Testing: Can mock this method in tests
        - Version compatibility: Handle PyTorch version differences
        """
        try:
            # Handle PyTorch 2.6+ compatibility issue
            # This is a common DevOps challenge: dependency version conflicts
            import torch
            
            # Check PyTorch version for compatibility handling
            torch_version = torch.__version__
            logger.info(f"Using PyTorch version: {torch_version}")
            
            # For PyTorch 2.6+, we need to allow YOLO model globals
            if hasattr(torch.serialization, 'add_safe_globals'):
                # Add YOLO-specific classes to safe globals
                from ultralytics.nn.tasks import DetectionModel
                torch.serialization.add_safe_globals([DetectionModel])
                logger.info("Added YOLO globals to PyTorch safe loading")
            
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except ImportError as e:
            raise ObjectDetectionError(f"Missing dependencies for model loading: {str(e)}")
        except Exception as e:
            # Provide helpful error message for common issues
            error_msg = str(e)
            if "weights_only" in error_msg or "WeightsUnpickler" in error_msg:
                helpful_msg = (
                    f"PyTorch version compatibility issue. "
                    f"Try updating ultralytics: 'pip install ultralytics --upgrade' "
                    f"Original error: {error_msg}"
                )
                raise ObjectDetectionError(helpful_msg)
            else:
                raise ObjectDetectionError(f"Failed to load model {self.model_name}: {str(e)}")
    
    def detect_from_image_path(self, image_path: str) -> List[Dict]:
        """
        Detect objects from an image file path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detection dictionaries
            
        Why separate methods for different input types?
        - Flexibility: API might send file paths, camera feeds send arrays
        - Error handling: Different input types have different failure modes
        - Testing: Easier to test individual methods
        """
        if not Path(image_path).exists():
            raise ObjectDetectionError(f"Image file not found: {image_path}")
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ObjectDetectionError(f"Could not read image: {image_path}")
                
            return self.detect_from_array(image)
        except ObjectDetectionError:
            raise  # Re-raise our custom errors
        except Exception as e:
            raise ObjectDetectionError(f"Error processing image {image_path}: {str(e)}")
    
    def detect_from_array(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects from a numpy array (image)
        
        Args:
            image: Image as numpy array (OpenCV format: BGR)
            
        Returns:
            List of detection dictionaries with keys:
            - class_name: Object class (e.g., 'car', 'person')
            - av_category: AV-relevant category ('vehicle', 'pedestrian', 'infrastructure')
            - confidence: Detection confidence (0.0 to 1.0)
            - bbox: Bounding box [x1, y1, x2, y2]
            - center: Object center point [x, y]
            
        Why this return format?
        - Structured: Easy to work with in API responses
        - Complete: Contains all information needed for AV decision-making
        - Consistent: Same format regardless of input type
        """
        if image is None or image.size == 0:
            raise ObjectDetectionError("Invalid image array")
            
        try:
            # Run YOLO detection
            results = self.model(image, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    detections.extend(self._process_boxes(boxes))
            
            logger.info(f"Detected {len(detections)} objects above confidence threshold")
            return detections
            
        except Exception as e:
            raise ObjectDetectionError(f"Detection failed: {str(e)}")
    
    def _process_boxes(self, boxes) -> List[Dict]:
        """
        Process YOLO detection boxes into our standard format
        
        Why this is a separate method?
        - Modularity: Easier to modify detection output format
        - Testing: Can test box processing independently
        - Readability: Keeps main detection method clean
        """
        detections = []
        
        for box in boxes:
            # Extract detection data
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.model.names[class_id]
            
            # Filter by confidence and relevance to AV applications
            if (confidence >= self.confidence_threshold and 
                class_name in self.AV_RELEVANT_CLASSES):
                
                # Calculate center point (useful for AV navigation)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detection = {
                    'class_name': class_name,
                    'av_category': self.AV_RELEVANT_CLASSES[class_name],
                    'confidence': confidence,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float(center_x), float(center_y)]
                }
                
                detections.append(detection)
        
        return detections
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model
        
        Why this method?
        - Debugging: Know which model version is running
        - Monitoring: Track model performance in production
        - API: Expose model info through web interface
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'supported_classes': list(self.AV_RELEVANT_CLASSES.keys())
        }


# Example usage and testing
if __name__ == "__main__":
    """
    This block runs only when the file is executed directly,
    not when imported as a module.
    
    Why this pattern?
    - Testing: Quick way to test the module during development
    - Documentation: Shows how to use the class
    - Debugging: Easy to run this file to check if everything works
    """
    
    print("=== Testing AVObjectDetector ===")
    
    try:
        # Initialize detector
        detector = AVObjectDetector()
        print("✓ Detector initialized successfully")
        
        # Show model info
        info = detector.get_model_info()
        print(f"✓ Model info: {info}")
        
        # Test with a sample image (if you have one)
        # detections = detector.detect_from_image_path("sample.jpg")
        # print(f"✓ Detected {len(detections)} objects")
        
        print("=== All tests passed! ===")
        
    except ObjectDetectionError as e:
        print(f"✗ Detection error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")