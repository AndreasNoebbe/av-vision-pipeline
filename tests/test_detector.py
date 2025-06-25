"""
tests/test_detector.py

Unit tests for our object detection module.

Why we write tests:
- Confidence: Know our code works as expected
- Regression Prevention: Changes don't break existing functionality  
- Documentation: Tests show how the code is supposed to work
- CI/CD: Automated testing in deployment pipelines
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.detector import AVObjectDetector, ObjectDetectionError


class TestAVObjectDetector:
    """
    Test class for AVObjectDetector
    
    Why group tests in a class?
    - Organization: Related tests stay together
    - Setup/Teardown: Shared test fixtures
    - Clarity: Clear test structure
    """
    
    def test_detector_initialization(self):
        """
        Test that detector initializes correctly with default parameters
        
        Why test initialization?
        - Basic functionality: If this fails, nothing else will work
        - Parameter validation: Ensure defaults are sensible
        - Error handling: Constructor should handle bad inputs gracefully
        """
        detector = AVObjectDetector()
        
        # Verify basic properties
        assert detector.model_name == "yolov8n.pt"
        assert detector.confidence_threshold == 0.5
        assert detector.device == "cpu"
        assert detector.model is not None
        
        print("✓ Detector initialization test passed")
    
    def test_detector_custom_parameters(self):
        """Test detector with custom parameters"""
        custom_detector = AVObjectDetector(
            model_name="yolov8s.pt",
            confidence_threshold=0.7,
            device="cpu"
        )
        
        assert custom_detector.model_name == "yolov8s.pt"
        assert custom_detector.confidence_threshold == 0.7
        assert custom_detector.device == "cpu"
        
        print("✓ Custom parameters test passed")
    
    def test_get_model_info(self):
        """
        Test model info retrieval
        
        Why test this?
        - API endpoints will use this method
        - Information accuracy is important for debugging
        - Return format needs to be consistent
        """
        detector = AVObjectDetector()
        info = detector.get_model_info()
        
        # Check required keys exist
        required_keys = ['model_name', 'device', 'confidence_threshold', 'supported_classes']
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
        
        # Check data types
        assert isinstance(info['supported_classes'], list)
        assert isinstance(info['confidence_threshold'], float)
        
        print("✓ Model info test passed")
    
    def test_invalid_image_path(self):
        """
        Test error handling for invalid image paths
        
        Why test error cases?
        - Robustness: Real-world usage will have invalid inputs
        - User Experience: Good error messages help debugging
        - Security: Prevent crashes from malicious inputs
        """
        detector = AVObjectDetector()
        
        with pytest.raises(ObjectDetectionError) as exc_info:
            detector.detect_from_image_path("nonexistent_image.jpg")
        
        # Verify error message is helpful
        assert "not found" in str(exc_info.value).lower()
        
        print("✓ Invalid image path test passed")
    
    def test_invalid_image_array(self):
        """Test error handling for invalid image arrays"""
        detector = AVObjectDetector()
        
        # Test None image
        with pytest.raises(ObjectDetectionError):
            detector.detect_from_array(None)
        
        # Test empty array
        empty_array = np.array([])
        with pytest.raises(ObjectDetectionError):
            detector.detect_from_array(empty_array)
        
        print("✓ Invalid image array test passed")
    
    def test_detection_output_format(self):
        """
        Test that detection output has correct format
        
        Why test output format?
        - API Contracts: Frontend expects specific data structure
        - Integration: Other parts of system depend on this format
        - Consistency: Format should be same regardless of input
        """
        detector = AVObjectDetector()
        
        # Create a simple test image (3 channels, 100x100 pixels)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Run detection (might not detect anything in a blank image)
        detections = detector.detect_from_array(test_image)
        
        # Verify output is a list
        assert isinstance(detections, list)
        
        # If there are detections, verify format
        for detection in detections:
            required_keys = ['class_name', 'av_category', 'confidence', 'bbox', 'center']
            for key in required_keys:
                assert key in detection, f"Missing key in detection: {key}"
            
            # Verify data types
            assert isinstance(detection['class_name'], str)
            assert isinstance(detection['confidence'], float)
            assert isinstance(detection['bbox'], list)
            assert len(detection['bbox']) == 4
            assert isinstance(detection['center'], list)
            assert len(detection['center']) == 2
        
        print("✓ Detection output format test passed")
    
    @pytest.fixture
    def sample_detector(self):
        """
        Pytest fixture - provides a detector instance for multiple tests
        
        Why use fixtures?
        - DRY: Don't repeat detector creation in every test
        - Consistency: Same detector configuration across tests
        - Cleanup: Fixtures can handle resource cleanup
        """
        return AVObjectDetector()


# Test runner for direct execution
if __name__ == "__main__":
    """
    Allow running tests directly with: python tests/test_detector.py
    
    Why this pattern?
    - Development: Quick way to run tests during coding
    - Debugging: Easy to add print statements and debug
    - CI/CD: Can run tests in different ways
    """
    
    print("=== Running Detector Tests ===")
    
    # Create test instance
    test_instance = TestAVObjectDetector()
    
    # Run each test method
    test_methods = [
        test_instance.test_detector_initialization,
        test_instance.test_detector_custom_parameters,
        test_instance.test_get_model_info,
        test_instance.test_invalid_image_path,
        test_instance.test_invalid_image_array,
        test_instance.test_detection_output_format
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_method()
            passed += 1
        except Exception as e:
            print(f"✗ {test_method.__name__} failed: {e}")
            failed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")