# src/__init__.py
"""
AV Vision Pipeline - Core Package

This file makes 'src' a Python package, allowing imports like:
    from src.detector import AVObjectDetector
    from src.api import app

Why __init__.py files?
- Package Recognition: Python knows this directory contains importable modules
- Namespace Control: Define what gets imported when someone does 'from src import *'
- Initialization: Run code when the package is first imported
"""

# Define what gets imported with 'from src import *'
__all__ = [
    'AVObjectDetector',
    'ObjectDetectionError'
]

# Import main classes for convenient access
from .detector import AVObjectDetector, ObjectDetectionError

# Package metadata
__version__ = "0.1.0"
__author__ = "Your Name - DTU Autonomous Systems Engineer"
__description__ = "Object Detection Pipeline for Autonomous Vehicles"


# tests/__init__.py
"""
Test Package for AV Vision Pipeline

This makes 'tests' a Python package and allows pytest to discover our tests.

Why separate test package?
- Organization: Keep tests separate from application code
- Import Testing: Can import and test our src modules
- Pytest Discovery: Pytest automatically finds test files in packages
"""

# Test configuration
import sys
from pathlib import Path

# Add src directory to Python path for testing
# This allows our tests to import from src package
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))