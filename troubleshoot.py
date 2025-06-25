#!/usr/bin/env python
"""
troubleshoot.py

DevOps troubleshooting tool for the AV Vision Pipeline

This script demonstrates proper DevOps practices:
- Environment validation
- Dependency checking  
- Version compatibility verification
- Clear error reporting
- Automated fixes where possible
"""

import sys
import subprocess
import importlib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible"""
    print("=== Python Version Check ===")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version compatible")
        return True


def check_virtual_environment():
    """Check if we're in a virtual environment"""
    print("\n=== Virtual Environment Check ===")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment active")
        print(f"Environment path: {sys.prefix}")
        return True
    else:
        print("❌ No virtual environment detected")
        print("Run: source venv/bin/activate")
        return False


def check_package_versions():
    """Check installed package versions and compatibility"""
    print("\n=== Package Version Check ===")
    
    required_packages = {
        'torch': 'PyTorch',
        'ultralytics': 'YOLO models',
        'cv2': 'OpenCV',
        'numpy': 'Numerical computing',
        'fastapi': 'Web API framework'
    }
    
    issues = []
    
    for package, description in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
                version = cv2.__version__
                module_name = 'opencv-python'
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'Unknown')
                module_name = package
            
            print(f"✅ {description}: {module_name} {version}")
            
            # Check for known compatibility issues
            if package == 'torch':
                major_version = version.split('.')[0]
                minor_version = version.split('.')[1] if '.' in version else '0'
                if int(major_version) >= 2 and int(minor_version) >= 6:
                    issues.append({
                        'package': 'torch',
                        'issue': 'PyTorch 2.6+ has stricter model loading security',
                        'solution': 'Our code handles this automatically, but you might see warnings'
                    })
            
        except ImportError:
            print(f"❌ {description}: {module_name} not installed")
            issues.append({
                'package': package,
                'issue': 'Package not installed',
                'solution': f'Run: pip install {module_name}'
            })
    
    return issues


def check_model_loading():
    """Test YOLO model loading with detailed error reporting"""
    print("\n=== Model Loading Test ===")
    
    try:
        # Import our detector
        from src.detector import AVObjectDetector
        
        print("✅ Detector module imported successfully")
        
        # Try to create detector instance
        detector = AVObjectDetector()
        print("✅ Detector initialized successfully")
        
        # Test model info
        info = detector.get_model_info()
        print(f"✅ Model info retrieved: {info['model_name']}")
        
        return True
        
    except Exception as e:
        error_str = str(e)
        print(f"❌ Model loading failed: {error_str}")
        
        # Provide specific solutions for common errors
        if "weights_only" in error_str or "WeightsUnpickler" in error_str:
            print("\n🔧 PyTorch Compatibility Issue Detected:")
            print("Solution 1: Update ultralytics to latest version")
            print("  pip install ultralytics --upgrade")
            print("\nSolution 2: If that doesn't work, downgrade PyTorch")
            print("  pip install 'torch<2.6.0' torchvision")
            
        elif "No module named" in error_str:
            print("\n🔧 Missing Dependency:")
            print("  pip install -r requirements.txt")
            
        elif "not found" in error_str:
            print("\n🔧 Model File Issue:")
            print("The YOLO model will be downloaded automatically on first run")
            print("Check your internet connection")
        
        return False


def run_basic_tests():
    """Run basic functionality tests"""
    print("\n=== Basic Functionality Tests ===")
    
    try:
        # Test imports
        from src.detector import AVObjectDetector, ObjectDetectionError
        print("✅ Module imports working")
        
        # Test error handling
        detector = AVObjectDetector()
        
        # Test invalid input handling
        try:
            detector.detect_from_image_path("nonexistent.jpg")
        except ObjectDetectionError:
            print("✅ Error handling working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        return False


def suggest_fixes(issues):
    """Suggest fixes for identified issues"""
    if not issues:
        return
        
    print("\n=== Suggested Fixes ===")
    
    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. {issue['package']} - {issue['issue']}")
        print(f"   Solution: {issue['solution']}")


def auto_fix_dependencies():
    """Attempt to automatically fix common dependency issues"""
    print("\n=== Attempting Automatic Fixes ===")
    
    try:
        # Update pip first
        print("Updating pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        print("✅ pip updated")
        
        # Reinstall requirements
        print("Reinstalling requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Requirements reinstalled")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Auto-fix failed: {e}")
        return False


def main():
    """Main troubleshooting routine"""
    print("🔧 AV Vision Pipeline - Troubleshooting Tool")
    print("=" * 50)
    
    # Check basic environment
    python_ok = check_python_version()
    venv_ok = check_virtual_environment()
    
    if not python_ok:
        print("\n❌ Python version incompatible. Exiting.")
        return False
    
    if not venv_ok:
        print("\n⚠️  Virtual environment recommended but continuing...")
    
    # Check packages
    issues = check_package_versions()
    
    # Test model loading
    model_ok = check_model_loading()
    
    # Run basic tests
    tests_ok = run_basic_tests()
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 TROUBLESHOOTING SUMMARY")
    print("=" * 50)
    
    if model_ok and tests_ok and not issues:
        print("🎉 All checks passed! Your environment is ready.")
        print("\nNext steps:")
        print("  python src/detector.py")
        print("  pytest tests/test_detector.py -v")
        return True
    else:
        print("⚠️  Issues detected:")
        suggest_fixes(issues)
        
        if issues:
            print(f"\n🔧 Try running: python troubleshoot.py --fix")
        
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        print("🔧 Running automatic fixes...")
        if auto_fix_dependencies():
            print("\n✅ Auto-fix completed. Run troubleshoot.py again to verify.")
        else:
            print("\n❌ Auto-fix failed. Manual intervention required.")
    else:
        main()