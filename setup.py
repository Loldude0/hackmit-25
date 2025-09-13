#!/usr/bin/env python3
"""
Simple setup script for the activity detection pipeline.

Usage:
1. Install dependencies: pip install -r requirements.txt
2. Set your ANTHROPIC_API_KEY environment variable
3. Run: python activity_detector.py
"""

import os
import sys

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import cv2
        import langgraph
        import langchain_anthropic
        import pydantic
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if API key is set"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print("✅ ANTHROPIC_API_KEY is set")
        return True
    else:
        print("❌ ANTHROPIC_API_KEY not found")
        print("Please set your API key:")
        print("export ANTHROPIC_API_KEY='your_api_key_here'")
        return False

def check_camera():
    """Check if camera is accessible"""
    try:
        import cv2
        camera = cv2.VideoCapture(0)
        ret, frame = camera.read()
        camera.release()
        
        if ret:
            print("✅ Camera is accessible")
            return True
        else:
            print("❌ Camera not accessible")
            return False
    except Exception as e:
        print(f"❌ Camera error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Checking system requirements...")
    
    deps_ok = check_dependencies()
    api_ok = check_api_key()
    camera_ok = check_camera()
    
    if deps_ok and api_ok and camera_ok:
        print("\n🎉 Everything looks good! You can run:")
        print("python activity_detector.py")
    else:
        print("\n⚠️  Please fix the issues above before running the pipeline.")
        sys.exit(1)
