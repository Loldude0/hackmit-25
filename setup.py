#!/usr/bin/env python3
"""
Setup script for Suno AI Music Generation Pipeline.

This script sets up the complete Suno AI system including:
- Activity detection using computer vision
- AI-powered music generation
- Integrated contextual music recommendations
- Command-line interfaces for all functionality

Usage:
1. Install dependencies: pip install -r requirements.txt
2. Set your ANTHROPIC_API_KEY environment variable
3. Run: python main.py [mode] [options]

Available modes:
- generate: Generate music with custom parameters
- activity: Generate music based on detected activity
- integrated: Combined activity detection + music generation
- interactive: Guided music generation
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

def check_suno_modules():
    """Check if Suno AI modules are available"""
    try:
        from suno.suno_ai import SunoAIPipeline
        from suno.suno_cli import setup_argument_parser
        print("✅ Suno AI modules are available")
        return True
    except ImportError as e:
        print(f"❌ Missing Suno AI module: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Checking Suno AI system requirements...")
    
    deps_ok = check_dependencies()
    api_ok = check_api_key()
    camera_ok = check_camera()
    suno_ok = check_suno_modules()
    
    if deps_ok and api_ok and camera_ok and suno_ok:
        print("\n🎉 Everything looks good! You can run:")
        print("🎵 Music Generation:")
        print("  python main.py generate --genre pop --mood upbeat")
        print("🎯 Activity-based Music:")
        print("  python main.py activity --activity 'working out'")
        print("🔄 Integrated Mode:")
        print("  python main.py integrated --monitor")
        print("🎮 Interactive Mode:")
        print("  python main.py interactive")
        print("\n📚 For help: python main.py --help")
    else:
        print("\n⚠️  Please fix the issues above before running Suno AI.")
        sys.exit(1)
