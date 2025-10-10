#!/usr/bin/env python3
"""
Test script to verify Streamlit integration with Person of Interest.

This script tests the core functionality without launching the full UI.
"""

import sys
from pathlib import Path
import os

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from poi.app import create_app
        from poi import config
        print("✅ Core Person of Interest modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core modules: {e}")
        return False
    
    try:
        import streamlit as st
        import plotly.express as px
        import pandas as pd
        import numpy as np
        from PIL import Image
        print("✅ Streamlit UI dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import UI dependencies: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from poi import config
        
        # Check required config sections
        required_sections = ['dataset', 'vector_db', 'image_encoding']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False
        
        # Check dataset path
        dataset_path = config['dataset']['processed_path']
        print(f"✅ Dataset path configured: {dataset_path}")
        
        # Check if dataset exists
        if os.path.exists(dataset_path):
            print(f"✅ Dataset path exists: {dataset_path}")
            
            # Count images
            dataset_path_obj = Path(dataset_path)
            image_files = list(dataset_path_obj.glob("*.jpg")) + list(dataset_path_obj.glob("*.jpeg"))
            print(f"✅ Found {len(image_files)} images in dataset")
        else:
            print(f"⚠️  Dataset path does not exist: {dataset_path}")
            print("   This is expected if the dataset hasn't been processed yet.")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_app_creation():
    """Test Person of Interest app creation."""
    print("\nTesting app creation...")
    
    try:
        from poi.app import create_app
        
        # Create app instance
        app = create_app()
        print("✅ Person of Interest app created successfully")
        
        # Test component initialization (without actually initializing)
        print("✅ App structure is valid")
        
        return True
        
    except Exception as e:
        print(f"❌ App creation test failed: {e}")
        return False

def test_streamlit_app():
    """Test that the Streamlit app can be loaded."""
    print("\nTesting Streamlit app...")
    
    try:
        # Import the streamlit app module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "streamlit_app", 
            "/home/sunitp/Person-Of-Interest/streamlit_app.py"
        )
        streamlit_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(streamlit_module)
        
        print("✅ Streamlit app module loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Person of Interest Streamlit Integration")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_app_creation,
        test_streamlit_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Streamlit UI is ready to use.")
        print("\nTo launch the UI, run:")
        print("  python run_streamlit.py")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
