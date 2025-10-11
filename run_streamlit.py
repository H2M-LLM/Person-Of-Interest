#!/usr/bin/env python3
"""
Launcher script for the Person of Interest Streamlit UI.

This script updates the configuration to use the correct dataset path
and launches the Streamlit application.

Usage:
    python run_streamlit.py
"""

import os
import sys
from pathlib import Path
import subprocess

def main():
    """Launch the Streamlit application with correct configuration."""
    
    # Set the dataset path
    dataset_path = "/home/biju/supportvectors/Person-Of-Interest/dataset/processed_faces"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        print("Please ensure the dataset is available at the specified path.")
        return 1
    
    print(f"✅ Dataset found at: {dataset_path}")
    
    # Set environment variables
    os.environ["POI_DATASET_PATH"] = dataset_path
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    streamlit_app = script_dir / "streamlit_app.py"
    
    if not streamlit_app.exists():
        print(f"❌ Streamlit app not found: {streamlit_app}")
        return 1
    
    print("🚀 Launching Person of Interest Streamlit UI...")
    print("📱 The application will open in your default web browser.")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n" + "="*60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
        return 0
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
