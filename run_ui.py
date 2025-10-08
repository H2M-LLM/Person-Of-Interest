#!/usr/bin/env python3
"""
Launcher script for the Person of Interest Streamlit UI.

This script provides an easy way to start the Streamlit web interface.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit UI."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    streamlit_app = script_dir / "src" / "simple_streamlit_app.py"
    
    if not streamlit_app.exists():
        print(f"Error: Streamlit app not found at {streamlit_app}")
        sys.exit(1)
    
    print("🚀 Starting Person of Interest Streamlit UI...")
    print(f"📁 App location: {streamlit_app}")
    print("🌐 The app will open in your default web browser")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down Person of Interest UI...")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
