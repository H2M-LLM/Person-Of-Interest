#!/usr/bin/env python3
"""
Configuration update script for Person of Interest.

This script updates the config.yaml file to use the correct dataset path
for the Streamlit UI.

Usage:
    python update_config.py
"""

import yaml
from pathlib import Path

def update_config():
    """Update the configuration file with the correct dataset path."""
    
    config_file = Path("config.yaml")
    dataset_path = "/home/biju/supportvectors/Person-Of-Interest/dataset/processed_faces"
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    try:
        # Read current config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update dataset path
        if 'dataset' not in config:
            config['dataset'] = {}
        
        config['dataset']['processed_path'] = dataset_path
        
        # Write updated config
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Updated config.yaml with dataset path: {dataset_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False

if __name__ == "__main__":
    success = update_config()
    if success:
        print("🎉 Configuration updated successfully!")
    else:
        print("💥 Failed to update configuration.")
        exit(1)
