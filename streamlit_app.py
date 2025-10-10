#!/usr/bin/env python3
"""
Streamlit UI for Person of Interest - Face Image Processing and Semantic Search

This application provides a web interface for:
- Face-based similarity search
- Text-based semantic search
- Metadata search (filename, source image)
- Database statistics and management
- Image browsing and visualization

Author: SupportVectors AI Training Team
"""

import streamlit as st
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the Person of Interest application
from poi.app import create_app
from poi import config

# Page configuration
st.set_page_config(
    page_title="Person of Interest - Face Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .search-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .face-image {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .face-image:hover {
        transform: scale(1.05);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'app' not in st.session_state:
    st.session_state.app = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'dataset_path' not in st.session_state:
    st.session_state.dataset_path = "/home/biju/supportvectors/Person-Of-Interest/dataset/processed_faces"

def initialize_app():
    """Initialize the Person of Interest application."""
    try:
        with st.spinner("Initializing Person of Interest Application..."):
            # Update config to use the specified dataset path
            config["dataset"]["processed_path"] = st.session_state.dataset_path
            
            # Create and initialize the app
            app = create_app()
            app.initialize_components()
            
            st.session_state.app = app
            st.session_state.initialized = True
            st.success("✅ Application initialized successfully!")
            return True
    except Exception as e:
        st.error(f"❌ Failed to initialize application: {str(e)}")
        return False

def get_image_from_path(image_path: str) -> Optional[Image.Image]:
    """Load image from file path."""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image {image_path}: {str(e)}")
    return None

def display_face_grid(images_data: List[Dict[str, Any]], title: str = "Search Results"):
    """Display a grid of face images with metadata."""
    if not images_data:
        st.warning("No images found.")
        return
    
    st.subheader(f"📸 {title} ({len(images_data)} results)")
    
    # Create columns for the grid
    cols = st.columns(4)
    
    for i, img_data in enumerate(images_data):
        col_idx = i % 4
        with cols[col_idx]:
            # Create result card
            with st.container():
                st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                
                # Display image
                if 'image' in img_data and img_data['image'] is not None:
                    st.image(
                        img_data['image'], 
                        caption=f"Score: {img_data.get('score', 'N/A'):.3f}" if 'score' in img_data else "N/A",
                        use_container_width=True,
                        output_format='JPEG'
                    )
                
                # Display metadata
                metadata = img_data.get('metadata', {})
                filename = metadata.get('filename', 'Unknown')
                source_image = metadata.get('source_image', 'Unknown')
                
                st.write(f"**File:** {filename}")
                st.write(f"**Source:** {source_image}")
                if 'score' in img_data:
                    st.write(f"**Similarity:** {img_data['score']:.3f}")
                
                st.markdown('</div>', unsafe_allow_html=True)

def display_database_stats():
    """Display database statistics."""
    try:
        if st.session_state.app is None:
            st.error("Application not initialized.")
            return
        
        stats = st.session_state.app.get_database_stats()
        
        if "error" in stats:
            st.error(f"Error getting database stats: {stats['error']}")
            return
        
        if not stats.get("exists", False):
            st.warning("Database collection does not exist.")
            return
        
        # Create metrics in separate rows with smaller font
        st.markdown(f"**Collection Name:** <small>{stats.get('name', 'N/A')}</small>", unsafe_allow_html=True)
        st.markdown(f"**Total Faces:** <small>{stats.get('point_count', 0)}</small>", unsafe_allow_html=True)
        st.markdown(f"**Vector Size:** <small>{stats.get('vector_size', 'N/A')}</small>", unsafe_allow_html=True)
        st.markdown(f"**Distance Metric:** <small>{stats.get('distance_metric', 'N/A')}</small>", unsafe_allow_html=True)
        
        # Additional info
        st.info(f"**Status:** {stats.get('status', 'Unknown')}")
        
    except Exception as e:
        st.error(f"Error getting database statistics: {str(e)}")

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Person of Interest - Face Search</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dataset path configuration
        st.subheader("📁 Dataset Settings")
        dataset_path = st.text_input(
            "Dataset Path", 
            value=st.session_state.dataset_path,
            help="Path to the processed faces dataset"
        )
        
        if dataset_path != st.session_state.dataset_path:
            st.session_state.dataset_path = dataset_path
            st.session_state.initialized = False  # Reset initialization
        
        # Initialize button
        if not st.session_state.initialized:
            if st.button("🚀 Initialize Application", type="primary"):
                initialize_app()
        else:
            st.success("✅ Application Ready")
            if st.button("🔄 Reinitialize"):
                st.session_state.initialized = False
                st.rerun()
        
        # Database stats
        st.subheader("📊 Database Statistics")
        if st.session_state.initialized:
            display_database_stats()
        else:
            st.info("Initialize the application to view database statistics.")
    
    # Main content area
    if not st.session_state.initialized:
        st.info("👈 Please initialize the application using the sidebar to begin.")
        return
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Face Search", 
        "📝 Text Search", 
        "🌐 Semantic Search", 
        "📁 Browse Dataset", 
        "📊 Analytics"
    ])
    
    with tab1:
        st.header("🔍 Face-Based Similarity Search")
        st.markdown("Upload an image to find similar faces in the dataset.")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a face image to search for similar faces"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            
            # Search parameters
            col1, col2 = st.columns(2)
            with col1:
                limit = st.slider("Number of results", 1, 50, 10)
            with col2:
                threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.6, 0.01)
            
            # Search button
            if st.button("🔍 Search Similar Faces", type="primary"):
                try:
                    with st.spinner("Searching for similar faces..."):
                        results = st.session_state.app.search_similar_faces(
                            uploaded_image, 
                            limit=limit, 
                            score_threshold=threshold
                        )
                    
                    if results:
                        # Load images for display
                        display_data = []
                        for result in results:
                            img_path = result['metadata'].get('filepath')
                            if img_path and os.path.exists(img_path):
                                img = Image.open(img_path)
                                display_data.append({
                                    'image': img,
                                    'score': result['score'],
                                    'metadata': result['metadata']
                                })
                        
                        display_face_grid(display_data, "Similar Faces Found")
                    else:
                        st.warning("No similar faces found.")
                        
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
    
    with tab2:
        st.header("📝 Text-Based Metadata Search")
        st.markdown("Search for faces using filename or source image metadata.")
        
        # Search type selection
        search_type = st.selectbox(
            "Search Type",
            ["filename", "source_image", "all"],
            help="Choose how to search the database"
        )
        
        # Query input
        if search_type == "all":
            query = ""
            st.info("Searching all faces in the database...")
        else:
            query = st.text_input(
                f"Enter {search_type} to search for:",
                placeholder=f"e.g., 000001 for {search_type}",
                help=f"Enter a {search_type} to search for"
            )
        
        # Search parameters
        col1, col2 = st.columns(2)
        with col1:
            limit = st.slider("Number of results", 1, 100, 20)
        
        # Search button
        if st.button("🔍 Search by Text", type="primary"):
            if search_type != "all" and not query.strip():
                st.warning("Please enter a search query.")
            else:
                try:
                    with st.spinner("Searching..."):
                        results = st.session_state.app.search_by_text(
                            query, 
                            search_type=search_type, 
                            limit=limit
                        )
                    
                    if results:
                        # Load images for display
                        display_data = []
                        for result in results:
                            img_path = result['metadata'].get('filepath')
                            if img_path and os.path.exists(img_path):
                                img = Image.open(img_path)
                                display_data.append({
                                    'image': img,
                                    'metadata': result['metadata']
                                })
                        
                        display_face_grid(display_data, f"Text Search Results ({search_type})")
                    else:
                        st.warning("No results found.")
                        
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
    
    with tab3:
        st.header("🌐 Semantic Text Search")
        st.markdown("Search for faces using natural language descriptions.")
        
        # Query input
        query = st.text_input(
            "Enter a semantic description:",
            placeholder="e.g., 'a happy person', 'someone with glasses', 'young woman'",
            help="Describe the type of face you're looking for"
        )
        
        # Search parameters
        col1, col2 = st.columns(2)
        with col1:
            limit = st.slider("Number of results", 1, 50, 10)
        with col2:
            threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.6, 0.01)
        
        # Search button
        if st.button("🔍 Semantic Search", type="primary"):
            if not query.strip():
                st.warning("Please enter a search query.")
            else:
                try:
                    with st.spinner("Performing semantic search..."):
                        results = st.session_state.app.search_by_semantic_text(
                            query, 
                            limit=limit, 
                            score_threshold=threshold
                        )
                    
                    if results:
                        # Load images for display
                        display_data = []
                        for result in results:
                            img_path = result['metadata'].get('filepath')
                            if img_path and os.path.exists(img_path):
                                img = Image.open(img_path)
                                display_data.append({
                                    'image': img,
                                    'score': result['score'],
                                    'metadata': result['metadata']
                                })
                        
                        display_face_grid(display_data, f"Semantic Search: '{query}'")
                    else:
                        st.warning("No results found.")
                        
                except Exception as e:
                    st.error(f"Semantic search failed: {str(e)}")
    
    with tab4:
        st.header("📁 Dataset Browser")
        st.markdown("Browse and explore the processed faces dataset.")
        
        # Dataset path info
        st.info(f"**Dataset Path:** `{st.session_state.dataset_path}`")
        
        # Check if dataset exists
        if not os.path.exists(st.session_state.dataset_path):
            st.error(f"Dataset path does not exist: {st.session_state.dataset_path}")
            return
        
        # Get list of images
        dataset_path = Path(st.session_state.dataset_path)
        image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.jpeg"))
        image_files.sort()
        
        if not image_files:
            st.warning("No images found in the dataset.")
            return
        
        st.success(f"Found {len(image_files)} images in the dataset.")
        
        # Pagination
        images_per_page = 20
        total_pages = (len(image_files) - 1) // images_per_page + 1
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page = st.selectbox("Page", range(1, total_pages + 1), index=0)
        
        # Display images for current page
        start_idx = (page - 1) * images_per_page
        end_idx = min(start_idx + images_per_page, len(image_files))
        current_images = image_files[start_idx:end_idx]
        
        # Create grid
        cols = st.columns(4)
        for i, img_path in enumerate(current_images):
            col_idx = i % 4
            with cols[col_idx]:
                try:
                    img = Image.open(img_path)
                    st.image(
                        img, 
                        caption=img_path.name,
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error loading {img_path.name}: {str(e)}")
    
    with tab5:
        st.header("📊 Analytics & Statistics")
        
        if st.session_state.app is None:
            st.error("Application not initialized.")
            return
        
        # Database statistics
        st.subheader("🗄️ Database Statistics")
        display_database_stats()
        
        # Dataset analysis
        st.subheader("📈 Dataset Analysis")
        
        try:
            # Get dataset info
            dataset_path = Path(st.session_state.dataset_path)
            if dataset_path.exists():
                image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.jpeg"))
                
                if image_files:
                    # File size analysis
                    file_sizes = [img.stat().st_size for img in image_files]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Images", len(image_files))
                        st.metric("Average File Size", f"{np.mean(file_sizes) / 1024:.1f} KB")
                        st.metric("Total Dataset Size", f"{sum(file_sizes) / (1024*1024):.1f} MB")
                    
                    with col2:
                        # File size distribution
                        fig = px.histogram(
                            x=file_sizes, 
                            title="File Size Distribution",
                            labels={'x': 'File Size (bytes)', 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sample images
                    st.subheader("🖼️ Sample Images")
                    sample_images = image_files[:6]  # Show first 6 images
                    cols = st.columns(3)
                    for i, img_path in enumerate(sample_images):
                        col_idx = i % 3
                        with cols[col_idx]:
                            try:
                                img = Image.open(img_path)
                                st.image(img, caption=img_path.name, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error loading {img_path.name}")
                else:
                    st.warning("No images found in the dataset.")
            else:
                st.error(f"Dataset path does not exist: {dataset_path}")
                
        except Exception as e:
            st.error(f"Error analyzing dataset: {str(e)}")

if __name__ == "__main__":
    main()
