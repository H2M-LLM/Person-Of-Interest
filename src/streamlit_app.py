#!/usr/bin/env python3
"""
Streamlit UI for Person of Interest Face Search Application

This module provides a comprehensive web interface for:
1. Face image upload and processing
2. Face similarity search
3. Text-based search
4. Database management and statistics
5. Batch processing capabilities
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import io

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np

# Optional plotly imports - fallback to matplotlib if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Charts will be simplified.")

# Import POI components
from poi import create_app, config
from poi.app import PersonOfInterestApp


# =============================================================================
#  Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Person of Interest - Face Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
#  Utility Functions
# =============================================================================
@st.cache_resource
def get_app() -> PersonOfInterestApp:
    """Get cached application instance."""
    try:
        return create_app()
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        st.stop()

def display_image_with_metadata(image: Image.Image, metadata: Dict[str, Any], 
                               similarity_score: Optional[float] = None) -> None:
    """Display image with metadata in a card format."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(image, use_container_width=True)
    
    with col2:
        if similarity_score is not None:
            st.metric("Similarity Score", f"{similarity_score:.4f}")
        
        st.write("**Metadata:**")
        for key, value in metadata.items():
            if key != "image_size":  # Skip image_size as it's not user-friendly
                st.write(f"• **{key}**: {value}")

def create_similarity_chart(results: List[Dict[str, Any]]):
    """Create a bar chart showing similarity scores."""
    if not results:
        return None
    
    scores = [r["score"] for r in results]
    labels = [f"Result {i+1}" for i in range(len(results))]
    
    if PLOTLY_AVAILABLE:
        fig = go.Figure(data=[
            go.Bar(x=labels, y=scores, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title="Similarity Scores",
            xaxis_title="Search Results",
            yaxis_title="Similarity Score",
            height=400
        )
        
        return fig
    else:
        # Fallback to simple text display
        st.subheader("Similarity Scores")
        for i, score in enumerate(scores):
            st.write(f"Result {i+1}: {score:.4f}")
        return None

# =============================================================================
#  Main Application
# =============================================================================
def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🔍 Person of Interest - Face Search")
    st.markdown("**Advanced face recognition and similarity search using deep learning**")
    
    # Initialize app
    app = get_app()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Search parameters
        st.subheader("Search Settings")
        max_results = st.slider("Max Results", 1, 50, 10)
        score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.0, 0.01)
        
        # Display options
        st.subheader("Display Options")
        show_metadata = st.checkbox("Show Metadata", True)
        show_similarity_chart = st.checkbox("Show Similarity Chart", True)
        grid_columns = st.slider("Grid Columns", 1, 6, 3)
        
        # Database info
        st.subheader("Database Status")
        try:
            stats = app.get_database_stats()
            if stats.get("exists", False):
                st.success(f"✅ {stats['point_count']} faces indexed")
                st.write(f"Vector size: {stats['vector_size']}")
                st.write(f"Distance: {stats['distance_metric']}")
            else:
                st.warning("⚠️ No faces indexed yet")
        except Exception as e:
            st.error(f"Database error: {str(e)}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Face Search", 
        "📝 Text Search", 
        "📊 Database", 
        "📁 Batch Process", 
        "ℹ️ About"
    ])
    
    # =============================================================================
    #  Tab 1: Face Search
    # =============================================================================
    with tab1:
        st.header("Face Similarity Search")
        st.markdown("Upload a face image to find similar faces in the database.")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose a face image", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing a face to search for similar faces"
        )
        
        if uploaded_file is not None:
            try:
                # Load and display uploaded image
                query_image = Image.open(uploaded_file)
                st.image(query_image, caption="Query Image", width=300)
                
                # Search button
                if st.button("🔍 Search Similar Faces", type="primary"):
                    with st.spinner("Searching for similar faces..."):
                        try:
                            # Perform face search
                            results = app.search_similar_faces(
                                query_image, 
                                limit=max_results,
                                score_threshold=score_threshold if score_threshold > 0 else None
                            )
                            
                            if results:
                                st.success(f"Found {len(results)} similar faces!")
                                
                                # Display results in grid
                                cols = st.columns(grid_columns)
                                for i, result in enumerate(results):
                                    col = cols[i % grid_columns]
                                    
                                    with col:
                                        st.subheader(f"Result {i+1}")
                                        
                                        # Load and display result image
                                        try:
                                            if "filepath" in result["metadata"]:
                                                result_image = Image.open(result["metadata"]["filepath"])
                                                st.image(result_image, use_container_width=True)
                                            
                                            # Display metadata
                                            if show_metadata:
                                                st.write(f"**Score:** {result['score']:.4f}")
                                                st.write(f"**Filename:** {result['filename']}")
                                                st.write(f"**Source:** {result['source_image']}")
                                                
                                        except Exception as e:
                                            st.error(f"Error loading image: {str(e)}")
                                
                                # Similarity chart
                                if show_similarity_chart and len(results) > 1:
                                    fig = create_similarity_chart(results)
                                    if fig is not None:
                                        st.plotly_chart(fig, use_container_width=True)
                                
                            else:
                                st.warning("No similar faces found. Try adjusting the score threshold.")
                                
                        except Exception as e:
                            st.error(f"Search failed: {str(e)}")
                            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    # =============================================================================
    #  Tab 2: Text Search
    # =============================================================================
    with tab2:
        st.header("Text-Based Search")
        st.markdown("Search for faces using text queries on filenames or metadata.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_query = st.text_input("Enter search query", placeholder="e.g., 000001, person_name, etc.")
        
        with col2:
            search_type = st.selectbox(
                "Search Type",
                ["filename", "source_image", "all"],
                help="filename: search by face filename, source_image: search by original image name, all: get all faces"
            )
        
        if st.button("🔍 Search by Text", type="primary"):
            if text_query or search_type == "all":
                with st.spinner("Searching..."):
                    try:
                        results = app.search_by_text(text_query, search_type, max_results)
                        
                        if results:
                            st.success(f"Found {len(results)} faces!")
                            
                            # Display results
                            for i, result in enumerate(results):
                                with st.expander(f"Result {i+1}: {result['filename']}"):
                                    try:
                                        if "filepath" in result["metadata"]:
                                            result_image = Image.open(result["metadata"]["filepath"])
                                            st.image(result_image, width=200)
                                        
                                        st.write("**Metadata:**")
                                        for key, value in result["metadata"].items():
                                            st.write(f"• **{key}**: {value}")
                                            
                                    except Exception as e:
                                        st.error(f"Error loading image: {str(e)}")
                        else:
                            st.warning("No faces found for this query.")
                            
                    except Exception as e:
                        st.error(f"Text search failed: {str(e)}")
            else:
                st.warning("Please enter a search query.")
    
    # =============================================================================
    #  Tab 3: Database Management
    # =============================================================================
    with tab3:
        st.header("Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Database Statistics")
            try:
                stats = app.get_database_stats()
                
                if stats.get("exists", False):
                    # Create metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("Total Faces", stats["point_count"])
                    
                    with metric_col2:
                        st.metric("Vector Size", stats["vector_size"])
                    
                    with metric_col3:
                        st.metric("Distance Metric", stats["distance_metric"])
                    
                    # Status
                    st.success(f"✅ Collection '{stats['name']}' is active")
                    
                    # Create a simple chart
                    if stats["point_count"] > 0:
                        if PLOTLY_AVAILABLE:
                            fig = go.Figure(data=[
                                go.Bar(x=["Indexed Faces"], y=[stats["point_count"]], 
                                       marker_color='lightgreen')
                            ])
                            fig.update_layout(
                                title="Database Overview",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write(f"📊 **Total Indexed Faces:** {stats['point_count']}")
                else:
                    st.warning("⚠️ No database collection found")
                    
            except Exception as e:
                st.error(f"Error getting database stats: {str(e)}")
        
        with col2:
            st.subheader("Configuration")
            st.json({
                "Input Path": config["dataset"]["input_path"],
                "Processed Path": config["dataset"]["processed_path"],
                "Vector DB Path": config["vector_db"]["path"],
                "Face Encoder": config["image_encoding"]["model"]
            })
    
    # =============================================================================
    #  Tab 4: Batch Processing
    # =============================================================================
    with tab4:
        st.header("Batch Processing")
        st.markdown("Process multiple images and manage the face database.")
        
        # File upload for batch processing
        uploaded_files = st.file_uploader(
            "Choose multiple face images for batch processing",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple face images to add them to the database"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files")
            
            if st.button("📥 Process and Index Images", type="primary"):
                with st.spinner("Processing images..."):
                    try:
                        # Process each uploaded file
                        processed_count = 0
                        errors = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            try:
                                # Load image
                                image = Image.open(uploaded_file)
                                
                                # Create metadata
                                metadata = {
                                    "filename": uploaded_file.name,
                                    "source": "uploaded",
                                    "upload_time": str(pd.Timestamp.now())
                                }
                                
                                # Index the face
                                point_id = app.face_search.index_face(image, metadata)
                                processed_count += 1
                                
                                status_text.text(f"Processed {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                                progress_bar.progress((i + 1) / len(uploaded_files))
                                
                            except Exception as e:
                                errors.append(f"{uploaded_file.name}: {str(e)}")
                        
                        # Display results
                        st.success(f"Successfully processed {processed_count} images!")
                        
                        if errors:
                            st.warning(f"Errors in {len(errors)} files:")
                            for error in errors:
                                st.write(f"• {error}")
                        
                        # Refresh database stats
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Batch processing failed: {str(e)}")
        
        # Pipeline controls
        st.subheader("Pipeline Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Run Complete Pipeline", type="secondary"):
                st.warning("This will process the entire dataset. This may take a long time.")
                if st.button("Confirm Pipeline Run", type="primary"):
                    with st.spinner("Running complete pipeline..."):
                        try:
                            stats = app.run_complete_pipeline()
                            st.success("Pipeline completed successfully!")
                            st.json(stats)
                        except Exception as e:
                            st.error(f"Pipeline failed: {str(e)}")
        
        with col2:
            if st.button("📊 Refresh Database Stats"):
                st.rerun()
    
    # =============================================================================
    #  Tab 5: About
    # =============================================================================
    with tab5:
        st.header("About Person of Interest")
        
        st.markdown("""
        **Person of Interest** is an advanced face recognition and similarity search application 
        built with modern deep learning techniques.
        
        ### Features
        - 🔍 **Face Similarity Search**: Upload a face image to find similar faces
        - 📝 **Text Search**: Search faces by filename or metadata
        - 🧠 **Deep Learning**: Uses state-of-the-art face encoders (FaceNet, ArcFace)
        - 🗄️ **Vector Database**: Powered by Qdrant for fast similarity search
        - 📊 **Analytics**: Comprehensive database statistics and visualization
        - 🔄 **Batch Processing**: Process multiple images at once
        
        ### Technology Stack
        - **Face Detection**: RetinaFace
        - **Face Encoding**: FaceNet, ArcFace, SigLIP
        - **Vector Database**: Qdrant
        - **Web Interface**: Streamlit
        - **Deep Learning**: PyTorch
        
        ### Configuration
        """)
        
        st.json({
            "Version": config["app"]["version"],
            "Description": config["app"]["description"],
            "Input Dataset": config["dataset"]["input_path"],
            "Processed Faces": config["dataset"]["processed_path"],
            "Vector Database": config["vector_db"]["path"],
            "Face Encoder Model": config["image_encoding"]["model"]
        })
        
        st.markdown("""
        ### Usage Tips
        1. **First Time**: Run the complete pipeline to index your face dataset
        2. **Face Search**: Upload clear face images for best results
        3. **Text Search**: Use partial filenames or source image names
        4. **Batch Processing**: Upload multiple images to add them to the database
        5. **Score Threshold**: Adjust to filter results by similarity
        
        ### Support
        For issues or questions, please refer to the project documentation.
        """)


if __name__ == "__main__":
    main()
