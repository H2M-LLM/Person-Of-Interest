#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#   This code is part of the training material and, therefore, part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
"""
Person of Interest - Face Image Processing and Semantic Search Application

This package provides functionality for:
1. Face detection and cropping using RetinaFace
2. Face encoding using FaceNet or ArcFace models
3. Vector database storage using Qdrant
4. Semantic search for similar faces

Usage:
    from poi import create_app
    
    app = create_app()
    app.run_complete_pipeline()  # Process dataset and create searchable database
    results = app.search_similar_faces("query_face.jpg")  # Search for similar faces
"""

from svlearn.config.configuration import ConfigurationMixin

from dotenv import load_dotenv
load_dotenv()

# Load configuration
config = ConfigurationMixin().load_config()

# Import main components
from .app import create_app, PersonOfInterestApp
from .preprocessing.face_detector import create_face_detector, FaceDetector
from .encoding.face_encoder import create_face_encoder, FaceEncoder, FaceNetEncoder, ArcFaceEncoder
from .encoding.siglip_encoder import create_siglip_face_encoder, SigLIPFaceEncoder
from .vector_db.face_vector_db import create_face_vector_db, FaceVectorDB
from .search.face_search import create_face_search, FaceSearch
from .search.text_search import create_text_search, TextSearch
from .search.enhanced_text_search import create_enhanced_text_search, EnhancedTextSearch

# Version information
__version__ = config["app"]["version"]
__author__ = "SupportVectors AI Training Team"
__description__ = config["app"]["description"]

# Public API
__all__ = [
    "config",
    "create_app",
    "PersonOfInterestApp",
    "create_face_detector", 
    "FaceDetector",
    "create_face_encoder",
    "FaceEncoder",
    "FaceNetEncoder", 
    "ArcFaceEncoder",
    "create_siglip_face_encoder",
    "SigLIPFaceEncoder",
    "create_face_vector_db",
    "FaceVectorDB",
    "create_face_search",
    "FaceSearch",
    "create_text_search",
    "TextSearch",
    "create_enhanced_text_search",
    "EnhancedTextSearch",
    "__version__",
    "__author__",
    "__description__"
]
