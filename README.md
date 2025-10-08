# Person of Interest - Face Image Processing and Semantic Search

A comprehensive application for face image processing and semantic search using MediaPipe, CLIP multimodal encoder, and vector databases. This application processes face images from the CelebA dataset, extracts face embeddings, and enables text-to-image semantic search through a Qdrant vector database.

## 🎯 Features

- **Face Detection & Cropping**: Uses MediaPipe for accurate face detection and cropping
- **Face Encoding**: Supports CLIP, FaceNet, and ArcFace models for generating face embeddings
- **Vector Database**: Uses Qdrant for efficient storage and retrieval of face embeddings
- **Text-Based Search**: Search for faces using text queries (filename, source image, etc.)
- **Semantic Text Search**: Multimodal text-to-image search using CLIP encoder
- **Image-Based Search**: Find similar faces using image queries
- **CLI Interface**: Easy-to-use command-line interface for all operations
- **Configurable**: All models and paths configurable via `config.yaml`
- **Complete Pipeline**: End-to-end processing from dataset to searchable database
- **GPU Acceleration**: Automatic GPU detection and usage for faster processing

## 🏗️ Architecture

The application follows a modular design inspired by the `rag_to_riches` project:

```
Person-Of-Interest/
├── src/poi/
│   ├── preprocessing/     # Face detection and cropping
│   ├── encoding/         # Face embedding generation
│   ├── vector_db/        # Vector database operations
│   ├── search/           # Semantic search functionality
│   ├── app.py           # Main application orchestrator
│   └── cli.py           # Command-line interface
├── config.yaml          # Configuration file
└── dataset/             # Input and processed images
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies using uv
uv sync

# Or install manually
pip install -e .
```

### 2. Configuration

Edit `config.yaml` to specify:
- Input dataset path (`dataset/img_align_celeba`)
- Model to use (CLIP, FaceNet, or ArcFace)
- Vector database location
- Processing parameters
- Device (auto-detects GPU if available)

### 3. Usage

#### Command Line Interface

```bash
# Run complete pipeline (preprocessing + encoding + storage)
python -m poi.cli pipeline

# Preprocess images only
python -m poi.cli preprocess

# Encode and store faces
python -m poi.cli encode --limit 100  # Encode 100 images
python -m poi.cli encode              # Encode all images

# Search for similar faces using image
python -m poi.cli search-image path/to/query_face.jpg

# Search for faces using text query (metadata search)
python -m poi.cli search-text "000001" --search-type filename
python -m poi.cli search-text "000001" --search-type source_image
python -m poi.cli search-text "" --search-type all

# Semantic text search (multimodal CLIP-based)
python -m poi.cli search-semantic "a happy man"
python -m poi.cli search-semantic "person with glasses"

# Get database statistics
python -m poi.cli stats

# Display application info
python -m poi.cli info
```

#### Python API

```python
from poi import create_app

# Create application instance
app = create_app()

# Run complete pipeline
stats = app.run_complete_pipeline()

# Search for similar faces using image
results = app.search_similar_faces("query_face.jpg", limit=10)

# Search for faces using text query (metadata search)
results = app.search_by_text("000001", "filename", limit=10)
results = app.search_by_text("000001", "source_image", limit=10)
results = app.search_by_text("", "all", limit=100)

# Semantic text search (multimodal CLIP-based)
results = app.search_by_semantic_text("a happy man", limit=10)
results = app.search_by_semantic_text("person with glasses", limit=10)

# Get database statistics
db_stats = app.get_database_stats()
```

## 📋 Requirements

- Python 3.12+
- CUDA-capable GPU (recommended for faster processing)
- CelebA dataset in `dataset/img_align_celeba/` directory

## 🔧 Configuration

The application is configured via `config.yaml`:

```yaml
# Dataset Configuration
dataset:
  input_path: "dataset/img_align_celeba"
  processed_path: "dataset/processed_faces"
  supported_formats: ["jpg", "jpeg", "png"]

# Face Detection Configuration
face_detection:
  model: "mediapipe"
  confidence_threshold: 0.8

# Image Encoding Configuration (using CLIP model)
image_encoding:
  model: "clip"  # Options: clip, siglip, facenet, arcface
  model_name: "ViT-B-32"  # CLIP model
  pretrained: "openai"
  embedding_dim: 512  # CLIP vector size
  device: "auto"  # auto, cpu, cuda

# Vector Database Configuration
vector_db:
  type: "qdrant"
  path: "data/clip_face_embeddings"
  face_embeddings:
    collection_name: "face_embeddings"
    vector_size: 512  # CLIP vector size
    distance: "cosine"
```

## 🎯 Workflow

1. **Preprocessing**: Scan input images → Detect faces → Crop and save
2. **Encoding**: Load cropped faces → Generate embeddings → Batch process
3. **Storage**: Store embeddings in vector database with metadata
4. **Search**: Enable semantic similarity search across all stored faces

## 📊 Example Output

```
Preprocessing Results
┌─────────────────┬───────┐
│ Metric          │ Value │
├─────────────────┼───────┤
│ Total Images    │ 1000  │
│ Processed Images│ 950   │
│ Faces Detected  │ 1200  │
│ Success Rate    │ 95%   │
└─────────────────┴───────┘

Similar Faces Found
┌──────┬────────┬─────────────┬──────────────┐
│ Rank │ Score  │ Filename    │ Source Image │
├──────┼────────┼─────────────┼──────────────┤
│ 1    │ 0.9234 │ face_001.jpg│ 000001       │
│ 2    │ 0.8901 │ face_045.jpg│ 000045       │
└──────┴────────┴─────────────┴──────────────┘
```

## 🛠️ Development

This project uses:
- **CLIP (OpenAI)** for multimodal image and text encoding
- **Qdrant** for vector database operations
- **MediaPipe** for face detection
- **Rich** for beautiful CLI output
- **icontract** for design-by-contract programming
- **loguru** for structured logging
- **PyTorch** with automatic GPU acceleration

## 📝 License

Copyright (c) 2016-2025. SupportVectors AI Lab

This code is part of the training material and, therefore, part of the intellectual property. It may not be reused or shared without the explicit, written permission of SupportVectors.
