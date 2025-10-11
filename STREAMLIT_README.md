# Person of Interest - Streamlit UI

A comprehensive web interface for the Person of Interest face processing and semantic search application.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Streamlit UI dependencies
pip install -r requirements_streamlit.txt

# Or install all dependencies including the main application
pip install -e .
pip install -r requirements_streamlit.txt
```

### 2. Launch the Application

```bash
# Option 1: Use the launcher script (recommended)
python run_streamlit.py

# Option 2: Launch directly with Streamlit
streamlit run streamlit_app.py --server.port 8501
```

### 3. Access the UI

Open your web browser and go to: **http://localhost:8501**

## 🎯 Features

### 🔍 Face-Based Similarity Search
- Upload an image to find similar faces in the dataset
- Adjustable similarity threshold and result count
- Visual comparison with similarity scores

### 📝 Text-Based Metadata Search
- Search by filename (e.g., "000001")
- Search by source image name
- Browse all faces in the database
- Configurable result limits

### 🌐 Semantic Text Search
- Natural language descriptions (e.g., "happy person", "someone with glasses")
- CLIP-based multimodal search
- Similarity scoring and threshold filtering

### 📁 Dataset Browser
- Browse all processed face images
- Paginated view with 20 images per page
- Direct access to individual face images

### 📊 Analytics & Statistics
- Database statistics (collection info, point count, vector size)
- Dataset analysis (file sizes, distribution)
- Sample image previews

## 🎨 UI Components

### Sidebar
- **Configuration**: Dataset path settings
- **Initialization**: App startup and reinitialization
- **Database Stats**: Real-time database statistics

### Main Tabs
1. **Face Search**: Upload and search for similar faces
2. **Text Search**: Metadata-based search functionality
3. **Semantic Search**: Natural language face descriptions
4. **Browse Dataset**: Paginated dataset exploration
5. **Analytics**: Statistics and visualizations

## ⚙️ Configuration

The application uses the dataset path: `/home/biju/supportvectors/Person-Of-Interest/dataset/processed_faces`

You can change this in the sidebar or by updating the `config.yaml` file.

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web UI framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Pillow**: Image processing

### Architecture
- **Frontend**: Streamlit web interface
- **Backend**: Person of Interest application (CLIP, Qdrant, MediaPipe)
- **Data**: Processed face images with CLIP embeddings

### Performance
- **Lazy Loading**: Images loaded on demand
- **Pagination**: Efficient browsing of large datasets
- **Caching**: Streamlit session state for app initialization
- **Batch Processing**: Efficient search operations

## 🐛 Troubleshooting

### Common Issues

1. **"Application not initialized"**
   - Click "Initialize Application" in the sidebar
   - Check that the dataset path exists
   - Ensure all dependencies are installed

2. **"Dataset path does not exist"**
   - Verify the dataset path in the sidebar
   - Check that processed faces are available
   - Update the path if necessary

3. **"No results found"**
   - Try lowering the similarity threshold
   - Increase the number of results
   - Check that the database is populated

4. **"Search failed"**
   - Ensure the vector database is initialized
   - Check that face embeddings are available
   - Verify the CLIP model is loaded

### Debug Mode

To run in debug mode with more verbose output:

```bash
streamlit run streamlit_app.py --logger.level debug
```

## 📱 Mobile Support

The UI is responsive and works on mobile devices:
- Touch-friendly interface
- Responsive image grids
- Mobile-optimized navigation

## 🔒 Security Notes

- The application runs locally by default
- No external data transmission
- All processing happens on your machine
- Dataset remains private and secure

## 🚀 Advanced Usage

### Custom Dataset Path

To use a different dataset:

1. Update the path in the sidebar
2. Or modify `config.yaml`:
   ```yaml
   dataset:
     processed_path: "/your/custom/path"
   ```

### Custom Search Parameters

- **Similarity Threshold**: Adjust for more/less strict matching
- **Result Count**: Control the number of returned results
- **Search Types**: Choose between different search methods

### Batch Operations

The UI supports:
- Multiple image uploads
- Batch similarity searches
- Bulk dataset browsing

## 📈 Performance Tips

1. **Initialize Once**: The app initializes components on first use
2. **Use Pagination**: Browse large datasets efficiently
3. **Adjust Thresholds**: Fine-tune search parameters
4. **Monitor Resources**: Check system resources for large operations

## 🤝 Contributing

To extend the UI:

1. **Add New Tabs**: Extend the tab structure in `main()`
2. **Custom Components**: Create reusable UI components
3. **New Search Types**: Integrate additional search methods
4. **Visualizations**: Add more analytics and charts

## 📄 License

Copyright (c) 2016-2025. SupportVectors AI Lab

This code is part of the training material and, therefore, part of the intellectual property. It may not be reused or shared without the explicit, written permission of SupportVectors.
