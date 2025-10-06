# =============================================================================
#  Filename: face_encoder.py
#
#  Short Description: Face encoding using various models (FaceNet, ArcFace) for generating embeddings.
#
#  Creation date: 2025-01-06
#  Author: SupportVectors AI Training Team
# =============================================================================

import torch
import torch.nn.functional as F
from typing import Union, List, Optional, Dict, Any
from PIL import Image
import numpy as np
from icontract import require, ensure
from loguru import logger
from qdrant_client import models
from uuid import uuid4

from poi import config


#============================================================================================
#  Abstract Base Class: FaceEncoder
#============================================================================================
class FaceEncoder:
    """Abstract base class for face encoders.
    
    Defines the interface for converting face images into vector embeddings
    following the patterns from rag_to_riches embedder architecture.
    """
    
    # ----------------------------------------------------------------------------------------
    #  Abstract Method: Encode Face
    # ----------------------------------------------------------------------------------------
    def encode_face(self, face_image: Image.Image) -> np.ndarray:
        """Convert a face image to a vector embedding.
        
        Args:
            face_image: PIL Image containing a face.
            
        Returns:
            Numpy array containing the face embedding.
            
        Raises:
            ValueError: If encoding fails.
        """
        raise NotImplementedError("Subclasses must implement encode_face method")
    
    # ----------------------------------------------------------------------------------------
    #  Abstract Method: Get Vector Size
    # ----------------------------------------------------------------------------------------
    def get_vector_size(self) -> int:
        """Get the dimensionality of the embedding vectors.
        
        Returns:
            Integer representing the vector dimension size.
        """
        raise NotImplementedError("Subclasses must implement get_vector_size method")
    
    # ----------------------------------------------------------------------------------------
    #  Abstract Method: Get Distance Metric
    # ----------------------------------------------------------------------------------------
    def get_distance_metric(self) -> str:
        """Get the recommended distance metric for this encoder.
        
        Returns:
            String representing the distance metric ('Cosine', 'Euclidean', 'Dot').
        """
        raise NotImplementedError("Subclasses must implement get_distance_metric method")
    
    # ----------------------------------------------------------------------------------------
    #  Encode Multiple Faces
    # ----------------------------------------------------------------------------------------
    def encode_faces_batch(self, face_images: List[Image.Image]) -> List[np.ndarray]:
        """Encode multiple face images in batch for efficiency.
        
        Args:
            face_images: List of PIL Images containing faces.
            
        Returns:
            List of numpy arrays containing face embeddings.
        """
        embeddings = []
        for face_image in face_images:
            embedding = self.encode_face(face_image)
            embeddings.append(embedding)
        return embeddings
    
    # ----------------------------------------------------------------------------------------
    #  Create Point Struct
    # ----------------------------------------------------------------------------------------
    def create_point(self, face_image: Image.Image, metadata: Optional[Dict[str, Any]] = None,
                    point_id: Optional[str] = None) -> models.PointStruct:
        """Create a PointStruct for vector database storage.
        
        Args:
            face_image: PIL Image containing a face.
            metadata: Optional metadata to store with the point.
            point_id: Optional custom ID for the point.
            
        Returns:
            PointStruct containing the embedding vector and metadata.
        """
        # Generate embedding
        embedding = self.encode_face(face_image)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add image metadata
        metadata.update({
            'content_type': 'face_image',
            'image_size': face_image.size,
            'image_mode': face_image.mode
        })
        
        # Generate point ID if not provided
        if point_id is None:
            point_id = str(uuid4())
        
        # Create point
        point = models.PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=metadata
        )
        
        return point


#============================================================================================
#  Class: FaceNetEncoder
#============================================================================================
class FaceNetEncoder(FaceEncoder):
    """Face encoder using FaceNet model for face recognition embeddings."""
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    @require(lambda: "image_encoding" in config, "Config must contain image_encoding section")
    def __init__(self) -> None:
        """Initialize the FaceNet encoder with configuration from config.yaml."""
        self.config = config["image_encoding"]
        
        # Load FaceNet model
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
            
            # Initialize MTCNN for face detection (if needed)
            self.mtcnn = MTCNN(
                image_size=self.config["input_size"][0],
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=self._get_device()
            )
            
            # Initialize FaceNet model
            self.model = InceptionResnetV1(pretrained='vggface2').eval()
            self.model.to(self._get_device())
            
            self.vector_size = self.config["embedding_dim"]
            self.distance_metric = "Cosine"
            
            logger.info(f"Initialized FaceNetEncoder with vector size: {self.vector_size}")
            
        except ImportError as e:
            raise ImportError(f"FaceNet dependencies not available: {e}")
        except Exception as e:
            raise ValueError(f"Failed to initialize FaceNet model: {e}")
    
    # ----------------------------------------------------------------------------------------
    #  Encode Face
    # ----------------------------------------------------------------------------------------
    @require(lambda face_image: isinstance(face_image, Image.Image), "Face image must be PIL Image")
    @ensure(lambda result: isinstance(result, np.ndarray), "Must return numpy array")
    def encode_face(self, face_image: Image.Image) -> np.ndarray:
        """Encode a face image using FaceNet.
        
        Args:
            face_image: PIL Image containing a face.
            
        Returns:
            Numpy array containing the face embedding.
        """
        try:
            # Convert PIL to tensor and preprocess
            face_tensor = self._preprocess_image(face_image)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model(face_tensor)
                # L2 normalize the embedding
                embedding = F.normalize(embedding, p=2, dim=1)
            
            # Convert to numpy
            embedding_np = embedding.cpu().numpy().flatten()
            
            logger.debug(f"Generated FaceNet embedding: shape {embedding_np.shape}")
            return embedding_np
            
        except Exception as e:
            logger.error(f"Error encoding face with FaceNet: {str(e)}")
            raise ValueError(f"FaceNet encoding failed: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Get Vector Size
    # ----------------------------------------------------------------------------------------
    def get_vector_size(self) -> int:
        """Get the dimensionality of FaceNet embeddings."""
        return self.vector_size
    
    # ----------------------------------------------------------------------------------------
    #  Get Distance Metric
    # ----------------------------------------------------------------------------------------
    def get_distance_metric(self) -> str:
        """Get the recommended distance metric for FaceNet."""
        return self.distance_metric
    
    # ----------------------------------------------------------------------------------------
    #  Helper Methods
    # ----------------------------------------------------------------------------------------
    def _get_device(self) -> torch.device:
        """Get the appropriate device for computation."""
        device_config = self.config.get("device", "auto")
        
        if device_config == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_config == "cuda":
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image for FaceNet model."""
        # Resize to required size
        target_size = tuple(self.config["input_size"])
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        image_tensor = torch.tensor(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self._get_device())


#============================================================================================
#  Class: ArcFaceEncoder
#============================================================================================
class ArcFaceEncoder(FaceEncoder):
    """Face encoder using ArcFace model for face recognition embeddings."""
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    @require(lambda: "image_encoding" in config, "Config must contain image_encoding section")
    def __init__(self) -> None:
        """Initialize the ArcFace encoder with configuration from config.yaml."""
        self.config = config["image_encoding"]
        
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            # Initialize InsightFace
            self.app = FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            
            self.vector_size = self.config["embedding_dim"]
            self.distance_metric = "Cosine"
            
            logger.info(f"Initialized ArcFaceEncoder with vector size: {self.vector_size}")
            
        except ImportError as e:
            raise ImportError(f"InsightFace dependencies not available: {e}")
        except Exception as e:
            raise ValueError(f"Failed to initialize ArcFace model: {e}")
    
    # ----------------------------------------------------------------------------------------
    #  Encode Face
    # ----------------------------------------------------------------------------------------
    @require(lambda face_image: isinstance(face_image, Image.Image), "Face image must be PIL Image")
    @ensure(lambda result: isinstance(result, np.ndarray), "Must return numpy array")
    def encode_face(self, face_image: Image.Image) -> np.ndarray:
        """Encode a face image using ArcFace.
        
        Args:
            face_image: PIL Image containing a face.
            
        Returns:
            Numpy array containing the face embedding.
        """
        try:
            # Convert PIL to numpy array
            image_array = np.array(face_image)
            
            # Get face embeddings
            faces = self.app.get(image_array)
            
            if not faces:
                raise ValueError("No face detected in image")
            
            # Use the first (and typically only) face
            face = faces[0]
            embedding = face.embedding
            
            # L2 normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            logger.debug(f"Generated ArcFace embedding: shape {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding face with ArcFace: {str(e)}")
            raise ValueError(f"ArcFace encoding failed: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Get Vector Size
    # ----------------------------------------------------------------------------------------
    def get_vector_size(self) -> int:
        """Get the dimensionality of ArcFace embeddings."""
        return self.vector_size
    
    # ----------------------------------------------------------------------------------------
    #  Get Distance Metric
    # ----------------------------------------------------------------------------------------
    def get_distance_metric(self) -> str:
        """Get the recommended distance metric for ArcFace."""
        return self.distance_metric


#============================================================================================
#  Factory Function
#============================================================================================
def create_face_encoder(model_name: Optional[str] = None) -> FaceEncoder:
    """Factory function to create a face encoder instance.
    
    Args:
        model_name: Name of the model to use. If None, uses config.
        
    Returns:
        Configured FaceEncoder instance.
        
    Raises:
        ValueError: If model name is not supported.
    """
    if model_name is None:
        model_name = config["image_encoding"]["model"]
    
    if model_name.lower() == "facenet":
        return FaceNetEncoder()
    elif model_name.lower() == "arcface":
        return ArcFaceEncoder()
    elif model_name.lower() == "siglip":
        # Import SigLIP encoder
        from .siglip_encoder import create_siglip_face_encoder
        return create_siglip_face_encoder()
    else:
        raise ValueError(f"Unsupported face encoder model: {model_name}")
