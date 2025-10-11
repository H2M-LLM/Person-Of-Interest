# =============================================================================
#  Filename: siglip_encoder.py
#
#  Short Description: SigLIP-based face encoder using the same model as rag_to_riches.
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
import open_clip

from poi import config
from .face_encoder import FaceEncoder


#============================================================================================
#  Class: SigLIPFaceEncoder
#============================================================================================
class SigLIPFaceEncoder(FaceEncoder):
    """SigLIP-based face encoder using the same model as rag_to_riches.
    
    This encoder uses the ViT-B-16-SigLIP2 model from Open-CLIP, which is
    the same multimodal model used in the rag_to_riches project.
    """
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    @require(lambda: "image_encoding" in config, "Config must contain image_encoding section")
    def __init__(self) -> None:
        """Initialize the SigLIP face encoder with configuration from config.yaml."""
        self.config = config["image_encoding"]
        
        # Load SigLIP model (same as rag_to_riches)
        try:
            model_name = self.config["model_name"]
            pretrained = self.config["pretrained"]
            
            # Load SigLIP model from open_clip (same as rag_to_riches)
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()
            
            # Move model to appropriate device
            self.device = self._get_device()
            self.model.to(self.device)
            
            self.vector_size = self.config["embedding_dim"]
            self.distance_metric = "Cosine"
            
            logger.info(f"Initialized SigLIPFaceEncoder with model '{model_name}' "
                       f"(pretrained: {pretrained}), vector size: {self.vector_size}")
            
        except Exception as e:
            raise ValueError(f"Failed to initialize SigLIP model: {e}")
    
    # ----------------------------------------------------------------------------------------
    #  Encode Face
    # ----------------------------------------------------------------------------------------
    @require(lambda face_image: isinstance(face_image, Image.Image), "Face image must be PIL Image")
    @ensure(lambda result: isinstance(result, np.ndarray), "Must return numpy array")
    def encode_face(self, face_image: Image.Image) -> np.ndarray:
        """Encode a face image using SigLIP.
        
        Args:
            face_image: PIL Image containing a face.
            
        Returns:
            Numpy array containing the face embedding.
        """
        try:
            # Preprocess image using SigLIP preprocessing
            image_tensor = self.preprocess(face_image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                # Normalize features
                image_features = F.normalize(image_features, p=2, dim=1)
            
            # Convert to numpy
            embedding_np = image_features.cpu().numpy().flatten()
            
            logger.debug(f"Generated SigLIP embedding: shape {embedding_np.shape}")
            return embedding_np
            
        except Exception as e:
            logger.error(f"Error encoding face with SigLIP: {str(e)}")
            raise ValueError(f"SigLIP encoding failed: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Encode Text (Multimodal)
    # ----------------------------------------------------------------------------------------
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using SigLIP (multimodal model).
        
        Args:
            text: Text string to encode.
            
        Returns:
            Numpy array containing the text embedding (same space as images).
        """
        try:
            # Generate text embedding using SigLIP
            with torch.no_grad():
                # Use the model's tokenizer with the correct context length (64 for SigLIP)
                text_tokens = open_clip.tokenize([text], context_length=64).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                # Normalize features
                text_features = F.normalize(text_features, p=2, dim=1)
            
            # Convert to numpy (remove batch dimension)
            embedding_np = text_features.squeeze(0).cpu().numpy()
            
            logger.debug(f"Generated SigLIP text embedding: shape {embedding_np.shape}")
            return embedding_np
            
        except Exception as e:
            logger.error(f"Error encoding text with SigLIP: {str(e)}")
            raise ValueError(f"SigLIP text encoding failed: {str(e)}")
    
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
    #  Get Vector Size
    # ----------------------------------------------------------------------------------------
    def get_vector_size(self) -> int:
        """Get the dimensionality of SigLIP embeddings."""
        return self.vector_size
    
    # ----------------------------------------------------------------------------------------
    #  Get Distance Metric
    # ----------------------------------------------------------------------------------------
    def get_distance_metric(self) -> str:
        """Get the recommended distance metric for SigLIP."""
        return self.distance_metric
    
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
            'image_mode': face_image.mode,
            'encoder_model': 'siglip',
            'model_name': self.config["model_name"]
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


#============================================================================================
#  Factory Function
#============================================================================================
def create_siglip_face_encoder() -> SigLIPFaceEncoder:
    """Factory function to create a SigLIPFaceEncoder instance.
    
    Returns:
        Configured SigLIPFaceEncoder instance.
    """
    return SigLIPFaceEncoder()
