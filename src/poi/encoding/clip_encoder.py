# =============================================================================
#  Filename: clip_encoder.py
#
#  Short Description: CLIP-based face encoder for testing/comparison.
#
#  Creation date: 2025-10-07
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
from poi.encoding.face_encoder import FaceEncoder


#============================================================================================
#  Class: CLIPFaceEncoder
#============================================================================================
class CLIPFaceEncoder(FaceEncoder):
    """CLIP-based face encoder for comparison with SigLIP.
    
    This encoder uses OpenAI's CLIP model (ViT-B-32) from Open-CLIP.
    """
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    def __init__(self) -> None:
        """Initialize the CLIP face encoder."""
        
        # Load CLIP model from open_clip
        try:
            model_name = "ViT-B-32"
            pretrained = "openai"
            
            # Load CLIP model from open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()
            
            # Move model to appropriate device
            self.device = self._get_device()
            self.model.to(self.device)
            
            self.vector_size = 512  # CLIP ViT-B-32 produces 512-dim embeddings
            self.distance_metric = "Cosine"
            
            logger.info(f"Initialized CLIPFaceEncoder with model '{model_name}' "
                       f"(pretrained: {pretrained}), vector size: {self.vector_size}")
            
        except Exception as e:
            raise ValueError(f"Failed to initialize CLIP model: {e}")
    
    # ----------------------------------------------------------------------------------------
    #  Encode Face
    # ----------------------------------------------------------------------------------------
    @require(lambda face_image: isinstance(face_image, Image.Image), "Face image must be PIL Image")
    @ensure(lambda result: isinstance(result, np.ndarray), "Must return numpy array")
    def encode_face(self, face_image: Image.Image) -> np.ndarray:
        """Encode a face image using CLIP.
        
        Args:
            face_image: PIL Image containing a face.
            
        Returns:
            Numpy array containing the face embedding.
        """
        try:
            # Preprocess image using CLIP preprocessing
            image_tensor = self.preprocess(face_image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                # Normalize features
                image_features = F.normalize(image_features, p=2, dim=1)
            
            # Convert to numpy
            embedding_np = image_features.cpu().numpy().flatten()
            
            logger.debug(f"Generated CLIP embedding: shape {embedding_np.shape}")
            return embedding_np
            
        except Exception as e:
            logger.error(f"Error encoding face with CLIP: {str(e)}")
            raise ValueError(f"CLIP encoding failed: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Encode Text (Multimodal)
    # ----------------------------------------------------------------------------------------
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using CLIP (multimodal model).
        
        Args:
            text: Text string to encode.
            
        Returns:
            Numpy array containing the text embedding (same space as images).
        """
        try:
            # Generate text embedding using CLIP
            with torch.no_grad():
                # Use the model's tokenizer with the correct context length (77 for CLIP)
                text_tokens = open_clip.tokenize([text], context_length=77).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                # Normalize features
                text_features = F.normalize(text_features, p=2, dim=1)
            
            # Convert to numpy (remove batch dimension)
            embedding_np = text_features.squeeze(0).cpu().numpy()
            
            logger.debug(f"Generated CLIP text embedding: shape {embedding_np.shape}")
            return embedding_np
            
        except Exception as e:
            logger.error(f"Error encoding text with CLIP: {str(e)}")
            raise ValueError(f"CLIP text encoding failed: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Get Vector Size
    # ----------------------------------------------------------------------------------------
    def get_vector_size(self) -> int:
        """Get the dimensionality of CLIP embeddings."""
        return self.vector_size
    
    # ----------------------------------------------------------------------------------------
    #  Get Distance Metric
    # ----------------------------------------------------------------------------------------
    def get_distance_metric(self) -> str:
        """Get the recommended distance metric for CLIP."""
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
            'encoder_model': 'clip',
            'model_name': 'ViT-B-32'
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
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


#============================================================================================
#  Factory Function
#============================================================================================
def create_clip_face_encoder() -> CLIPFaceEncoder:
    """Factory function to create a CLIPFaceEncoder instance.
    
    Returns:
        Configured CLIPFaceEncoder instance.
    """
    return CLIPFaceEncoder()



