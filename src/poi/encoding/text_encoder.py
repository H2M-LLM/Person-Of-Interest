# =============================================================================
#  Filename: text_encoder.py
#
#  Short Description: Text encoder using sentence transformers (same as rag_to_riches).
#
#  Creation date: 2025-01-06
#  Author: SupportVectors AI Training Team
# =============================================================================

import torch
from typing import Union, List, Optional, Dict, Any
from icontract import require, ensure
from loguru import logger
from qdrant_client import models
from uuid import uuid4
from sentence_transformers import SentenceTransformer

from poi import config


#============================================================================================
#  Class: TextEncoder
#============================================================================================
class TextEncoder:
    """Text encoder using sentence transformers (same as rag_to_riches).
    
    This encoder uses the all-MiniLM-L6-v2 model, which is the same
    text embedder used in the rag_to_riches project.
    """
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    @require(lambda: "text_encoding" in config, "Config must contain text_encoding section")
    def __init__(self) -> None:
        """Initialize the text encoder with configuration from config.yaml."""
        self.config = config["text_encoding"]
        
        # Load sentence transformers model (same as rag_to_riches)
        try:
            model_name = self.config["model_name"]
            
            # Load sentence transformers model
            self.model = SentenceTransformer(model_name)
            
            # Get device and move model
            self.device = self._get_device()
            self.model.to(self.device)
            
            # Determine vector size by running a test embedding
            with torch.no_grad():
                test_embedding = self.model.encode("test")
                self.vector_size = test_embedding.shape[0]
            
            self.distance_metric = "Cosine"
            
            logger.info(f"Initialized TextEncoder with model '{model_name}', "
                       f"vector size: {self.vector_size}")
            
        except Exception as e:
            raise ValueError(f"Failed to initialize text encoder: {e}")
    
    # ----------------------------------------------------------------------------------------
    #  Encode Text
    # ----------------------------------------------------------------------------------------
    @require(lambda text: isinstance(text, str) and len(text.strip()) > 0,
             "Text must be a non-empty string")
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text using sentence transformers.
        
        Args:
            text: Text string to encode.
            
        Returns:
            Torch tensor containing the text embedding.
        """
        try:
            # Generate embedding using sentence transformers
            with torch.no_grad():
                embedding = self.model.encode(text, convert_to_tensor=True)
                # Normalize embedding
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
            
            logger.debug(f"Generated text embedding: shape {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise ValueError(f"Text encoding failed: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Encode Multiple Texts
    # ----------------------------------------------------------------------------------------
    def encode_texts_batch(self, texts: List[str]) -> List[torch.Tensor]:
        """Encode multiple texts in batch for efficiency.
        
        Args:
            texts: List of text strings to encode.
            
        Returns:
            List of torch tensors containing text embeddings.
        """
        embeddings = []
        for text in texts:
            embedding = self.encode_text(text)
            embeddings.append(embedding)
        return embeddings
    
    # ----------------------------------------------------------------------------------------
    #  Get Vector Size
    # ----------------------------------------------------------------------------------------
    def get_vector_size(self) -> int:
        """Get the dimensionality of text embeddings."""
        return self.vector_size
    
    # ----------------------------------------------------------------------------------------
    #  Get Distance Metric
    # ----------------------------------------------------------------------------------------
    def get_distance_metric(self) -> str:
        """Get the recommended distance metric for text embeddings."""
        return self.distance_metric
    
    # ----------------------------------------------------------------------------------------
    #  Create Point Struct
    # ----------------------------------------------------------------------------------------
    def create_point(self, text: str, metadata: Optional[Dict[str, Any]] = None,
                    point_id: Optional[str] = None) -> models.PointStruct:
        """Create a PointStruct for vector database storage.
        
        Args:
            text: Text string to encode.
            metadata: Optional metadata to store with the point.
            point_id: Optional custom ID for the point.
            
        Returns:
            PointStruct containing the embedding vector and metadata.
        """
        # Generate embedding
        embedding = self.encode_text(text)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add text metadata
        metadata.update({
            'content_type': 'text',
            'text_content': text,
            'text_length': len(text),
            'encoder_model': 'sentence-transformers',
            'model_name': self.config["model_name"]
        })
        
        # Generate point ID if not provided
        if point_id is None:
            point_id = str(uuid4())
        
        # Create point
        point = models.PointStruct(
            id=point_id,
            vector=embedding.cpu().numpy().flatten().tolist(),
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
def create_text_encoder() -> TextEncoder:
    """Factory function to create a TextEncoder instance.
    
    Returns:
        Configured TextEncoder instance.
    """
    return TextEncoder()
