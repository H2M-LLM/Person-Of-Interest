# =============================================================================
#  Filename: face_search.py
#
#  Short Description: High-level face search interface combining face encoders and vector database.
#
#  Creation date: 2025-01-06
#  Author: SupportVectors AI Training Team
# =============================================================================

from typing import List, Dict, Any, Optional
from PIL import Image
from icontract import require, ensure
from qdrant_client import models
from loguru import logger

from poi.encoding.face_encoder import FaceEncoder
from poi.vector_db.face_vector_db import FaceVectorDB
from poi import config


#============================================================================================
#  Class: FaceSearch
#============================================================================================
class FaceSearch:
    """High-level face search interface combining face encoders and vector database.
    
    This class provides a unified interface for indexing and searching face images
    using embeddings stored in a Qdrant vector database. It follows the patterns
    from rag_to_riches semantic_search.py for consistency.
    """
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    @require(lambda face_encoder: isinstance(face_encoder, FaceEncoder), 
             "Face encoder must be a FaceEncoder instance")
    @require(lambda vector_db: isinstance(vector_db, FaceVectorDB), 
             "Vector DB must be a FaceVectorDB instance")
    @require(lambda collection_name: isinstance(collection_name, str) and len(collection_name.strip()) > 0,
             "Collection name must be a non-empty string")
    def __init__(self, face_encoder: FaceEncoder, vector_db: FaceVectorDB, 
                 collection_name: str) -> None:
        """Initialize the face search system.
        
        Args:
            face_encoder: FaceEncoder instance for converting face images to vectors.
            vector_db: FaceVectorDB instance for storage and retrieval.
            collection_name: Name of the collection to work with.
        """
        self.face_encoder = face_encoder
        self.vector_db = vector_db
        self.collection_name = collection_name
        
        # Ensure collection exists with correct parameters
        self._ensure_collection_setup()
        
        logger.info(f"Initialized FaceSearch for collection '{collection_name}' "
                   f"with {type(face_encoder).__name__}")

    # ----------------------------------------------------------------------------------------
    #  Consistency Check
    # ----------------------------------------------------------------------------------------
    def consistency_check(self) -> bool:
        """Ensure that the vector size and distance metric of the collection is compatible with the encoder.
        
        Returns:
            True if collection parameters match encoder requirements.
            
        Raises:
            ValueError: If collection doesn't exist or parameters don't match.
        """
        if not self.vector_db.collection_exists(self.collection_name):
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist. "
                f"Available collections: {self.vector_db.list_collections()}"
            )
        
        # Get collection info
        collection_info = self.vector_db.get_collection_info(self.collection_name)
        collection_vector_size = collection_info.config.params.vectors.size
        collection_distance = collection_info.config.params.vectors.distance
        
        # Get encoder requirements
        encoder_vector_size = self.face_encoder.get_vector_size()
        encoder_distance = self.face_encoder.get_distance_metric()
        
        # Check for mismatches
        if collection_vector_size != encoder_vector_size:
            raise ValueError(
                f"Collection '{self.collection_name}' has vector size {collection_vector_size}, "
                f"but encoder requires {encoder_vector_size}"
            )
        
        if collection_distance != encoder_distance:
            raise ValueError(
                f"Collection '{self.collection_name}' has distance metric {collection_distance}, "
                f"but encoder requires {encoder_distance}"
            )
        
        logger.info(f"Consistency check passed for collection '{self.collection_name}'")
        return True

    # ----------------------------------------------------------------------------------------
    #  Index Face Image
    # ----------------------------------------------------------------------------------------
    @require(lambda face_image: isinstance(face_image, Image.Image), 
             "Face image must be a PIL Image instance")
    def index_face(self, face_image: Image.Image, metadata: Optional[Dict[str, Any]] = None,
                   point_id: Optional[str] = None) -> str:
        """Index a face image into the collection.
        
        Args:
            face_image: PIL Image containing a face.
            metadata: Optional metadata to store with the face.
            point_id: Optional custom ID for the point.
            
        Returns:
            The ID of the indexed point.
            
        Raises:
            ValueError: If face cannot be encoded or indexed.
        """
        try:
            # Create point using the face encoder
            point = self.face_encoder.create_point(face_image, metadata, point_id)
            
            # Store in vector database
            self.vector_db.upsert_points(self.collection_name, [point])
            
            logger.debug(f"Indexed face image (size: {face_image.size}) with ID: {point.id}")
            return point.id
            
        except Exception as e:
            logger.error(f"Error indexing face image: {str(e)}")
            raise ValueError(f"Failed to index face image: {str(e)}")

    # ----------------------------------------------------------------------------------------
    #  Search with Face Image
    # ----------------------------------------------------------------------------------------
    @require(lambda query_face: isinstance(query_face, Image.Image),
             "Query face must be a PIL Image instance")
    @require(lambda limit: isinstance(limit, int) and limit > 0,
             "Limit must be a positive integer")
    @ensure(lambda result: isinstance(result, list), "Must return a list")
    def search_with_face(self, query_face: Image.Image, limit: int = 10,
                        score_threshold: Optional[float] = None) -> List[models.ScoredPoint]:
        """Search for similar faces using a face image query.
        
        Args:
            query_face: PIL Image containing a face to search for.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold.
            
        Returns:
            List of scored points sorted by similarity.
            
        Raises:
            ValueError: If query cannot be processed.
        """
        try:
            # Create query embedding
            query_embedding = self.face_encoder.encode_face(query_face)
            
            # Convert numpy array to list if needed
            if hasattr(query_embedding, 'tolist'):
                query_vector = query_embedding.tolist()
            else:
                query_vector = query_embedding
            
            # Search in vector database
            results = self.vector_db.search_points(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            logger.info(f"Face search (size: {query_face.size}) returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching with face image: {str(e)}")
            raise ValueError(f"Failed to search with face image: {str(e)}")

    # ----------------------------------------------------------------------------------------
    #  Index Multiple Face Images
    # ----------------------------------------------------------------------------------------
    @require(lambda face_images: isinstance(face_images, list) and len(face_images) > 0,
             "Face images must be a non-empty list")
    @require(lambda face_images: all(isinstance(img, Image.Image) for img in face_images),
             "All items must be PIL Image instances")
    def index_all_faces(self, face_images: List[Image.Image],
                       metadata_list: Optional[List[Dict[str, Any]]] = None,
                       point_ids: Optional[List[str]] = None) -> List[str]:
        """Index multiple face images into the collection.
        
        Args:
            face_images: List of PIL Images containing faces.
            metadata_list: Optional list of metadata for each face.
            point_ids: Optional list of custom IDs for the points.
            
        Returns:
            List of IDs of the indexed points.
            
        Raises:
            ValueError: If faces cannot be encoded or indexed.
        """
        try:
            # Validate input lengths
            if metadata_list and len(metadata_list) != len(face_images):
                raise ValueError("Metadata list length must match face images list length")
            
            if point_ids and len(point_ids) != len(face_images):
                raise ValueError("Point IDs list length must match face images list length")
            
            # Create points for all face images
            points = []
            indexed_ids = []
            
            for i, face_image in enumerate(face_images):
                metadata = metadata_list[i] if metadata_list else None
                point_id = point_ids[i] if point_ids else None
                
                point = self.face_encoder.create_point(face_image, metadata, point_id)
                points.append(point)
                indexed_ids.append(point.id)
            
            # Batch upsert to vector database
            self.vector_db.upsert_points(self.collection_name, points)
            
            logger.info(f"Indexed {len(face_images)} face images into collection '{self.collection_name}'")
            return indexed_ids
            
        except Exception as e:
            logger.error(f"Error indexing face images: {str(e)}")
            raise ValueError(f"Failed to index face images: {str(e)}")

    # ----------------------------------------------------------------------------------------
    #  Get Collection Statistics
    # ----------------------------------------------------------------------------------------
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary containing collection statistics.
        """
        try:
            if not self.vector_db.collection_exists(self.collection_name):
                return {"exists": False}
            
            # Get collection info
            collection_info = self.vector_db.get_collection_info(self.collection_name)
            point_count = self.vector_db.count_points(self.collection_name)
            
            stats = {
                "exists": True,
                "name": self.collection_name,
                "point_count": point_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
                "status": collection_info.status
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}

    # ----------------------------------------------------------------------------------------
    #  Helper Methods (Private)
    # ----------------------------------------------------------------------------------------
    def _ensure_collection_setup(self) -> None:
        """Helper to ensure collection exists with correct parameters."""
        vector_size = self.face_encoder.get_vector_size()
        distance_metric = self.face_encoder.get_distance_metric()
        
        # Ensure collection exists with correct parameters
        self.vector_db.ensure_collection(
            collection_name=self.collection_name,
            vector_size=vector_size,
            distance=distance_metric
        )
        
        # Run consistency check to verify everything is correct
        self.consistency_check()


#============================================================================================
#  Factory Function
#============================================================================================
def create_face_search(face_encoder: FaceEncoder, vector_db: FaceVectorDB, 
                      collection_name: Optional[str] = None) -> FaceSearch:
    """Factory function to create a FaceSearch instance.
    
    Args:
        face_encoder: FaceEncoder instance.
        vector_db: FaceVectorDB instance.
        collection_name: Name of the collection. If None, uses config.
        
    Returns:
        Configured FaceSearch instance.
    """
    if collection_name is None:
        collection_name = config["vector_db"]["face_embeddings"]["collection_name"]
    
    return FaceSearch(face_encoder, vector_db, collection_name)
