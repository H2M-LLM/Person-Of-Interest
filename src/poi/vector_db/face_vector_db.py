# =============================================================================
#  Filename: face_vector_db.py
#
#  Short Description: Vector database client using Qdrant for storing and searching face embeddings.
#
#  Creation date: 2025-01-06
#  Author: SupportVectors AI Training Team
# =============================================================================

from pathlib import Path
from typing import List, Optional, Dict, Any
from icontract import require, ensure, invariant
from qdrant_client import QdrantClient, models
from loguru import logger

from poi import config


#============================================================================================
#  Class: FaceVectorDB
#============================================================================================
@invariant(lambda self: hasattr(self, 'client') and self.client is not None, 
           "Client must be initialized and not None")
class FaceVectorDB:
    """Vector database client using Qdrant for storing and searching face embeddings.
    
    This class provides a simple interface to interact with a local Qdrant
    vector database specifically for face embeddings, following the patterns
    from rag_to_riches embedded_vectordb.py.
    """
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    @require(lambda: "vector_db" in config, "Config must contain vector_db section")
    @require(lambda: "path" in config["vector_db"], "Config must contain vector_db.path")
    @ensure(lambda self: hasattr(self, 'client'), "Client must be initialized after construction")
    def __init__(self) -> None:
        """Initialize the face vector database client.
        
        Raises:
            ValueError: If the vector database path doesn't exist or cannot be created.
        """
        self.config = config["vector_db"]
        path = self.config["path"]
        
        logger.info(f"Connecting to face vector database at {path}")
        
        # Ensure the directory exists
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        # Initialize Qdrant client
        self.client = QdrantClient(path=str(path_obj))
        logger.info(f"Connected to face vector database at {path}")
    
    # ----------------------------------------------------------------------------------------
    #  Collection Exists
    # ----------------------------------------------------------------------------------------
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.
        
        Args:
            collection_name: Name of the collection to check.
            
        Returns:
            True if collection exists, False otherwise.
        """
        return self.client.collection_exists(collection_name)
    
    # ----------------------------------------------------------------------------------------
    #  Count Points
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    def count_points(self, collection_name: str) -> int:
        """Count the number of points in a collection.
        
        Args:
            collection_name: Name of the collection.
            
        Returns:
            Number of points in the collection.
            
        Raises:
            ValueError: If collection doesn't exist.
        """
        if not self.client.collection_exists(collection_name):
            try:
                available_collections = [col.name for col in self.client.get_collections().collections]
            except Exception:
                available_collections = None
            
            raise ValueError(
                f"Collection '{collection_name}' does not exist. "
                f"Available collections: {available_collections}"
            )
        
        result = self.client.count(collection_name)
        return result.count
    
    # ----------------------------------------------------------------------------------------
    #  Get Collection Info
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    def get_collection_info(self, collection_name: str) -> models.CollectionInfo:
        """Get information about a collection.
        
        Args:
            collection_name: Name of the collection.
            
        Returns:
            CollectionInfo object containing collection details.
            
        Raises:
            ValueError: If collection doesn't exist.
        """
        if not self.client.collection_exists(collection_name):
            try:
                available_collections = [col.name for col in self.client.get_collections().collections]
            except Exception:
                available_collections = None
            
            raise ValueError(
                f"Collection '{collection_name}' does not exist. "
                f"Available collections: {available_collections}"
            )
        
        return self.client.get_collection(collection_name)
    
    # ----------------------------------------------------------------------------------------
    #  List Collections
    # ----------------------------------------------------------------------------------------
    def list_collections(self) -> List[str]:
        """List all available collections.
        
        Returns:
            List of collection names.
        """
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.warning(f"Error listing collections: {e}")
            return []
    
    # ----------------------------------------------------------------------------------------
    #  Ensure Collection
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    @require(lambda vector_size: isinstance(vector_size, int) and vector_size > 0,
             "Vector size must be a positive integer")
    @require(lambda distance: distance in ["Cosine", "Euclidean", "Dot"],
             "Distance metric must be Cosine, Euclidean, or Dot")
    def ensure_collection(self, collection_name: str, vector_size: int, 
                         distance: str = "Cosine") -> None:
        """Ensure a collection exists with the specified parameters.
        
        Args:
            collection_name: Name of the collection.
            vector_size: Size of the vectors in the collection.
            distance: Distance metric to use.
            
        Raises:
            ValueError: If collection parameters are invalid.
        """
        try:
            if self.client.collection_exists(collection_name):
                # Check if existing collection has correct parameters
                collection_info = self.get_collection_info(collection_name)
                existing_size = collection_info.config.params.vectors.size
                existing_distance = collection_info.config.params.vectors.distance
                
                if existing_size != vector_size:
                    raise ValueError(
                        f"Collection '{collection_name}' exists with vector size {existing_size}, "
                        f"but requested size is {vector_size}"
                    )
                
                if existing_distance != distance:
                    raise ValueError(
                        f"Collection '{collection_name}' exists with distance metric {existing_distance}, "
                        f"but requested metric is {distance}"
                    )
                
                logger.info(f"Collection '{collection_name}' already exists with correct parameters")
            else:
                # Create new collection
                # Map distance string to Qdrant distance enum
                distance_map = {
                    "Cosine": models.Distance.COSINE,
                    "Euclidean": models.Distance.EUCLID,
                    "Dot": models.Distance.DOT
                }
                
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance_map.get(distance, models.Distance.COSINE)
                    )
                )
                logger.info(f"Created collection '{collection_name}' with vector size {vector_size} and distance {distance}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection '{collection_name}': {e}")
            raise ValueError(f"Failed to ensure collection '{collection_name}': {e}")
    
    # ----------------------------------------------------------------------------------------
    #  Upsert Points
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    @require(lambda points: isinstance(points, list) and len(points) > 0,
             "Points must be a non-empty list")
    def upsert_points(self, collection_name: str, points: List[models.PointStruct]) -> None:
        """Insert or update points in the collection.
        
        Args:
            collection_name: Name of the collection.
            points: List of PointStruct objects to insert/update.
            
        Raises:
            ValueError: If collection doesn't exist or points are invalid.
        """
        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} points to collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error upserting points to collection '{collection_name}': {e}")
            raise ValueError(f"Failed to upsert points: {e}")
    
    # ----------------------------------------------------------------------------------------
    #  Search Points
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    @require(lambda query_vector: isinstance(query_vector, (list, tuple)) and len(query_vector) > 0,
             "Query vector must be a non-empty list or tuple")
    @require(lambda limit: isinstance(limit, int) and limit > 0,
             "Limit must be a positive integer")
    def search_points(self, collection_name: str, query_vector: List[float], 
                     limit: int = 10, score_threshold: Optional[float] = None) -> List[models.ScoredPoint]:
        """Search for similar points in the collection.
        
        Args:
            collection_name: Name of the collection.
            query_vector: Query vector for similarity search.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold.
            
        Returns:
            List of ScoredPoint objects sorted by similarity.
            
        Raises:
            ValueError: If collection doesn't exist or search fails.
        """
        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        try:
            # Prepare search parameters
            search_params = {
                "collection_name": collection_name,
                "query_vector": query_vector,
                "limit": limit,
                "with_payload": True,
                "with_vectors": False
            }
            
            # Add score threshold if provided
            if score_threshold is not None:
                search_params["score_threshold"] = score_threshold
            
            # Perform search
            results = self.client.search(**search_params)
            
            logger.info(f"Search in collection '{collection_name}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching in collection '{collection_name}': {e}")
            raise ValueError(f"Search failed: {e}")
    
    # ----------------------------------------------------------------------------------------
    #  Delete Collection
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete.
            
        Raises:
            ValueError: If collection doesn't exist or deletion fails.
        """
        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {e}")
            raise ValueError(f"Failed to delete collection: {e}")
    
    # ----------------------------------------------------------------------------------------
    #  Get Point by ID
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    @require(lambda point_id: isinstance(point_id, str) and len(point_id.strip()) > 0,
             "Point ID must be a non-empty string")
    def get_point(self, collection_name: str, point_id: str) -> Optional[models.Record]:
        """Get a specific point by its ID.
        
        Args:
            collection_name: Name of the collection.
            point_id: ID of the point to retrieve.
            
        Returns:
            Record object if found, None otherwise.
            
        Raises:
            ValueError: If collection doesn't exist.
        """
        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        try:
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False
            )
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error retrieving point '{point_id}' from collection '{collection_name}': {e}")
            raise ValueError(f"Failed to retrieve point: {e}")


#============================================================================================
#  Factory Function
#============================================================================================
def create_face_vector_db() -> FaceVectorDB:
    """Factory function to create a FaceVectorDB instance.
    
    Returns:
        Configured FaceVectorDB instance.
    """
    return FaceVectorDB()
