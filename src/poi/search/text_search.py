# =============================================================================
#  Filename: text_search.py
#
#  Short Description: Text-based search functionality for face descriptions.
#
#  Creation date: 2025-01-06
#  Author: SupportVectors AI Training Team
# =============================================================================

from typing import List, Dict, Any, Optional
from icontract import require, ensure
from loguru import logger
from qdrant_client import models

from poi.vector_db.face_vector_db import FaceVectorDB
from poi import config


#============================================================================================
#  Class: TextSearch
#============================================================================================
class TextSearch:
    """Text-based search functionality for face descriptions.
    
    This class provides the ability to search for faces using text descriptions
    by leveraging the metadata stored with face embeddings.
    """
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    @require(lambda vector_db: isinstance(vector_db, FaceVectorDB), 
             "Vector DB must be a FaceVectorDB instance")
    @require(lambda collection_name: isinstance(collection_name, str) and len(collection_name.strip()) > 0,
             "Collection name must be a non-empty string")
    def __init__(self, vector_db: FaceVectorDB, collection_name: str) -> None:
        """Initialize the text search system.
        
        Args:
            vector_db: FaceVectorDB instance for database operations.
            collection_name: Name of the collection to search.
        """
        self.vector_db = vector_db
        self.collection_name = collection_name
        
        logger.info(f"Initialized TextSearch for collection '{collection_name}'")

    # ----------------------------------------------------------------------------------------
    #  Search by Filename
    # ----------------------------------------------------------------------------------------
    @require(lambda filename: isinstance(filename, str) and len(filename.strip()) > 0,
             "Filename must be a non-empty string")
    def search_by_filename(self, filename: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for faces by filename.
        
        Args:
            filename: Filename to search for (supports partial matching).
            limit: Maximum number of results to return.
            
        Returns:
            List of search results with metadata.
        """
        try:
            # Get all points and filter by filename
            # Note: This is a simple implementation. For large datasets, 
            # you might want to use Qdrant's filtering capabilities
            results = self._search_by_metadata_field("filename", filename, limit)
            
            logger.info(f"Filename search for '{filename}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching by filename: {str(e)}")
            raise ValueError(f"Filename search failed: {str(e)}")

    # ----------------------------------------------------------------------------------------
    #  Search by Source Image
    # ----------------------------------------------------------------------------------------
    @require(lambda source_image: isinstance(source_image, str) and len(source_image.strip()) > 0,
             "Source image must be a non-empty string")
    def search_by_source_image(self, source_image: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for faces by source image name.
        
        Args:
            source_image: Source image name to search for.
            limit: Maximum number of results to return.
            
        Returns:
            List of search results with metadata.
        """
        try:
            results = self._search_by_metadata_field("source_image", source_image, limit)
            
            logger.info(f"Source image search for '{source_image}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching by source image: {str(e)}")
            raise ValueError(f"Source image search failed: {str(e)}")

    # ----------------------------------------------------------------------------------------
    #  Search by Custom Metadata
    # ----------------------------------------------------------------------------------------
    @require(lambda field_name: isinstance(field_name, str) and len(field_name.strip()) > 0,
             "Field name must be a non-empty string")
    @require(lambda field_value: isinstance(field_value, str) and len(field_value.strip()) > 0,
             "Field value must be a non-empty string")
    def search_by_metadata(self, field_name: str, field_value: str, 
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Search for faces by custom metadata field.
        
        Args:
            field_name: Name of the metadata field to search.
            field_value: Value to search for in the field.
            limit: Maximum number of results to return.
            
        Returns:
            List of search results with metadata.
        """
        try:
            results = self._search_by_metadata_field(field_name, field_value, limit)
            
            logger.info(f"Metadata search for '{field_name}={field_value}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {str(e)}")
            raise ValueError(f"Metadata search failed: {str(e)}")

    # ----------------------------------------------------------------------------------------
    #  Get All Faces
    # ----------------------------------------------------------------------------------------
    @require(lambda limit: isinstance(limit, int) and limit > 0,
             "Limit must be a positive integer")
    def get_all_faces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all faces in the database.
        
        Args:
            limit: Maximum number of results to return.
            
        Returns:
            List of all faces with metadata.
        """
        try:
            if not self.vector_db.collection_exists(self.collection_name):
                logger.warning(f"Collection '{self.collection_name}' does not exist")
                return []
            
            # Get collection info to determine vector size
            collection_info = self.vector_db.get_collection_info(self.collection_name)
            vector_size = collection_info.config.params.vectors.size
            
            # Create a dummy query vector (all zeros) to get all results
            dummy_vector = [0.0] * vector_size
            
            # Search with very low threshold to get all results
            results = self.vector_db.search_points(
                collection_name=self.collection_name,
                query_vector=dummy_vector,
                limit=limit,
                score_threshold=0.0
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.payload,
                    "filename": result.payload.get("filename", "unknown"),
                    "source_image": result.payload.get("source_image", "unknown")
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Retrieved {len(formatted_results)} faces from database")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error getting all faces: {str(e)}")
            raise ValueError(f"Failed to get all faces: {str(e)}")

    # ----------------------------------------------------------------------------------------
    #  Helper Methods (Private)
    # ----------------------------------------------------------------------------------------
    def _search_by_metadata_field(self, field_name: str, field_value: str, 
                                 limit: int) -> List[Dict[str, Any]]:
        """Helper method to search by metadata field."""
        try:
            # Get all faces first (this is a simple implementation)
            all_faces = self.get_all_faces(limit=1000)  # Get more to filter
            
            # Filter by metadata field
            filtered_results = []
            for face in all_faces:
                if field_name in face["metadata"]:
                    metadata_value = str(face["metadata"][field_name])
                    if field_value.lower() in metadata_value.lower():
                        filtered_results.append(face)
                        
                        if len(filtered_results) >= limit:
                            break
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in metadata field search: {str(e)}")
            raise ValueError(f"Metadata field search failed: {str(e)}")


#============================================================================================
#  Factory Function
#============================================================================================
def create_text_search(vector_db: FaceVectorDB, 
                      collection_name: Optional[str] = None) -> TextSearch:
    """Factory function to create a TextSearch instance.
    
    Args:
        vector_db: FaceVectorDB instance.
        collection_name: Name of the collection. If None, uses config.
        
    Returns:
        Configured TextSearch instance.
    """
    if collection_name is None:
        collection_name = config["vector_db"]["face_embeddings"]["collection_name"]
    
    return TextSearch(vector_db, collection_name)





