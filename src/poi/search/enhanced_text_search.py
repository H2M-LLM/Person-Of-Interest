# =============================================================================
#  Filename: enhanced_text_search.py
#
#  Short Description: Enhanced text search using sentence transformers (same as rag_to_riches).
#
#  Creation date: 2025-01-06
#  Author: SupportVectors AI Training Team
# =============================================================================

from typing import List, Dict, Any, Optional
from icontract import require, ensure
from loguru import logger
from qdrant_client import models

from poi.vector_db.face_vector_db import FaceVectorDB
from poi.encoding.face_encoder import create_face_encoder
from poi import config


#============================================================================================
#  Class: EnhancedTextSearch
#============================================================================================
class EnhancedTextSearch:
    """Enhanced text search using sentence transformers (same as rag_to_riches).
    
    This class provides text-based search functionality that can:
    1. Search face metadata using text queries
    2. Create text embeddings for semantic search
    3. Find faces based on text descriptions
    """
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    @require(lambda vector_db: isinstance(vector_db, FaceVectorDB), 
             "Vector DB must be a FaceVectorDB instance")
    def __init__(self, vector_db: FaceVectorDB) -> None:
        """Initialize the enhanced text search system.
        
        Args:
            vector_db: FaceVectorDB instance for database operations.
        """
        self.vector_db = vector_db
        self.face_encoder = None
        
        # Collection name from config
        self.face_collection = config["vector_db"]["face_embeddings"]["collection_name"]
        
        logger.info("Initialized EnhancedTextSearch with SigLIP multimodal model")

    # ----------------------------------------------------------------------------------------
    #  Initialize Text Encoder
    # ----------------------------------------------------------------------------------------
    def _ensure_face_encoder(self) -> None:
        """Ensure face encoder (SigLIP multimodal) is initialized."""
        if self.face_encoder is None:
            self.face_encoder = create_face_encoder()

    # ----------------------------------------------------------------------------------------
    #  Search by Text Query (Semantic)
    # ----------------------------------------------------------------------------------------
    @require(lambda query: isinstance(query, str) and len(query.strip()) > 0,
             "Query must be a non-empty string")
    def search_by_text_query(self, query: str, limit: int = 10,
                           score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search for faces using semantic text queries.
        
        Args:
            query: Text query to search for.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold.
            
        Returns:
            List of search results with metadata.
        """
        try:
            self._ensure_face_encoder()
            
            # Create text embedding using SigLIP (multimodal model)
            query_embedding = self.face_encoder.encode_text(query)
            query_vector = query_embedding.tolist()
            
            # Print the number of points in the face collection before searching
            try:
                # Use get_collection_info if get_collection_stats is not available
                if hasattr(self.vector_db, "get_collection_info"):
                    info = self.vector_db.get_collection_info(self.face_collection)
                    num_points = info.get("point_count", "unknown")
                    print(f"[EnhancedTextSearch] Number of points in '{self.face_collection}': {num_points}")
                else:
                    print(f"[EnhancedTextSearch] Could not retrieve collection stats: 'FaceVectorDB' object has no attribute 'get_collection_stats'")
            except Exception as e:
                print(f"[EnhancedTextSearch] Could not retrieve collection stats: {e}")
            # Search directly in face_embeddings collection using multimodal SigLIP!
            results = self.vector_db.search_points(
                collection_name=self.face_collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.payload,
                    "filename": result.payload.get("filename", "unknown"),
                    "source_image": result.payload.get("source_image", "unknown"),
                    "search_type": "semantic_text_multimodal"
                }
                formatted_results.append(formatted_result)
            
            results = formatted_results
            
            logger.info(f"Semantic text search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic text search: {str(e)}")
            raise ValueError(f"Semantic text search failed: {str(e)}")

    # ----------------------------------------------------------------------------------------
    #  Search by Filename (Metadata)
    # ----------------------------------------------------------------------------------------
    @require(lambda filename: isinstance(filename, str) and len(filename.strip()) > 0,
             "Filename must be a non-empty string")
    def search_by_filename(self, filename: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for faces by filename (metadata search).
        
        Args:
            filename: Filename to search for (supports partial matching).
            limit: Maximum number of results to return.
            
        Returns:
            List of search results with metadata.
        """
        try:
            # Get all faces and filter by filename
            all_faces = self._get_all_faces(limit=1000)
            
            # Filter by filename
            filtered_results = []
            for face in all_faces:
                if "filename" in face["metadata"]:
                    face_filename = str(face["metadata"]["filename"])
                    if filename.lower() in face_filename.lower():
                        face["search_type"] = "filename_metadata"
                        filtered_results.append(face)
                        
                        if len(filtered_results) >= limit:
                            break
            
            logger.info(f"Filename search for '{filename}' returned {len(filtered_results)} results")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching by filename: {str(e)}")
            raise ValueError(f"Filename search failed: {str(e)}")

    # ----------------------------------------------------------------------------------------
    #  Search by Source Image (Metadata)
    # ----------------------------------------------------------------------------------------
    @require(lambda source_image: isinstance(source_image, str) and len(source_image.strip()) > 0,
             "Source image must be a non-empty string")
    def search_by_source_image(self, source_image: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for faces by source image name (metadata search).
        
        Args:
            source_image: Source image name to search for.
            limit: Maximum number of results to return.
            
        Returns:
            List of search results with metadata.
        """
        try:
            # Get all faces and filter by source image
            all_faces = self._get_all_faces(limit=1000)
            
            # Filter by source image
            filtered_results = []
            for face in all_faces:
                if "source_image" in face["metadata"]:
                    face_source = str(face["metadata"]["source_image"])
                    if source_image.lower() in face_source.lower():
                        face["search_type"] = "source_image_metadata"
                        filtered_results.append(face)
                        
                        if len(filtered_results) >= limit:
                            break
            
            logger.info(f"Source image search for '{source_image}' returned {len(filtered_results)} results")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching by source image: {str(e)}")
            raise ValueError(f"Source image search failed: {str(e)}")

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
            results = self._get_all_faces(limit)
            for result in results:
                result["search_type"] = "all_faces"
            
            logger.info(f"Retrieved {len(results)} faces from database")
            return results
            
        except Exception as e:
            logger.error(f"Error getting all faces: {str(e)}")
            raise ValueError(f"Failed to get all faces: {str(e)}")

    # ----------------------------------------------------------------------------------------
    #  Helper Methods (Private)
    # ----------------------------------------------------------------------------------------
    def _get_all_faces(self, limit: int) -> List[Dict[str, Any]]:
        """Helper method to get all faces from database."""
        try:
            if not self.vector_db.collection_exists(self.face_collection):
                logger.warning(f"Collection '{self.face_collection}' does not exist")
                return []
            
            # Get collection info to determine vector size
            collection_info = self.vector_db.get_collection_info(self.face_collection)
            vector_size = collection_info.config.params.vectors.size
            
            # Create a dummy query vector (all zeros) to get all results
            dummy_vector = [0.0] * vector_size
            
            # Search with very low threshold to get all results
            results = self.vector_db.search_points(
                collection_name=self.face_collection,
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
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error getting all faces: {str(e)}")
            raise ValueError(f"Failed to get all faces: {str(e)}")


#============================================================================================
#  Factory Function
#============================================================================================
def create_enhanced_text_search(vector_db: FaceVectorDB) -> EnhancedTextSearch:
    """Factory function to create an EnhancedTextSearch instance.
    
    Args:
        vector_db: FaceVectorDB instance.
        
    Returns:
        Configured EnhancedTextSearch instance.
    """
    return EnhancedTextSearch(vector_db)
