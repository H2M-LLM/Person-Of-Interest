# =============================================================================
#  Filename: app.py
#
#  Short Description: Main application orchestrator for Person of Interest face processing pipeline.
#
#  Creation date: 2025-01-06
#  Author: SupportVectors AI Training Team
# =============================================================================

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from icontract import require, ensure
from loguru import logger
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.panel import Panel

from poi import config
from poi.preprocessing.face_detector import create_face_detector
from poi.encoding.face_encoder import create_face_encoder
from poi.vector_db.face_vector_db import create_face_vector_db
from poi.search.face_search import create_face_search
from poi.search.text_search import create_text_search
from poi.search.enhanced_text_search import create_enhanced_text_search


#============================================================================================
#  Class: PersonOfInterestApp
#============================================================================================
class PersonOfInterestApp:
    """Main application orchestrator for Person of Interest face processing pipeline.
    
    This class coordinates the entire workflow from preprocessing images to
    enabling semantic search, following the patterns from rag_to_riches.
    """
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    @require(lambda: "dataset" in config, "Config must contain dataset section")
    @require(lambda: "vector_db" in config, "Config must contain vector_db section")
    def __init__(self) -> None:
        """Initialize the Person of Interest application.
        
        Raises:
            ValueError: If required configuration is missing.
        """
        self.config = config
        self.console = Console()
        
        # Initialize components
        self.face_detector = None
        self.face_encoder = None
        self.vector_db = None
        self.face_search = None
        self.text_search = None
        self.enhanced_text_search = None
        
        # Configuration
        self.dataset_config = config["dataset"]
        self.vector_db_config = config["vector_db"]
        
        logger.info("Initialized PersonOfInterestApp")
    
    # ----------------------------------------------------------------------------------------
    #  Initialize Components
    # ----------------------------------------------------------------------------------------
    def initialize_components(self) -> None:
        """Initialize all application components."""
        try:
            self.console.print("[bold blue]Initializing Person of Interest Application...[/bold blue]")
            
            # Initialize face detector
            self.console.print("🔍 Initializing face detector...")
            self.face_detector = create_face_detector()
            
            # Initialize face encoder
            self.console.print("🧠 Initializing face encoder...")
            self.face_encoder = create_face_encoder()
            
            # Initialize vector database
            self.console.print("🗄️  Initializing vector database...")
            self.vector_db = create_face_vector_db()
            
            # Initialize face search
            self.console.print("🔎 Initializing face search...")
            self.face_search = create_face_search(
                self.face_encoder, 
                self.vector_db
            )
            
            # Initialize text search
            self.console.print("📝 Initializing text search...")
            self.text_search = create_text_search(self.vector_db)
            
            # Initialize enhanced text search
            self.console.print("🔤 Initializing enhanced text search...")
            self.enhanced_text_search = create_enhanced_text_search(self.vector_db)
            
            self.console.print("[bold green]✅ All components initialized successfully![/bold green]")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise ValueError(f"Failed to initialize components: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Preprocess Dataset
    # ----------------------------------------------------------------------------------------
    def preprocess_dataset(self, input_path: Optional[str] = None, 
                          output_path: Optional[str] = None) -> Dict[str, Any]:
        """Preprocess the dataset by detecting and cropping faces.
        
        Args:
            input_path: Path to input images. If None, uses config.
            output_path: Path to save processed faces. If None, uses config.
            
        Returns:
            Dictionary with preprocessing statistics.
        """
        if input_path is None:
            input_path = self.dataset_config["input_path"]
        if output_path is None:
            output_path = self.dataset_config["processed_path"]
        
        # Validate paths after config defaults are applied
        if not isinstance(input_path, (str, Path)):
            raise ValueError(f"Input path must be string or Path, got {type(input_path)}")
        if not isinstance(output_path, (str, Path)):
            raise ValueError(f"Output path must be string or Path, got {type(output_path)}")
        
        try:
            self.console.print(f"[bold blue]Preprocessing dataset from {input_path}...[/bold blue]")
            
            # Ensure face detector is initialized
            if self.face_detector is None:
                self.face_detector = create_face_detector()
            
            # Process dataset
            stats = self.face_detector.process_dataset(
                input_dir=input_path,
                output_dir=output_path,
                supported_formats=self.dataset_config["supported_formats"]
            )
            
            # Display results
            self._display_preprocessing_results(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            raise ValueError(f"Dataset preprocessing failed: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Encode and Store Faces
    # ----------------------------------------------------------------------------------------
    def encode_and_store_faces(self, processed_path: Optional[str] = None) -> Dict[str, Any]:
        """Encode processed face images and store them in the vector database.
        
        Args:
            processed_path: Path to processed face images. If None, uses config.
            
        Returns:
            Dictionary with encoding and storage statistics.
        """
        if processed_path is None:
            processed_path = self.dataset_config["processed_path"]
        
        # Validate path after config defaults are applied
        if not isinstance(processed_path, (str, Path)):
            raise ValueError(f"Processed path must be string or Path, got {type(processed_path)}")
        
        try:
            self.console.print(f"[bold blue]Encoding and storing faces from {processed_path}...[/bold blue]")
            
            # Ensure components are initialized
            if self.face_encoder is None:
                self.face_encoder = create_face_encoder()
            if self.vector_db is None:
                self.vector_db = create_face_vector_db()
            if self.face_search is None:
                self.face_search = create_face_search(self.face_encoder, self.vector_db)
            
            # Find all processed face images
            processed_path_obj = Path(processed_path)
            if not processed_path_obj.exists():
                raise ValueError(f"Processed path does not exist: {processed_path}")
            
            face_images = list(processed_path_obj.glob("*.jpg")) + list(processed_path_obj.glob("*.jpeg"))
            
            if not face_images:
                raise ValueError(f"No face images found in {processed_path}")
            
            self.console.print(f"Found {len(face_images)} face images to encode")
            
            # Process images in batches
            batch_size = self.config["image_encoding"]["batch_size"]
            encoded_count = 0
            errors = []
            
            with Progress() as progress:
                task = progress.add_task("Encoding faces...", total=len(face_images))
                
                for i in range(0, len(face_images), batch_size):
                    batch = face_images[i:i + batch_size]
                    
                    try:
                        # Load images
                        images = []
                        metadata_list = []
                        
                        for img_path in batch:
                            try:
                                image = Image.open(img_path)
                                images.append(image)
                                
                                # Create metadata
                                metadata = {
                                    "filename": img_path.name,
                                    "filepath": str(img_path),
                                    "source_image": img_path.stem.split('_face')[0] if '_face' in img_path.stem else img_path.stem
                                }
                                metadata_list.append(metadata)
                                
                            except Exception as e:
                                error_msg = f"Failed to load {img_path}: {str(e)}"
                                logger.error(error_msg)
                                errors.append(error_msg)
                        
                        # Encode and store batch
                        if images:
                            self.face_search.index_all_faces(images, metadata_list)
                            encoded_count += len(images)
                        
                        progress.update(task, advance=len(batch))
                        
                    except Exception as e:
                        error_msg = f"Failed to process batch {i//batch_size + 1}: {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        progress.update(task, advance=len(batch))
            
            # Compile statistics
            stats = {
                "total_images": len(face_images),
                "encoded_images": encoded_count,
                "errors": errors,
                "success_rate": encoded_count / len(face_images) if face_images else 0
            }
            
            # Display results
            self._display_encoding_results(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error encoding and storing faces: {str(e)}")
            raise ValueError(f"Face encoding and storage failed: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Search Similar Faces
    # ----------------------------------------------------------------------------------------
    @require(lambda query_image: isinstance(query_image, (str, Path, Image.Image)), 
             "Query image must be string, Path, or PIL Image")
    def search_similar_faces(self, query_image, limit: int = 10, 
                           score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search for similar faces using a query image.
        
        Args:
            query_image: Path to query image or PIL Image.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold.
            
        Returns:
            List of search results with metadata.
        """
        try:
            self.console.print("[bold blue]Searching for similar faces...[/bold blue]")
            
            # Ensure components are initialized
            if self.face_search is None:
                if self.face_encoder is None:
                    self.face_encoder = create_face_encoder()
                if self.vector_db is None:
                    self.vector_db = create_face_vector_db()
                self.face_search = create_face_search(self.face_encoder, self.vector_db)
            
            # Load query image if it's a path
            if isinstance(query_image, (str, Path)):
                query_image = Image.open(query_image)
            
            # Perform search
            results = self.face_search.search_with_face(
                query_image, 
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
                    "source_image": result.payload.get("source_image", "unknown")
                }
                formatted_results.append(formatted_result)
            
            # Display results
            self._display_search_results(formatted_results)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching for similar faces: {str(e)}")
            raise ValueError(f"Face search failed: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Search by Text Query
    # ----------------------------------------------------------------------------------------
    @require(lambda query: isinstance(query, str) and len(query.strip()) > 0,
             "Query must be a non-empty string")
    def search_by_text(self, query: str, search_type: str = "filename", 
                      limit: int = 10) -> List[Dict[str, Any]]:
        """Search for faces using text queries.
        
        Args:
            query: Text query to search for.
            search_type: Type of search ('filename', 'source_image', 'all').
            limit: Maximum number of results to return.
            
        Returns:
            List of search results with metadata.
        """
        try:
            self.console.print(f"[bold blue]Searching for faces with query: '{query}'...[/bold blue]")
            
            # Ensure components are initialized
            if self.text_search is None:
                if self.vector_db is None:
                    self.vector_db = create_face_vector_db()
                self.text_search = create_text_search(self.vector_db)
            
            # Perform search based on type
            if search_type == "filename":
                results = self.text_search.search_by_filename(query, limit)
            elif search_type == "source_image":
                results = self.text_search.search_by_source_image(query, limit)
            elif search_type == "all":
                results = self.text_search.get_all_faces(limit)
            else:
                raise ValueError(f"Invalid search type: {search_type}")
            
            # Display results
            self._display_text_search_results(results, query, search_type)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching by text: {str(e)}")
            raise ValueError(f"Text search failed: {str(e)}")

    # ----------------------------------------------------------------------------------------
    #  Enhanced Text Search (Semantic)
    # ----------------------------------------------------------------------------------------
    @require(lambda query: isinstance(query, str) and len(query.strip()) > 0,
             "Query must be a non-empty string")
    def search_by_semantic_text(self, query: str, limit: int = 10,
                              score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search for faces using semantic text queries (same as rag_to_riches).
        
        Args:
            query: Text query to search for.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold.
            
        Returns:
            List of search results with metadata.
        """
        try:
            self.console.print(f"[bold blue]Semantic text search for: '{query}'...[/bold blue]")
            
            # Ensure components are initialized
            if self.enhanced_text_search is None:
                if self.vector_db is None:
                    self.vector_db = create_face_vector_db()
                self.enhanced_text_search = create_enhanced_text_search(self.vector_db)
            
            # Perform semantic search
            results = self.enhanced_text_search.search_by_text_query(
                query, limit, score_threshold
            )
            
            # Display results
            self._display_semantic_search_results(results, query)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic text search: {str(e)}")
            raise ValueError(f"Semantic text search failed: {str(e)}")

    # ----------------------------------------------------------------------------------------
    #  Add Text Description to Face
    # ----------------------------------------------------------------------------------------
    @require(lambda face_id: isinstance(face_id, str) and len(face_id.strip()) > 0,
             "Face ID must be a non-empty string")
    @require(lambda description: isinstance(description, str) and len(description.strip()) > 0,
             "Description must be a non-empty string")
    def add_face_description(self, face_id: str, description: str) -> str:
        """Add a text description to a face for enhanced search.
        
        Args:
            face_id: ID of the face to add description to.
            description: Text description of the face.
            
        Returns:
            ID of the created text embedding.
        """
        try:
            self.console.print(f"[bold blue]Adding description to face {face_id}...[/bold blue]")
            
            # Ensure components are initialized
            if self.enhanced_text_search is None:
                if self.vector_db is None:
                    self.vector_db = create_face_vector_db()
                self.enhanced_text_search = create_enhanced_text_search(self.vector_db)
            
            # Add description
            text_id = self.enhanced_text_search.add_text_description(face_id, description)
            
            self.console.print(f"[green]✅ Added description: '{description}'[/green]")
            return text_id
            
        except Exception as e:
            logger.error(f"Error adding face description: {str(e)}")
            raise ValueError(f"Failed to add face description: {str(e)}")

    # ----------------------------------------------------------------------------------------
    #  Search Text Descriptions
    # ----------------------------------------------------------------------------------------
    @require(lambda query: isinstance(query, str) and len(query.strip()) > 0,
             "Query must be a non-empty string")
    def search_face_descriptions(self, query: str, limit: int = 10,
                               score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search through text descriptions of faces.
        
        Args:
            query: Text query to search for.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold.
            
        Returns:
            List of search results with metadata.
        """
        try:
            self.console.print(f"[bold blue]Searching face descriptions for: '{query}'...[/bold blue]")
            
            # Ensure components are initialized
            if self.enhanced_text_search is None:
                if self.vector_db is None:
                    self.vector_db = create_face_vector_db()
                self.enhanced_text_search = create_enhanced_text_search(self.vector_db)
            
            # Search descriptions
            results = self.enhanced_text_search.search_text_descriptions(
                query, limit, score_threshold
            )
            
            # Display results
            self._display_description_search_results(results, query)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching face descriptions: {str(e)}")
            raise ValueError(f"Face description search failed: {str(e)}")

    # ----------------------------------------------------------------------------------------
    #  Get Database Statistics
    # ----------------------------------------------------------------------------------------
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database.
        
        Returns:
            Dictionary containing database statistics.
        """
        try:
            if self.face_search is None:
                if self.face_encoder is None:
                    self.face_encoder = create_face_encoder()
                if self.vector_db is None:
                    self.vector_db = create_face_vector_db()
                self.face_search = create_face_search(self.face_encoder, self.vector_db)
            
            stats = self.face_search.get_collection_stats()
            self._display_database_stats(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            raise ValueError(f"Failed to get database stats: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Run Complete Pipeline
    # ----------------------------------------------------------------------------------------
    def run_complete_pipeline(self, input_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete pipeline: preprocessing, encoding, and storage.
        
        Args:
            input_path: Path to input images. If None, uses config.
            
        Returns:
            Dictionary with complete pipeline statistics.
        """
        try:
            self.console.print("[bold green]🚀 Starting Complete Person of Interest Pipeline[/bold green]")
            
            # Step 1: Preprocess dataset
            preprocessing_stats = self.preprocess_dataset(input_path)
            
            # Step 2: Encode and store faces
            encoding_stats = self.encode_and_store_faces()
            
            # Step 3: Get final database stats
            db_stats = self.get_database_stats()
            
            # Compile complete statistics
            complete_stats = {
                "preprocessing": preprocessing_stats,
                "encoding": encoding_stats,
                "database": db_stats,
                "pipeline_completed": True
            }
            
            self.console.print("[bold green]✅ Complete pipeline finished successfully![/bold green]")
            return complete_stats
            
        except Exception as e:
            logger.error(f"Error in complete pipeline: {str(e)}")
            raise ValueError(f"Complete pipeline failed: {str(e)}")
    
    # ----------------------------------------------------------------------------------------
    #  Display Methods
    # ----------------------------------------------------------------------------------------
    def _display_preprocessing_results(self, stats: Dict[str, Any]) -> None:
        """Display preprocessing results in a formatted table."""
        table = Table(title="Preprocessing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Images", str(stats["total_images"]))
        table.add_row("Processed Images", str(stats["processed_images"]))
        table.add_row("Faces Detected", str(stats["total_faces_detected"]))
        table.add_row("Success Rate", f"{stats['success_rate']:.2%}")
        table.add_row("Errors", str(len(stats["errors"])))
        
        self.console.print(table)
    
    def _display_encoding_results(self, stats: Dict[str, Any]) -> None:
        """Display encoding results in a formatted table."""
        table = Table(title="Encoding Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Images", str(stats["total_images"]))
        table.add_row("Encoded Images", str(stats["encoded_images"]))
        table.add_row("Success Rate", f"{stats['success_rate']:.2%}")
        table.add_row("Errors", str(len(stats["errors"])))
        
        self.console.print(table)
    
    def _display_search_results(self, results: List[Dict[str, Any]]) -> None:
        """Display search results in a formatted table."""
        if not results:
            self.console.print("[yellow]No similar faces found.[/yellow]")
            return
        
        table = Table(title="Similar Faces Found")
        table.add_column("Rank", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Filename", style="magenta")
        table.add_column("Source Image", style="blue")
        
        for i, result in enumerate(results, 1):
            table.add_row(
                str(i),
                f"{result['score']:.4f}",
                result["filename"],
                result["source_image"]
            )
        
        self.console.print(table)
    
    def _display_text_search_results(self, results: List[Dict[str, Any]], 
                                   query: str, search_type: str) -> None:
        """Display text search results in a formatted table."""
        if not results:
            self.console.print(f"[yellow]No faces found for query '{query}' with search type '{search_type}'.[/yellow]")
            return
        
        table = Table(title=f"Text Search Results: '{query}' ({search_type})")
        table.add_column("Rank", style="cyan")
        table.add_column("ID", style="blue")
        table.add_column("Filename", style="magenta")
        table.add_column("Source Image", style="green")
        table.add_column("Image Size", style="yellow")
        
        for i, result in enumerate(results, 1):
            image_size = result["metadata"].get("image_size", "unknown")
            table.add_row(
                str(i),
                result["id"][:8] + "...",  # Truncate ID for display
                result["filename"],
                result["source_image"],
                str(image_size)
            )
        
        self.console.print(table)
    
    def _display_semantic_search_results(self, results: List[Dict[str, Any]], query: str) -> None:
        """Display semantic search results in a formatted table."""
        if not results:
            self.console.print(f"[yellow]No faces found for semantic query '{query}'.[/yellow]")
            return
        
        table = Table(title=f"Semantic Text Search Results: '{query}'")
        table.add_column("Rank", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("ID", style="blue")
        table.add_column("Filename", style="magenta")
        table.add_column("Source Image", style="yellow")
        
        for i, result in enumerate(results, 1):
            table.add_row(
                str(i),
                f"{result['score']:.4f}",
                result["id"][:8] + "...",  # Truncate ID for display
                result["filename"],
                result["source_image"]
            )
        
        self.console.print(table)
    
    def _display_description_search_results(self, results: List[Dict[str, Any]], query: str) -> None:
        """Display description search results in a formatted table."""
        if not results:
            self.console.print(f"[yellow]No face descriptions found for query '{query}'.[/yellow]")
            return
        
        table = Table(title=f"Face Description Search Results: '{query}'")
        table.add_column("Rank", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Description", style="blue")
        table.add_column("Face ID", style="magenta")
        table.add_column("Face Filename", style="yellow")
        
        for i, result in enumerate(results, 1):
            description = result["description"]
            if len(description) > 50:
                description = description[:47] + "..."
            
            table.add_row(
                str(i),
                f"{result['score']:.4f}",
                description,
                result["face_id"][:8] + "...",  # Truncate ID for display
                result["face_filename"]
            )
        
        self.console.print(table)
    
    def _display_database_stats(self, stats: Dict[str, Any]) -> None:
        """Display database statistics in a formatted panel."""
        if "error" in stats:
            self.console.print(f"[red]Error getting database stats: {stats['error']}[/red]")
            return
        
        if not stats.get("exists", False):
            self.console.print("[yellow]Database collection does not exist.[/yellow]")
            return
        
        content = f"""
Collection: {stats['name']}
Points: {stats['point_count']}
Vector Size: {stats['vector_size']}
Distance Metric: {stats['distance_metric']}
Status: {stats['status']}
        """.strip()
        
        panel = Panel(content, title="Database Statistics", border_style="green")
        self.console.print(panel)


#============================================================================================
#  Factory Function
#============================================================================================
def create_app() -> PersonOfInterestApp:
    """Factory function to create a PersonOfInterestApp instance.
    
    Returns:
        Configured PersonOfInterestApp instance.
    """
    return PersonOfInterestApp()
