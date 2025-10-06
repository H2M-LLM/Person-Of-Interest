#!/usr/bin/env python3
"""
Example usage of the Person of Interest application.

This script demonstrates how to use the Person of Interest application
to process face images and perform semantic search.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from poi import create_app, config
from rich.console import Console

console = Console()

def main():
    """Main example function."""
    console.print("[bold green]Person of Interest - Example Usage[/bold green]")
    console.print("=" * 50)
    
    try:
        # Create application instance
        console.print("Creating application instance...")
        app = create_app()
        
        # Display configuration
        console.print("\n[bold blue]Application Configuration:[/bold blue]")
        console.print(f"Input dataset path: {config['dataset']['input_path']}")
        console.print(f"Processed faces path: {config['dataset']['processed_path']}")
        console.print(f"Vector database path: {config['vector_db']['path']}")
        console.print(f"Face encoder model: {config['image_encoding']['model']}")
        
        # Initialize components
        console.print("\n[bold blue]Initializing components...[/bold blue]")
        app.initialize_components()
        
        # Check if dataset exists
        input_path = Path(config['dataset']['input_path'])
        if not input_path.exists():
            console.print(f"[yellow]⚠️  Dataset path does not exist: {input_path}[/yellow]")
            console.print("Please ensure the CelebA dataset is available at the configured path.")
            return
        
        # Run complete pipeline
        console.print("\n[bold blue]Running complete pipeline...[/bold blue]")
        console.print("This will:")
        console.print("1. Detect and crop faces from images")
        console.print("2. Encode faces using the configured model")
        console.print("3. Store embeddings in the vector database")
        
        # Uncomment the following line to run the complete pipeline
        # stats = app.run_complete_pipeline()
        
        # For demo purposes, just show what would happen
        console.print("\n[bold green]✅ Pipeline would process the dataset and create searchable embeddings![/bold green]")
        
        # Example search (if database exists)
        console.print("\n[bold blue]Example search functionality:[/bold blue]")
        console.print("To search for similar faces, you would use:")
        console.print("results = app.search_similar_faces('path/to/query_face.jpg')")
        
        # Display database stats
        console.print("\n[bold blue]Database statistics:[/bold blue]")
        app.get_database_stats()
        
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        return 1
    
    console.print("\n[bold green]Example completed successfully![/bold green]")
    return 0

if __name__ == "__main__":
    sys.exit(main())
