#!/usr/bin/env python3
"""
Complete example of the Person of Interest application.

This script demonstrates the complete workflow:
1. Read from dataset (CelebA images)
2. Encode faces using the configured model
3. Store embeddings in vector database
4. Provide text-based search functionality
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from poi import create_app, config
from rich.console import Console

console = Console()

def main():
    """Main example function demonstrating complete functionality."""
    console.print("[bold green]Person of Interest - Complete Example[/bold green]")
    console.print("=" * 60)
    
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
        
        # Check if dataset exists
        input_path = Path(config['dataset']['input_path'])
        if not input_path.exists():
            console.print(f"[yellow]⚠️  Dataset path does not exist: {input_path}[/yellow]")
            console.print("Please ensure the CelebA dataset is available at the configured path.")
            console.print("You can create a test dataset by copying some images to the dataset/img_align_celeba/ directory.")
            return 1
        
        # Count images in dataset
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))
        console.print(f"\n[bold green]Found {len(image_files)} images in dataset[/bold green]")
        
        if len(image_files) == 0:
            console.print("[yellow]No images found in dataset directory.[/yellow]")
            return 1
        
        # Run complete pipeline
        console.print("\n[bold blue]🚀 Running Complete Pipeline...[/bold blue]")
        console.print("This will:")
        console.print("1. Detect and crop faces from images")
        console.print("2. Encode faces using the configured model")
        console.print("3. Store embeddings in the vector database")
        
        # Uncomment the following line to run the complete pipeline
        # stats = app.run_complete_pipeline()
        
        # For demo purposes, let's show what the pipeline would do
        console.print("\n[bold green]✅ Pipeline would process the dataset and create searchable embeddings![/bold green]")
        
        # Example text search functionality
        console.print("\n[bold blue]📝 Text Search Functionality:[/bold blue]")
        console.print("Once the pipeline is run, you can search for faces using text queries:")
        console.print("")
        console.print("Examples:")
        console.print("  # Search by filename")
        console.print("  results = app.search_by_text('000001', 'filename')")
        console.print("")
        console.print("  # Search by source image")
        console.print("  results = app.search_by_text('000001', 'source_image')")
        console.print("")
        console.print("  # Get all faces")
        console.print("  results = app.search_by_text('', 'all')")
        
        # Display database stats (will show empty if not run)
        console.print("\n[bold blue]📊 Database Statistics:[/bold blue]")
        app.get_database_stats()
        
        # Show CLI usage
        console.print("\n[bold blue]💻 CLI Usage Examples:[/bold blue]")
        console.print("")
        console.print("# Run complete pipeline")
        console.print("python -m poi.cli pipeline")
        console.print("")
        console.print("# Search by text query")
        console.print("python -m poi.cli search-text '000001' --search-type filename")
        console.print("")
        console.print("# Search by source image")
        console.print("python -m poi.cli search-text '000001' --search-type source_image")
        console.print("")
        console.print("# Get all faces")
        console.print("python -m poi.cli search-text '' --search-type all")
        console.print("")
        console.print("# Get database statistics")
        console.print("python -m poi.cli stats")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        return 1
    
    console.print("\n[bold green]✅ Complete example finished successfully![/bold green]")
    console.print("\n[bold yellow]To run the actual pipeline, uncomment the line in the script:[/bold yellow]")
    console.print("[bold yellow]stats = app.run_complete_pipeline()[/bold yellow]")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
