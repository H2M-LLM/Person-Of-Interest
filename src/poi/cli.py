# =============================================================================
#  Filename: cli.py
#
#  Short Description: Command-line interface for Person of Interest application.
#
#  Creation date: 2025-01-06
#  Author: SupportVectors AI Training Team
# =============================================================================

import sys
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from poi.app import create_app
from poi import config


#============================================================================================
#  CLI Application
#============================================================================================
console = Console()

@click.group()
@click.version_option(version=config["app"]["version"])
def cli():
    """Person of Interest - Face Image Processing and Semantic Search Application.
    
    This application processes face images from the CelebA dataset, extracts face embeddings,
    and enables semantic search through a vector database using RetinaFace and Qdrant.
    """
    pass


@cli.command()
@click.option('--input-path', '-i', 
              help='Path to input images directory (default: from config)')
@click.option('--output-path', '-o',
              help='Path to save processed faces (default: from config)')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
def preprocess(input_path: Optional[str], output_path: Optional[str], verbose: bool):
    """Preprocess images: detect and crop faces using RetinaFace."""
    try:
        console.print("[bold blue]🔍 Starting Face Preprocessing...[/bold blue]")
        
        app = create_app()
        stats = app.preprocess_dataset(input_path, output_path)
        
        if verbose:
            console.print(f"[green]✅ Preprocessing completed successfully![/green]")
            console.print(f"Processed {stats['processed_images']} images with {stats['total_faces_detected']} faces detected")
        
    except Exception as e:
        console.print(f"[red]❌ Error during preprocessing: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--processed-path', '-p',
              help='Path to processed face images (default: from config)')
@click.option('--limit', '-l', type=int,
              help='Limit number of images to encode (optional, encodes all if not specified)')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
def encode(processed_path: Optional[str], limit: Optional[int], verbose: bool):
    """Encode processed face images and store in vector database."""
    try:
        console.print("[bold blue]🧠 Starting Face Encoding and Storage...[/bold blue]")
        if limit:
            console.print(f"[dim]Limiting to {limit} images[/dim]")
        
        app = create_app()
        stats = app.encode_and_store_faces(processed_path, limit=limit)
        
        if verbose:
            console.print(f"[green]✅ Encoding completed successfully![/green]")
            console.print(f"Encoded and stored {stats['encoded_images']} face images")
        
    except Exception as e:
        console.print(f"[red]❌ Error during encoding: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('query_image', type=click.Path(exists=True))
@click.option('--limit', '-l', default=10,
              help='Maximum number of results to return (default: 10)')
@click.option('--threshold', '-t', type=float,
              help='Minimum similarity score threshold')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
def search_image(query_image: str, limit: int, threshold: Optional[float], verbose: bool):
    """Search for similar faces using a query image."""
    try:
        console.print(f"[bold blue]🔎 Searching for similar faces using {query_image}...[/bold blue]")
        
        app = create_app()
        results = app.search_similar_faces(query_image, limit, threshold)
        
        if verbose:
            console.print(f"[green]✅ Search completed! Found {len(results)} similar faces[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error during search: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('query', type=str)
@click.option('--search-type', '-t', default='filename',
              type=click.Choice(['filename', 'source_image', 'all']),
              help='Type of text search to perform (default: filename)')
@click.option('--limit', '-l', default=10,
              help='Maximum number of results to return (default: 10)')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
def search_text(query: str, search_type: str, limit: int, verbose: bool):
    """Search for faces using text queries (metadata search)."""
    try:
        console.print(f"[bold blue]📝 Searching for faces with text query: '{query}'...[/bold blue]")
        
        app = create_app()
        results = app.search_by_text(query, search_type, limit)
        
        if verbose:
            console.print(f"[green]✅ Text search completed! Found {len(results)} faces[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error during text search: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('query', type=str)
@click.option('--limit', '-l', default=10,
              help='Maximum number of results to return (default: 10)')
@click.option('--threshold', '-t', type=float,
              help='Minimum similarity score threshold')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
def search_semantic(query: str, limit: int, threshold: Optional[float], verbose: bool):
    """Search for faces using semantic text queries (same as rag_to_riches)."""
    try:
        console.print(f"[bold blue]🔤 Semantic text search for: '{query}'...[/bold blue]")
        
        app = create_app()
        results = app.search_by_semantic_text(query, limit, threshold)
        
        if verbose:
            console.print(f"[green]✅ Semantic search completed! Found {len(results)} faces[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error during semantic search: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--input-path', '-i',
              help='Path to input images directory (default: from config)')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
def pipeline(input_path: Optional[str], verbose: bool):
    """Run the complete pipeline: preprocessing, encoding, and storage."""
    try:
        console.print("[bold green]🚀 Starting Complete Pipeline...[/bold green]")
        
        app = create_app()
        stats = app.run_complete_pipeline(input_path)
        
        if verbose:
            console.print(f"[green]✅ Complete pipeline finished successfully![/green]")
            console.print(f"Database now contains {stats['database']['point_count']} face embeddings")
        
    except Exception as e:
        console.print(f"[red]❌ Error during pipeline: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
def stats():
    """Display database statistics."""
    try:
        console.print("[bold blue]📊 Getting Database Statistics...[/bold blue]")
        
        app = create_app()
        app.get_database_stats()
        
    except Exception as e:
        console.print(f"[red]❌ Error getting stats: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
def info():
    """Display application information and configuration."""
    try:
        import torch
        
        # Application info
        app_info = config["app"]
        dataset_info = config["dataset"]
        face_detection_info = config["face_detection"]
        image_encoding_info = config["image_encoding"]
        vector_db_info = config["vector_db"]
        
        # GPU/CUDA info
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device_count = torch.cuda.device_count()
            cuda_device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            gpu_info = f"""
[bold]GPU/CUDA Information[/bold]
CUDA Available: ✅ Yes
CUDA Version: {cuda_version}
GPU Device Count: {cuda_device_count}
GPU Name: {cuda_device_name}
Current Device: cuda:0
PyTorch Version: {torch.__version__}"""
        else:
            gpu_info = f"""
[bold]GPU/CUDA Information[/bold]
CUDA Available: ❌ No (using CPU)
PyTorch Version: {torch.__version__}"""
        
        info_text = f"""
[bold]Application Information[/bold]
Name: {app_info['name']}
Version: {app_info['version']}
Description: {app_info['description']}

[bold]Dataset Configuration[/bold]
Input Path: {dataset_info['input_path']}
Processed Path: {dataset_info['processed_path']}
Supported Formats: {', '.join(dataset_info['supported_formats'])}
Batch Size: {dataset_info['batch_size']}

[bold]Face Detection Configuration[/bold]
Model: {face_detection_info['model']}
Confidence Threshold: {face_detection_info['confidence_threshold']}
Min Face Size: {face_detection_info['min_face_size']}
Max Face Size: {face_detection_info['max_face_size']}

[bold]Image Encoding Configuration[/bold]
Model: {image_encoding_info['model']}
Model Name: {image_encoding_info.get('model_name', 'N/A')}
Input Size: {image_encoding_info['input_size']}
Embedding Dimension: {image_encoding_info['embedding_dim']}
Device: {image_encoding_info['device']}

[bold]Vector Database Configuration[/bold]
Type: {vector_db_info['type']}
Path: {vector_db_info['path']}
Collection: {vector_db_info['face_embeddings']['collection_name']}
Vector Size: {vector_db_info['face_embeddings']['vector_size']}
Distance Metric: {vector_db_info['face_embeddings']['distance']}
{gpu_info}
        """.strip()
        
        panel = Panel(info_text, title="Person of Interest Application Info", border_style="blue")
        console.print(panel)
        
    except Exception as e:
        console.print(f"[red]❌ Error displaying info: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
def init():
    """Initialize the application components and check configuration."""
    try:
        console.print("[bold blue]🔧 Initializing Application Components...[/bold blue]")
        
        app = create_app()
        app.initialize_components()
        
        console.print("[green]✅ Application initialization completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error during initialization: {str(e)}[/red]")
        sys.exit(1)


#============================================================================================
#  Main Entry Point
#============================================================================================
def main():
    """Main entry point for the CLI application."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
