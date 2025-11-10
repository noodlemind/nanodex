"""
RAG (Retrieval-Augmented Generation) CLI commands.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from pathlib import Path

from ..utils import Config
from ..analyzers import CodeAnalyzer
from ..rag import SemanticRetriever

console = Console()


@click.group()
def rag_cmd():
    """
    🔍 RAG (Retrieval-Augmented Generation) commands

    Build vector index and perform semantic search on your codebase.

    \b
    Steps:
    1. turbo-code-gpt rag index     # Index codebase
    2. turbo-code-gpt rag search    # Search for code
    3. turbo-code-gpt rag query     # Ask questions
    """
    pass


@rag_cmd.command('index')
@click.option('--config', default='config.yaml', help='Configuration file path')
@click.option('--output', default='./models/rag_index', help='Output path for index')
@click.option('--embedding-model', default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model')
@click.option('--chunk-strategy', default='hybrid', type=click.Choice(['function', 'class', 'file', 'hybrid']), help='Chunking strategy')
def index_cmd(config, output, embedding_model, chunk_strategy):
    """
    Build RAG index from codebase.

    This command:
    1. Analyzes your codebase
    2. Chunks code intelligently (by function/class)
    3. Embeds chunks into vectors
    4. Builds FAISS index for fast search
    5. Saves index to disk

    \b
    Example:
        turbo-code-gpt rag index
        turbo-code-gpt rag index --embedding-model sentence-transformers/all-MiniLM-L6-v2
    """
    try:
        console.print("\n[bold cyan]Building RAG Index...[/bold cyan]\n")

        # Load configuration
        cfg = Config(config)
        repo_config = cfg.get_repository_config()

        # Analyze codebase
        console.print("Step 1: Analyzing codebase...")
        analyzer = CodeAnalyzer(repo_config)
        code_samples = analyzer.analyze()

        stats = analyzer.get_statistics(code_samples)
        console.print(f"  Found {stats['total_files']} files with {stats['total_lines']:,} lines\n")

        if stats['total_files'] == 0:
            console.print("[yellow]No code files found![/yellow]")
            return

        # Initialize retriever
        console.print("Step 2: Initializing retriever...")
        console.print(f"  Embedding model: [cyan]{embedding_model}[/cyan]")
        console.print(f"  Chunk strategy: [cyan]{chunk_strategy}[/cyan]\n")

        from ..rag import CodeEmbedder, CodeChunker, VectorIndexer, SemanticRetriever

        retriever = SemanticRetriever(
            embedding_model=embedding_model,
            chunk_strategy=chunk_strategy
        )

        # Index codebase
        console.print("Step 3: Indexing codebase...")
        console.print("[dim]This may take a few minutes...[/dim]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Indexing...", total=None)

            index_stats = retriever.index_codebase(code_samples, show_progress=True)

            progress.update(task, completed=True)

        # Save index
        console.print(f"\nStep 4: Saving index to {output}...")
        retriever.save(output)

        # Display results
        console.print("\n")
        table = Table(title="Indexing Results", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Code Samples", str(index_stats['num_samples']))
        table.add_row("Chunks Created", str(index_stats['num_chunks']))
        table.add_row("Chunk Types", str(len(index_stats['chunk_stats'].get('types', {}))))
        table.add_row("Avg Chunk Size", f"{index_stats['chunk_stats'].get('avg_chunk_size', 0):.0f} chars")

        console.print(table)

        # Success
        console.print("\n")
        console.print(Panel.fit(
            f"[bold green]✓ RAG Index Built Successfully![/bold green]\n\n"
            f"Index saved to: [cyan]{output}[/cyan]\n"
            f"Chunks indexed: [cyan]{index_stats['num_chunks']}[/cyan]\n\n"
            "[bold]Next steps:[/bold]\n"
            f"  turbo-code-gpt rag search \"your query\" --index {output}\n"
            f"  turbo-code-gpt rag query \"your question\" --index {output}",
            title="Success",
            border_style="green"
        ))

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        console.print(f"\n[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


@rag_cmd.command('search')
@click.argument('query')
@click.option('--index', default='./models/rag_index', help='Path to RAG index')
@click.option('-k', '--top-k', default=5, help='Number of results')
@click.option('--type', help='Filter by type (function, class, file)')
@click.option('--language', help='Filter by language (python, javascript, etc.)')
def search_cmd(query, index, top_k, type, language):
    """
    Search for relevant code.

    Performs semantic search to find code relevant to your query.

    \b
    Examples:
        turbo-code-gpt rag search "authentication logic"
        turbo-code-gpt rag search "parse JSON" -k 10
        turbo-code-gpt rag search "database connection" --type function
    """
    try:
        console.print(f"\n[bold cyan]Searching:[/bold cyan] {query}\n")

        # Load index
        console.print(f"Loading index from {index}...")

        from ..rag import SemanticRetriever

        retriever = SemanticRetriever()
        retriever.load(index)

        console.print(f"  Loaded {len(retriever.indexer.chunks)} chunks\n")

        # Search
        console.print("Searching...")
        results = retriever.search(
            query,
            k=top_k,
            filter_type=type,
            filter_language=language
        )

        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        # Display results
        console.print(f"\n[bold green]Found {len(results)} results:[/bold green]\n")

        for i, result in enumerate(results):
            # Header
            chunk_type = result.get('type', 'code')
            name = result.get('name', 'unknown')
            file_path = result.get('file_path', '')
            score = result.get('score', 0)

            header = f"[{i+1}] {chunk_type.upper()}: {name}"
            if file_path:
                header += f"\nFile: {file_path}"
            header += f"\nRelevance: {score:.2f}"

            # Content preview
            content = result.get('content', '')
            preview = content[:300] + ('...' if len(content) > 300 else '')

            console.print(Panel(
                f"{header}\n\n```\n{preview}\n```",
                border_style="cyan",
                title=f"Result {i+1}"
            ))

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise click.Abort()


@rag_cmd.command('query')
@click.argument('question')
@click.option('--index', default='./models/rag_index', help='Path to RAG index')
@click.option('-k', '--top-k', default=3, help='Number of context chunks')
@click.option('--show-context', is_flag=True, help='Show retrieved context')
def query_cmd(question, index, top_k, show_context):
    """
    Ask a question about your codebase.

    Uses RAG to retrieve relevant code and provide context-aware answers.

    \b
    Examples:
        turbo-code-gpt rag query "How does authentication work?"
        turbo-code-gpt rag query "Where is database connection handled?" --show-context
    """
    try:
        console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")

        # Load index
        console.print(f"Loading index from {index}...")

        from ..rag import SemanticRetriever
        from ..inference import RAGInference

        retriever = SemanticRetriever()
        retriever.load(index)

        console.print(f"  Loaded {len(retriever.indexer.chunks)} chunks\n")

        # Create inference engine (without model for now)
        inference = RAGInference(
            retriever=retriever,
            retrieval_k=top_k
        )

        # Query
        console.print("Retrieving relevant context...")
        result = inference.query(question, use_rag=True)

        # Display answer
        console.print("\n")
        console.print(Panel(
            f"[bold]Question:[/bold]\n{result['question']}\n\n"
            f"[bold]Answer:[/bold]\n{result['answer']}",
            title="RAG Response",
            border_style="green"
        ))

        # Show retrieved chunks if requested
        if show_context and result['retrieved_chunks']:
            console.print("\n[bold cyan]Retrieved Context:[/bold cyan]\n")

            for i, chunk in enumerate(result['retrieved_chunks']):
                chunk_type = chunk.get('type', 'code')
                name = chunk.get('name', 'unknown')
                file_path = chunk.get('file_path', '')
                score = chunk.get('score', 0)

                console.print(Panel(
                    f"Type: {chunk_type}\n"
                    f"Name: {name}\n"
                    f"File: {file_path}\n"
                    f"Relevance: {score:.2f}\n\n"
                    f"```\n{chunk.get('content', '')[:200]}...\n```",
                    title=f"Context {i+1}",
                    border_style="dim"
                ))

        # Note about model
        if result['answer'].startswith("Context retrieved"):
            console.print("\n[yellow]Note:[/yellow] No fine-tuned model loaded. Only context retrieval shown.")
            console.print("[dim]Train a model with 'turbo-code-gpt train' for full RAG inference.[/dim]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise click.Abort()


@rag_cmd.command('stats')
@click.option('--index', default='./models/rag_index', help='Path to RAG index')
def stats_cmd(index):
    """
    Show RAG index statistics.

    \b
    Example:
        turbo-code-gpt rag stats
        turbo-code-gpt rag stats --index ./my_index
    """
    try:
        console.print(f"\n[bold cyan]RAG Index Statistics[/bold cyan]\n")

        # Load index
        from ..rag import SemanticRetriever

        retriever = SemanticRetriever()
        retriever.load(index)

        # Get stats
        stats = retriever.get_stats()

        # Display
        table = Table(title="Index Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Chunks", str(stats['indexer']['num_chunks']))
        table.add_row("Embedding Dimension", str(stats['embedder']['embedding_dim']))
        table.add_row("Embedding Model", stats['embedder']['model'])
        table.add_row("Chunk Strategy", stats['chunker']['strategy'])
        table.add_row("Max Chunk Size", f"{stats['chunker']['max_chunk_size']} chars")

        console.print(table)

        # Chunk type distribution
        chunks = retriever.indexer.chunks
        types = {}
        languages = {}

        for chunk in chunks:
            chunk_type = chunk.get('type', 'unknown')
            types[chunk_type] = types.get(chunk_type, 0) + 1

            lang = chunk.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1

        # Types table
        if types:
            console.print()
            table = Table(title="Chunk Types", box=box.ROUNDED)
            table.add_column("Type", style="cyan")
            table.add_column("Count", justify="right")
            table.add_column("Percentage", justify="right", style="green")

            for chunk_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(chunks) * 100) if chunks else 0
                table.add_row(chunk_type, str(count), f"{pct:.1f}%")

            console.print(table)

        # Languages table
        if languages:
            console.print()
            table = Table(title="Languages", box=box.ROUNDED)
            table.add_column("Language", style="cyan")
            table.add_column("Count", justify="right")
            table.add_column("Percentage", justify="right", style="green")

            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(chunks) * 100) if chunks else 0
                table.add_row(lang, str(count), f"{pct:.1f}%")

            console.print(table)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise click.Abort()
