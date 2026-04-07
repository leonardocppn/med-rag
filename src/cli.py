"""
cli.py

Command-line interface for med-rag.

Commands:
  index <pdf>                          Index a PDF
  index-multi --corpus <name> <pdf...> Index multiple PDFs into a corpus
  ask <pdf> <question>                 Ask a question about an indexed PDF
  ask-corpus <corpus> <question>       Ask a question across a multi-PDF corpus
  summary <pdf>                        Generate a structured summary
  chat <pdf>                           Interactive multi-turn session
  inspect <pdf>                        Show extracted blocks (debug)
"""

import os
import click
from dotenv import load_dotenv

load_dotenv()

from .parser import extract_blocks, blocks_to_chunks
from .indexer import (
    index_chunks, index_chunks_to_corpus,
    _collection_name, corpus_collection_name,
)
from .chain import ask_stream, summarize_stream, chat_stream, ask_corpus_stream

DB_PATH = os.path.join(os.path.dirname(__file__), "../chroma_db")


@click.group()
def cli():
    """med-rag — ask questions about any PDF document."""
    pass


# ── Indexing ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("pdf_path")
def index(pdf_path: str):
    """Index a PDF for subsequent queries."""
    click.echo(f"Parsing {pdf_path}...")
    blocks = extract_blocks(pdf_path)
    chunks = blocks_to_chunks(blocks, overlap=1)
    click.echo(f"Extracted {len(chunks)} chunks from {len(blocks)} blocks.")
    index_chunks(chunks, pdf_path, db_path=DB_PATH)
    click.echo("Indexing complete.")


@cli.command("index-multi")
@click.argument("pdf_paths", nargs=-1, required=True)
@click.option("--corpus", required=True, help="Corpus name.")
def index_multi(pdf_paths: tuple, corpus: str):
    """Index multiple PDFs into a shared corpus for cross-document queries."""
    for pdf_path in pdf_paths:
        click.echo(f"Parsing {pdf_path}...")
        blocks = extract_blocks(pdf_path)
        chunks = blocks_to_chunks(blocks, overlap=1)
        click.echo(f"  {len(chunks)} chunks extracted.")
        index_chunks_to_corpus(chunks, corpus, pdf_path, db_path=DB_PATH)
    click.echo(f"\nCorpus '{corpus}' updated with {len(pdf_paths)} document(s).")


# ── Query ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("pdf_path")
@click.argument("query")
@click.option("--n", default=5, help="Number of chunks to retrieve.")
@click.option("--rerank", is_flag=True, help="Enable cross-encoder re-ranking.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt for Claude.")
def ask(pdf_path: str, query: str, n: int, rerank: bool, system_prompt: str | None):
    """Ask a question about an indexed PDF."""
    col_name = _collection_name(pdf_path)
    click.echo(f"\nSearching: '{query}'\n")
    for chunk in ask_stream(query, col_name, db_path=DB_PATH, n_results=n, use_rerank=rerank, system_prompt=system_prompt):
        click.echo(chunk, nl=False)
    click.echo("\n")


@cli.command("ask-corpus")
@click.argument("corpus")
@click.argument("query")
@click.option("--n", default=5, help="Number of chunks to retrieve.")
@click.option("--rerank", is_flag=True, help="Enable cross-encoder re-ranking.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt for Claude.")
def ask_corpus(corpus: str, query: str, n: int, rerank: bool, system_prompt: str | None):
    """Ask a question across a multi-PDF corpus."""
    click.echo(f"\nSearching corpus '{corpus}': '{query}'\n")
    for chunk in ask_corpus_stream(query, corpus, db_path=DB_PATH, n_results=n, use_rerank=rerank, system_prompt=system_prompt):
        click.echo(chunk, nl=False)
    click.echo("\n")


@cli.command()
@click.argument("pdf_path")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt for Claude.")
def summary(pdf_path: str, system_prompt: str | None):
    """Generate a structured summary of an indexed PDF."""
    col_name = _collection_name(pdf_path)
    click.echo(f"\nGenerating summary of '{os.path.basename(pdf_path)}'...\n")
    for chunk in summarize_stream(col_name, db_path=DB_PATH, system_prompt=system_prompt):
        click.echo(chunk, nl=False)
    click.echo("\n")


@cli.command()
@click.argument("pdf_path")
@click.option("--n", default=5, help="Number of chunks to retrieve per turn.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt for Claude.")
def chat(pdf_path: str, n: int, system_prompt: str | None):
    """Interactive multi-turn chat session about an indexed PDF."""
    col_name = _collection_name(pdf_path)
    messages: list[dict] = []

    click.echo(f"\nChatting with '{os.path.basename(pdf_path)}'. Type 'exit' to quit.\n")

    while True:
        try:
            query = click.prompt("You")
        except (KeyboardInterrupt, EOFError):
            click.echo("\nExiting.")
            break

        if query.strip().lower() in ("exit", "quit", "q"):
            break

        messages.append({"role": "user", "content": query})

        click.echo("\nClaude: ", nl=False)
        full_response = ""
        for chunk in chat_stream(messages, col_name, db_path=DB_PATH, n_results=n, system_prompt=system_prompt):
            click.echo(chunk, nl=False)
            full_response += chunk
        click.echo("\n")

        messages.append({"role": "assistant", "content": full_response})


# ── Debug ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("pdf_path")
def inspect(pdf_path: str):
    """Show extracted blocks from the parser (useful for debugging)."""
    blocks = extract_blocks(pdf_path)
    for b in blocks:
        click.echo(
            f"[p{b.page:02d}] [{b.region:6s}] "
            f"(top={b.top:6.1f}, size={b.font_size:.1f}) "
            f"{b.text[:80]}"
        )


if __name__ == "__main__":
    cli()
