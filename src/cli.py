"""
cli.py

Command-line interface for med-rag.

Commands:
  index <path...>                        Index one or more PDFs (or directories)
  index-multi --corpus <name> <pdf...>   [alias] Index multiple PDFs into a corpus
  sync <directory> [--corpus <name>]     Index only new PDFs in a directory
  ask <target> <question>                Ask a question about a PDF or corpus
  ask-corpus <corpus> <question>         [alias] Ask a question across a multi-PDF corpus
  summary <pdf>                          Generate a structured summary
  chat <target>                          Interactive multi-turn session (PDF or corpus)
  chat-corpus <corpus>                   [alias] Interactive multi-turn session on a corpus
  delete <pdf>                           Delete the collection for a PDF
  delete-corpus <corpus>                 Delete a corpus (or a single PDF with --pdf)
  profile <pdf>                          Show layout profile and derived parameters
  cluster [pdf...]                       Cluster PDFs by layout similarity
  inspect <pdf>                          Show extracted blocks (debug)
"""

import os
import re
import glob as _glob
import click
from dotenv import load_dotenv

load_dotenv()

from .parser import extract_blocks, blocks_to_chunks
from .profiler import profile_pdf, profile_and_params, derive_params, ProfileStore
from . import indexer as _indexer
from .indexer import (
    index_chunks, index_chunks_to_corpus,
    _collection_name, corpus_collection_name,
    file_hash, corpus_has_file,
)
from .chain import ask_stream, summarize_stream, chat_stream, ask_corpus_stream, chat_corpus_stream

DB_PATH = os.path.join(os.path.dirname(__file__), "../chroma_db")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_pdfs(paths: tuple) -> list[str]:
    """Expands directories into a list of PDFs; keeps direct PDF file paths."""
    pdfs = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(_glob.glob(os.path.join(p, "*.pdf")))
            if not found:
                click.echo(f"Warning: no PDFs found in '{p}'.")
            pdfs.extend(found)
        elif p.lower().endswith(".pdf"):
            pdfs.append(p)
        else:
            click.echo(f"Warning: '{p}' is not a PDF or directory, skipped.")
    return pdfs


def _corpus_exists(corpus_name: str, db_path: str) -> bool:
    """Checks whether a corpus exists in the database."""
    import chromadb
    col_name = corpus_collection_name(corpus_name)
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.get_collection(col_name)
        return True
    except Exception:
        return False


def _next_corpus_name(db_path: str) -> str:
    """Returns the next automatic corpus name: corpus_1, corpus_2, ..."""
    import chromadb
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    nums = []
    for col in collections:
        name = col.name if hasattr(col, "name") else str(col)
        m = re.match(r"^corpus_corpus_(\d+)$", name)
        if m:
            nums.append(int(m.group(1)))
    return f"corpus_{max(nums) + 1 if nums else 1}"


def _do_index_single(pdf_path: str):
    """Indexes a single PDF (shared logic)."""
    prof, params = profile_and_params(pdf_path)
    click.echo(f"Profile: {prof.num_pages} pages, {prof.words_per_page:.0f} words/page, "
               f"header<{params.header_threshold:.2f}, footer>{params.footer_threshold:.2f}")
    click.echo(f"Parsing {pdf_path}...")
    blocks = extract_blocks(
        pdf_path,
        header_threshold=params.header_threshold,
        footer_threshold=params.footer_threshold,
        title_font_ratio=params.title_font_ratio,
        line_gap=params.line_gap,
        block_gap_multiplier=params.block_gap_multiplier,
        block_gap_minimum=params.block_gap_minimum,
        min_columns_for_table=params.min_columns_for_table,
    )
    chunks = blocks_to_chunks(blocks, skip_regions=params.skip_regions, overlap=params.chunk_overlap)
    click.echo(f"Extracted {len(chunks)} chunks from {len(blocks)} blocks.")
    index_chunks(chunks, pdf_path, db_path=DB_PATH)
    ProfileStore(db_path=DB_PATH).add(prof)
    click.echo("Indexing complete.")


def _do_index_corpus(pdf_paths: list[str], corpus_name: str):
    """Indexes multiple PDFs into a corpus (shared logic)."""
    store = ProfileStore(db_path=DB_PATH)
    for pdf_path in pdf_paths:
        prof, params = profile_and_params(pdf_path)
        click.echo(f"Parsing {pdf_path} ({prof.words_per_page:.0f} words/page)...")
        blocks = extract_blocks(
            pdf_path,
            header_threshold=params.header_threshold,
            footer_threshold=params.footer_threshold,
            title_font_ratio=params.title_font_ratio,
            line_gap=params.line_gap,
            block_gap_multiplier=params.block_gap_multiplier,
            block_gap_minimum=params.block_gap_minimum,
            min_columns_for_table=params.min_columns_for_table,
        )
        chunks = blocks_to_chunks(blocks, skip_regions=params.skip_regions, overlap=params.chunk_overlap)
        click.echo(f"  {len(chunks)} chunks extracted.")
        index_chunks_to_corpus(chunks, corpus_name, pdf_path, db_path=DB_PATH)
        store.add(prof)
    click.echo(f"\nCorpus '{corpus_name}' updated with {len(pdf_paths)} document(s).")


def _run_chat(target: str, n: int, system_prompt: str | None):
    """Shared logic for chat sessions (PDF or corpus)."""
    if os.path.isfile(target):
        mode = "pdf"
        col_name = _collection_name(target)
        label = os.path.basename(target)
    elif _corpus_exists(target, DB_PATH):
        mode = "corpus"
        label = target
    else:
        click.echo(f"Error: '{target}' is not a PDF file or an existing corpus.")
        return

    messages: list[dict] = []
    click.echo(f"\nChatting with '{label}'. Type 'exit' to quit.\n")

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
        if mode == "pdf":
            stream = chat_stream(messages, col_name, db_path=DB_PATH, n_results=n, system_prompt=system_prompt)
        else:
            stream = chat_corpus_stream(messages, target, db_path=DB_PATH, n_results=n, system_prompt=system_prompt)
        for chunk in stream:
            click.echo(chunk, nl=False)
            full_response += chunk
        click.echo("\n")

        messages.append({"role": "assistant", "content": full_response})


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group()
@click.option("--show", is_flag=True, default=False,
              help="Show verbose output from embedding/re-ranking models.")
@click.pass_context
def cli(ctx: click.Context, show: bool):
    """med-rag — ask questions about any PDF document."""
    _indexer.VERBOSE = show


# ── Indexing ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--corpus", "corpus_name", default=None, help="Corpus name (forces corpus mode).")
def index(paths: tuple, corpus_name: str | None):
    """Index one or more PDFs (or directories).

    \b
    Examples:
      index data/1.pdf              → single PDF
      index data/                   → auto: single if 1 PDF, corpus if more
      index data/1.pdf data/2.pdf   → automatic corpus
      index data/ --corpus name     → corpus with explicit name
    """
    pdfs = _collect_pdfs(paths)
    if not pdfs:
        click.echo("No PDFs found.")
        return

    if len(pdfs) == 1 and corpus_name is None:
        _do_index_single(pdfs[0])
    else:
        if corpus_name is None:
            corpus_name = _next_corpus_name(DB_PATH)
            click.echo(f"Auto-assigned corpus name: '{corpus_name}'")
        _do_index_corpus(pdfs, corpus_name)


@cli.command("index-multi")
@click.argument("pdf_paths", nargs=-1, required=True)
@click.option("--corpus", required=True, help="Corpus name.")
def index_multi(pdf_paths: tuple, corpus: str):
    """[Alias] Index multiple PDFs into a shared corpus."""
    _do_index_corpus(list(pdf_paths), corpus)


@cli.command()
@click.argument("directory")
@click.option("--corpus", "corpus_name", default=None,
              help="Corpus name. If omitted, uses or creates an automatic corpus.")
def sync(directory: str, corpus_name: str | None):
    """Index only new PDFs in a directory, skipping those already in the corpus.

    \b
    Examples:
      sync data/                      → auto corpus, only new PDFs
      sync data/ --corpus my-corpus   → explicit corpus, only new PDFs
    """
    if not os.path.isdir(directory):
        click.echo(f"Error: '{directory}' is not a directory.")
        return

    all_pdfs = sorted(_glob.glob(os.path.join(directory, "*.pdf")))
    if not all_pdfs:
        click.echo(f"No PDFs found in '{directory}'.")
        return

    if corpus_name is None:
        corpus_name = _next_corpus_name(DB_PATH)
        click.echo(f"Corpus: '{corpus_name}' (auto)")
    else:
        click.echo(f"Corpus: '{corpus_name}'")

    new_pdfs = []
    skipped = 0
    for pdf_path in all_pdfs:
        fhash = file_hash(pdf_path)
        if corpus_has_file(corpus_name, fhash, DB_PATH):
            click.echo(f"  [already indexed] {os.path.basename(pdf_path)}")
            skipped += 1
        else:
            click.echo(f"  [new]             {os.path.basename(pdf_path)}")
            new_pdfs.append(pdf_path)

    if not new_pdfs:
        click.echo(f"\nNo new PDFs to index ({skipped} already indexed).")
        return

    click.echo(f"\nIndexing {len(new_pdfs)} new PDF(s)...\n")
    _do_index_corpus(new_pdfs, corpus_name)
    if skipped:
        click.echo(f"({skipped} already-indexed PDF(s) skipped)")


# ── Query ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("target")
@click.argument("query")
@click.option("--n", default=5, help="Number of chunks to retrieve.")
@click.option("--rerank", is_flag=True, help="Enable cross-encoder re-ranking.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt for Claude.")
def ask(target: str, query: str, n: int, rerank: bool, system_prompt: str | None):
    """Ask a question about a PDF or corpus.

    \b
    Examples:
      ask data/1.pdf "What is the PSA value?"   → single PDF
      ask my-corpus "What medications?"          → corpus
    """
    click.echo(f"\nSearching: '{query}'\n")
    if os.path.isfile(target):
        col_name = _collection_name(target)
        stream = ask_stream(query, col_name, db_path=DB_PATH, n_results=n,
                            use_rerank=rerank, system_prompt=system_prompt)
    elif _corpus_exists(target, DB_PATH):
        stream = ask_corpus_stream(query, target, db_path=DB_PATH, n_results=n,
                                   use_rerank=rerank, system_prompt=system_prompt)
    else:
        click.echo(f"Error: '{target}' is not a PDF file or an existing corpus.")
        return
    for chunk in stream:
        click.echo(chunk, nl=False)
    click.echo("\n")


@cli.command("ask-corpus")
@click.argument("corpus")
@click.argument("query")
@click.option("--n", default=5, help="Number of chunks to retrieve.")
@click.option("--rerank", is_flag=True, help="Enable cross-encoder re-ranking.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt for Claude.")
def ask_corpus(corpus: str, query: str, n: int, rerank: bool, system_prompt: str | None):
    """[Alias] Ask a question across a multi-PDF corpus."""
    click.echo(f"\nSearching corpus '{corpus}': '{query}'\n")
    for chunk in ask_corpus_stream(query, corpus, db_path=DB_PATH, n_results=n,
                                   use_rerank=rerank, system_prompt=system_prompt):
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
@click.argument("target")
@click.option("--n", default=5, help="Number of chunks to retrieve per turn.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt for Claude.")
def chat(target: str, n: int, system_prompt: str | None):
    """Interactive multi-turn chat session on a PDF or corpus.

    \b
    Examples:
      chat data/1.pdf            → chat about a single PDF
      chat my-corpus             → chat across a corpus
    """
    _run_chat(target, n, system_prompt)


@cli.command("chat-corpus")
@click.argument("corpus")
@click.option("--n", default=5, help="Number of chunks to retrieve per turn.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt for Claude.")
def chat_corpus(corpus: str, n: int, system_prompt: str | None):
    """[Alias] Interactive multi-turn chat session on a corpus."""
    _run_chat(corpus, n, system_prompt)


# ── Delete ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("pdf_path")
def delete(pdf_path: str):
    """Delete the collection for an indexed PDF."""
    col_name = _collection_name(pdf_path)
    client = __import__("chromadb").PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(col_name)
        click.echo(f"Collection '{col_name}' deleted.")
    except Exception:
        click.echo(f"No collection found for '{pdf_path}'.")


@cli.command("delete-corpus")
@click.argument("corpus")
@click.option("--pdf", "pdf_path", default=None, help="Remove only chunks from this PDF in the corpus.")
def delete_corpus(corpus: str, pdf_path: str | None):
    """Delete a multi-PDF corpus, or just one PDF within it using --pdf."""
    import hashlib
    col_name = corpus_collection_name(corpus)
    client = __import__("chromadb").PersistentClient(path=DB_PATH)

    if pdf_path is None:
        try:
            client.delete_collection(col_name)
            click.echo(f"Corpus '{corpus}' (collection '{col_name}') deleted.")
        except Exception:
            click.echo(f"No corpus found with name '{corpus}'.")
        return

    try:
        collection = client.get_collection(col_name)
    except Exception:
        click.echo(f"No corpus found with name '{corpus}'.")
        return

    pdf_id = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
    existing = collection.get(where={"pdf_id": pdf_id})
    if not existing["ids"]:
        click.echo(f"No chunks found for '{pdf_path}' in corpus '{corpus}'.")
        return

    collection.delete(ids=existing["ids"])
    click.echo(f"Removed {len(existing['ids'])} chunks of '{pdf_path}' from corpus '{corpus}'.")


# ── Debug ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("pdf_path")
def profile(pdf_path: str):
    """Show the layout profile and derived parsing parameters for a PDF."""
    prof, params = profile_and_params(pdf_path)
    click.echo(f"Layout metrics:")
    click.echo(f"  Pages:             {prof.num_pages}")
    click.echo(f"  Words/page:        {prof.words_per_page:.0f}")
    click.echo(f"  Median font:       {prof.font_size_median:.1f} pt (std: {prof.font_size_std:.1f})")
    click.echo(f"  Font levels:       {prof.font_size_levels}")
    click.echo(f"  Line gap (med):    {prof.line_gap_median:.1f} pt (std: {prof.line_gap_std:.1f})")
    click.echo(f"  Block gap (med):   {prof.block_gap_median:.1f} pt")
    click.echo(f"  Column density:    {prof.column_density:.2f}")
    click.echo(f"  Header zone:       0 - {prof.header_zone:.3f}")
    click.echo(f"  Footer zone:       {prof.footer_zone:.3f} - 1")
    click.echo(f"  Content density:   {prof.content_density:.2f}")
    click.echo(f"  Title gap (ratio): {prof.title_font_gap:.2f}")
    click.echo(f"\nDerived parsing parameters:")
    click.echo(f"  header_threshold:     {params.header_threshold:.3f}")
    click.echo(f"  footer_threshold:     {params.footer_threshold:.3f}")
    click.echo(f"  title_font_ratio:     {params.title_font_ratio:.2f}")
    click.echo(f"  line_gap:             {params.line_gap:.1f}")
    click.echo(f"  block_gap_multiplier: {params.block_gap_multiplier:.2f}")
    click.echo(f"  block_gap_minimum:    {params.block_gap_minimum:.1f}")
    click.echo(f"  chunk_overlap:        {params.chunk_overlap}")
    click.echo(f"  min_columns_table:    {params.min_columns_for_table}")


@cli.command()
@click.argument("pdf_path")
def inspect(pdf_path: str):
    """Show extracted blocks from the parser (useful for debugging)."""
    prof, params = profile_and_params(pdf_path)
    click.echo(f"[header<{params.header_threshold:.3f}, footer>{params.footer_threshold:.3f}]\n")
    blocks = extract_blocks(
        pdf_path,
        header_threshold=params.header_threshold,
        footer_threshold=params.footer_threshold,
        title_font_ratio=params.title_font_ratio,
        line_gap=params.line_gap,
        block_gap_multiplier=params.block_gap_multiplier,
        block_gap_minimum=params.block_gap_minimum,
        min_columns_for_table=params.min_columns_for_table,
    )
    for b in blocks:
        click.echo(
            f"[p{b.page:02d}] [{b.region:6s}] "
            f"(top={b.top:6.1f}, size={b.font_size:.1f}) "
            f"{b.text[:80]}"
        )


@cli.command()
@click.option("--eps", default=None, type=float, help="DBSCAN radius (0-1). Default: automatic.")
@click.argument("pdf_paths", nargs=-1)
def cluster(pdf_paths: tuple, eps: float | None):
    """Cluster PDFs by layout similarity.

    Without arguments, uses profiles already saved in the store.
    With arguments, profiles the given PDFs and clusters them on the fly.
    """
    store = ProfileStore(db_path=DB_PATH)

    if pdf_paths:
        for pdf_path in pdf_paths:
            prof = profile_pdf(pdf_path)
            store.add(prof)

    clusters = store.cluster(eps=eps)
    if not clusters:
        click.echo("No saved profiles. Index or profile at least one PDF.")
        return

    click.echo(f"Found {len(clusters)} cluster(s):\n")
    for i, group in enumerate(clusters):
        click.echo(f"  Cluster {i + 1} ({len(group)} PDF(s)):")
        for prof in group:
            params = derive_params(prof)
            click.echo(
                f"    {os.path.basename(prof.pdf_path):20s}  "
                f"{prof.words_per_page:5.0f} w/page  "
                f"font={prof.font_size_median:.1f}  "
                f"cols={prof.column_density:.2f}  "
                f"h<{params.header_threshold:.3f} f>{params.footer_threshold:.3f}"
            )
        click.echo()


if __name__ == "__main__":
    cli()
