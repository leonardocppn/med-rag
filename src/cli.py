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
  search <target> <query>                BM25 keyword search (no LLM, instant results)
  delete <pdf>                           Delete the collection for a PDF
  delete-corpus <corpus>                 Delete a corpus (or a single PDF with --pdf)
  profile <pdf>                          Show layout profile and derived parameters
  cluster [pdf...]                       Cluster PDFs by layout similarity
  inspect <pdf>                          Show extracted blocks (debug)
"""

import itertools
import os
import re
import glob as _glob
import sys
import threading
import time
import click
from dotenv import load_dotenv

load_dotenv()

from .parser import extract_blocks, extract_blocks_plain, blocks_to_chunks
from .profiler import profile_pdf, profile_and_params, derive_params, ProfileStore
from . import indexer as _indexer
from .indexer import (
    index_chunks, index_chunks_to_corpus,
    _collection_name, corpus_collection_name,
    file_hash, corpus_has_file, pdf_id as _pdf_id,
    collection_exists,
    search_bm25, search_bm25_corpus,
)
from .chain import ask_stream, summarize_stream, summarize_corpus_stream, chat_stream, ask_corpus_stream, chat_corpus_stream, PHASE_LLM, PHASE_PROGRESS

DB_PATH = os.path.join(os.path.dirname(__file__), "../chroma_db")


# -- Spinner -------------------------------------------------------------------

class _Spinner:
    r"""Animated spinner \|/- on stdout to show status of slow operations."""

    FRAMES = r"\|/-"

    def __init__(self):
        self._label = ""
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self, label: str = ""):
        self._label = label
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, label: str):
        self._label = label

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

    def _run(self):
        for frame in itertools.cycle(self.FRAMES):
            if not self._running:
                break
            sys.stdout.write(f"\r{frame} {self._label}  ")
            sys.stdout.flush()
            time.sleep(0.12)


def _stream_to_console(stream) -> str:
    """Consumes a stream with phase-aware spinner, prints tokens, returns full text.

    Recognized phases via sentinel:
      - Default:   "Searching documents..."
      - PHASE_LLM: "LLM generating..."  (retrieval done, LLM is producing output)
    """
    spinner = _Spinner()
    spinner.start("Searching documents...")

    result: list[str] = []
    first_token = True

    try:
        for chunk in stream:
            if chunk == PHASE_LLM:
                spinner.update("LLM generating...")
                continue
            if chunk.startswith(PHASE_PROGRESS):
                spinner.update(chunk[len(PHASE_PROGRESS):])
                continue
            if first_token:
                spinner.stop()
                first_token = False
            sys.stdout.write(chunk)
            sys.stdout.flush()
            result.append(chunk)
    except KeyboardInterrupt:
        stream.close()
        raise
    finally:
        if first_token:
            spinner.stop()

    return "".join(result)


# -- Helpers -------------------------------------------------------------------

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


def _do_index_single(pdf_path: str, plain: bool = False):
    """Indexes a single PDF (shared logic)."""
    if plain:
        click.echo(f"Parsing {pdf_path} (plain mode)...")
        blocks = extract_blocks_plain(pdf_path)
        chunks = blocks_to_chunks(blocks, skip_regions=set(), overlap=0)
    else:
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
    if not plain:
        ProfileStore(db_path=DB_PATH).add(prof)
    click.echo("Indexing complete.")


def _do_index_corpus(pdf_paths: list[str], corpus_name: str, plain: bool = False):
    """Indexes multiple PDFs into a corpus (shared logic)."""
    store = ProfileStore(db_path=DB_PATH)
    for pdf_path in pdf_paths:
        if plain:
            click.echo(f"Parsing {pdf_path} (plain mode)...")
            blocks = extract_blocks_plain(pdf_path)
            chunks = blocks_to_chunks(blocks, skip_regions=set(), overlap=0)
        else:
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
        if not plain:
            store.add(prof)
    click.echo(f"\nCorpus '{corpus_name}' updated with {len(pdf_paths)} document(s).")


def _run_chat(target: str, n: int, system_prompt: str | None, backend: str = "local"):
    """Shared logic for chat sessions (PDF or corpus)."""
    if os.path.isfile(target):
        mode = "pdf"
        col_name = _collection_name(target)
        label = os.path.basename(target)
    elif collection_exists(corpus_collection_name(target), DB_PATH):
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

        click.echo("\nAssistant: ", nl=False)
        if mode == "pdf":
            stream = chat_stream(messages, col_name, db_path=DB_PATH, n_results=n,
                                 system_prompt=system_prompt, backend=backend)
        else:
            stream = chat_corpus_stream(messages, target, db_path=DB_PATH, n_results=n,
                                        system_prompt=system_prompt, backend=backend)
        full_response = _stream_to_console(stream)
        click.echo("\n")

        messages.append({"role": "assistant", "content": full_response})


# -- CLI group -----------------------------------------------------------------

@click.group()
@click.option("--show", is_flag=True, default=False,
              help="Show verbose output from embedding/re-ranking models.")
@click.pass_context
def cli(ctx: click.Context, show: bool):
    """med-rag -- ask questions about any PDF document."""
    _indexer.VERBOSE = show


# -- Indexing ------------------------------------------------------------------

@cli.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--corpus", "corpus_name", default=None, help="Corpus name (forces corpus mode).")
@click.option("--plain", is_flag=True, default=False,
              help="Plain-text extraction (no layout analysis). Best for simple one-column or OCR documents.")
def index(paths: tuple, corpus_name: str | None, plain: bool):
    """Index one or more PDFs (or directories).

    \b
    Examples:
      index data/1.pdf              -> single PDF
      index data/                   -> auto: single if 1 PDF, corpus if more
      index data/1.pdf data/2.pdf   -> automatic corpus
      index data/ --corpus name     -> corpus with explicit name
      index data/ --plain           -> plain-text extraction (no layout)
    """
    pdfs = _collect_pdfs(paths)
    if not pdfs:
        click.echo("No PDFs found.")
        return

    if len(pdfs) == 1 and corpus_name is None:
        _do_index_single(pdfs[0], plain=plain)
    else:
        if corpus_name is None:
            corpus_name = _next_corpus_name(DB_PATH)
            click.echo(f"Auto-assigned corpus name: '{corpus_name}'")
        _do_index_corpus(pdfs, corpus_name, plain=plain)


@cli.command("index-multi")
@click.argument("pdf_paths", nargs=-1, required=True)
@click.option("--corpus", required=True, help="Corpus name.")
@click.option("--plain", is_flag=True, default=False,
              help="Plain-text extraction (no layout analysis).")
def index_multi(pdf_paths: tuple, corpus: str, plain: bool):
    """[Alias] Index multiple PDFs into a shared corpus."""
    _do_index_corpus(list(pdf_paths), corpus, plain=plain)


@cli.command()
@click.argument("directory")
@click.option("--corpus", "corpus_name", default=None,
              help="Corpus name. If omitted, uses or creates an automatic corpus.")
@click.option("--plain", is_flag=True, default=False,
              help="Plain-text extraction (no layout analysis).")
def sync(directory: str, corpus_name: str | None, plain: bool):
    """Index only new PDFs in a directory, skipping those already in the corpus.

    \b
    Examples:
      sync data/                      -> auto corpus, only new PDFs
      sync data/ --corpus my-corpus   -> explicit corpus, only new PDFs
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
    _do_index_corpus(new_pdfs, corpus_name, plain=plain)
    if skipped:
        click.echo(f"({skipped} already-indexed PDF(s) skipped)")


# -- Query ---------------------------------------------------------------------

@cli.command()
@click.argument("target")
@click.argument("query")
@click.option("--n", default=5, help="Number of chunks to retrieve.")
@click.option("--rerank", is_flag=True, help="Enable cross-encoder re-ranking.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt.")
@click.option("--model", "backend", default="local",
              type=click.Choice(["local", "claude"], case_sensitive=False),
              help="LLM backend: local (Ollama) or claude (API).")
def ask(target: str, query: str, n: int, rerank: bool, system_prompt: str | None, backend: str):
    """Ask a question about a PDF or corpus.

    \b
    Examples:
      ask data/1.pdf "What is the PSA value?"   -> single PDF
      ask my-corpus "What medications?"          -> corpus
      ask data/1.pdf "..." --model claude        -> use Claude API
    """
    click.echo(f"\nSearching: '{query}'\n")
    if os.path.isfile(target):
        col_name = _collection_name(target)
        if not collection_exists(col_name, DB_PATH):
            click.echo(f"Error: '{target}' has not been indexed yet. Run: index {target}")
            return
        stream = ask_stream(query, col_name, db_path=DB_PATH, n_results=n,
                            use_rerank=rerank, system_prompt=system_prompt, backend=backend)
    elif collection_exists(corpus_collection_name(target), DB_PATH):
        stream = ask_corpus_stream(query, target, db_path=DB_PATH, n_results=n,
                                   use_rerank=rerank, system_prompt=system_prompt, backend=backend)
    else:
        click.echo(f"Error: '{target}' is not a PDF file or an existing corpus.")
        return
    _stream_to_console(stream)
    click.echo("\n")


@cli.command("ask-corpus")
@click.argument("corpus")
@click.argument("query")
@click.option("--n", default=5, help="Number of chunks to retrieve.")
@click.option("--rerank", is_flag=True, help="Enable cross-encoder re-ranking.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt.")
@click.option("--model", "backend", default="local",
              type=click.Choice(["local", "claude"], case_sensitive=False),
              help="LLM backend: local (Ollama) or claude (API).")
def ask_corpus(corpus: str, query: str, n: int, rerank: bool, system_prompt: str | None, backend: str):
    """[Alias] Ask a question across a multi-PDF corpus."""
    click.echo(f"\nSearching corpus '{corpus}': '{query}'\n")
    _stream_to_console(ask_corpus_stream(query, corpus, db_path=DB_PATH, n_results=n,
                                         use_rerank=rerank, system_prompt=system_prompt, backend=backend))
    click.echo("\n")


@cli.command()
@click.argument("target")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt.")
@click.option("--model", "backend", default="local",
              type=click.Choice(["local", "claude"], case_sensitive=False),
              help="LLM backend: local (Ollama) or claude (API).")
def summary(target: str, system_prompt: str | None, backend: str):
    """Generate a structured summary of an indexed PDF or corpus."""
    if os.path.isfile(target):
        col_name = _collection_name(target)
        if not collection_exists(col_name, DB_PATH):
            click.echo(f"Error: '{target}' has not been indexed yet. Run: index {target}")
            return
        label = os.path.basename(target)
        stream = summarize_stream(col_name, db_path=DB_PATH, system_prompt=system_prompt, backend=backend)
    elif collection_exists(corpus_collection_name(target), DB_PATH):
        label = target
        stream = summarize_corpus_stream(target, db_path=DB_PATH, system_prompt=system_prompt, backend=backend)
    else:
        click.echo(f"Error: '{target}' is not a PDF file or an existing corpus.")
        return
    click.echo(f"\nGenerating summary of '{label}'...\n")
    _stream_to_console(stream)
    click.echo("\n")


@cli.command()
@click.argument("target")
@click.option("--n", default=5, help="Number of chunks to retrieve per turn.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt.")
@click.option("--model", "backend", default="local",
              type=click.Choice(["local", "claude"], case_sensitive=False),
              help="LLM backend: local (Ollama) or claude (API).")
def chat(target: str, n: int, system_prompt: str | None, backend: str):
    """Interactive multi-turn chat session on a PDF or corpus.

    \b
    Examples:
      chat data/1.pdf            -> chat about a single PDF
      chat my-corpus             -> chat across a corpus
      chat data/1.pdf --model claude
    """
    _run_chat(target, n, system_prompt, backend)


@cli.command("chat-corpus")
@click.argument("corpus")
@click.option("--n", default=5, help="Number of chunks to retrieve per turn.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt.")
@click.option("--model", "backend", default="local",
              type=click.Choice(["local", "claude"], case_sensitive=False),
              help="LLM backend: local (Ollama) or claude (API).")
def chat_corpus(corpus: str, n: int, system_prompt: str | None, backend: str):
    """[Alias] Interactive multi-turn chat session on a corpus."""
    _run_chat(corpus, n, system_prompt, backend)


# -- Search (BM25, no LLM) ----------------------------------------------------

@cli.command()
@click.argument("target")
@click.argument("query")
@click.option("--n", default=5, help="Number of results to show.")
def search(target: str, query: str, n: int):
    """BM25 keyword search -- no LLM, instant results.

    \b
    Examples:
      search data/1.pdf "hemoglobin"
      search my-corpus "vitamin D"
    """
    if os.path.isfile(target):
        results = search_bm25(query, _collection_name(target), db_path=DB_PATH, n_results=n)
        label = os.path.basename(target)
    elif collection_exists(corpus_collection_name(target), DB_PATH):
        results = search_bm25_corpus(query, target, db_path=DB_PATH, n_results=n)
        label = target
    else:
        click.echo(f"Error: '{target}' is not a PDF file or an existing corpus.")
        return

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nResults for '{query}' in '{label}':\n")
    for i, r in enumerate(results, 1):
        source = f"p.{r['page']}"
        if "pdf_path" in r:
            source += f"  [{os.path.basename(r['pdf_path'])}]"
        click.echo(f"  {i}.  {source:<30} score {r['score']:.3f}  [{r['region']}]")
        click.echo(f"      {r['snippet']}")
        click.echo()


# -- Delete --------------------------------------------------------------------

@cli.command()
@click.argument("pdf_path")
def delete(pdf_path: str):
    """Delete the collection for an indexed PDF."""
    import chromadb
    col_name = _collection_name(pdf_path)
    client = chromadb.PersistentClient(path=DB_PATH)
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
    import chromadb
    col_name = corpus_collection_name(corpus)
    client = chromadb.PersistentClient(path=DB_PATH)

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

    existing = collection.get(where={"pdf_id": _pdf_id(pdf_path)})
    if not existing["ids"]:
        click.echo(f"No chunks found for '{pdf_path}' in corpus '{corpus}'.")
        return

    collection.delete(ids=existing["ids"])
    click.echo(f"Removed {len(existing['ids'])} chunks of '{pdf_path}' from corpus '{corpus}'.")


# -- Debug ---------------------------------------------------------------------

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
