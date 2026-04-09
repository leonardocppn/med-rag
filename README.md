# med-rag

A RAG pipeline for querying PDF documents in natural language, powered by Claude.

Built with: Python · pdfplumber · sentence-transformers · ChromaDB · Anthropic Claude API

---

## How it works

1. **Profile** — Each PDF is analyzed to extract numerical layout metrics (font sizes, line spacing, column density, header/footer boundaries). These metrics drive adaptive parsing parameters — no hardcoded thresholds or document-type lookup tables.
2. **Parse** — `pdfplumber` extracts every text element from the PDF with its coordinates (position, font size). Elements are grouped into lines, then into paragraphs, then classified into regions: `title`, `header`, `footer`, `body`. Multi-column layouts are detected and formatted as structured tables. Headers, footers, and layout artifacts are discarded before indexing.
3. **Index** — Each text block is embedded with a multilingual sentence-transformers model (ONNX, no GPU required) and stored in a local ChromaDB vector database.
4. **Retrieve** — A query is embedded the same way; the most semantically similar chunks are retrieved. Optionally, a cross-encoder re-ranker refines the results.
5. **Generate** — Retrieved chunks are passed to Claude with the question; the answer always cites the page number.

## Setup

```bash
git clone https://github.com/leonardocppn/med-rag
cd med-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

Get your API key at [console.anthropic.com](https://console.anthropic.com).

## Usage

### Index a PDF

```bash
python -m src.cli index path/to/document.pdf
```

The PDF is automatically profiled to derive optimal parsing parameters (header/footer boundaries, block gap thresholds, column detection, etc.).

### Ask a question

```bash
# default
python -m src.cli ask path/to/document.pdf "What are the main conclusions?"

# with cross-encoder re-ranking (more precise, downloads ~100MB on first run)
python -m src.cli ask path/to/document.pdf "What are the main conclusions?" --rerank

# with a custom system prompt
python -m src.cli ask path/to/document.pdf "What are the main conclusions?" \
  --system "You are a medical expert. Answer in technical terms."
```

### Generate a structured summary

```bash
python -m src.cli summary path/to/document.pdf
```

### Interactive multi-turn chat

```bash
python -m src.cli chat path/to/document.pdf
```

Type `exit` to quit the session.

### Multi-document corpus

```bash
# Index multiple PDFs into a named corpus
python -m src.cli index-multi --corpus my-corpus doc1.pdf doc2.pdf doc3.pdf

# Ask a question across all documents
python -m src.cli ask-corpus my-corpus "What do these documents have in common?"
```

### Delete indexed data

```bash
# Delete a single PDF's collection
python -m src.cli delete path/to/document.pdf

# Delete an entire corpus
python -m src.cli delete-corpus my-corpus

# Remove only one PDF from a corpus
python -m src.cli delete-corpus my-corpus --pdf path/to/document.pdf
```

### Profile a PDF (layout analysis)

```bash
python -m src.cli profile path/to/document.pdf
```

Displays measured layout metrics (font distribution, spacing, column density, header/footer zones) and the parsing parameters derived from them.

### Cluster PDFs by layout similarity

```bash
# Cluster all previously profiled PDFs
python -m src.cli cluster

# Profile and cluster specific PDFs
python -m src.cli cluster doc1.pdf doc2.pdf doc3.pdf

# With a custom DBSCAN radius
python -m src.cli cluster --eps 0.3
```

Groups PDFs into clusters based on layout similarity using DBSCAN on normalized feature vectors.

### Inspect extracted blocks (debug)

```bash
python -m src.cli inspect path/to/document.pdf
```

## Options

| Option | Commands | Description |
|--------|----------|-------------|
| `--n N` | `ask`, `ask-corpus`, `chat` | Number of chunks to retrieve (default: 5) |
| `--rerank` | `ask`, `ask-corpus` | Enable cross-encoder re-ranking |
| `--system "..."` | `ask`, `ask-corpus`, `summary`, `chat` | Optional system prompt for Claude |
| `--eps N` | `cluster` | DBSCAN radius (0-1, default: automatic) |
| `--pdf <path>` | `delete-corpus` | Remove only one PDF from a corpus |

## Architecture

| Module | Responsibility |
|--------|----------------|
| `src/profiler.py` | PDF layout analysis, adaptive parameter derivation, profile persistence and DBSCAN clustering |
| `src/parser.py` | PDF text extraction, line/block grouping, column detection, region classification, artifact cleanup |
| `src/indexer.py` | Embedding with fastembed, ChromaDB storage, vector retrieval, cross-encoder re-ranking |
| `src/chain.py` | RAG orchestration, Claude API streaming, multi-turn conversation management |
| `src/cli.py` | Click-based CLI with all user-facing commands |

## Notes

- Works best with digitally-generated PDFs. Scanned PDFs require OCR pre-processing (**not included**).
- Everything stays local: embeddings and chunks are stored in `./chroma_db`. Only the final query is sent to the Anthropic API.
- The embedding model (~50MB) is downloaded on first run from HuggingFace. The re-ranking model (~100MB) is downloaded only if `--rerank` is used.
- Layout profiling is fully adaptive: every parameter is derived from the PDF's own metrics, with no hardcoded document types.
