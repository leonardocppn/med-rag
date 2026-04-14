# med-rag

A RAG pipeline for querying PDF documents in natural language, powered by Claude or any local model via Ollama.<br>
Designed as a tool to navigate archives of personal medical reports spanning years, but adaptable to any use case.

Handles two document types:<br>
- **Structured documents** — PDFs with tables, multi-column layouts, headers and footers (e.g. lab reports). Layout is analyzed adaptively; parsing parameters are derived from each document's own metrics, with no hardcoded thresholds.<br>
- **Plain text documents** — Narrative PDFs with simple linear text (e.g. clinical notes, discharge letters, radiology reports). Extracted page by page without layout analysis.

Built with: Python · pdfplumber · sentence-transformers · ChromaDB · Anthropic Claude API · Ollama

---

## How it works

**Shared pipeline:**

1. **Index** — Text blocks are embedded with a multilingual sentence-transformers model (ONNX, no GPU required) and stored in a local ChromaDB vector database.
2. **Retrieve** — A query is embedded the same way; the most semantically similar chunks are retrieved. Optionally refined by a cross-encoder re-ranker.
3. **Generate** — Retrieved chunks are passed to the LLM with the question; the answer always cites the page number.

**For structured documents**, two steps run before indexing:

- **Profile** — Numerical layout metrics are extracted (font distribution, line spacing, column density, header/footer zones) and used to derive adaptive parsing parameters — no hardcoded thresholds.
- **Parse** — Text elements are extracted with coordinates, grouped into lines and paragraphs, and classified into regions (`title`, `header`, `footer`, `body`). Multi-column layouts are detected and formatted as tables. Headers, footers, and artifacts are discarded before indexing.

**For plain text documents**, `--plain` mode extracts raw text page by page, skipping layout analysis entirely.

## Setup

```bash
git clone https://github.com/leonardocppn/med-rag
cd med-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your keys
```

Edit `.env`:

```
ANTHROPIC_API_KEY=your_key_here   # required for --model claude
OLLAMA_MODEL=gemma3:12b           # optional, default local model
```

For the Claude backend, get your API key at [console.anthropic.com](https://console.anthropic.com).  
For the local backend, install [Ollama](https://ollama.com) and pull a model: `ollama pull gemma3:12b`.

## LLM backends

All query commands (`ask`, `summary`, `chat`) accept a `--model` flag:

```bash
--model local    # local model via Ollama (default)
--model claude   # Claude API (Anthropic)
```

The local model is configured via `OLLAMA_MODEL` in `.env`. The default is `gemma3:12b`. You can set any model available in your Ollama installation.

## Usage

### Indexing

Index a single PDF:

```bash
python -m src.cli index data/document.pdf
```

Index multiple PDFs into a shared corpus:

```bash
python -m src.cli index data/1.pdf data/2.pdf --corpus my-corpus
```

If `--corpus` is omitted, a name is auto-assigned (e.g. `corpus_1`, `corpus_2`, ...).

You can also pass a directory — all PDFs inside it will be indexed into a single corpus:

```bash
python -m src.cli index data/ --corpus my-data
```

For narrative PDFs, use `--plain` to skip layout analysis — see [Plain text documents](#plain-text-documents).

> `index-multi --corpus <name>` is available as an explicit alias for scripting.

### Sync a directory (incremental indexing)

If you add new PDFs to a folder over time, `sync` indexes only the ones that haven't been indexed yet. Duplicates are detected by file content, not by filename.

```bash
python -m src.cli sync data/
```

Same as `index`: you can specify a corpus name after `--corpus`, otherwise one is auto-assigned. The `--plain` flag is also supported.

### Querying

Ask a question about a single PDF:

```bash
python -m src.cli ask data/1.pdf "What is the reference range for hemoglobin?"
```

Or about an entire corpus:

```bash
python -m src.cli ask my-corpus "Was I ok with vitamin D in October 2023?"
```

Use Claude instead of the local model:

```bash
python -m src.cli ask data/1.pdf "What is the reference range for hemoglobin?" --model claude
```

For more precise results, enable cross-encoder re-ranking (downloads ~100MB on first run):

```bash
python -m src.cli ask data/1.pdf "What is the reference range for hemoglobin?" --rerank
```

You can also pass a custom system prompt:

```bash
python -m src.cli ask data/1.pdf "What is the reference range for hemoglobin?" --system "You are a medical expert. Answer in technical terms."
```

> `ask-corpus` is available as an explicit alias that only targets corpora.

### Summary

Generate a structured summary of an **entire** indexed PDF:

```bash
python -m src.cli summary data/document.pdf
python -m src.cli summary data/document.pdf --model claude
```

### Chat

Start an interactive multi-turn conversation about a PDF or corpus:

```bash
python -m src.cli chat data/document.pdf
python -m src.cli chat my-corpus --model claude
```

Type `exit` to quit the session.

> `chat-corpus` is available as an explicit alias that only targets corpora.

### Deleting indexed data

Delete a single PDF's collection:

```bash
python -m src.cli delete data/document.pdf
```

Delete an entire corpus:

```bash
python -m src.cli delete-corpus my-corpus
```

Remove only one PDF from a corpus, keeping the rest:

```bash
python -m src.cli delete-corpus my-corpus --pdf data/document.pdf
```

---

## Structured documents

Commands specific to PDFs with complex layouts (tables, multi-column text, headers and footers).

### Profiling (layout analysis)

Show the measured layout metrics and derived parsing parameters for a PDF:

```bash
python -m src.cli profile data/document.pdf
```

Displays font distribution, spacing, column density, header/footer zones, and the thresholds the parser will use.

### Clustering

Group PDFs by layout similarity. Useful to understand which documents share the same structure:

```bash
python -m src.cli cluster
```

This uses all previously profiled PDFs. You can also profile and cluster specific files on the fly:

```bash
python -m src.cli cluster data/1.pdf data/2.pdf data/3.pdf
```

Optionally set a custom DBSCAN radius (0-1, default: automatic):

```bash
python -m src.cli cluster --eps 0.3
```

### Inspect (debug)

Show the raw blocks extracted by the parser, useful for debugging parsing issues:

```bash
python -m src.cli inspect data/document.pdf
```

---

## Plain text documents

For narrative PDFs where layout analysis is unnecessary or unreliable.

### Indexing in plain-text mode

Use `--plain` to extract raw text page by page, without layout or column detection:

```bash
python -m src.cli index data/document.pdf --plain
```

Works well for clinical notes, discharge letters, and general reports. Also useful for scanned PDFs where the embedded OCR text has no reliable spatial structure.

### Keyword search (via BM25)

For instant keyword-based lookup without any LLM, use `search`.<br> Works best on narrative documents rather than tabular lab results, where keyword frequency alone gives poor ranking:

```bash
python -m src.cli search data/report.pdf "atrial fibrillation"
python -m src.cli search archive "anticoagulant therapy"
```

---

## Options

| Option | Commands | Description |
|--------|----------|-------------|
| `--show` | (global) | Show verbose model loading output (suppressed by default) |
| `--model local\|claude` | `ask`, `ask-corpus`, `summary`, `chat`, `chat-corpus` | LLM backend (default: `local`) |
| `--plain` | `index`, `index-multi`, `sync` | Plain-text extraction, no layout analysis |
| `--corpus <name>` | `index`, `sync` | Corpus name (auto-assigned if omitted) |
| `--n N` | `ask`, `chat`, `search` | Number of chunks/results to retrieve (default: 5) |
| `--rerank` | `ask`, `ask-corpus` | Enable cross-encoder re-ranking |
| `--system "..."` | `ask`, `summary`, `chat` | Custom system prompt |
| `--eps N` | `cluster` | DBSCAN radius (0-1, default: automatic) |
| `--pdf <path>` | `delete-corpus` | Remove only one PDF from a corpus |

## Architecture

| Module | Responsibility |
|--------|----------------|
| `src/profiler.py` | PDF layout analysis, adaptive parameter derivation, profile persistence and DBSCAN clustering |
| `src/parser.py` | PDF text extraction, line/block grouping, column detection, region classification, artifact cleanup; plain-text extraction mode |
| `src/indexer.py` | Embedding with fastembed, ChromaDB storage, vector retrieval, cross-encoder re-ranking, BM25 keyword search |
| `src/chain.py` | RAG orchestration, dual-backend LLM streaming (Claude / Ollama), multi-turn conversation management |
| `src/cli.py` | Click-based CLI with all user-facing commands and animated spinner |

## Notes

- Works best with digitally-generated PDFs. For scanned documents, use `--plain` to skip layout analysis and rely on the raw OCR text embedded in the PDF.
- Everything stays local: embeddings and chunks are stored in `./chroma_db`. Only the final query is sent to the Anthropic API (when using `--model claude`).
- The embedding model (~50MB) is downloaded on first run from HuggingFace. The re-ranking model (~100MB) is downloaded only if `--rerank` is used.
- Layout profiling is fully adaptive: every parameter is derived from the PDF's own metrics, with no hardcoded document types.
- Model loading output is suppressed by default for a cleaner experience. Use `--show` to see it.
