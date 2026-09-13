# med-rag

A RAG pipeline that answers questions in natural language about an archive of PDFs, with the language model running locally through Ollama or remotely through the Claude API. It was written to search years of personal medical reports, and nothing in it is specific to that use.

Two kinds of document are handled. PDFs with tables, multiple columns, headers and footers go through a layout analysis, where every parsing threshold is derived from metrics measured on the document itself. Narrative PDFs such as clinical notes, discharge letters and radiology reports are extracted page by page in `--plain` mode, which skips the layout work.

Built with Python, pdfplumber, sentence-transformers, ChromaDB, the Anthropic Claude API and Ollama.

---

## How it works

Three steps are shared by every document:

1. Index. Text blocks are embedded with a multilingual sentence-transformers model (ONNX, no GPU required) and stored in a local ChromaDB vector database.
2. Retrieve. The question is embedded the same way, and the closest chunks come back. A cross-encoder can re-rank them when precision matters more than speed.
3. Generate. The retrieved chunks reach the model together with the question. Each one carries the page it came from, so the answer can point back to it.

Documents with a structured layout get two more steps before indexing. Profiling measures font distribution, line spacing, column density and the extent of the header and footer zones, and turns those numbers into parsing parameters for that specific document. A scanned document betrays itself in those same numbers, since OCR noise on the word coordinates reads as very high column density and dozens of font size levels, and it gets a parameter set of its own: column and title detection off, and the footer kept, because in a scanned book the footnotes hold the citations. Parsing then extracts every text element with its coordinates, groups words into lines and lines into blocks, classifies each block as `title`, `header`, `footer`, `table` or `body`, detects multi-column stretches and formats them as tables, and drops headers, footers and artifacts before indexing.

In `--plain` mode none of that runs: the page text is taken as pdfplumber returns it.

## Setup

```bash
git clone https://github.com/leonardocppn/med-rag
cd med-rag
python -m venv venv && source venv/bin/activate
pip install -e .
cp .env.example .env
```

Python 3.10 or later. The install puts a `medrag` command in the virtual environment, so every example below works from anywhere as long as that environment is active. Then fill in `.env`:

```
ANTHROPIC_API_KEY=your_key_here   # required for --model claude
OLLAMA_MODEL=gemma3:12b           # optional, default local model
```

For the Claude backend, get an API key at [console.anthropic.com](https://console.anthropic.com). For the local backend, install [Ollama](https://ollama.com) and pull a model with `ollama pull gemma3:12b`.

## LLM backends

`ask`, `ask-corpus`, `summary`, `chat` and `chat-corpus` all take a `--model` flag:

```bash
--model local    # local model via Ollama (default)
--model claude   # Claude API (Anthropic)
```

The local model is whatever `OLLAMA_MODEL` names in `.env`, `gemma3:12b` by default, and any model your Ollama installation has will do.

## Usage

### Indexing

One PDF:

```bash
medrag index data/document.pdf
```

Several PDFs in a shared corpus:

```bash
medrag index data/1.pdf data/2.pdf --corpus my-corpus
```

Without `--corpus` a name is assigned automatically (`corpus_1`, `corpus_2`, and so on). A directory works too, and every PDF inside it goes into a single corpus:

```bash
medrag index data/ --corpus my-data
```

For narrative PDFs add `--plain`, described under [Plain text documents](#plain-text-documents).

> `index-multi --corpus <name>` exists as an explicit alias for scripting.

### Sync a directory (incremental indexing)

When PDFs keep arriving in a folder, `sync` indexes only the ones that are not in the corpus yet, recognising them by the hash of their content rather than by filename.

```bash
medrag sync data/
```

It takes `--corpus` and `--plain` like `index`.

### Querying

About a single PDF:

```bash
medrag ask data/1.pdf "What is the reference range for hemoglobin?"
```

About a whole corpus:

```bash
medrag ask my-corpus "Was I ok with vitamin D in October 2023?"
```

Against Claude instead of the local model:

```bash
medrag ask data/1.pdf "What is the reference range for hemoglobin?" --model claude
```

With cross-encoder re-ranking, which downloads about 100MB the first time:

```bash
medrag ask data/1.pdf "What is the reference range for hemoglobin?" --rerank
```

With a system prompt of your own:

```bash
medrag ask data/1.pdf "What is the reference range for hemoglobin?" --system "You are a medical expert. Answer in technical terms."
```

> `ask-corpus` exists as an explicit alias that only accepts corpora.

### Summary

A structured summary of an indexed PDF or corpus:

```bash
medrag summary data/document.pdf
medrag summary my-corpus --model claude
```

On the Claude backend a document too large for one request is summarised in batches, each pass feeding the next.

### Chat

An interactive session, with the conversation kept across turns:

```bash
medrag chat data/document.pdf
medrag chat my-corpus --model claude
```

Type `exit` to leave.

> `chat-corpus` exists as an explicit alias that only accepts corpora.

### Deleting indexed data

```bash
medrag delete data/document.pdf
medrag delete-corpus my-corpus
medrag delete-corpus my-corpus --pdf data/document.pdf
```

The last form drops one PDF from a corpus and leaves the rest indexed.

---

## Structured documents

These commands are for PDFs with complex layouts.

### Profiling

```bash
medrag profile data/document.pdf
```

Shows the measured metrics, font distribution, spacing, column density, header and footer zones, next to the thresholds the parser derives from them.

### Clustering

Groups PDFs by layout similarity, which tells you which documents share a structure:

```bash
medrag cluster
```

With no arguments it uses every PDF profiled so far. Named files are profiled on the spot:

```bash
medrag cluster data/1.pdf data/2.pdf data/3.pdf
```

The DBSCAN radius is automatic and can be set by hand:

```bash
medrag cluster --eps 0.3
```

### Inspect

```bash
medrag inspect data/document.pdf
```

Prints the raw blocks the parser extracted, for when the parsing goes wrong and you need to see where.

---

## Plain text documents

### Indexing in plain-text mode

`--plain` takes the text page by page, with no layout or column detection:

```bash
medrag index data/document.pdf --plain
```

This suits clinical notes, discharge letters and general reports, and also scanned PDFs, where the OCR text carries no reliable spatial structure.

### Keyword search (BM25)

`search` looks up keywords without calling any model, and answers instantly. It works on narrative documents; on tabular lab results, where a term repeats across every panel, the ranking is poor.

```bash
medrag search data/report.pdf "atrial fibrillation"
medrag search archive "anticoagulant therapy"
```

---

## Options

| Option | Commands | Description |
|--------|----------|-------------|
| `--show` | (global) | Show the verbose model loading output, suppressed by default |
| `--model local\|claude` | `ask`, `ask-corpus`, `summary`, `chat`, `chat-corpus` | LLM backend (default: `local`) |
| `--plain` | `index`, `index-multi`, `sync` | Plain-text extraction, no layout analysis |
| `--corpus <name>` | `index`, `index-multi`, `sync` | Corpus name (assigned automatically if omitted, required by `index-multi`) |
| `--n N` | `ask`, `ask-corpus`, `chat`, `chat-corpus`, `search` | Number of chunks or results to retrieve (default: 5) |
| `--rerank` | `ask`, `ask-corpus` | Cross-encoder re-ranking |
| `--system "..."` | `ask`, `ask-corpus`, `summary`, `chat`, `chat-corpus` | Custom system prompt |
| `--eps N` | `cluster` | DBSCAN radius, 0 to 1 (default: automatic) |
| `--pdf <path>` | `delete-corpus` | Remove a single PDF from a corpus |

## Architecture

| Module | Responsibility |
|--------|----------------|
| `medrag/profiler.py` | Layout analysis, adaptive parameter derivation, profile persistence, DBSCAN clustering |
| `medrag/parser.py` | Text extraction with coordinates, line and block grouping, column detection, region classification, artifact cleanup, plain-text mode |
| `medrag/indexer.py` | Embedding with fastembed, ChromaDB storage, vector retrieval, cross-encoder re-ranking, BM25 keyword search |
| `medrag/chain.py` | RAG orchestration, streaming from either backend, batched summaries, multi-turn conversation |
| `medrag/cli.py` | Click commands and terminal output |

## Notes

- Digitally generated PDFs work best. Scans are recognised by the profiler and parsed with thresholds of their own, and `--plain` is there for when you would rather have no layout analysis at all.
- Embeddings and chunks stay in `./chroma_db` on your machine. With `--model claude` the retrieved excerpts and the question go to the Anthropic API; with `--model local` nothing leaves the machine.
- The embedding model (about 50MB) is downloaded from HuggingFace on first run. The re-ranking model (about 100MB) only when `--rerank` is used.
- Model loading output is hidden by default, and `--show` brings it back.

## License

MIT, see [LICENSE](LICENSE).
