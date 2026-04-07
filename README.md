# med-rag

A RAG pipeline for querying PDF documents in natural language, powered by Claude.

Built with: Python · pdfplumber · sentence-transformers · ChromaDB · Anthropic Claude API

---

## How it works

1. **Parse** — `pdfplumber` extracts every text element from the PDF with its coordinates (position, font size). Elements are grouped into lines, then into paragraphs, then classified into regions: `title`, `header`, `footer`, `body`. Headers and footers are discarded before indexing.
2. **Index** — Each text block is embedded with a multilingual sentence-transformers model (ONNX, no GPU required) and stored in a local ChromaDB vector database.
3. **Retrieve** — A query is embedded the same way; the most semantically similar chunks are retrieved. Optionally, a cross-encoder re-ranker refines the results.
4. **Generate** — Retrieved chunks are passed to Claude with the question; the answer always cites the page number.

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

## Notes

- Works best with digitally-generated PDFs. Scanned PDFs require OCR pre-processing (**not included**).
- Everything stays local: embeddings and chunks are stored in `./chroma_db`. Only the final query is sent to the Anthropic API.
- The embedding model (~50MB) is downloaded on first run from HuggingFace. The re-ranking model (~100MB) is downloaded only if `--rerank` is used.
