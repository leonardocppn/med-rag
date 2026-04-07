"""
indexer.py

Takes chunks produced by the parser, converts them to embeddings
with sentence-transformers, and stores them in a local ChromaDB database.

Each document (PDF) gets its own ChromaDB collection, identified by filename.
Multiple PDFs can also be grouped into a named corpus collection.
"""

import hashlib
import chromadb
from fastembed import TextEmbedding

# Lightweight multilingual model, ONNX (no CUDA required, ~50MB)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Cross-encoder model for re-ranking (multilingual, ~100MB)
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_model: TextEmbedding | None = None
_rerank_model = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        print(f"Loading embedding model {MODEL_NAME} (first run, ~50MB)...")
        _model = TextEmbedding(MODEL_NAME)
    return _model


def _get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder
        print("Loading re-ranking model (first run, ~100MB)...")
        _rerank_model = CrossEncoder(RERANK_MODEL)
    return _rerank_model


def _collection_name(pdf_path: str) -> str:
    """Collection name = hash of the path (Chroma requires simple names)."""
    h = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
    import os
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    # Chroma accepts only [a-zA-Z0-9_-], max 63 chars
    safe = "".join(c if c.isalnum() else "_" for c in base)[:40]
    return f"{safe}_{h}"


def corpus_collection_name(corpus_name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in corpus_name)[:55]
    return f"corpus_{safe}"


def index_chunks(chunks: list[dict], pdf_path: str,
                 db_path: str = "./chroma_db") -> str:
    """
    Indexes chunks into the ChromaDB database.
    Returns the name of the created collection.
    If the collection already exists, it is replaced.
    """
    model = _get_model()
    client = chromadb.PersistentClient(path=db_path)
    col_name = _collection_name(pdf_path)

    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    collection = client.create_collection(col_name)

    texts = [c["text"] for c in chunks]
    embeddings = list(model.embed(texts))

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"page": c["page"], "region": c["region"]}
                   for c in chunks],
    )

    print(f"Indexed {len(chunks)} chunks → collection '{col_name}'")
    return col_name


def index_chunks_to_corpus(chunks: list[dict], corpus_name: str,
                            pdf_path: str,
                            db_path: str = "./chroma_db") -> str:
    """
    Adds chunks from a PDF to a shared multi-document corpus.
    If the PDF was already in the corpus, its chunks are replaced (idempotent).
    Returns the name of the corpus collection.
    """
    model = _get_model()
    client = chromadb.PersistentClient(path=db_path)
    col_name = corpus_collection_name(corpus_name)

    try:
        collection = client.get_collection(col_name)
    except Exception:
        collection = client.create_collection(col_name)

    # Remove existing chunks from this PDF before re-indexing
    pdf_id = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
    try:
        existing = collection.get(where={"pdf_id": pdf_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    texts = [c["text"] for c in chunks]
    embeddings = list(model.embed(texts))

    collection.add(
        ids=[f"{pdf_id}_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "page": c["page"],
            "region": c["region"],
            "pdf_path": pdf_path,
            "pdf_id": pdf_id,
        } for c in chunks],
    )

    print(f"Indexed {len(chunks)} chunks from '{pdf_path}' → corpus '{col_name}'")
    return col_name


def retrieve(query: str, col_name: str,
             db_path: str = "./chroma_db",
             n_results: int = 5) -> list[dict]:
    """Returns the most relevant chunks for a query, with metadata."""
    model = _get_model()
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(col_name)

    query_embedding = list(model.embed([query]))
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "page": meta["page"],
            "region": meta["region"],
            "score": round(1 - dist, 3),
        })

    return chunks


def retrieve_from_corpus(query: str, corpus_name: str,
                          db_path: str = "./chroma_db",
                          n_results: int = 5) -> list[dict]:
    """Retrieves chunks from a multi-document corpus."""
    model = _get_model()
    client = chromadb.PersistentClient(path=db_path)
    col_name = corpus_collection_name(corpus_name)
    collection = client.get_collection(col_name)

    query_embedding = list(model.embed([query]))
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "page": meta["page"],
            "region": meta["region"],
            "pdf_path": meta.get("pdf_path", "unknown"),
            "score": round(1 - dist, 3),
        })

    return chunks


def retrieve_all(col_name: str, db_path: str = "./chroma_db") -> list[dict]:
    """Returns all chunks in a collection, sorted by page number."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(col_name)
    results = collection.get(include=["documents", "metadatas"])
    chunks = [
        {"text": doc, "page": meta["page"], "region": meta["region"]}
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]
    return sorted(chunks, key=lambda c: c["page"])


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """
    Re-ranks chunks with a multilingual cross-encoder.
    Retrieves more candidates than needed, then selects the best ones.
    """
    model = _get_rerank_model()
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]
