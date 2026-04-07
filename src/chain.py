"""
chain.py

Takes chunks retrieved from the indexer and passes them to Claude
to answer questions grounded in the document content.
"""

from typing import Iterator
import anthropic
from .indexer import retrieve, retrieve_all, retrieve_from_corpus, rerank as rerank_chunks


def _build_context(chunks: list[dict], with_source: bool = False) -> str:
    parts = []
    for c in chunks:
        source = f"[Page {c['page']}, {c['region']}]"
        if with_source and "pdf_path" in c:
            import os
            source += f" [{os.path.basename(c['pdf_path'])}]"
        parts.append(f"{source}\n{c['text']}")
    return "\n\n".join(parts)


def ask_stream(query: str, col_name: str,
               db_path: str = "./chroma_db",
               n_results: int = 5,
               use_rerank: bool = False,
               system_prompt: str | None = None) -> Iterator[str]:
    """Runs the RAG pipeline and returns the answer as a stream."""
    if use_rerank:
        candidates = retrieve(query, col_name, db_path=db_path, n_results=n_results * 3)
        chunks = rerank_chunks(query, candidates, top_k=n_results)
    else:
        chunks = retrieve(query, col_name, db_path=db_path, n_results=n_results)

    if not chunks:
        yield "No relevant content found in the document."
        return

    context = _build_context(chunks)
    user_message = f"Excerpts from the document:\n\n{context}\n\n---\nQuestion: {query}"

    client = anthropic.Anthropic()
    kwargs = dict(model="claude-haiku-4-5-20251001", max_tokens=1024,
                  messages=[{"role": "user", "content": user_message}])
    if system_prompt:
        kwargs["system"] = system_prompt
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text


def summarize_stream(col_name: str,
                     db_path: str = "./chroma_db",
                     system_prompt: str | None = None) -> Iterator[str]:
    """Generates a structured summary of the entire document as a stream."""
    chunks = retrieve_all(col_name, db_path=db_path)

    if not chunks:
        yield "No content found in the document."
        return

    context = _build_context(chunks)
    client = anthropic.Anthropic()
    kwargs = dict(model="claude-haiku-4-5-20251001", max_tokens=2048,
                  messages=[{"role": "user", "content": f"Document:\n\n{context}\n\nGenerate a structured summary."}])
    if system_prompt:
        kwargs["system"] = system_prompt
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text


def chat_stream(messages: list[dict], col_name: str,
                db_path: str = "./chroma_db",
                n_results: int = 5,
                system_prompt: str | None = None) -> Iterator[str]:
    """
    Multi-turn chat: retrieves chunks based on the latest question
    and maintains the full conversation history.

    messages: list of {"role": "user"|"assistant", "content": str}
    The last message must have role "user".
    """
    query = messages[-1]["content"]
    chunks = retrieve(query, col_name, db_path=db_path, n_results=n_results)

    if not chunks:
        yield "No relevant content found in the document."
        return

    context = _build_context(chunks)
    augmented_user_message = f"Excerpts from the document:\n\n{context}\n\n---\nQuestion: {query}"

    # Replace the last user message with the context-enriched version
    augmented_messages = messages[:-1] + [{"role": "user", "content": augmented_user_message}]

    client = anthropic.Anthropic()
    kwargs = dict(model="claude-haiku-4-5-20251001", max_tokens=1024,
                  messages=augmented_messages)
    if system_prompt:
        kwargs["system"] = system_prompt
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text


def ask_corpus_stream(query: str, corpus_name: str,
                      db_path: str = "./chroma_db",
                      n_results: int = 5,
                      use_rerank: bool = False,
                      system_prompt: str | None = None) -> Iterator[str]:
    """Runs RAG over a multi-document corpus and streams the answer."""
    if use_rerank:
        candidates = retrieve_from_corpus(query, corpus_name, db_path=db_path, n_results=n_results * 3)
        chunks = rerank_chunks(query, candidates, top_k=n_results)
    else:
        chunks = retrieve_from_corpus(query, corpus_name, db_path=db_path, n_results=n_results)

    if not chunks:
        yield "No relevant content found in the corpus."
        return

    context = _build_context(chunks, with_source=True)
    user_message = f"Excerpts from the corpus:\n\n{context}\n\n---\nQuestion: {query}"

    client = anthropic.Anthropic()
    kwargs = dict(model="claude-haiku-4-5-20251001", max_tokens=1024,
                  messages=[{"role": "user", "content": user_message}])
    if system_prompt:
        kwargs["system"] = system_prompt
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text
