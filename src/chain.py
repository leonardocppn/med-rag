"""
chain.py

Takes chunks retrieved from the indexer and passes them to an LLM
(Claude API or local Ollama) to answer questions grounded in the
document content.
"""

import os
import time
from typing import Iterator
import ollama
import anthropic
from .indexer import retrieve, retrieve_all, retrieve_all_corpus, retrieve_from_corpus, rerank as rerank_chunks, corpus_collection_name

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:12b")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# Sentinels emitted by the stream; the CLI uses them to update the spinner.
PHASE_LLM = "\x00LLM\x00"          # retrieval done, LLM is generating
PHASE_PROGRESS = "\x00PROGRESS\x00" # progress message follows (map-reduce batches)

# Map-reduce parameters for large documents with the Claude backend.
# 160k chars ≈ 40-50k tokens, just under the 50k TPM rate limit.
_BATCH_CHARS = 160_000
_BATCH_PAUSE_SECS = 70  # pause between batches to let the rate-limit window roll over


def _build_context(chunks: list[dict], with_source: bool = False) -> str:
    parts = []
    for c in chunks:
        source = f"[Page {c['page']}, {c['region']}]"
        if with_source and "pdf_path" in c:
            import os
            source += f" [{os.path.basename(c['pdf_path'])}]"
        parts.append(f"{source}\n{c['text']}")
    return "\n\n".join(parts)


def _call_ollama(messages: list[dict], system_prompt: str | None = None) -> Iterator[str]:
    """Calls Ollama in streaming and yields tokens."""
    ollama_messages = []
    if system_prompt:
        ollama_messages.append({"role": "system", "content": system_prompt})
    ollama_messages.extend(messages)

    stream = ollama.chat(model=OLLAMA_MODEL, messages=ollama_messages, stream=True)
    for chunk in stream:
        text = chunk["message"]["content"]
        if text:
            yield text


def _call_claude(messages: list[dict], system_prompt: str | None = None,
                 max_tokens: int = 1024) -> Iterator[str]:
    """Calls Claude API in streaming and yields tokens."""
    client = anthropic.Anthropic()
    kwargs: dict = dict(model=CLAUDE_MODEL, max_tokens=max_tokens, messages=messages)
    if system_prompt:
        kwargs["system"] = system_prompt
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text


def _call_llm(messages: list[dict], system_prompt: str | None,
              backend: str, max_tokens: int = 1024) -> Iterator[str]:
    if backend == "claude":
        yield from _call_claude(messages, system_prompt, max_tokens=max_tokens)
    else:
        yield from _call_ollama(messages, system_prompt)


def ask_stream(query: str, col_name: str,
               db_path: str = "./chroma_db",
               n_results: int = 5,
               use_rerank: bool = False,
               system_prompt: str | None = None,
               backend: str = "local") -> Iterator[str]:
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
    yield PHASE_LLM
    yield from _call_llm([{"role": "user", "content": user_message}], system_prompt, backend)


def _summarize_batch_sync(batch_text: str, max_tokens: int = 2048) -> str:
    """Calls Claude synchronously to summarize one batch. Raises on error."""
    client = anthropic.Anthropic()
    msg = (
        "Summarize the following section of a document. "
        "Be thorough and preserve key facts, values, and dates:\n\n" + batch_text
    )
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": msg}],
    )
    return response.content[0].text


def _summarize_mapreduce_claude(chunks: list[dict],
                                system_prompt: str | None) -> Iterator[str]:
    """Map-reduce summarization for large documents via the Claude backend.

    Splits the full context into batches of ~160k characters, summarizes each
    one synchronously (respecting the 50k TPM rate limit with a pause between
    batches), then streams a final unified synthesis.
    """
    context = _build_context(chunks)

    # Fast path: fits in a single call
    if len(context) <= _BATCH_CHARS:
        user_message = f"Document:\n\n{context}\n\nGenerate a structured summary."
        yield PHASE_LLM
        yield from _call_claude([{"role": "user", "content": user_message}],
                                system_prompt, max_tokens=4096)
        return

    # Split into batches
    batches = [context[i:i + _BATCH_CHARS] for i in range(0, len(context), _BATCH_CHARS)]
    n = len(batches)
    estimated_secs = n * 30 + (n - 1) * _BATCH_PAUSE_SECS
    estimated_min = max(1, estimated_secs // 60)
    yield f"{PHASE_PROGRESS}Large document — {n} batches (~{estimated_min} min estimated)"

    partial_summaries: list[str] = []
    for i, batch in enumerate(batches):
        yield f"{PHASE_PROGRESS}Batch {i + 1}/{n}: summarizing..."

        # Retry with exponential backoff on rate-limit errors
        for attempt in range(4):
            try:
                partial = _summarize_batch_sync(batch)
                partial_summaries.append(partial)
                break
            except anthropic.RateLimitError:
                if attempt == 3:
                    raise
                wait = _BATCH_PAUSE_SECS * (2 ** attempt)
                for remaining in range(wait, 0, -10):
                    yield f"{PHASE_PROGRESS}Rate limit — retrying in {remaining}s..."
                    time.sleep(min(10, remaining))

        if i < n - 1:
            for remaining in range(_BATCH_PAUSE_SECS, 0, -10):
                yield f"{PHASE_PROGRESS}Batch {i + 1}/{n} done — waiting {remaining}s..."
                time.sleep(min(10, remaining))

    # Final synthesis (streamed)
    combined = "\n\n---\n\n".join(
        f"[Section {i + 1}]\n{s}" for i, s in enumerate(partial_summaries)
    )
    final_msg = (
        "The following are summaries of consecutive sections of the same document. "
        "Produce a single unified structured summary:\n\n" + combined
    )
    yield f"{PHASE_PROGRESS}Synthesizing {n} sections..."
    yield PHASE_LLM
    yield from _call_claude([{"role": "user", "content": final_msg}],
                            system_prompt, max_tokens=4096)


def summarize_stream(col_name: str,
                     db_path: str = "./chroma_db",
                     system_prompt: str | None = None,
                     backend: str = "local") -> Iterator[str]:
    """Generates a structured summary of the entire document as a stream.

    For the Claude backend, uses map-reduce if the document exceeds the
    rate-limit window (~160k chars per batch, 70s pause between batches).
    """
    chunks = retrieve_all(col_name, db_path=db_path)

    if not chunks:
        yield "No content found in the document."
        return

    if backend == "claude":
        yield from _summarize_mapreduce_claude(chunks, system_prompt)
        return

    context = _build_context(chunks)
    user_message = f"Document:\n\n{context}\n\nGenerate a structured summary."
    yield PHASE_LLM
    yield from _call_llm([{"role": "user", "content": user_message}], system_prompt, backend,
                         max_tokens=4096)


def summarize_corpus_stream(corpus_name: str,
                            db_path: str = "./chroma_db",
                            system_prompt: str | None = None,
                            backend: str = "local") -> Iterator[str]:
    """Generates a structured summary of an entire multi-document corpus as a stream."""
    chunks = retrieve_all_corpus(corpus_name, db_path=db_path)

    if not chunks:
        yield "No content found in the corpus."
        return

    if backend == "claude":
        yield from _summarize_mapreduce_claude(chunks, system_prompt)
        return

    context = _build_context(chunks, with_source=True)
    user_message = f"Document corpus:\n\n{context}\n\nGenerate a structured summary."
    yield PHASE_LLM
    yield from _call_llm([{"role": "user", "content": user_message}], system_prompt, backend,
                         max_tokens=4096)


def chat_stream(messages: list[dict], col_name: str,
                db_path: str = "./chroma_db",
                n_results: int = 5,
                system_prompt: str | None = None,
                backend: str = "local") -> Iterator[str]:
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
    augmented_messages = messages[:-1] + [{"role": "user", "content": augmented_user_message}]
    yield PHASE_LLM
    yield from _call_llm(augmented_messages, system_prompt, backend)


def chat_corpus_stream(messages: list[dict], corpus_name: str,
                       db_path: str = "./chroma_db",
                       n_results: int = 5,
                       system_prompt: str | None = None,
                       backend: str = "local") -> Iterator[str]:
    """
    Multi-turn chat over a multi-document corpus.

    messages: list of {"role": "user"|"assistant", "content": str}
    The last message must have role "user".
    """
    query = messages[-1]["content"]
    chunks = retrieve_from_corpus(query, corpus_name, db_path=db_path, n_results=n_results)

    if not chunks:
        yield "No relevant content found in the corpus."
        return

    context = _build_context(chunks, with_source=True)
    augmented_user_message = f"Excerpts from the corpus:\n\n{context}\n\n---\nQuestion: {query}"
    augmented_messages = messages[:-1] + [{"role": "user", "content": augmented_user_message}]
    yield PHASE_LLM
    yield from _call_llm(augmented_messages, system_prompt, backend)


def ask_corpus_stream(query: str, corpus_name: str,
                      db_path: str = "./chroma_db",
                      n_results: int = 5,
                      use_rerank: bool = False,
                      system_prompt: str | None = None,
                      backend: str = "local") -> Iterator[str]:
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
    yield PHASE_LLM
    yield from _call_llm([{"role": "user", "content": user_message}], system_prompt, backend)
