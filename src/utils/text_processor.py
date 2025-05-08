import tiktoken
from typing import List, Dict, Any, Union, Tuple


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text using the cl100k_base encoding.

    Args:
        text: The input text to tokenize

    Returns:
        The number of tokens in the text
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)


def chunk_text(text: str, max_tokens: int = 8000) -> List[str]:
    """
    Split text into chunks that don't exceed the maximum token limit.

    Args:
        text: The input text to chunk
        max_tokens: Maximum number of tokens per chunk (default: 8000)

    Returns:
        A list of text chunks
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    current_chunk_tokens = []

    for token in tokens:
        current_chunk_tokens.append(token)

        if len(current_chunk_tokens) >= max_tokens:
            chunk_text = encoding.decode(current_chunk_tokens)
            chunks.append(chunk_text)
            current_chunk_tokens = []

    # Add any remaining tokens
    if current_chunk_tokens:
        chunk_text = encoding.decode(current_chunk_tokens)
        chunks.append(chunk_text)

    return chunks


def chunk_document(
    document: Dict[str, Any], text_key: str, max_tokens: int = 8000
) -> List[Dict[str, Any]]:
    """
    Split a document into chunks based on token count while preserving metadata.

    Args:
        document: The document containing text and metadata
        text_key: The key in the document that contains the text to chunk
        max_tokens: Maximum number of tokens per chunk (default: 8000)

    Returns:
        A list of document chunks with preserved metadata
    """
    if text_key not in document:
        raise KeyError(f"Text key '{text_key}' not found in document")

    text = document[text_key]
    text_chunks = chunk_text(text, max_tokens)

    # If text doesn't need chunking, return the original document
    if len(text_chunks) == 1:
        return [document]

    # Create chunks with preserved metadata
    document_chunks = []
    for i, chunk in enumerate(text_chunks):
        chunk_doc = document.copy()
        chunk_doc[text_key] = chunk
        chunk_doc["chunk_id"] = i
        chunk_doc["total_chunks"] = len(text_chunks)
        document_chunks.append(chunk_doc)

    return document_chunks
