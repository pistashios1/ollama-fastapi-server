"""Ingestion, chunking and retrieval helpers for the RAG application."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ChunkingStrategy = Literal["fixed", "recursive", "semantic"]


def load_document(path: str | Path) -> list[Document]:
    """Load a UTF-8 text file or every text-bearing page of a PDF."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Document not found: {source}")

    if source.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        pages = PdfReader(str(source)).pages
        documents = [
            Document(page_content=page.extract_text() or "", metadata={"source": str(source), "page": number})
            for number, page in enumerate(pages, start=1)
        ]
        documents = [document for document in documents if document.page_content.strip()]
        if not documents:
            raise ValueError(f"No extractable text found in PDF: {source}")
        return documents

    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Document is empty: {source}")
    return [Document(page_content=text, metadata={"source": str(source), "page": 1})]


def fixed_size_chunks(documents: list[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> list[Document]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be in [0, chunk_size)")
    step = chunk_size - chunk_overlap
    result: list[Document] = []
    for document in documents:
        for offset in range(0, len(document.page_content), step):
            text = document.page_content[offset : offset + chunk_size]
            if text.strip():
                result.append(Document(page_content=text, metadata={**document.metadata, "offset": offset}))
    return result


def recursive_chunks(documents: list[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def semantic_chunks(documents: list[Document], embeddings: OllamaEmbeddings, similarity_threshold: float = 0.72) -> list[Document]:
    """Join adjacent sentences whose embeddings are sufficiently similar."""
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")
    result: list[Document] = []
    for document in documents:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", document.page_content) if sentence.strip()]
        if not sentences:
            continue
        vectors = embeddings.embed_documents(sentences)
        current = [sentences[0]]
        for sentence, prior, current_vector in zip(sentences[1:], vectors, vectors[1:]):
            dot = sum(left * right for left, right in zip(prior, current_vector))
            prior_norm = sum(value * value for value in prior) ** 0.5
            current_norm = sum(value * value for value in current_vector) ** 0.5
            similarity = dot / (prior_norm * current_norm) if prior_norm and current_norm else 0.0
            if similarity >= similarity_threshold:
                current.append(sentence)
            else:
                result.append(Document(page_content=" ".join(current), metadata=document.metadata.copy()))
                current = [sentence]
        result.append(Document(page_content=" ".join(current), metadata=document.metadata.copy()))
    return result


def chunk_documents(
    documents: list[Document], strategy: str = "recursive", chunk_size: int = 800, chunk_overlap: int = 100,
    embeddings: OllamaEmbeddings | None = None,
) -> list[Document]:
    if strategy == "fixed":
        return fixed_size_chunks(documents, chunk_size, chunk_overlap)
    if strategy == "recursive":
        return recursive_chunks(documents, chunk_size, chunk_overlap)
    if strategy == "semantic":
        if embeddings is None:
            raise ValueError("semantic chunking requires embeddings")
        return semantic_chunks(documents, embeddings)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
