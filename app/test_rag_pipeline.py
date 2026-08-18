from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.ragfunc import chunk_documents, fixed_size_chunks, load_document, recursive_chunks


def test_fixed_size_chunks_preserve_source_metadata_and_overlap():
    chunks = fixed_size_chunks([Document(page_content="abcdefghij", metadata={"page": 2})], chunk_size=5, chunk_overlap=2)

    assert [chunk.page_content for chunk in chunks] == ["abcde", "defgh", "ghij"]
    assert [chunk.metadata["offset"] for chunk in chunks] == [0, 3, 6]
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)


def test_recursive_strategy_returns_documents():
    chunks = recursive_chunks([Document(page_content="One paragraph.\n\nAnother paragraph.", metadata={"page": 1})], chunk_size=20, chunk_overlap=0)

    assert len(chunks) >= 2
    assert all(chunk.metadata["page"] == 1 for chunk in chunks)


def test_unknown_chunking_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_documents([Document(page_content="facts")], strategy="invalid")  # type: ignore[arg-type]


def test_load_document_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.pdf")
