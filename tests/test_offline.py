"""Offline tests: no network, no API key, no OpenAI calls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from app import config, rag


def test_citation_extraction():
    text = "The revenue grew 10% in 2023 (page 3) and again in Q4 (Page 12)."
    assert rag._extract_page_citations(text) == [3, 12]


def test_citation_extraction_empty():
    assert rag._extract_page_citations("No citations here.") == []


def test_format_docs_keeps_page_numbers():
    docs = [
        Document(page_content="alpha", metadata={"page": 0, "source": "a.pdf"}),
        Document(page_content="beta", metadata={"page": 4, "source": "b.pdf"}),
    ]
    formatted = rag._format_docs(docs)
    assert "[a.pdf | page 1]" in formatted
    assert "[b.pdf | page 5]" in formatted
    assert "alpha" in formatted and "beta" in formatted


def test_splitter_chunks_with_overlap():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
    )
    text = "word " * 2000  # ~10,000 chars
    chunks = splitter.split_documents(
        [Document(page_content=text, metadata={"page": 0, "source": "x"})]
    )
    assert len(chunks) > 1
    assert all(len(c.page_content) <= config.CHUNK_SIZE * 1.05 for c in chunks)
    # Overlap: consecutive chunks share some text
    assert chunks[0].page_content[-20:] != chunks[1].page_content[:20]


def test_config_paths_exist():
    assert config.UPLOAD_DIR.exists()
    assert config.CHROMA_DIR.exists()
