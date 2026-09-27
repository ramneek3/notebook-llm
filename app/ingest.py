"""Ingest PDFs: load -> chunk -> embed -> store in a persistent Chroma collection."""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config


def get_embeddings():
    """Return the embedding model chosen by EMBEDDING_BACKEND.

    - "local": FastEmbed (ONNX, torch-free) — free, offline, private,
      and small enough for 1GB-RAM hosts like Streamlit Community Cloud.
    - "openai": text-embedding-3-small — needs OPENAI_API_KEY + credits.
    """
    if config.EMBEDDING_BACKEND == "openai":
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.require_api_key(),
        )
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)


def get_vectorstore(collection: str = "pdf_docs") -> Chroma:
    """Open (creating if needed) the persistent Chroma vector store."""
    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def ingest_pdf(pdf_path: str | Path, collection: str = "pdf_docs") -> dict:
    """Load a PDF, split it into chunks, and index it in Chroma.

    Returns basic stats about the ingestion.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"No such file: {pdf_path}")

    # 1. Load page-by-page (keeps page numbers for source citations)
    pages = PyPDFLoader(str(pdf_path)).load()

    # 2. Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,  # store chunk position in metadata
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError(f"No extractable text found in {pdf_path.name} (scanned image PDF?)")

    # 3. Tag every chunk with its source document
    source = pdf_path.name
    for chunk in chunks:
        chunk.metadata["source"] = source

    # 4. Embed + persist in Chroma
    vectorstore = get_vectorstore(collection)
    ids = vectorstore.add_documents(chunks)

    return {
        "source": source,
        "pages": len(pages),
        "chunks": len(chunks),
        "stored_ids": len(ids),
    }


def reset_collection(collection: str = "pdf_docs") -> None:
    """Delete every document in the given collection."""
    get_vectorstore(collection).delete_collection()
