"""Question answering over the indexed PDFs using a RAG pipeline."""
import re

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from app import config
from app.ingest import get_vectorstore

PROMPT_TEMPLATE = """You are a precise assistant that answers questions strictly from the provided document excerpts.

Rules:
- Answer only from the excerpts below. Do not use outside knowledge.
- If the excerpts do not contain the answer, say you don't know and suggest what to look for.
- Cite the pages you used, like (page 3).

Excerpts:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)


def _format_docs(docs):
    """Join retrieved chunks, keeping page numbers for inline citations."""
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{source} | page {int(page) + 1}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _extract_page_citations(answer: str) -> list[int]:
    """Pull page numbers mentioned in the answer like (page 3)."""
    return sorted({int(m) for m in re.findall(r"page\s+(\d+)", answer, flags=re.I)})


def ask(
    question: str,
    collection: str = "pdf_docs",
    k: int | None = None,
) -> dict:
    """Answer a question from the indexed PDFs.

    Returns a dict with the answer text, cited pages, and the retrieved chunks.
    """
    k = k or config.RETRIEVER_K
    vectorstore: Chroma = get_vectorstore(collection)

    if vectorstore._collection.count() == 0:
        raise RuntimeError("No documents indexed yet. Upload a PDF first.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = ChatOpenAI(
        model=config.CHAT_MODEL,
        temperature=0,
        **config.chat_credentials(),
    )

    chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)

    return {
        "answer": answer,
        "cited_pages": _extract_page_citations(answer),
        "source_docs": source_docs,
    }
