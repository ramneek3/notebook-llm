---
title: PDF Q&A
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.64.0
app_file: app/streamlit_app.py
pinned: false
---

# 📄 PDF Q&A

Upload a PDF, ask questions about it, and get answers grounded in the document — with page citations.

**Stack:** Python · LangChain · Chroma · Streamlit · OpenAI-compatible chat API (OpenRouter free models or OpenAI) + local embeddings

## How it works

1. **Ingest** — `pypdf` extracts text page by page; chunks are tagged with the page they came from.
2. **Index** — chunks are embedded **locally** (`all-MiniLM-L6-v2` via sentence-transformers — free, offline, private) and stored in a persistent **Chroma** collection under `data/chroma`.
3. **Retrieve + Answer** — each question retrieves the top-k relevant chunks; the chat model answers strictly from them and cites pages inline, like `(page 3)`.

```
PDF → pypdf → chunks (+page metadata) → local MiniLM embeddings → Chroma
Question → embed → similarity search → top-k chunks → chat model → answer + citations
```

## Setup

Requires Python 3.10+ (3.11 recommended).

```bash
# 1. Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add a chat API key (free option: OpenRouter)
cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY=sk-or-...
# (create one at https://openrouter.ai/settings/keys — free models, no card needed)
#
# Prefer OpenAI instead? Set OPENAI_API_KEY=sk-... and CHAT_MODEL=gpt-4o-mini,
# optionally EMBEDDING_BACKEND=openai.
```

## Usage

### Web UI (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

- Upload a PDF in the sidebar → it is indexed automatically.
- Ask questions in the chat box; expand **🔍 Sources** under any answer to see the exact chunks and pages used.
- "Clear everything" wipes the vector store.

### Terminal (CLI)

```bash
python cli.py index path/to/report.pdf        # index one or more PDFs
python cli.py ask "What are the key findings?" --show-sources
python cli.py reset                           # clear the index
```

## Configuration

Set in `.env` (see `.env.example`):

| Variable             | Default                                      | Purpose                        |
| -------------------- | -------------------------------------------- | ------------------------------ |
| `OPENROUTER_API_KEY` | —                                            | Free chat models via OpenRouter |
| `OPENAI_API_KEY`     | —                                            | Fallback chat/embedding provider |
| `CHAT_MODEL`         | `nvidia/nemotron-3-ultra-550b-a55b:free`     | Answering model                |
| `EMBEDDING_BACKEND`  | `local`                                      | `local` (free) or `openai`     |
| `EMBEDDING_MODEL`    | `sentence-transformers/all-MiniLM-L6-v2`     | Embedding model                |

Chunk size / overlap / retrieval count can be tuned in `app/config.py`.

## Project structure

```
app/
  config.py          # env + paths + tunables
  ingest.py          # PDF → chunks → Chroma
  rag.py             # retrieval + GPT answer with citations
  streamlit_app.py   # web UI
cli.py               # terminal interface
data/
  uploads/           # uploaded PDFs
  chroma/            # persistent vector store
```

## Notes & limits

- **Scanned/image-only PDFs** have no extractable text — OCR support is not included (easy to add with `pytesseract` + `pdf2image`).
- Multi-query retrieval (rephrasing the question several ways) noticeably improves recall on hard questions — a good first upgrade.
