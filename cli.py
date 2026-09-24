"""Terminal interface.

Examples:
    python cli.py index path/to/file.pdf
    python cli.py ask "What is this document about?"
    python cli.py reset
"""
import argparse
import sys


def cmd_index(args: argparse.Namespace) -> None:
    from app import ingest

    for path in args.files:
        print(f"Indexing {path} ...")
        stats = ingest.ingest_pdf(path)
        print(
            f"  ✔ {stats['source']}: {stats['pages']} pages -> "
            f"{stats['chunks']} chunks (total stored: {stats['stored_ids']})"
        )


def cmd_ask(args: argparse.Namespace) -> None:
    from app import rag

    result = rag.ask(args.question, k=args.k)
    print("\n--- ANSWER ---")
    print(result["answer"])
    if result["cited_pages"]:
        print(f"\nCited pages: {result['cited_pages']}")
    if args.show_sources:
        print("\n--- RETRIEVED CHUNKS ---")
        for i, doc in enumerate(result["source_docs"], start=1):
            page = int(doc.metadata.get("page", 0)) + 1
            src = doc.metadata.get("source", "unknown")
            print(f"\n[{i}] {src} (page {page}):\n{doc.page_content[:400]}")


def cmd_reset(args: argparse.Namespace) -> None:
    from app import ingest

    ingest.reset_collection()
    print("Collection cleared.")


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Q&A from the terminal")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index one or more PDFs")
    p_index.add_argument("files", nargs="+", help="Paths to PDF files")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="Ask a question about indexed PDFs")
    p_ask.add_argument("question", help="Your question")
    p_ask.add_argument("--k", type=int, default=None, help="Number of chunks to retrieve")
    p_ask.add_argument("--show-sources", action="store_true", help="Print retrieved chunks")
    p_ask.set_defaults(func=cmd_ask)

    p_reset = sub.add_parser("reset", help="Delete the indexed collection")
    p_reset.set_defaults(func=cmd_reset)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
