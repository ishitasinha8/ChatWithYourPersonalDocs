#!/usr/bin/env python3
"""Run a small set of demo questions against sample_docs using the ask_docs pipeline."""

from pathlib import Path

from openai import OpenAI

from ask_docs import (
    answer_from_chunks,
    cannot_answer_from_docs,
    chunk_text,
    load_documents,
    relevant_sources,
    require_api_key,
    retrieve_chunks_embedding,
)

SAMPLE_DOCS = Path(__file__).parent / "sample_docs"

DEMO_QUERIES = [
    (
        "specific (notes.md)",
        "What are the key concerns about migrating to GraphQL?",
    ),
    (
        "summary (across documents)",
        "Summarize what these documents are about.",
    ),
    (
        "unanswerable",
        "What was the company's revenue last quarter?",
    ),
]


def build_chunks(folder: Path):
    docs = load_documents(folder)
    if not docs:
        raise SystemExit(f"No documents found in {folder}")

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc.text, doc.filename))
    if not all_chunks:
        raise SystemExit(f"No chunkable text in {folder}")
    return all_chunks


def ask_question(client: OpenAI, all_chunks, question: str) -> tuple[str, list]:
    scored_retrieved = retrieve_chunks_embedding(client, all_chunks, question)
    sources = relevant_sources(scored_retrieved)
    context_chunks = [item.chunk for item in sources] or [
        item.chunk for item in scored_retrieved
    ]
    answer = answer_from_chunks(client, question, context_chunks)
    return answer, sources


def print_result(question: str, answer: str, sources: list) -> None:
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    if cannot_answer_from_docs(answer):
        return
    if sources:
        print("Sources:")
        for item in sources:
            print(f"- {item.chunk.filename} (chunk {item.chunk.chunk_index})")


def main() -> None:
    folder = SAMPLE_DOCS.resolve()
    if not folder.is_dir():
        raise SystemExit(f"sample_docs folder not found: {folder}")

    api_key = require_api_key()
    client = OpenAI(api_key=api_key)
    all_chunks = build_chunks(folder)

    print(f"Running {len(DEMO_QUERIES)} demo queries on {folder}\n")

    for label, question in DEMO_QUERIES:
        print("=" * 60)
        print(label)
        print("=" * 60)
        answer, sources = ask_question(client, all_chunks, question)
        print_result(question, answer, sources)
        print()


if __name__ == "__main__":
    main()
