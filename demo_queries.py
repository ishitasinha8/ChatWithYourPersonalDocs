#!/usr/bin/env python3
"""Run a small set of demo questions against sample_docs using the ask_docs pipeline."""

from pathlib import Path

from openai import OpenAI

from ask_docs import (
    ask_question,
    build_chunks,
    cannot_answer_from_docs,
    require_api_key,
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

    client = OpenAI(api_key=require_api_key())
    all_chunks = build_chunks(folder)

    print(f"Running {len(DEMO_QUERIES)} demo queries on {folder}\n")

    for label, question in DEMO_QUERIES:
        print("=" * 60)
        print(label)
        print("=" * 60)
        result = ask_question(client, folder, all_chunks, question)
        print_result(question, result.answer, result.sources)
        print()


if __name__ == "__main__":
    main()
