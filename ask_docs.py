#!/usr/bin/env python3
"""Ask questions about documents in a folder using retrieval-augmented generation.

Expects OPENAI_API_KEY in the environment or in a .env file (loaded via python-dotenv).
"""

import argparse
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import APIError, OpenAI
from pypdf import PdfReader

load_dotenv()

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
SOURCE_SCORE_RATIO = 0.85
EMBED_BATCH_SIZE = 100
PREVIEW_LENGTH = 120
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CACHE_FILENAME = ".embeddings_cache.pkl"
PLACEHOLDER_API_KEYS = frozenset({"your_key_here", "sk-your-key-here"})

STOPWORDS = frozenset({"the", "is", "what", "and", "of", "in"})


@dataclass
class Document:
    filename: str
    text: str


@dataclass
class Chunk:
    filename: str
    chunk_index: int
    text: str


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class AskResult:
    answer: str
    sources: list[ScoredChunk]
    retrieved: list[ScoredChunk]


def die(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def get_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.lower() in PLACEHOLDER_API_KEYS:
        raise ValueError(
            "OPENAI_API_KEY is not set or is still a placeholder. "
            "Set it in .env or your environment."
        )
    return api_key


def require_api_key() -> str:
    try:
        return get_api_key()
    except ValueError as exc:
        die(str(exc))
    raise AssertionError("unreachable")


def resolve_folder(folder_input: str) -> Path:
    folder_input = folder_input.strip()
    if not folder_input:
        raise ValueError("folder_path cannot be empty.")
    folder = Path(folder_input).expanduser().resolve()
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")
    return folder


def resolve_question(question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("question cannot be empty.")
    return question


def build_chunks(folder: Path) -> list[Chunk]:
    docs = load_documents(folder)
    if not docs:
        raise ValueError(
            f"No supported documents found in {folder}. "
            f"Add .txt, .md, or .pdf files and try again."
        )

    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc.text, doc.filename))

    if not all_chunks:
        raise ValueError(f"Documents in {folder} contain no chunkable text.")
    return all_chunks


def ask_question(
    client: OpenAI, folder: Path, all_chunks: list[Chunk], question: str
) -> AskResult:
    scored_retrieved = retrieve_chunks_embedding(client, folder, all_chunks, question)
    sources = relevant_sources(scored_retrieved)
    context_chunks = [item.chunk for item in scored_retrieved]
    answer = answer_from_chunks(client, question, context_chunks)
    return AskResult(answer=answer, sources=sources, retrieved=scored_retrieved)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            parts.append(page_text.strip())
    return clean_text("\n\n".join(parts))


def load_documents(folder: Path) -> list[Document]:
    docs: list[Document] = []

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        rel_path = str(path.relative_to(folder))
        try:
            if path.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(path)
            else:
                raw = path.read_text(encoding="utf-8", errors="replace")
                text = clean_text(raw)
        except Exception as exc:
            print(f"Warning: could not read {rel_path}: {exc}", file=sys.stderr)
            continue

        if not text:
            continue

        docs.append(Document(filename=rel_path, text=text))

    return docs


def chunk_text(
    text: str,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    if not text:
        return []
    if overlap >= chunk_size:
        die(f"chunk overlap ({overlap}) must be smaller than chunk size ({chunk_size})")

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap

    return [
        Chunk(filename=filename, chunk_index=index, text=piece)
        for index, piece in enumerate(pieces)
    ]


def keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in STOPWORDS}


def preview(text: str, max_length: int = PREVIEW_LENGTH) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= max_length:
        return one_line
    return one_line[:max_length].rstrip() + "..."


def retrieve_chunks_keyword(
    chunks: list[Chunk],
    question: str,
    top_k: int = TOP_K,
) -> list[Chunk]:
    question_keywords = keywords(question)
    if not question_keywords:
        return chunks[:top_k]

    scored: list[tuple[int, Chunk]] = []
    for chunk in chunks:
        overlap = len(question_keywords & keywords(chunk.text))
        scored.append((overlap, chunk))

    scored.sort(key=lambda item: (-item[0], item[1].filename, item[1].chunk_index))
    return [chunk for score, chunk in scored[:top_k] if score > 0] or [
        chunk for _, chunk in scored[:top_k]
    ]


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    embeddings: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in response.data)
    return embeddings


def log_info(message: str) -> None:
    print(message, file=sys.stderr)


def cache_path(folder: Path) -> Path:
    return folder / CACHE_FILENAME


def load_embedding_cache(folder: Path) -> dict:
    path = cache_path(folder)
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.PickleError) as exc:
        log_info(f"Warning: could not load embedding cache: {exc}")
        return {}


def save_embedding_cache(folder: Path, cache: dict) -> None:
    path = cache_path(folder)
    with path.open("wb") as handle:
        pickle.dump(cache, handle)


def group_chunks_by_file(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.filename, []).append(chunk)
    for file_chunks in grouped.values():
        file_chunks.sort(key=lambda item: item.chunk_index)
    return grouped


def get_chunk_embeddings(
    client: OpenAI, folder: Path, chunks: list[Chunk]
) -> list[list[float]]:
    cache = load_embedding_cache(folder)
    grouped = group_chunks_by_file(chunks)
    new_cache: dict = {}
    embedding_by_key: dict[tuple[str, int], list[float]] = {}

    used_cache = False
    recomputed_files: list[str] = []

    for filename, file_chunks in grouped.items():
        file_path = folder / filename
        mtime = file_path.stat().st_mtime
        signature = [(chunk.chunk_index, chunk.text) for chunk in file_chunks]

        entry = cache.get(filename)
        if entry and entry.get("modified_timestamp") == mtime:
            cached_chunks = entry.get("chunks", [])
            cached_signature = [
                (item["chunk_index"], item["chunk_text"]) for item in cached_chunks
            ]
            if cached_signature == signature:
                for item in cached_chunks:
                    embedding_by_key[(filename, item["chunk_index"])] = item["embedding"]
                new_cache[filename] = entry
                used_cache = True
                continue

        embeddings = embed_texts(client, [chunk.text for chunk in file_chunks])
        new_cache[filename] = {
            "file_path": filename,
            "modified_timestamp": mtime,
            "chunks": [
                {
                    "chunk_index": chunk.chunk_index,
                    "chunk_text": chunk.text,
                    "embedding": embedding,
                }
                for chunk, embedding in zip(file_chunks, embeddings)
            ],
        }
        for chunk, embedding in zip(file_chunks, embeddings):
            embedding_by_key[(filename, chunk.chunk_index)] = embedding
        recomputed_files.append(filename)

    save_embedding_cache(folder, new_cache)

    if used_cache and not recomputed_files:
        log_info("Loaded embeddings from cache")
    if recomputed_files:
        log_info(
            "Recomputed embeddings for changed files: "
            + ", ".join(sorted(recomputed_files))
        )

    return [embedding_by_key[(chunk.filename, chunk.chunk_index)] for chunk in chunks]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    vec_a = np.array(a, dtype=np.float64)
    vec_b = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def retrieve_chunks_embedding(
    client: OpenAI,
    folder: Path,
    chunks: list[Chunk],
    question: str,
    top_k: int = TOP_K,
) -> list[ScoredChunk]:
    try:
        chunk_embeddings = get_chunk_embeddings(client, folder, chunks)
        question_embedding = embed_texts(client, [question])[0]
    except APIError as exc:
        die(f"OpenAI embedding request failed: {exc}")

    scored = [
        (cosine_similarity(question_embedding, embedding), chunk)
        for chunk, embedding in zip(chunks, chunk_embeddings)
    ]
    scored.sort(key=lambda item: (-item[0], item[1].filename, item[1].chunk_index))
    return [ScoredChunk(chunk=chunk, score=score) for score, chunk in scored[:top_k]]


def relevant_sources(
    scored_chunks: list[ScoredChunk],
    ratio: float = SOURCE_SCORE_RATIO,
) -> list[ScoredChunk]:
    if not scored_chunks:
        return []
    top_score = scored_chunks[0].score
    if top_score <= 0:
        return scored_chunks[:1]
    threshold = top_score * ratio
    filtered = [item for item in scored_chunks if item.score >= threshold]
    return filtered or [scored_chunks[0]]


def answer_from_chunks(client: OpenAI, question: str, retrieved: list[Chunk]) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[{i}] {chunk.filename} (chunk {chunk.chunk_index})\n{chunk.text}"
        )

    context = "\n\n".join(context_blocks)
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using only the provided excerpts. "
                        "If the excerpts do not contain enough information to answer, "
                        "say that you cannot answer from the documents. "
                        "Do not use outside knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Excerpts:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.2,
        )
    except APIError as exc:
        die(f"OpenAI chat request failed: {exc}")
    return response.choices[0].message.content or ""


def cannot_answer_from_docs(answer: str) -> bool:
    lower = answer.lower()
    return any(
        phrase in lower
        for phrase in ("cannot answer", "not contain", "not in the documents")
    )


def print_keyword_debug(question: str, matches: list[Chunk]) -> None:
    print(f"Question: {question}")
    print()
    print("Top matching chunks (keyword):")
    for rank, chunk in enumerate(matches, start=1):
        print(f"\n{rank}. {chunk.filename} (chunk {chunk.chunk_index})")
        print(f"   {preview(chunk.text)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions about documents in a folder (.txt, .md, .pdf)."
    )
    parser.add_argument("folder_path", help="Path to a folder of documents")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument(
        "--keyword",
        action="store_true",
        help="Use keyword retrieval instead of embeddings (debug/fallback)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print similarity scores for retrieved chunks",
    )
    args = parser.parse_args()

    try:
        folder = resolve_folder(args.folder_path)
        question = resolve_question(args.question)
    except ValueError as exc:
        die(str(exc))

    if not args.keyword:
        api_key = require_api_key()
    else:
        api_key = ""

    try:
        all_chunks = build_chunks(folder)
    except ValueError as exc:
        die(str(exc))

    if args.keyword:
        matches = retrieve_chunks_keyword(all_chunks, question)
        print_keyword_debug(question, matches)
        return

    client = OpenAI(api_key=api_key)
    result = ask_question(client, folder, all_chunks, question)

    print("Answer:")
    print(result.answer)

    if cannot_answer_from_docs(result.answer):
        return

    print()
    print("Sources:")
    for item in result.sources:
        line = f"- {item.chunk.filename} (chunk {item.chunk.chunk_index})"
        if args.debug:
            line += f"  [score: {item.score:.4f}]"
        print(line)


if __name__ == "__main__":
    main()
