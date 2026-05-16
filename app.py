#!/usr/bin/env python3
"""Streamlit UI for document question answering."""

from pathlib import Path

import streamlit as st
from openai import APIError, OpenAI

from ask_docs import (
    AskResult,
    ask_question,
    build_chunks,
    cannot_answer_from_docs,
    get_api_key,
    preview,
    resolve_folder,
    resolve_question,
)

DEFAULT_FOLDER = str(Path(__file__).parent / "sample_docs")


st.set_page_config(page_title="Chat With Your Docs", layout="centered")
st.title("Chat With Your Personal Docs")
st.caption("Ask questions over `.txt`, `.md`, and `.pdf` files in a folder.")

folder_path = st.text_input("Folder path", value=DEFAULT_FOLDER)
question = st.text_area("Question", placeholder="What are the key concerns about GraphQL?")

if st.button("Ask", type="primary"):
    try:
        folder = resolve_folder(folder_path)
        resolved_question = resolve_question(question)
        api_key = get_api_key()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("Loading documents and generating answer..."):
        try:
            all_chunks = build_chunks(folder)
            client = OpenAI(api_key=api_key)
            result: AskResult = ask_question(
                client, folder, all_chunks, resolved_question
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        except APIError as exc:
            st.error(f"OpenAI API error: {exc}")
            st.stop()

    st.subheader("Answer")
    st.write(result.answer)

    if not cannot_answer_from_docs(result.answer):
        st.subheader("Sources")
        for item in result.sources:
            st.markdown(f"- **{item.chunk.filename}** (chunk {item.chunk.chunk_index})")

    with st.expander("Retrieved context previews"):
        for rank, item in enumerate(result.retrieved, start=1):
            st.markdown(
                f"**{rank}. {item.chunk.filename}** "
                f"(chunk {item.chunk.chunk_index}, score {item.score:.4f})"
            )
            st.text(preview(item.chunk.text, max_length=400))
