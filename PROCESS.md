# PROCESS.md

## Goal

The assignment was to build a small document question-answering tool over a local folder of personal files. The core requirement was a working RAG-style flow: load documents, retrieve relevant context, answer from that context, and show where the answer came from.

I optimized for an end-to-end pipeline. That meant prioritizing:

- reliable document loading and chunking
- retrieval that worked on real files
- grounded answers with source attribution
- simple setup on a fresh clone


## Initial Scope Decision

I started with a CLI instead of a web app because the risky part of the project was the document pipeline itself—parsing, chunking, retrieval, and grounding—not UI polish. A CLI made it easy to iterate quickly, inspect chunk counts, and compare retrieval behavior without extra layers.

I also intentionally avoided over-engineering at the start:

- no vector database
- no file upload service
- no auth or deployment setup
- no hybrid search until the basic embedding path worked

That kept the first version small enough to reason about. Once the CLI worked reliably, adding Streamlit was a thin wrapper over the same functions rather than a rewrite.

## Development Timeline

### 1. Basic parsing + chunking

**What I implemented**

- `load_documents()` for `.txt`, `.md`, and `.pdf`
- PDF extraction with `pypdf`
- fixed-size chunking (~800 characters, 150 overlap)
- chunk metadata: filename and chunk index

**Why**

I wanted to validate the hardest boring part first: can the app actually read my files and produce sensible chunks?

**Tradeoff**

I skipped fancy semantic chunking (by paragraph/heading) in favor of predictable character-based chunks. Simpler to implement and debug, but less ideal for structured documents.

---

### 2. Keyword retrieval prototype

**What I implemented**

- `retrieve_chunks_keyword()` using stopword-filtered overlap
- CLI output showing top chunks and short previews

**Why**

Before paying for embeddings, I wanted a cheap retrieval baseline. Keyword search helped verify that chunking and file loading were wired correctly.

**Tradeoff**

Keyword retrieval is brittle for paraphrases and summaries, but fast and useful as a debug/fallback path (`--keyword`).

---

### 3. OpenAI embeddings + cosine similarity retrieval

**What I implemented**

- `text-embedding-3-small` for chunk and question embeddings
- in-memory cosine similarity
- top-5 chunk retrieval

**Why**

This was the core retrieval upgrade. Embeddings handled semantic matches much better than keyword overlap for real questions.

**Tradeoff**

Every query still embeds the question, and all chunk embeddings live in memory during a run. Fine for a take-home scale, not ideal for very large corpora.

---

### 4. LLM answer generation with grounding

**What I implemented**

- `gpt-4o-mini` answering from retrieved excerpts only
- explicit instruction to refuse when context is insufficient
- printed answer + sources

**Why**

Retrieval alone is not the product; the user needs a natural-language answer constrained to their documents.

**Tradeoff**

Grounding is prompt-based, not enforced structurally. The model can still overreach if weak chunks are included in context.

---

### 5. Error handling and edge cases

**What I implemented**

- validation for missing folder, empty question, missing API key
- safer PDF loading (warn and skip bad files)
- OpenAI API error handling
- refusal detection to hide sources on unanswerable responses

**Why**

A demo that crashes on empty input or a corrupt PDF is worse than a demo with fewer features. I did a focused reliability pass after the happy path worked.

**Tradeoff**

Some checks are pragmatic rather than exhaustive—for example, refusal detection uses simple phrase matching, not structured model output.

---

### 6. Embedding cache persistence

**What I implemented**

- pickle cache at `<folder>/.embeddings_cache.pkl`
- per-file cache entries with modified time, chunk text, and embeddings
- re-embed only changed/new files

**Why**

Repeated questions on the same folder were re-embedding everything. A local cache improved iteration speed without introducing a vector DB.

**Tradeoff**

Pickle is simple and good enough here, but not portable, version-safe, or ideal for multi-process use.

---

### 7. Streamlit UI

**What I implemented**

- `app.py` with folder path + question inputs
- answer, sources, and expandable retrieved-context previews
- shared helpers imported from `ask_docs.py`

**Why**

The CLI proved the pipeline; Streamlit made it easier to demo interactively.

**Tradeoff**

The UI reloads and re-chunks on each query instead of keeping a rich session index. Simpler code, more repeated work.

---

### 8. Source filtering improvements

**What I implemented**

- similarity score attached to retrieved chunks
- display-only source filter: `score >= max_score * 0.85`
- always show at least one source when sources are shown
- keep passing all top-5 chunks to the LLM for answer generation

**Why**

Specific questions like “Who wrote Federalist 1?” produced correct answers but listed unrelated sources. Separating “context for the model” from “sources shown to the user” fixed the UX without changing retrieval breadth.

**Tradeoff**

The threshold is heuristic. It improves display precision but does not guarantee the model only used the shown chunks.

---

### 9. Demo/eval script

**What I implemented**

- `demo_queries.py` with three representative questions:
  - specific lookup
  - cross-document summary
  - unanswerable query

**Why**

I wanted a repeatable smoke test that did not require manually retyping commands before a demo.

**Tradeoff**

It is a lightweight script, not a formal eval harness with expected outputs and pass/fail metrics.

## AI Tool Usage

I used **Cursor heavily** throughout the project—not as a one-shot code generator, but as a fast collaborator while I stayed responsible for scope and design decisions.

### How I used it

- scaffold repetitive boilerplate (argparse CLI, chunk loops, Streamlit wiring)
- suggest module structure and helper boundaries
- review edge cases and failure modes
- accelerate small refactors (e.g., extracting `ask_question()`, `build_chunks()`, cache helpers)

### How I worked with it

I iteratively refined prompts instead of accepting the first output. Typical loop:

1. ask for the smallest working version
2. run it on `sample_docs`
3. fix what broke or felt wrong
4. ask for a targeted improvement only

### Concrete examples

| Area | How AI helped | How I constrained or changed it |
|------|----------------|----------------------------------|
| Chunking | suggested overlap loop and metadata dataclasses | kept fixed char chunks; added overlap safety guard |
| Embedding cache | proposed pickle-per-folder structure | required per-file mtime + chunk signature invalidation |
| Streamlit UI | scaffolded `app.py` importing `ask_docs` | refused file upload/auth/session index scope creep |
| Source filtering | helped add score thresholding | tuned to 0.85 and split “LLM context” vs “displayed sources” |
| OpenAI billing/API | helped interpret `billing_not_active` and key setup | switched to real platform key; removed leaked key from git history |
| Reliability review | surfaced missing PDF/API/error cases | accepted only minimal fixes needed for demo reliability |

### Places I deliberately pushed back

- no external vector DB
- no hybrid search in the main path (keyword mode stays debug-only)
- no folder watcher or upload pipeline
- kept logic in a readable single core module with thin wrappers

## What AI Helped Most With

- scaffolding repetitive code quickly (CLI, cache batching, UI wiring)
- surfacing edge cases I might delay until late (empty folder, corrupt PDF, placeholder API key)
- speeding up experimentation with retrieval and output formatting
- drafting README content after the implementation stabilized

## Where AI Was Less Helpful

- **source precision tuning** — suggestions were reasonable, but the right threshold only became clear after running real queries (GraphQL, Federalist Papers, summary questions)
- **scope control** — AI will happily add vector DBs, hybrid search, rerankers, and upload flows; I had to keep cutting scope manually
- **2-hour prioritization** — deciding what not to build mattered more than generating more code
- **eval quality** — a script of example questions is easy; knowing which ones actually stress the system required manual judgment

## Testing Performed

### Functional / happy path

- specific lookup: `What are the key concerns about GraphQL?`
- long PDF lookup: `Who wrote Federalist 1?` (Federalist Papers PDF in `sample_docs`)
- summary across files: `Summarize what these documents are about.`
- unanswerable: `What was the company's revenue last quarter?`

### Error / edge cases

- missing folder path
- empty folder / no supported documents
- empty or whitespace-only question
- missing or placeholder `OPENAI_API_KEY`
- repeated runs on the same folder to verify cache behavior

### Cache validation

- first run: `Recomputed embeddings for changed files: ...`
- second run (unchanged files): `Loaded embeddings from cache`

### Representative outcomes

**GraphQL question**

```text
Answer:
The key concerns about GraphQL are caching, authentication, and developer tooling.

Sources:
- notes.md (chunk 0)
```

**Unanswerable question**

```text
Answer:
I cannot answer from the documents.
```

(no Sources section)

## What I Would Improve Next

- hybrid retrieval (embeddings + keyword/BM25) in the main path
- better PDF extraction for layout-heavy or scanned documents
- explicit chunk citations from the model (chunk IDs returned in structured output)
- reranking before answer generation
- Streamlit session caching for loaded chunks/embeddings
- small automated eval harness with expected source files and pass/fail checks

## Biggest Tradeoff

### Lightweight local cache vs full vector DB

I chose a pickle cache keyed by file modification time and chunk contents because the expected scale (~20 documents, ~50k words) did not justify Docker, hosted vector infrastructure, or extra setup. The cache gives most of the repeated-query benefit with far less complexity.

### OpenAI APIs vs local model hosting

I used OpenAI embeddings and chat models to reduce time spent on model serving and focus on document loading, retrieval, grounding, and source attribution—the parts most likely to break in a short project.

### End-to-end flow vs infrastructure breadth

The biggest deliberate tradeoff was optimizing for a **clean, explainable pipeline** over feature breadth. That is why the project has:

- a working CLI and Streamlit UI
- local embedding persistence
- source attribution with display filtering

—but not hybrid search, uploads, reranking, or a formal eval suite. Those are the next steps, not the foundation I optimized for in the initial build.
