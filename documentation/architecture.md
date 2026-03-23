# md-notebook — Comprehensive Application Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Layout](#2-repository-layout)
3. [Architecture](#3-architecture)
   - 3.1 [High-Level Diagram](#31-high-level-diagram)
   - 3.2 [Data Flow — Indexing](#32-data-flow--indexing)
   - 3.3 [Data Flow — Query](#33-data-flow--query)
4. [Modules](#4-modules)
   - 4.1 [vectorizer](#41-vectorizer)
   - 4.2 [notebook_lm](#42-notebook_lm)
   - 4.3 [notebook_api](#43-notebook_api)
   - 4.4 [Entry Points](#44-entry-points)
5. [Environment Variables](#5-environment-variables)
6. [LLM Providers](#6-llm-providers)
7. [API Reference](#7-api-reference)
8. [Running the Application](#8-running-the-application)
   - 8.1 [Local (CLI)](#81-local-cli)
   - 8.2 [Local (REST API)](#82-local-rest-api)
   - 8.3 [Docker](#83-docker)
9. [Dependencies](#9-dependencies)
10. [Configuration Files](#10-configuration-files)
11. [Vector Database](#11-vector-database)
12. [Agent Behaviour & Prompt Design](#12-agent-behaviour--prompt-design)
13. [Adding New Notes](#13-adding-new-notes)
14. [Source Notes Format](#14-source-notes-format)

---

## 1. Overview

**md-notebook** is a self-hosted, local-first NotebookLM clone. It enables natural-language Q&A over a personal collection of Markdown notes by combining:

- **Semantic search** — notes are embedded with a sentence-transformer model and stored in a FAISS vector index.
- **LLM reasoning** — an AI agent (Anthropic Claude or AWS Bedrock) receives the retrieved note snippets as its sole knowledge source and generates grounded, citation-backed answers.
- **Zero hallucination policy** — the agent is instructed to answer exclusively from retrieved content; if nothing relevant is found it says so explicitly.

The application ships two interfaces:

| Interface | Entry point | Purpose |
|---|---|---|
| Interactive CLI | `main.py` | Run from terminal; interactive prompt loop |
| REST API | `main_api.py` + `notebook_api/` | HTTP server; integrates with other tools or front-ends |

---

## 2. Repository Layout

```
md-notebook/
├── main.py                  # CLI entry point
├── main_api.py              # REST API entry point
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container image definition
├── .env                     # Local environment variables (not committed)
├── .env.example             # Environment variable template
├── .gitignore
├── .dockerignore
│
├── notebook_lm/             # LLM agent package
│   ├── __init__.py          # Public surface: ask(), startup()
│   └── agent.py             # Agent definition, system prompt, search tool
│
├── notebook_api/            # FastAPI application package
│   ├── __init__.py          # Re-exports app
│   └── app.py               # FastAPI routes, request/response models, lifespan
│
├── vectorizer/              # Embedding + retrieval package
│   ├── __init__.py          # Public surface: run(), search()
│   ├── vectorize.py         # Reads source-md/, builds FAISS index + metadata
│   └── retriever.py         # Loads index on demand, runs similarity search
│
├── source-md/               # User's Markdown notes (not committed)
│   └── *.md
│
├── vector-db/               # Persisted FAISS artefacts (auto-generated)
│   ├── index.faiss          # FAISS flat L2 index (binary)
│   └── metadata.json        # Array of {filename, content} records
│
└── documentation/
    └── architecture.md      # This file
```

---

## 3. Architecture

### 3.1 High-Level Diagram

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                        User interfaces                           │
 │                                                                  │
 │   CLI (main.py)              REST API (main_api.py)             │
 │   stdin/stdout loop          FastAPI  POST /ask                 │
 └──────────────────┬───────────────────────┬──────────────────────┘
                    │                       │
                    └──────────┬────────────┘
                               │  notebook_lm.ask(query)
                               ▼
              ┌────────────────────────────────┐
              │        notebook_lm / agent.py  │
              │   Agent (agent_framework)      │
              │   System prompt: no-hallucinate│
              │   Tool: search_notes()         │
              └───────────────┬────────────────┘
                              │ search_notes(query)
                              ▼
              ┌────────────────────────────────┐
              │   vectorizer / retriever.py    │
              │   FAISS IndexFlatL2 (L2 dist)  │
              │   SentenceTransformer encode   │
              └───────────────┬────────────────┘
                              │ top-5 results [{filename, content, score}]
                              ▼
              ┌────────────────────────────────┐
              │       vector-db/               │
              │   index.faiss + metadata.json  │
              └────────────────────────────────┘

 Indexing (once, at startup if vector-db is empty):
              ┌────────────────────────────────┐
              │   vectorizer / vectorize.py    │
              │   source-md/*.md               │
              │   → SentenceTransformer embed  │
              │   → FAISS index build          │
              │   → write index.faiss          │
              │   → write metadata.json        │
              └────────────────────────────────┘

 LLM back-end (configured via LLM_PROVIDER env var):
              ┌──────────────────┐   ┌──────────────────────────┐
              │  Anthropic API   │   │  AWS Bedrock API         │
              │  (default)       │   │  (LLM_PROVIDER=bedrock)  │
              └──────────────────┘   └──────────────────────────┘
```

### 3.2 Data Flow — Indexing

1. `startup()` in `notebook_lm/agent.py` is called once on application start.
2. `vectorizer._is_vectorized()` checks whether `vector-db/` is non-empty.
3. If empty, `vectorizer.run()` is called:
   - All `.md` files in `source-md/` are read (sorted alphabetically; empty files are skipped).
   - Each document is encoded by the `all-MiniLM-L6-v2` sentence-transformer model into a 384-dimensional float32 vector.
   - Vectors are stacked into a NumPy matrix and added to a `faiss.IndexFlatL2` (exact nearest-neighbour search, L2 distance metric).
   - The FAISS index is written to `vector-db/index.faiss`.
   - The full list of `{filename, content}` records is written to `vector-db/metadata.json`.
4. On subsequent starts, the existing index is reused (no re-embedding).

### 3.3 Data Flow — Query

1. User submits a query (CLI input or HTTP POST body).
2. `notebook_lm.ask(query)` is called.
3. The agent sends the query to the LLM with the strict system prompt.
4. The LLM **must** invoke the `search_notes` tool before answering. The tool:
   - Encodes the query with the same `all-MiniLM-L6-v2` model (singleton, loaded once).
   - Calls `faiss_index.search(vec, top_k=5)` — returns the 5 closest vectors by L2 distance.
   - Maps result indices back to `{filename, content}` via `metadata.json`.
   - Returns a formatted string of retrieved snippets to the LLM.
5. The LLM synthesises an answer using only the retrieved snippets, then appends a `Sources:` section listing the filenames.
6. The answer string is returned to the caller.

---

## 4. Modules

### 4.1 `vectorizer`

| File | Responsibility |
|---|---|
| `vectorize.py` | Reads `source-md/`, embeds documents, builds and saves FAISS index + metadata |
| `retriever.py` | Loads FAISS index + metadata lazily (once); encodes queries; performs similarity search |
| `__init__.py` | Exports `run()` (from `vectorize`) and `search()` (from `retriever`) |

**Key constants:**

| Constant | Value | Description |
|---|---|---|
| `MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformer embedding model (384-dim) |
| `SOURCE_DIR` | `../source-md/` | Where Markdown notes are read from |
| `VECTOR_DB_DIR` | `../vector-db/` | Where FAISS artefacts are written/read |
| `INDEX_FILE` | `vector-db/index.faiss` | Binary FAISS index |
| `METADATA_FILE` | `vector-db/metadata.json` | JSON array of note records |

**`vectorize.run()`** — Build the index (destructive; overwrites existing files).

**`retriever.search(query, top_k=5)`** — Returns `list[dict]` where each dict is:
```python
{"filename": str, "content": str, "score": float}  # score = L2 distance (lower = more similar)
```

Module-level singletons (`_index`, `_metadata`, `_model`) are loaded on the first `search()` call and reused thereafter.

---

### 4.2 `notebook_lm`

| File | Responsibility |
|---|---|
| `agent.py` | Defines `_SYSTEM_PROMPT`, `search_notes` tool, agent factory, `ask()`, `startup()` |
| `__init__.py` | Exports `ask()` and `startup()` |

**`startup()`**  
Checks whether the vector-db is populated. If not, triggers `vectorizer.run()`. Must be called before the first `ask()`.

**`ask(query: str) → str`**  
Lazily builds the `_agent` singleton (Anthropic or Bedrock client, depending on `LLM_PROVIDER`), then calls `await _agent.run(query)` and returns the result as a string.

**System Prompt Summary:**

The `_SYSTEM_PROMPT` enforces:
1. `search_notes()` must be called before every answer.
2. Answers are drawn exclusively from retrieved content.
3. No relevant content → fixed canned response.
4. Source filenames always cited in a `Sources:` section.
5. Response depth adapts to user keywords (`brief` / `short` → concise; default / `detailed` → comprehensive).

**`search_notes` tool** (decorated with `@tool(approval_mode="never_require")`):  
- Calls `vectorizer.search(query, top_k=5)`.
- Formats results as a multi-block string with source headers.
- Returns `"No relevant notes found."` if the result set is empty.

**Agent factory (`_build_agent`):**

```
LLM_PROVIDER=anthropic  →  AnthropicClient().as_agent(...)   (default)
LLM_PROVIDER=bedrock    →  Agent(client=BedrockChatClient(), ...)
```

---

### 4.3 `notebook_api`

| File | Responsibility |
|---|---|
| `app.py` | FastAPI application, lifespan hook, route definitions, Pydantic models |
| `__init__.py` | Re-exports the `app` object |

**Routes:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Redirects to `/docs` (Swagger UI) |
| `GET` | `/swagger` | Redirects to `/docs` (Swagger UI) |
| `POST` | `/ask` | Accept a query, return an LLM-generated answer |

**Lifespan hook:**  
`notebook_lm.startup()` is called when the FastAPI application starts up (using `@asynccontextmanager lifespan`), ensuring the vector-db is ready before the first request is served.

**Request / Response models:**

```python
class AskRequest(BaseModel):
    query: str         # The natural-language question

class AskResponse(BaseModel):
    answer: str        # The LLM-generated, citation-backed answer
```

---

### 4.4 Entry Points

#### `main.py` — CLI

```
python main.py
```

- Calls `dotenv.load_dotenv()` then `asyncio.run(main())`.
- Calls `notebook_lm.startup()` once.
- Prints usage hints.
- Enters an infinite `input()` loop accepting queries.
- Prefixing the query with `brief:` triggers the agent's concise response mode.
- Type `quit`, `exit`, or `q` to terminate.

#### `main_api.py` — REST API server

```
python main_api.py
```

- Calls `dotenv.load_dotenv()`.
- Reads `PORT` from environment (default: `8000`).
- Starts `uvicorn` serving the FastAPI `app` on `0.0.0.0:{PORT}`.
- Lifespan hook on the FastAPI app handles `startup()` automatically.

---

## 5. Environment Variables

Copy `.env.example` to `.env` and fill in your credentials.

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | If `LLM_PROVIDER=anthropic` | — | Anthropic API key |
| `ANTHROPIC_CHAT_MODEL_ID` | No | (agent_framework default) | Anthropic model ID, e.g. `claude-sonnet-4-5-20250929` |
| `LLM_PROVIDER` | No | `anthropic` | LLM backend: `anthropic` or `bedrock` |
| `BEDROCK_CHAT_MODEL_ID` | If `LLM_PROVIDER=bedrock` | — | Bedrock model ID, e.g. `anthropic.claude-3-5-sonnet-20240620-v1:0` |
| `BEDROCK_REGION` | If `LLM_PROVIDER=bedrock` | — | AWS region, e.g. `us-east-1` |
| `AWS_ACCESS_KEY_ID` | If `LLM_PROVIDER=bedrock` | — | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | If `LLM_PROVIDER=bedrock` | — | AWS secret key |
| `AWS_SESSION_TOKEN` | No | — | AWS session token (temporary credentials only) |
| `PORT` | No | `8000` | Port the REST API server listens on |

---

## 6. LLM Providers

### Anthropic (default)

Set `LLM_PROVIDER=anthropic` (or omit — it is the default).  
Provide `ANTHROPIC_API_KEY` and optionally `ANTHROPIC_CHAT_MODEL_ID`.  
The agent is created via `AnthropicClient().as_agent(...)` from the `agent_framework` package.

### AWS Bedrock

Set `LLM_PROVIDER=bedrock`.  
Provide `BEDROCK_CHAT_MODEL_ID`, `BEDROCK_REGION`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY`.  
The agent is created via `Agent(client=BedrockChatClient(), ...)`.  
AWS credentials are picked up by the standard boto3 credential chain (env vars, `~/.aws/credentials`, IAM role, etc.).

---

## 7. API Reference

### `POST /ask`

Ask the NotebookLM agent a question.

**Request body** (`application/json`):

```json
{
  "query": "What are the main themes in my notes?"
}
```

**Response body** (`application/json`):

```json
{
  "answer": "Based on your notes, the main themes are ...\n\nSources:\n- example-note.md"
}
```

**Status codes:**

| Code | Meaning |
|---|---|
| `200` | Successful answer returned |
| `422` | Validation error (malformed request body) |
| `500` | Internal server error (LLM call or vector search failure) |

### `GET /docs`

Swagger UI — interactive API documentation generated by FastAPI.

### `GET /` or `GET /swagger`

Redirects to `/docs`.

---

## 8. Running the Application

### 8.1 Local (CLI)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
# Edit .env and fill in ANTHROPIC_API_KEY (or Bedrock credentials)

# 4. Add your Markdown notes
# Place *.md files in source-md/

# 5. Run the CLI
python main.py
```

On first run, all notes in `source-md/` are automatically vectorized and the FAISS index is written to `vector-db/`. Subsequent runs skip the indexing step.

### 8.2 Local (REST API)

```bash
# (after completing steps 1–4 above)
python main_api.py
# Server starts on http://0.0.0.0:8000
# Swagger UI: http://localhost:8000/docs
```

To use a different port:

```bash
# In .env:
PORT=9000

# Or inline:
PORT=9000 python main_api.py
```

### 8.3 Docker

The [Dockerfile](../Dockerfile) builds a self-contained image. `source-md/` and `.env` are excluded from the image (via `.dockerignore`); mount them at runtime.

```bash
# Build
docker build -t md-notebook .

# Run — mount notes and env file; override port if desired
docker run -p 8000:8000 \
  -v "$(pwd)/source-md:/app/source-md" \
  -v "$(pwd)/vector-db:/app/vector-db" \
  --env-file .env \
  md-notebook

# Override port
docker run -p 9000:9000 \
  -e PORT=9000 \
  -v "$(pwd)/source-md:/app/source-md" \
  -v "$(pwd)/vector-db:/app/vector-db" \
  --env-file .env \
  md-notebook
```

> **Note:** Mount `vector-db/` as a volume so the FAISS index persists between container restarts and is not re-built on every run.

**Dockerfile summary:**

| Instruction | Value / Purpose |
|---|---|
| `FROM` | `python:3.14-slim` |
| `WORKDIR` | `/app` |
| `COPY` | `requirements.txt`, `vectorizer/`, `notebook_lm/`, `notebook_api/`, `vector-db/`, `main_api.py` |
| `ENV PORT` | `8000` (default; overridable at `docker run`) |
| `EXPOSE` | `${PORT}` |
| `CMD` | `python main_api.py` |

`source-md/` is **not** copied into the image (it is in `.dockerignore`). You must bind-mount `source-md/` and `vector-db/` as shown above.

---

## 9. Dependencies

| Package | Role |
|---|---|
| `faiss-cpu` | Efficient similarity search over float32 vectors (FAISS IndexFlatL2) |
| `sentence-transformers` | Hugging Face wrapper for `all-MiniLM-L6-v2`; produces 384-dim embeddings |
| `agent-framework >= 1.0.0rc5` | Custom agent framework: `Agent`, `@tool` decorator, `AnthropicClient`, `BedrockChatClient` |
| `python-dotenv` | Loads `.env` file into `os.environ` at startup |
| `boto3` | AWS SDK; used by `BedrockChatClient` when `LLM_PROVIDER=bedrock` |
| `fastapi` | ASGI web framework for the REST API |
| `uvicorn[standard]` | ASGI server that serves the FastAPI app |

---

## 10. Configuration Files

| File | Purpose |
|---|---|
| `.env` | Local secrets and runtime configuration (not committed) |
| `.env.example` | Template showing all supported env vars with safe placeholder values |
| `.gitignore` | Excludes `.env`, `source-md/`, `.venv/`, `__pycache__/`, etc. from git |
| `.dockerignore` | Excludes `.venv/`, `source-md/`, `__pycache__/`, `*.pyc`, `.env` from Docker build context |

---

## 11. Vector Database

The vector-db stores two files, both located in `vector-db/`:

| File | Format | Contents |
|---|---|---|
| `index.faiss` | Binary (FAISS native) | `IndexFlatL2` — exact nearest-neighbour index; stores all document embeddings as float32 vectors |
| `metadata.json` | JSON array | `[{"filename": "...", "content": "..."}, ...]` — parallel array to the FAISS index; maps index position → note details |

**Embedding model:** `all-MiniLM-L6-v2`  
- Produces 384-dimensional sentence embeddings.  
- Downloaded automatically from HuggingFace Hub on first use (cached in the sentence-transformers model cache).

**Search metric:** L2 (Euclidean) distance. Lower score = higher similarity.

**Top-K:** 5 results returned per query by default.

**Re-indexing:** Delete or empty `vector-db/` and restart the application. `startup()` detects an empty directory and re-runs `vectorizer.run()` automatically.

---

## 12. Agent Behaviour & Prompt Design

The LLM agent is governed by `_SYSTEM_PROMPT` in `notebook_lm/agent.py`.

**Core rules (enforced via prompt):**

1. **Mandatory tool call** — The agent must call `search_notes()` before producing any response. The `agent_framework` routes this tool call to the FAISS retriever.
2. **Exclusive sourcing** — The LLM is instructed to use only the content returned by `search_notes()`. It is explicitly forbidden from using its pre-training knowledge.
3. **No-match response** — If the retriever returns nothing relevant, the response is fixed: `"I could not find relevant information in the notes."`
4. **Citations** — Every answer ends with a `Sources:` section listing the filenames of the retrieved notes.
5. **Adaptive verbosity** — Keywords like `brief`, `short`, `quick`, `one line` in the query trigger a concise answer. Absence of such keywords, or keywords like `detailed`, `thorough`, `explain`, trigger a comprehensive response.
6. **Zero hallucination** — The prompt explicitly instructs the model not to fabricate, infer, or extend beyond note content.

**Tool approval mode:** `"never_require"` — the agent executes `search_notes` automatically without pausing for user confirmation.

---

## 13. Adding New Notes

1. Place one or more `.md` files in `source-md/`.
2. Delete the contents of (or the entire) `vector-db/` directory so the existence check fails.
3. Restart the application — `startup()` will detect the empty `vector-db/` and re-index all notes automatically.

> There is currently no incremental indexing. All notes are re-embedded on every re-index run.

---

## 14. Source Notes Format

Notes are plain Markdown (`.md`) files. Any valid Markdown is accepted — headings, lists, tables, blockquotes, footnotes, links, code blocks. The full raw text content of each file is used as the document for embedding; Markdown formatting is not stripped before embedding.

File naming is unrestricted. The filename (including the `.md` extension) is used as the citation reference in agent responses.

Empty files are silently skipped during indexing.
