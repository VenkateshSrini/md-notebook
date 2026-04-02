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
   - 4.4 [notebook_ui](#44-notebook_ui)
   - 4.5 [Entry Points](#45-entry-points)
5. [Environment Variables](#5-environment-variables)
6. [LLM Providers](#6-llm-providers)
7. [API Reference](#7-api-reference)
8. [Running the Application](#8-running-the-application)
   - 8.1 [Local (CLI)](#81-local-cli)
   - 8.2 [Local (REST API)](#82-local-rest-api)
   - 8.3 [Local (Gradio UI)](#83-local-gradio-ui)
   - 8.4 [Docker](#84-docker)
9. [Dependencies](#9-dependencies)
10. [Configuration Files](#10-configuration-files)
11. [Vector Database](#11-vector-database)
12. [Agent Behaviour & Prompt Design](#12-agent-behaviour--prompt-design)
13. [Adding New Notes](#13-adding-new-notes)
14. [Source Notes Format](#14-source-notes-format)
15. [Multi-turn Conversations \& Memory Design](#15-multi-turn-conversations--memory-design)

---

## 1. Overview

**md-notebook** is a self-hosted, local-first NotebookLM clone. It enables natural-language Q&A over a personal collection of Markdown notes by combining:

- **Semantic search** — notes are embedded with a sentence-transformer model and stored in a FAISS vector index.
- **LLM reasoning** — an AI agent (Anthropic Claude or AWS Bedrock) receives the retrieved note snippets as its sole knowledge source and generates grounded, citation-backed answers.
- **Zero hallucination policy** — the agent is instructed to answer exclusively from retrieved content; if nothing relevant is found it says so explicitly.
- **Multi-turn conversation** — each conversation session is identified by a `thread_id`. Bounded message history (last 6 raw messages) and a compact rolling summary (≤ 100 lines) are maintained per thread entirely in RAM. The rolling summary is injected as context on every turn so the LLM can answer coherently across turns without ever seeing the full transcript.

The application ships three interfaces:

| Interface | Entry point | Purpose |
|---|---|---|
| Interactive CLI | `main.py` | Run from terminal; interactive prompt loop |
| REST API | `main_api.py` + `notebook_api/` | HTTP server; integrates with other tools or front-ends |
| Gradio Web UI | `main_ui.py` + `notebook_ui/` | Browser-based chat interface with theme switching and PDF export |

---

## 2. Repository Layout

```
md-notebook/
├── main.py                  # CLI entry point
├── main_api.py              # REST API entry point
├── main_ui.py               # Gradio web UI entry point
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
├── notebook_ui/             # Gradio web UI package
│   ├── __init__.py          # Public surface: create_ui()
│   └── ui.py                # Gradio Blocks layout, theme CSS/JS, PDF export
│
├── vectorizer/              # Embedding + retrieval package
│   ├── __init__.py          # Public surface: run(), search()
│   ├── vectorize.py         # Reads source-md/, builds FAISS index + metadata
│   └── retriever.py         # Loads index on demand, runs similarity search
│
├── wireframes/              # UI wireframe (browser preview)
│   └── wireframe.html       # Standalone HTML wireframe with all three themes
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
 │  Gradio Web UI (main_ui.py)                                      │
 │  browser  →  notebook_ui/ui.py  →  notebook_lm.ask()  (direct)  │
 │                                                                  │
 │  CLI (main.py)              REST API (main_api.py)              │
 │  stdin/stdout loop          FastAPI  POST /ask                  │
 └────────────┬─────────────────────────────┬──────────────────────┘
              │                             │
              └──────────────┬──────────────┘
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

**`AskResult` (dataclass)**

| Field | Type | Description |
|---|---|---|
| `answer` | `str` | LLM-generated, citation-backed answer (summary markers stripped) |
| `summary` | `str` | Latest cumulative rolling summary (≤ 100 bullet lines) |
| `thread_id` | `str` | UUID identifying the conversation thread |

**`startup()`**  
Checks whether the vector-db is populated. If not, triggers `vectorizer.run()`. Must be called before the first `ask()`.

**`ask(query: str, thread_id: str | None = None) → AskResult`**  
Lazily builds the `_agent` singleton, resolves or creates an `AgentSession` for the given `thread_id` (empty/unknown → new session), calls `await _agent.run(query, session=session)`, and returns an `AskResult`. Passing the same `thread_id` on subsequent calls maintains conversation continuity.

**`BoundedHistoryProvider(BaseHistoryProvider)`**  
Stores the last `max_messages` (default: 6, overridable via `HISTORY_MAX_MESSAGES` env var) raw `Message` objects in `session.state["bounded_history"]["messages"]`. `save_messages()` appends then trims, so RAM usage is hard-capped regardless of conversation length.

**`SummaryContextProvider(BaseContextProvider)`**  
Before each run: if `session.state["rolling_summary"]["summary"]` is non-empty, injects it as a system instruction block (`[CONVERSATION SUMMARY]…[END SUMMARY]`). After each run: parses the `---SUMMARY---` / `---END SUMMARY---` delimiters from the LLM response and updates the stored summary. Fallback: if the LLM omits the markers, the previous summary is preserved unchanged.

**`_sessions: dict[str, AgentSession]`** — module-level in-memory registry mapping `thread_id → AgentSession`. Cleared on process restart.

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
    query: str                  # The natural-language question
    thread_id: str | None = None  # Resume an existing thread; omit to start a new one

class AskResponse(BaseModel):
    answer: str      # The LLM-generated, citation-backed answer
    thread_id: str   # The conversation thread UUID (new or existing)
    summary: str     # Latest cumulative rolling summary (empty on first turn if LLM hasn't responded yet)
```

---

### 4.4 `notebook_ui`

| File | Responsibility |
|---|---|
| `ui.py` | Gradio Blocks layout, custom CSS/JS, chat handler, PDF export |
| `__init__.py` | Exports `create_ui()` |

**`create_ui() → gr.Blocks`**  
Builds the full Gradio application and returns it. Called by `main_ui.py`.

**Layout:**

| Component | Purpose |
|---|---|
| `gr.HTML` header | App title and subtitle |
| `gr.Radio(["Gradient","Dark","Light"])` | Theme switcher — pure JS, no Python round-trip |
| `gr.Chatbot(type="messages")` | Message history (Gradio 5 format) |
| `gr.Textbox` + `gr.Button("Send")` | Query input and submit |
| `gr.Button("Clear Chat")` | Resets chat history and input box |
| `gr.Button("Export Chat as PDF")` | Triggers PDF generation |
| `gr.File(visible=False)` | Hidden; becomes a download link once the PDF is generated |

**Themes:**

| Theme | Background | Text |
|---|---|---|
| Gradient (default) | `linear-gradient(135deg, #0f2027, #203a43, #2c5364)` | White |
| Dark | `#000000` | White |
| Light | `#ffffff` | Black |

Switching is driven by a JS attribute (`data-theme`) on `<body>`. The switch is instant with no server round-trip. The default theme is applied via `INIT_JS` which runs once on page load.

**`_chat(message, history, thread_id)` (async generator)**  
1. Appends the user's message to history and yields immediately.
2. Inserts an `"Agent is thinking…"` placeholder and yields again.
3. Calls `await notebook_lm.ask(message, thread_id=thread_id or None)` directly (no HTTP).
4. Replaces the placeholder with the final answer and yields `(history, "", result.thread_id, result.summary)`.

**Thread state & summary:** A `gr.State("")` component tracks the active `thread_id` across Gradio events. A collapsible `gr.Accordion` below the chatbot contains a read-only `gr.Textbox` that displays the latest rolling summary. Clearing the chat also resets both the thread state and the summary box, starting a fresh conversation.

**`_export_pdf(history)`**  
Uses `fpdf2.FPDF` to produce an A4, plain-white, black-text PDF:  
- Header: app title + export timestamp.  
- Each turn labelled `You:` or `Agent:` with wrapped body text.  
Returns the path of a temp file; Gradio serves it as a browser download named `chat.pdf`.

---

### 4.5 Entry Points

#### `main_ui.py` — Gradio web UI

```
python main_ui.py
```

- Calls `dotenv.load_dotenv()`.
- Calls `asyncio.run(notebook_lm.startup())` — vectorises if needed; safe before Gradio's event loop starts.
- Reads `UI_HOST` (default: `0.0.0.0`) and `UI_PORT` (default: `7860`) from environment.
- Calls `create_ui().launch(server_name=host, server_port=port, inbrowser=True)`.
- `inbrowser=True` opens the default browser automatically on startup.

#### `main.py` — CLI

```
python main.py
```

- Calls `dotenv.load_dotenv()` then `asyncio.run(main())`.
- Calls `notebook_lm.startup()` once.
- Prints usage hints.
- Enters an infinite `input()` loop accepting queries.
- Prefixing the query with `brief:` triggers the agent's concise response mode.
- Tracks `thread_id` across turns; prints the thread UUID on the first response.
- Prints a `[Summary]` block after each response (when non-empty).
- Type `new` to reset the conversation (new thread, no prior context).
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
| `UI_HOST` | No | `0.0.0.0` | Host the Gradio UI binds to (`127.0.0.1` for localhost only) |
| `UI_PORT` | No | `7860` | Port the Gradio UI listens on |

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
  "query": "What are the main themes in my notes?",
  "thread_id": null
}
```

Omit `thread_id` (or pass `null`) to start a new conversation. Pass the `thread_id` returned from a previous response to continue an existing thread.

**Response body** (`application/json`):

```json
{
  "answer": "Based on your notes, the main themes are ...\n\nSources:\n- example-note.md",
  "thread_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "summary": "• Q: main themes in notes → A: themes are X, Y, Z"
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

### 8.3 Local (Gradio UI)

```bash
# (after completing steps 1–4 from 8.1 above)
python main_ui.py
# Gradio starts on http://0.0.0.0:7860
# Browser opens automatically
```

To use a different host or port:

```bash
# In .env:
UI_HOST=127.0.0.1
UI_PORT=8080
```

The UI calls `notebook_lm.ask()` directly in-process — no REST API server is required when using the Gradio UI.

### 8.4 Docker

The [Dockerfile](../Dockerfile) builds a self-contained image. `source-md/` and `.env` are excluded from the image (via `.dockerignore`); mount them at runtime.

```bash
# Build
docker build -t md-notebook .

# Run REST API (default)
docker run -p 8000:8000 \
  -v "$(pwd)/source-md:/app/source-md" \
  -v "$(pwd)/vector-db:/app/vector-db" \
  --env-file .env \
  md-notebook

# Run Gradio UI instead
docker run -p 7860:7860 \
  -v "$(pwd)/source-md:/app/source-md" \
  -v "$(pwd)/vector-db:/app/vector-db" \
  --env-file .env \
  md-notebook python main_ui.py
```

> **Note:** Mount `vector-db/` as a volume so the FAISS index persists between container restarts and is not re-built on every run.

**Dockerfile summary:**

| Instruction | Value / Purpose |
|---|---|
| `FROM` | `python:3.12-slim` |
| `WORKDIR` | `/app` |
| `COPY` | `requirements.txt`, `vectorizer/`, `notebook_lm/`, `notebook_api/`, `notebook_ui/`, `vector-db/`, `main_api.py`, `main_ui.py` |
| `ENV PORT` | `8000` (REST API; overridable at `docker run`) |
| `ENV UI_HOST` | `0.0.0.0` |
| `ENV UI_PORT` | `7860` (Gradio UI; overridable at `docker run`) |
| `EXPOSE` | `${PORT}` and `${UI_PORT}` |
| `CMD` | `python main_api.py` (override with `python main_ui.py` to run the Gradio UI) |

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
| `gradio >= 5.0.0` | Browser-based UI framework; powers the chat interface in `notebook_ui/` |
| `fpdf2 >= 2.7.0` | Pure-Python PDF generation; used by the Export Chat as PDF feature |

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
7. **Structured summary output** — Every response must end with a `---SUMMARY---` / `---END SUMMARY---` block containing cumulative `• Q: → A:` bullets (max 100). The prior summary (injected via `[CONVERSATION SUMMARY]` block in the system instructions by `SummaryContextProvider`) must be copied then extended. This embeds summarization into every normal LLM call — zero extra API requests.

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

---

## 15. Multi-turn Conversations & Memory Design

### 15.1 Overview

As of the multi-turn update, `md-notebook` supports persistent, thread-based conversations. Each conversation is identified by a `thread_id` (UUID). The same `thread_id` can be passed across CLI turns, API calls, or UI interactions to maintain context.

A conversation is **stateless from the caller's perspective** — the server holds session state in-memory, keyed by `thread_id`. Sending an unknown or empty `thread_id` silently creates a new thread.

> **Note:** Sessions are in-memory only. Restarting the server clears all threads.

---

### 15.2 Memory Design Options Considered

Three approaches were evaluated. All use MAF's `AgentSession` for per-thread state isolation and require **zero extra LLM calls** for memory management.

| Option | What is stored | RAM per session | Context quality | Container safe? |
|---|---|---|---|---|
| **A — Summary only** | 100-line rolling summary string | ~8 KB | Good — but no verbatim recent turns | ✅ Yes |
| **B — Bounded history + Summary** *(chosen)* | Last 6 raw messages + rolling summary | ~50–80 KB | **Best** — verbatim recent context + historical summary | ✅ Yes |
| **C — InMemoryHistoryProvider + Summary** | All raw messages (unbounded) + summary | Unbounded | Best but wasteful | ⚠️ Risk |

**Why Option C was rejected:**  
`InMemoryHistoryProvider.save_messages()` is an unbounded append (`state["messages"] = [*existing, *messages]`) with no `max_messages` parameter. Over a long conversation this grows indefinitely — a risk in containerised deployments.

**Why not `SlidingWindowStrategy`?**  
Inspection of the MAF source confirms it only marks old messages `_excluded=True` in annotations; it does **not** remove them from `state["messages"]`. No actual RAM saving.

**Why not `SummarizationStrategy`?**  
Inspection of the MAF source confirms it calls `await self.client.get_response(...)` — a separate LLM API call for every summarization trigger. Explicitly excluded by requirement.

**Why Option B was chosen:**  
- Hard RAM cap: `save_messages()` trims on every write — `combined[-max_messages:]`.
- Highest context quality: the LLM sees verbatim recent turns (exact phrasing + citations) **plus** a compressed summary of all older turns.
- Zero extra LLM calls: the rolling summary is embedded inside every normal response via System Prompt Rule 7.
- Container-safe: worst-case ~80 KB per session × 20 concurrent users = ~1.6 MB.

---

### 15.3 MAF Components Used

| Component | Role |
|---|---|
| `AgentSession` | Per-thread state container (`session_id` + mutable `state` dict) |
| `BaseHistoryProvider` | Abstract base for `BoundedHistoryProvider`; framework calls `before_run`/`after_run` automatically |
| `BaseContextProvider` | Abstract base for `SummaryContextProvider`; injected into agent's context pipeline |
| `context_providers=[...]` | Suppresses the framework's default `InMemoryHistoryProvider`; our two providers replace it |

**Provider-scoped state mapping:**

| Provider | `source_id` | State key | Content |
|---|---|---|---|
| `BoundedHistoryProvider` | `bounded_history` | `session.state["bounded_history"]["messages"]` | `list[Message]` — last 6 raw messages |
| `SummaryContextProvider` | `rolling_summary` | `session.state["rolling_summary"]["summary"]` | `str` — latest cumulative summary |

---

### 15.4 Execution Flow per Turn

```
ask(query, thread_id?)
  │
  ├─ empty/unknown thread_id → AgentSession()  (new session_id = new thread_id)
  └─ known thread_id         → _sessions[thread_id]
          │
          ▼
  agent.run(query, session=session)
          │
          ├─ BoundedHistoryProvider.before_run()
          │     reads state["bounded_history"]["messages"]  (last 6 msgs)
          │     → context.extend_messages(...)              (verbatim recent turns)
          │
          ├─ SummaryContextProvider.before_run()
          │     reads state["rolling_summary"]["summary"]
          │     → context.extend_instructions(...)          (compact history)
          │
          ▼
  LLM call (single):
    [summary instructions] + [last 6 messages] + [user query]
    → tool call: search_notes(query)
    → tool result: top-5 note snippets
    → final answer + ---SUMMARY--- block
          │
          ├─ BoundedHistoryProvider.after_run()
          │     appends new messages; trims to last max_messages
          │
          ├─ SummaryContextProvider.after_run()
          │     parses ---SUMMARY--- / ---END SUMMARY--- from response
          │     → saves new summary to state["rolling_summary"]["summary"]
          │
          ▼
  _parse_response(result.text) → (answer, _)
  summary = session.state["rolling_summary"].get("summary", "")
  return AskResult(answer, summary, thread_id)
```

---

### 15.5 Summary Format (System Prompt Rule 7)

The LLM is instructed to always append the following block at the end of every response:

```
---SUMMARY---
• Q: <one-line question summary> → A: <one-line answer summary>
(prior bullets copied from [CONVERSATION SUMMARY] if provided, new bullet appended last)
---END SUMMARY---
```

Rules enforced by the prompt:
- Each bullet on a single line.
- Total bullets ≤ 100.
- Maximally compact — no filler words.
- No source citations inside the block.

If the LLM omits the markers (e.g., on a refusal), `_parse_response()` returns `(full_text, "")` and the previously stored summary is preserved unchanged.

---

### 15.6 RAM Estimates

| Component | Per session | 20 sessions |
|---|---|---|
| `BoundedHistoryProvider` (6 msgs × ~10 KB avg) | ~60 KB | ~1.2 MB |
| `SummaryContextProvider` (100-line max) | ~8 KB | ~160 KB |
| **Total (worst case)** | **~68 KB** | **~1.4 MB** |
