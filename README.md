# md-notebook

A self-hosted, local-first **NotebookLM clone** — ask natural-language questions over your own Markdown notes using semantic search and an LLM.

## Quick Start

```bash
# Install dependencies
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Configure (copy and fill in your API key)
cp .env.example .env

# Drop your *.md notes into source-md/ then run
python main.py          # interactive CLI
python main_api.py      # REST API  →  http://localhost:8000/docs
```

## How it works

1. On first startup, all Markdown files in `source-md/` are embedded with `all-MiniLM-L6-v2` and saved to a local FAISS vector index.
2. Each query is semantically matched against the index (top-5 results).
3. An LLM (Anthropic Claude or AWS Bedrock) answers using **only** the retrieved note content, with source citations — zero hallucination.

## Documentation

For a full architectural breakdown, module reference, API docs, Docker usage, and configuration guide see:

**[documentation/architecture.md](documentation/architecture.md)**
