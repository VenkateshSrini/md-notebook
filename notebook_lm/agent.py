import os
from typing import Annotated

import vectorizer
from agent_framework import tool, Agent
from agent_framework.anthropic import AnthropicClient

_VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "vector-db")


def _is_vectorized() -> bool:
    if not os.path.isdir(_VECTOR_DB_DIR):
        return False
    return any(os.scandir(_VECTOR_DB_DIR))


async def startup() -> None:
    """Ensure vector-db is ready. Call once before the first ask()."""
    if _is_vectorized():
        print("Vectorization already complete. Proceeding to agent...")
    else:
        vectorizer.run()

_SYSTEM_PROMPT = """\
You are a NotebookLM assistant. Your ONLY knowledge source is the user's personal notes \
stored in a vector knowledge base.

STRICT RULES — never break any of these:
1. For EVERY user question, you MUST call search_notes() before responding.
2. Answer ONLY from the content returned by search_notes(). Never use external knowledge or prior training.
3. If search_notes() returns no relevant content, respond exactly: \
"I could not find relevant information in the notes."
4. Always cite the source filename(s) at the end of your response under a "Sources:" section.
5. Match response depth to the user's instruction:
   - "brief", "short", "quick", "one line" → concise answer only.
   - No instruction given, or "detailed", "thorough", "explain" → comprehensive, well-structured answer.
6. Do NOT fabricate, infer, or extend beyond what the notes contain. Zero hallucination.
"""


@tool(approval_mode="never_require")
def search_notes(
    query: Annotated[str, "The search query to find relevant notes in the knowledge base"],
) -> str:
    """Search the knowledge base (FAISS vector store) and return relevant note content."""
    results = vectorizer.search(query, top_k=5)
    if not results:
        return "No relevant notes found."
    parts = [f"--- Source: {r['filename']} ---\n{r['content']}" for r in results]
    return "\n\n".join(parts)


def _build_agent():
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "bedrock":
        from agent_framework.amazon import BedrockChatClient
        return Agent(
            client=BedrockChatClient(),
            name="NotebookLM",
            instructions=_SYSTEM_PROMPT,
            tools=[search_notes],
        )
    # Default: Anthropic (reads ANTHROPIC_API_KEY + ANTHROPIC_CHAT_MODEL_ID from env)
    return AnthropicClient().as_agent(
        name="NotebookLM",
        instructions=_SYSTEM_PROMPT,
        tools=[search_notes],
    )


_agent = None


async def ask(query: str) -> str:
    """Send a query to the NotebookLM agent. Returns the response as a string."""
    global _agent
    if _agent is None:
        _agent = _build_agent()
    result = await _agent.run(query)
    return str(result)
