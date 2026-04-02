import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any

import vectorizer
from agent_framework import (
    Agent,
    AgentSession,
    BaseContextProvider,
    BaseHistoryProvider,
    Message,
    SessionContext,
    SupportsAgentRun,
    tool,
)
from agent_framework.anthropic import AnthropicClient

_VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "vector-db")
_SUMMARY_START = "---SUMMARY---"
_SUMMARY_END = "---END SUMMARY---"
_MAX_HISTORY_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "6"))


# ── Startup ───────────────────────────────────────────────────────────────────

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


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class AskResult:
    """Return value of ask(). Carries the answer, rolling summary, and thread id."""
    answer: str
    summary: str
    thread_id: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_response(raw: str) -> tuple[str, str]:
    """Split LLM output into (answer, summary). Falls back to (raw, '') if markers absent."""
    if _SUMMARY_START in raw and _SUMMARY_END in raw:
        before, rest = raw.split(_SUMMARY_START, 1)
        summary, _ = rest.split(_SUMMARY_END, 1)
        return before.strip(), summary.strip()
    return raw.strip(), ""


# ── Context providers ─────────────────────────────────────────────────────────

class BoundedHistoryProvider(BaseHistoryProvider):
    """Keeps only the last `max_messages` raw messages per session (RAM-capped history)."""

    def __init__(self, max_messages: int = _MAX_HISTORY_MESSAGES) -> None:
        super().__init__(source_id="bounded_history")
        self.max_messages = max_messages

    async def get_messages(
        self,
        session_id: str | None,
        *,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Message]:
        if state is None:
            return []
        return list(state.get("messages", []))

    async def save_messages(
        self,
        session_id: str | None,
        messages: Sequence[Message],
        *,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if state is None:
            return
        combined = list(state.get("messages", [])) + list(messages)
        trimmed = combined[-self.max_messages :]
        # Anthropic requires the first message in context to be a plain user message.
        # Trimming can split a tool_use / tool_result pair, leaving an orphaned
        # role="tool" message (or a role="assistant" tool_use) at position 0.
        # Advance past any such leading non-user messages to ensure a clean boundary.
        start = 0
        while start < len(trimmed) and trimmed[start].role != "user":
            start += 1
        state["messages"] = trimmed[start:]


class SummaryContextProvider(BaseContextProvider):
    """Injects rolling summary as instructions before each run; extracts and stores it after."""

    def __init__(self) -> None:
        super().__init__(source_id="rolling_summary")

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        summary = state.get("summary", "")
        if summary:
            context.extend_instructions(
                self.source_id,
                f"[CONVERSATION SUMMARY — previous turns]\n{summary}\n[END SUMMARY]",
            )

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        if context.response is None:
            return
        _, new_summary = _parse_response(context.response.text or "")
        if new_summary:
            state["summary"] = new_summary


# ── System prompt ─────────────────────────────────────────────────────────────

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
7. After your answer (including the Sources section), append a cumulative conversation summary \
in this EXACT format — no text after the closing marker:
   ---SUMMARY---
   • Q: <one-line question summary> → A: <one-line answer summary>
   (copy all prior bullets from [CONVERSATION SUMMARY] if one was provided, then append the new bullet last)
   ---END SUMMARY---
   Summary rules:
   - Each bullet must fit on a single line.
   - Total bullet count must never exceed 100.
   - Be maximally compact — omit filler words.
   - Do NOT include source citations inside the summary block.
"""


# ── Search tool ───────────────────────────────────────────────────────────────

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


# ── Agent factory ─────────────────────────────────────────────────────────────

def _build_agent() -> Agent:
    providers = [BoundedHistoryProvider(), SummaryContextProvider()]
    if os.getenv("LLM_PROVIDER", "anthropic").lower() == "bedrock":
        from agent_framework.amazon import BedrockChatClient
        return Agent(
            client=BedrockChatClient(),
            name="NotebookLM",
            instructions=_SYSTEM_PROMPT,
            tools=[search_notes],
            context_providers=providers,
        )
    return AnthropicClient().as_agent(
        name="NotebookLM",
        instructions=_SYSTEM_PROMPT,
        tools=[search_notes],
        context_providers=providers,
    )


# ── Session registry ──────────────────────────────────────────────────────────

_agent: Agent | None = None
_sessions: dict[str, AgentSession] = {}


def _get_or_create_session(thread_id: str | None) -> tuple[str, AgentSession]:
    """Return (thread_id, session). Creates a new session when thread_id is empty or unknown."""
    if thread_id and thread_id in _sessions:
        return thread_id, _sessions[thread_id]
    session = AgentSession()
    tid = session.session_id
    _sessions[tid] = session
    return tid, session


# ── Public API ────────────────────────────────────────────────────────────────

async def ask(query: str, thread_id: str | None = None) -> AskResult:
    """Send a query to the NotebookLM agent. Returns AskResult(answer, summary, thread_id)."""
    global _agent
    if _agent is None:
        _agent = _build_agent()
    tid, session = _get_or_create_session(thread_id)
    result = await _agent.run(query, session=session)
    answer, _ = _parse_response(result.text or str(result))
    summary = session.state.get("rolling_summary", {}).get("summary", "")
    return AskResult(answer=answer, summary=summary, thread_id=tid)
