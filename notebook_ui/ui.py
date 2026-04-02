"""
notebook_ui/ui.py
~~~~~~~~~~~~~~~~~
Gradio Blocks UI for md-notebook.

Themes
------
  Gradient (default)  — teal/ocean linear-gradient, white text
  Dark                — pure black background, white text
  Light               — white background, black text

Theme switching is driven entirely by a JS attribute on <body>; no Python
round-trip is required so the switch is instant.

PDF export
----------
Uses fpdf2 to produce an A4, plain-white, black-text document with a
timestamped header and each turn labelled "You:" / "Agent:".
"""

import asyncio
import os
import tempfile
from datetime import datetime

import gradio as gr
from fpdf import FPDF

import notebook_lm

# ── CSS ───────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ================================================================
   Gradio 5 theming via data-theme attribute on <body>.
   Three strategies are combined for maximum reliability:
     1. CSS custom-property overrides (Gradio 5 reads these).
     2. Direct element / structural selectors with !important.
     3. Message-bubble wildcard (*) to prevent color inheritance
        from Gradio's default stylesheet leaking through.
   ================================================================ */

/* ── Gradient theme — CSS variables ─────────────────────────────────────── */
body[data-theme="gradient"] {
    --body-background-fill:        linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    --body-text-color:             #ffffff;
    --body-text-color-subdued:     rgba(255,255,255,0.7);
    --background-fill-primary:     rgba(255,255,255,0.07);
    --background-fill-secondary:   rgba(255,255,255,0.04);
    --border-color-primary:        rgba(255,255,255,0.15);
    --border-color-accent:         rgba(255,255,255,0.30);
    --input-background-fill:       rgba(255,255,255,0.10);
    --input-border-color:          rgba(255,255,255,0.25);
    --input-placeholder-color:     rgba(255,255,255,0.40);
    --block-background-fill:       rgba(255,255,255,0.06);
    --block-border-color:          rgba(255,255,255,0.12);
    --block-label-text-color:      rgba(255,255,255,0.85);
    --button-secondary-background-fill: rgba(255,255,255,0.12);
    --button-secondary-text-color: #ffffff;
    --button-secondary-border-color: rgba(255,255,255,0.25);
    --color-accent:                #4db8ff;
    --link-text-color:             #7dd3fc;
    --link-text-color-hover:       #93dafc;
    --link-text-color-visited:     #a5b4fc;
    --link-text-color-active:      #7dd3fc;
    --table-odd-background-fill:   rgba(255,255,255,0.10);
    --table-even-background-fill:  rgba(255,255,255,0.05);
    --shadow-drop:                 none;
}

/* ── Dark theme — CSS variables ──────────────────────────────────────────── */
body[data-theme="dark"] {
    --body-background-fill:        #000000;
    --body-text-color:             #ffffff;
    --body-text-color-subdued:     rgba(255,255,255,0.65);
    --background-fill-primary:     #111111;
    --background-fill-secondary:   #1a1a1a;
    --border-color-primary:        #333333;
    --border-color-accent:         #444444;
    --input-background-fill:       #1a1a1a;
    --input-border-color:          #444444;
    --input-placeholder-color:     #888888;
    --block-background-fill:       #111111;
    --block-border-color:          #333333;
    --block-label-text-color:      rgba(255,255,255,0.75);
    --button-secondary-background-fill: #2a2a2a;
    --button-secondary-text-color: #ffffff;
    --button-secondary-border-color: #444444;
    --color-accent:                #4db8ff;
    --link-text-color:             #7dd3fc;
    --link-text-color-hover:       #93dafc;
    --link-text-color-visited:     #a5b4fc;
    --link-text-color-active:      #7dd3fc;
    --table-odd-background-fill:   #1e1e1e;
    --table-even-background-fill:  #2a2a2a;
    --shadow-drop:                 none;
}

/* ── Light theme — CSS variables ─────────────────────────────────────────── */
body[data-theme="light"] {
    --body-background-fill:        #ffffff;
    --body-text-color:             #111111;
    --body-text-color-subdued:     #555555;
    --background-fill-primary:     #f7f7f7;
    --background-fill-secondary:   #eeeeee;
    --border-color-primary:        #dddddd;
    --border-color-accent:         #cccccc;
    --input-background-fill:       #ffffff;
    --input-border-color:          #cccccc;
    --input-placeholder-color:     #999999;
    --block-background-fill:       #f7f7f7;
    --block-border-color:          #dddddd;
    --block-label-text-color:      #333333;
    --button-secondary-background-fill: #eeeeee;
    --button-secondary-text-color: #111111;
    --button-secondary-border-color: #cccccc;
    --color-accent:                #0f3460;
    --link-text-color:             #0f3460;
    --link-text-color-hover:       #1a4a80;
    --link-text-color-visited:     #5b21b6;
    --link-text-color-active:      #0f3460;
    --table-odd-background-fill:   #f0f0f0;
    --table-even-background-fill:  #e8e8e8;
}

/* ── Backgrounds ─────────────────────────────────────────────────────────── */
body[data-theme="gradient"],
body[data-theme="gradient"] .gradio-container,
body[data-theme="gradient"] .main,
body[data-theme="gradient"] footer {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364) !important;
    background-attachment: fixed !important;
    color: #ffffff !important;
    min-height: 100vh;
}
body[data-theme="dark"],
body[data-theme="dark"] .gradio-container,
body[data-theme="dark"] .main,
body[data-theme="dark"] footer {
    background: #000000 !important;
    color: #ffffff !important;
    min-height: 100vh;
}
body[data-theme="light"],
body[data-theme="light"] .gradio-container,
body[data-theme="light"] .main,
body[data-theme="light"] footer {
    background: #ffffff !important;
    color: #111111 !important;
    min-height: 100vh;
}

/* ── Component blocks (Gradio 5 uses .block, fieldset, .gap) ─────────────── */
body[data-theme="gradient"] .block,
body[data-theme="gradient"] fieldset,
body[data-theme="gradient"] .form,
body[data-theme="gradient"] .gap {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
}
body[data-theme="dark"] .block,
body[data-theme="dark"] fieldset,
body[data-theme="dark"] .form,
body[data-theme="dark"] .gap {
    background: #111111 !important;
    border-color: #333333 !important;
}
body[data-theme="light"] .block,
body[data-theme="light"] fieldset,
body[data-theme="light"] .form,
body[data-theme="light"] .gap {
    background: #f7f7f7 !important;
    border-color: #dddddd !important;
}

/* ── Text elements ───────────────────────────────────────────────────────── */
body[data-theme="gradient"] label,
body[data-theme="gradient"] span,
body[data-theme="gradient"] p,
body[data-theme="gradient"] h1, body[data-theme="gradient"] h2,
body[data-theme="gradient"] h3, body[data-theme="gradient"] h4,
body[data-theme="gradient"] li, body[data-theme="gradient"] td,
body[data-theme="gradient"] th, body[data-theme="gradient"] strong,
body[data-theme="gradient"] em, body[data-theme="gradient"] code,
body[data-theme="gradient"] .prose * { color: #ffffff !important; }

body[data-theme="dark"] label,
body[data-theme="dark"] span,
body[data-theme="dark"] p,
body[data-theme="dark"] h1, body[data-theme="dark"] h2,
body[data-theme="dark"] h3, body[data-theme="dark"] h4,
body[data-theme="dark"] li, body[data-theme="dark"] td,
body[data-theme="dark"] th, body[data-theme="dark"] strong,
body[data-theme="dark"] em, body[data-theme="dark"] code,
body[data-theme="dark"] .prose * { color: #ffffff !important; }

body[data-theme="light"] label,
body[data-theme="light"] span,
body[data-theme="light"] p,
body[data-theme="light"] h1, body[data-theme="light"] h2,
body[data-theme="light"] h3, body[data-theme="light"] h4,
body[data-theme="light"] li, body[data-theme="light"] td,
body[data-theme="light"] th, body[data-theme="light"] strong,
body[data-theme="light"] em, body[data-theme="light"] code,
body[data-theme="light"] .prose * { color: #111111 !important; }

/* ── Inputs & textareas ──────────────────────────────────────────────────── */
body[data-theme="gradient"] input,
body[data-theme="gradient"] textarea,
body[data-theme="gradient"] select {
    background: rgba(255,255,255,0.10) !important;
    color: #ffffff !important;
    border-color: rgba(255,255,255,0.25) !important;
}
body[data-theme="gradient"] input::placeholder,
body[data-theme="gradient"] textarea::placeholder { color: rgba(255,255,255,0.40) !important; }

body[data-theme="dark"] input,
body[data-theme="dark"] textarea,
body[data-theme="dark"] select {
    background: #1a1a1a !important;
    color: #ffffff !important;
    border-color: #444444 !important;
}
body[data-theme="dark"] input::placeholder,
body[data-theme="dark"] textarea::placeholder { color: #888888 !important; }

body[data-theme="light"] input,
body[data-theme="light"] textarea,
body[data-theme="light"] select {
    background: #ffffff !important;
    color: #111111 !important;
    border-color: #cccccc !important;
}
body[data-theme="light"] input::placeholder,
body[data-theme="light"] textarea::placeholder { color: #999999 !important; }

/* ── Chatbot container ───────────────────────────────────────────────────── */
body[data-theme="gradient"] .chatbot { background: rgba(255,255,255,0.04) !important; }
body[data-theme="dark"]     .chatbot { background: #111111 !important; }
body[data-theme="light"]    .chatbot { background: #f7f7f7 !important; }

/* ── Chat message bubbles  (Gradio 5: .message.user / .message.bot        ── */
/* ──  also cover .assistant variant and [data-testid] fallback)          ── */
body[data-theme="gradient"] .message.user,
body[data-theme="gradient"] [data-testid="user"] {
    background: rgba(255,255,255,0.20) !important;
}
body[data-theme="gradient"] .message.user *,
body[data-theme="gradient"] [data-testid="user"] * { color: #ffffff !important; }

body[data-theme="gradient"] .message.bot,
body[data-theme="gradient"] .message.assistant,
body[data-theme="gradient"] [data-testid="bot"] {
    background: rgba(44,83,100,0.75) !important;
}
body[data-theme="gradient"] .message.bot *,
body[data-theme="gradient"] .message.assistant *,
body[data-theme="gradient"] [data-testid="bot"] * { color: #e0f4ff !important; }

body[data-theme="dark"] .message.user,
body[data-theme="dark"] [data-testid="user"] { background: #1e3a5f !important; }
body[data-theme="dark"] .message.user *,
body[data-theme="dark"] [data-testid="user"] * { color: #ffffff !important; }

body[data-theme="dark"] .message.bot,
body[data-theme="dark"] .message.assistant,
body[data-theme="dark"] [data-testid="bot"] { background: #2a2a2a !important; }
body[data-theme="dark"] .message.bot *,
body[data-theme="dark"] .message.assistant *,
body[data-theme="dark"] [data-testid="bot"] * { color: #e0e0e0 !important; }

body[data-theme="light"] .message.user,
body[data-theme="light"] [data-testid="user"] { background: #0f3460 !important; }
body[data-theme="light"] .message.user *,
body[data-theme="light"] [data-testid="user"] * { color: #ffffff !important; }

body[data-theme="light"] .message.bot,
body[data-theme="light"] .message.assistant,
body[data-theme="light"] [data-testid="bot"] { background: #e9e9e9 !important; }
body[data-theme="light"] .message.bot *,
body[data-theme="light"] .message.assistant *,
body[data-theme="light"] [data-testid="bot"] * { color: #111111 !important; }

/* ── Accordion / details ─────────────────────────────────────────────────── */
body[data-theme="gradient"] details,
body[data-theme="gradient"] details > summary {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #ffffff !important;
}
body[data-theme="dark"] details,
body[data-theme="dark"] details > summary {
    background: #111111 !important;
    border-color: #333333 !important;
    color: #ffffff !important;
}
body[data-theme="light"] details,
body[data-theme="light"] details > summary {
    background: #f7f7f7 !important;
    border-color: #dddddd !important;
    color: #111111 !important;
}

/* ── Shared header ───────────────────────────────────────────────────────── */
.md-nb-header {
    text-align: center;
    padding: 16px 0 8px;
}
.md-nb-header h1 {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 0.4px;
}
.md-nb-header p {
    font-size: 13px;
    opacity: 0.75;
    margin-top: 4px;
}
"""

# ── JS ────────────────────────────────────────────────────────────────────────

# Set gradient theme immediately on page load so the default takes effect
# before any user interaction.
INIT_JS = """
() => {
    document.body.setAttribute('data-theme', 'gradient');
}
"""

# Theme radio → attribute on <body>; runs entirely in the browser.
THEME_SWITCH_JS = """
(value) => {
    document.body.setAttribute('data-theme', value.toLowerCase());
    return [];
}
"""

# ── PDF export ────────────────────────────────────────────────────────────────

def _export_pdf(history: list) -> str:
    """Render the chat history to a temporary PDF file and return its path."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, "md-notebook - Chat Export", ln=True, align="C")
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(
        0, 6,
        f"Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ln=True, align="C",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    if not history:
        pdf.set_font("Helvetica", style="I", size=11)
        pdf.cell(0, 8, "No messages to export.", ln=True)
    else:
        for turn in history:
            # Gradio 5 messages format: {"role": "user"|"assistant", "content": str}
            role = turn.get("role", "")
            content = str(turn.get("content", ""))
            label = "You:" if role == "user" else "Agent:"

            # Label
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.set_text_color(15, 32, 39)
            pdf.cell(0, 7, label, ln=True)

            # Content — multi_cell handles wrapping
            pdf.set_font("Helvetica", size=10)
            pdf.set_text_color(30, 30, 30)
            # Encode to latin-1 best-effort; replace characters that fpdf2
            # cannot render in the built-in font.
            safe = content.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 6, safe)
            pdf.ln(3)

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf", prefix="chat_"
    )
    pdf.output(tmp.name)
    return tmp.name


# ── Chat handler ──────────────────────────────────────────────────────────────

async def _chat(message: str, history: list, thread_id: str):
    """
    Append the user message, insert a thinking placeholder, yield, then
    replace the placeholder with the real agent response.
    Yields: (history, cleared_input, thread_id, summary)
    """
    if not message.strip():
        yield history, "", thread_id, ""
        return

    # 1. Add user turn
    history = history + [{"role": "user", "content": message}]

    # 2. Show thinking placeholder
    thinking_history = history + [
        {"role": "assistant", "content": "Agent is thinking\u2026"}
    ]
    yield thinking_history, "", thread_id, ""

    # 3. Call the agent (thread_id=None starts a new conversation)
    result = await notebook_lm.ask(message, thread_id=thread_id or None)

    # 4. Replace placeholder with real answer
    final_history = history + [{"role": "assistant", "content": result.answer}]
    yield final_history, "", result.thread_id, result.summary


# ── UI builder ────────────────────────────────────────────────────────────────

def create_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""

    with gr.Blocks(
        title="md-notebook",
        css=CUSTOM_CSS,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
            <div class="md-nb-header">
                <h1>&#128211; md-notebook</h1>
                <p>Ask questions about your Markdown notes &mdash;
                   grounded answers, zero hallucination</p>
            </div>
        """)

        # ── Theme switcher ────────────────────────────────────────────────────
        with gr.Row():
            theme_radio = gr.Radio(
                choices=["Gradient", "Dark", "Light"],
                value="Gradient",
                label="Theme",
                interactive=True,
            )

        # ── Chat panel ────────────────────────────────────────────────────────
        # ── Thread-id state (invisible) ────────────────────────────────────
        thread_state = gr.State("")

        chatbot = gr.Chatbot(
            value=[],
            height=480,
            label="Chat",
        )

        # ── Conversation summary ───────────────────────────────────────────
        with gr.Accordion("Conversation Summary", open=False):
            summary_box = gr.Textbox(
                value="",
                label="",
                interactive=False,
                lines=6,
                placeholder="Compact rolling summary will appear here after the first response\u2026",
            )

        # ── Input row ─────────────────────────────────────────────────────────
        with gr.Row():
            msg_box = gr.Textbox(
                placeholder=(
                    "Ask a question about your notes\u2026 "
                    "(prefix with \u2018brief:\u2019 for a concise answer)"
                ),
                label="",
                lines=1,
                max_lines=5,
                scale=8,
                show_label=False,
                container=False,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)
            clear_btn = gr.Button("Clear Chat", variant="secondary", scale=1, min_width=100)

        # ── Action row ────────────────────────────────────────────────────────
        with gr.Row():
            export_btn = gr.Button("⬇  Export Chat as PDF", variant="secondary")
            pdf_file = gr.File(
                label="Download",
                visible=False,
                interactive=False,
            )

        # ── Event: set gradient theme on initial page load ────────────────────
        demo.load(fn=None, js=INIT_JS)

        # ── Event: theme switch (pure JS, no Python) ──────────────────────────
        theme_radio.change(
            fn=None,
            inputs=theme_radio,
            outputs=None,
            js=THEME_SWITCH_JS,
        )

        # ── Event: send message ───────────────────────────────────────────────
        send_event = msg_box.submit(
            fn=_chat,
            inputs=[msg_box, chatbot, thread_state],
            outputs=[chatbot, msg_box, thread_state, summary_box],
        )
        send_btn.click(
            fn=_chat,
            inputs=[msg_box, chatbot, thread_state],
            outputs=[chatbot, msg_box, thread_state, summary_box],
        )

        # ── Event: clear chat (also resets thread + summary) ─────────────────
        clear_btn.click(
            fn=lambda: ([], "", "", ""),
            inputs=None,
            outputs=[chatbot, msg_box, thread_state, summary_box],
        )

        # ── Event: export PDF ─────────────────────────────────────────────────
        def _do_export(history):
            if not history:
                return gr.File(visible=False, value=None)
            path = _export_pdf(history)
            return gr.File(visible=True, value=path, label="chat.pdf")

        export_btn.click(
            fn=_do_export,
            inputs=[chatbot],
            outputs=[pdf_file],
        )

    return demo
