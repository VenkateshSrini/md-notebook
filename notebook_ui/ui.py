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
/* ── Theme: gradient (default) ──────────────────────────────────────────── */
body[data-theme="gradient"],
body[data-theme="gradient"] .gradio-container,
body[data-theme="gradient"] .main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
    background-attachment: fixed !important;
    color: #ffffff !important;
    min-height: 100vh;
}
body[data-theme="gradient"] .gr-box,
body[data-theme="gradient"] .gr-panel,
body[data-theme="gradient"] .contain {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.15) !important;
}
body[data-theme="gradient"] label,
body[data-theme="gradient"] span,
body[data-theme="gradient"] p,
body[data-theme="gradient"] h1,
body[data-theme="gradient"] h2,
body[data-theme="gradient"] h3 {
    color: #ffffff !important;
}
body[data-theme="gradient"] input,
body[data-theme="gradient"] textarea {
    background: rgba(255,255,255,0.10) !important;
    color: #ffffff !important;
    border-color: rgba(255,255,255,0.25) !important;
}
body[data-theme="gradient"] .chatbot {
    background: rgba(255,255,255,0.06) !important;
}
body[data-theme="gradient"] .message.user > div {
    background: rgba(255,255,255,0.20) !important;
    color: #ffffff !important;
}
body[data-theme="gradient"] .message.bot > div {
    background: rgba(44,83,100,0.75) !important;
    color: #e0f4ff !important;
}

/* ── Theme: dark ─────────────────────────────────────────────────────────── */
body[data-theme="dark"],
body[data-theme="dark"] .gradio-container,
body[data-theme="dark"] .main {
    background: #000000 !important;
    color: #ffffff !important;
    min-height: 100vh;
}
body[data-theme="dark"] .gr-box,
body[data-theme="dark"] .gr-panel,
body[data-theme="dark"] .contain {
    background: #111111 !important;
    border-color: #333333 !important;
}
body[data-theme="dark"] label,
body[data-theme="dark"] span,
body[data-theme="dark"] p,
body[data-theme="dark"] h1,
body[data-theme="dark"] h2,
body[data-theme="dark"] h3 {
    color: #ffffff !important;
}
body[data-theme="dark"] input,
body[data-theme="dark"] textarea {
    background: #1a1a1a !important;
    color: #ffffff !important;
    border-color: #444444 !important;
}
body[data-theme="dark"] .chatbot {
    background: #111111 !important;
}
body[data-theme="dark"] .message.user > div {
    background: #1e3a5f !important;
    color: #ffffff !important;
}
body[data-theme="dark"] .message.bot > div {
    background: #2a2a2a !important;
    color: #e0e0e0 !important;
}

/* ── Theme: light ────────────────────────────────────────────────────────── */
body[data-theme="light"],
body[data-theme="light"] .gradio-container,
body[data-theme="light"] .main {
    background: #ffffff !important;
    color: #000000 !important;
    min-height: 100vh;
}
body[data-theme="light"] .gr-box,
body[data-theme="light"] .gr-panel,
body[data-theme="light"] .contain {
    background: #f7f7f7 !important;
    border-color: #dddddd !important;
}
body[data-theme="light"] label,
body[data-theme="light"] span,
body[data-theme="light"] p,
body[data-theme="light"] h1,
body[data-theme="light"] h2,
body[data-theme="light"] h3 {
    color: #000000 !important;
}
body[data-theme="light"] input,
body[data-theme="light"] textarea {
    background: #ffffff !important;
    color: #000000 !important;
    border-color: #cccccc !important;
}
body[data-theme="light"] .chatbot {
    background: #f7f7f7 !important;
}
body[data-theme="light"] .message.user > div {
    background: #0f3460 !important;
    color: #ffffff !important;
}
body[data-theme="light"] .message.bot > div {
    background: #e9e9e9 !important;
    color: #111111 !important;
}

/* ── Shared overrides ────────────────────────────────────────────────────── */
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
    pdf.cell(0, 10, "md-notebook — Chat Export", ln=True, align="C")
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

async def _chat(message: str, history: list):
    """
    Append the user message, insert a thinking placeholder, yield, then
    replace the placeholder with the real agent response.
    """
    if not message.strip():
        yield history, ""
        return

    # 1. Add user turn
    history = history + [{"role": "user", "content": message}]

    # 2. Show thinking placeholder
    thinking_history = history + [
        {"role": "assistant", "content": "Agent is thinking\u2026"}
    ]
    yield thinking_history, ""

    # 3. Call the agent directly (no HTTP)
    response = await notebook_lm.ask(message)

    # 4. Replace placeholder with real answer
    final_history = history + [{"role": "assistant", "content": response}]
    yield final_history, ""


# ── UI builder ────────────────────────────────────────────────────────────────

def create_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""

    with gr.Blocks(
        title="md-notebook",
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
        chatbot = gr.Chatbot(
            value=[],
            height=480,
            label="Chat",
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
            inputs=[msg_box, chatbot],
            outputs=[chatbot, msg_box],
        )
        send_btn.click(
            fn=_chat,
            inputs=[msg_box, chatbot],
            outputs=[chatbot, msg_box],
        )

        # ── Event: clear chat ─────────────────────────────────────────────────
        clear_btn.click(
            fn=lambda: ([], ""),
            inputs=None,
            outputs=[chatbot, msg_box],
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
