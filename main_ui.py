"""
main_ui.py
~~~~~~~~~~
Entry point for the Gradio web UI.

Configuration is read exclusively from the .env file (or environment):

    UI_HOST   — host to bind the Gradio server (default: 0.0.0.0)
    UI_PORT   — port for the Gradio server    (default: 7860)

Start:
    python main_ui.py
"""

import asyncio
import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

import notebook_lm
from notebook_ui import create_ui, CUSTOM_CSS, INIT_JS


def main() -> None:
    # Ensure the vector database is ready before the Gradio event loop starts.
    asyncio.run(notebook_lm.startup())

    host = os.getenv("UI_HOST", "0.0.0.0")
    port = int(os.getenv("UI_PORT", 7860))

    ui = create_ui()
    ui.launch(
        server_name=host,
        server_port=port,
        inbrowser=True,
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
        js=INIT_JS,
    )


if __name__ == "__main__":
    main()
