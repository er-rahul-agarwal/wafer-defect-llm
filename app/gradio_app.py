"""
app/gradio_app.py -- the web interface for the wafer defect assistant.

Shows a gallery of example wafers; the user picks one, then chats with the
assistant about it. Each selection starts a fresh conversation (a new
WaferAssistant) bound to that wafer. Chat is routed through the orchestrator,
so the LLM calls the detector via tool-calling exactly as in our tests.

Run on the server (binds a local web server):
    CUDA_VISIBLE_DEVICES=0 python app/gradio_app.py
"""

import os
import sys
import json

import numpy as np
import gradio as gr

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import config
from assistant.orchestrator import WaferAssistant

EXAMPLES_DIR = os.path.join(ROOT_DIR, "app", "examples")

# Load the example index prepared by prepare_examples.py.
with open(os.path.join(EXAMPLES_DIR, "index.json")) as f:
    EXAMPLES = json.load(f)

# Gallery images (paths) with a caption each.
GALLERY_ITEMS = [
    (os.path.join(EXAMPLES_DIR, ex["png"]), f"Example {ex['id']}")
    for ex in EXAMPLES
]


def select_wafer(evt: gr.SelectData):
    """A gallery image was clicked: load that wafer, start a fresh assistant."""
    ex = EXAMPLES[evt.index]
    wafer = np.load(os.path.join(EXAMPLES_DIR, ex["npy"]))
    assistant = WaferAssistant(wafer)
    greeting = (f"Selected Example {ex['id']}. "
                f"Ask me what defects it has, or anything about it.")
    # Return: the new assistant (kept in state), a cleared chat with a greeting.
    return assistant, [(None, greeting)]


def respond(message, chat_history, assistant):
    """Send the user's message through the orchestrator; append the reply."""
    if assistant is None:
        chat_history = chat_history + [
            (message, "Please select a wafer from the gallery first.")]
        return "", chat_history
    answer = assistant.ask(message)
    chat_history = chat_history + [(message, answer)]
    return "", chat_history


with gr.Blocks(title="Wafer Defect Assistant") as demo:
    gr.Markdown("# Wafer Defect Assistant")
    gr.Markdown(
        "Pick a wafer below, then ask about its defects. A Vision Transformer "
        "detects the defect patterns; a local LLM explains them via tool-calling."
    )

    # Per-session state: the current WaferAssistant.
    assistant_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Choose a wafer")
            gallery = gr.Gallery(value=GALLERY_ITEMS, columns=3,
                                 height="auto", allow_preview=False)
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=420)
            msg = gr.Textbox(placeholder="Ask about the selected wafer...",
                             label="Your question")

    gallery.select(select_wafer, outputs=[assistant_state, chatbot])
    msg.submit(respond, inputs=[msg, chatbot, assistant_state],
               outputs=[msg, chatbot])


if __name__ == "__main__":
    # server_name 0.0.0.0 so it is reachable via SSH port-forwarding.
    demo.launch(server_name="127.0.0.1", server_port=7860)