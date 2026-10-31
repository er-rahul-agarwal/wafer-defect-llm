"""
app/gradio_app.py -- the web interface for the wafer defect assistant.
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

with open(os.path.join(EXAMPLES_DIR, "index.json")) as f:
    EXAMPLES = json.load(f)

GALLERY_ITEMS = [
    (os.path.join(EXAMPLES_DIR, ex["png"]), f"Example {ex['id']}")
    for ex in EXAMPLES
]


def select_wafer(evt: gr.SelectData):
    ex = EXAMPLES[evt.index]
    wafer = np.load(os.path.join(EXAMPLES_DIR, ex["npy"]))
    assistant = WaferAssistant(wafer)
    greeting = (f"Selected Example {ex['id']}. "
                f"Ask me what defects it has, or anything about it.")
    return assistant, [{"role": "assistant", "content": greeting}]


def respond(message, chat_history, assistant):
    if assistant is None:
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant",
             "content": "Please select a wafer from the gallery first."}]
        return "", chat_history
    answer = assistant.ask(message)
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}]
    return "", chat_history


with gr.Blocks(title="Wafer Defect Assistant") as demo:
    gr.Markdown("# Wafer Defect Assistant")
    gr.Markdown(
        "Pick a wafer below, then ask about its defects. A Vision Transformer "
        "detects the defect patterns; a local LLM explains them via tool-calling."
    )

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
    demo.launch(server_name="127.0.0.1", server_port=7860)
