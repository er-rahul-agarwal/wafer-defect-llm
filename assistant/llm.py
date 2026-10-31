"""
assistant/llm.py -- a thin client wrapping Ollama for our assistant.

One job: send a message history (and optional tool definitions) to the local
Qwen model via Ollama, and return the model's response. It does NOT contain
the tool-calling loop -- that lives in the orchestrator. This keeps the
LLM-transport concern separate from the orchestration logic.

Qwen's "thinking" mode is disabled: for tool-calling we want fast, direct
decisions, not paragraphs of visible reasoning.

Self-test: run directly to send a simple prompt and print the reply.
    python assistant/llm.py
"""

import os
import sys

import ollama

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import config


def chat(messages, tools=None):
    """Send a conversation to the model and return its response message.

    Args:
        messages: list of {"role": ..., "content": ...} dicts (the history).
        tools:    optional list of tool definitions (JSON schemas) the model
                  may choose to call. When given, the returned message may
                  contain tool_calls instead of (or alongside) content.

    Returns:
        The response 'message' object. Key fields:
          .content      -- the text reply (may be empty if it called a tool)
          .tool_calls   -- a list of requested tool calls, or None
    """
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=messages,
        tools=tools,
        think=False,                 # no visible chain-of-thought; direct answers
        options={"temperature": config.LLM_TEMPERATURE},
    )
    return response["message"]


# ----------------------------------------------------------------------
# Self-test: a plain round-trip with no tools, to confirm we can talk to
# the model through Ollama from our own code.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Model:", config.LLM_MODEL)
    msgs = [{"role": "user",
             "content": "In one sentence, what is a semiconductor wafer?"}]
    reply = chat(msgs)
    print("\nReply:", reply["content"])