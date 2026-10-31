"""
assistant/orchestrator.py -- the tool-calling loop (Layer 5): the heart of
the assistant. Ties together the user, the LLM, and the detect_defects tool.

The loop:
  1. The current wafer is set (our code holds the image out-of-band).
  2. The user's message is added to the conversation history.
  3. History + tool schema go to the LLM.
  4. If the LLM requests a tool call, we run the tool, append its result to
     the history, and loop back so the LLM can answer using that result.
     If the LLM returns a text answer instead, we return it to the user.

A system prompt tells the LLM what it is and how to behave. The prompt is
kept in prompts/ (versioned config), not hardcoded here.

Self-test: load a wafer and ask it a question end to end.
    python assistant/orchestrator.py
"""

import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import config
from assistant.llm import chat
from assistant.tools import (TOOL_SCHEMA, TOOL_IMPLEMENTATIONS,
                             set_current_wafer)


def _load_system_prompt():
    """Read the system prompt from prompts/system_prompt.txt."""
    path = os.path.join(config.PROMPT_DIR, "system_prompt.txt")
    with open(path, "r") as f:
        return f.read().strip()


class WaferAssistant:
    """Holds one conversation about one wafer."""

    def __init__(self, wafer_map):
        # Hand the wafer to the tool layer (out-of-band; never via the LLM).
        set_current_wafer(wafer_map)
        # Seed the history with the system prompt.
        self.messages = [{"role": "system", "content": _load_system_prompt()}]

    def ask(self, user_message, max_tool_rounds=5):
        """Send a user message through the tool-calling loop; return the reply."""
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(max_tool_rounds):
            reply = chat(self.messages, tools=[TOOL_SCHEMA])
            self.messages.append(reply)        # record what the model said

            tool_calls = reply.get("tool_calls")
            if not tool_calls:
                # No tool requested -> this is the final text answer.
                return reply["content"]

            # The model asked to call one or more tools. Run each, feed back.
            for call in tool_calls:
                name = call["function"]["name"]
                impl = TOOL_IMPLEMENTATIONS.get(name)
                result = impl() if impl else {"error": f"Unknown tool: {name}"}
                self.messages.append({
                    "role": "tool",
                    "content": json.dumps(result),
                })
            # Loop back: the model now answers using the tool result(s).

        return "(Stopped: too many tool-calling rounds.)"


# ----------------------------------------------------------------------
# Self-test: the whole thing end to end -- load a wafer, ask about it.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    test = np.load(os.path.join(config.PROCESSED_DIR, "test.npz"))
    assistant = WaferAssistant(test["maps"][2])   # the Center/Edge-Ring/Loc wafer

    question = "What defects does this wafer have, and what might they indicate?"
    print("USER:", question, "\n")
    answer = assistant.ask(question)
    print("ASSISTANT:", answer)