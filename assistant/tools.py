"""
assistant/tools.py -- the detect_defects tool: the boundary (Layer 3) where
the LLM meets the vision detector.

Two halves:
  1. TOOL_SCHEMA  -- the structured description the LLM sees, so it knows the
     tool exists and when to call it. The LLM never passes the image; it just
     decides to invoke detection.
  2. run_detect_defects() -- what actually executes when the LLM calls the
     tool. It reads the "current wafer" that the orchestrator has set, runs
     the detector, and returns a text-friendly result for the LLM to reason
     about.

The image reaches the tool out-of-band: the orchestrator calls
set_current_wafer(array) before the conversation, and the tool reads it here.
This is why the schema below takes no image argument.

Self-test: run directly to set a wafer and call the tool by hand.
    python assistant/tools.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from detector.infer import detect_defects

# --- The current wafer the conversation is about (set by the orchestrator) ---
_current_wafer = None


def set_current_wafer(wafer_map):
    """Orchestrator calls this to load the wafer the LLM will analyze."""
    global _current_wafer
    _current_wafer = wafer_map


# --- Half 1: the schema the LLM sees ---
# Note: no image parameter. The LLM only decides WHETHER to detect; the wafer
# is supplied by our code. 'parameters' is an empty object because the LLM
# passes no arguments.
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "detect_defects",
        "description": (
            "Analyze the current semiconductor wafer map and report which "
            "manufacturing defect patterns are present. Call this whenever the "
            "user asks what is wrong with the wafer, what defects it has, or "
            "to analyze or inspect it. Returns the detected defect types and "
            "the model's confidence in each."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


# --- Half 2: what runs when the LLM calls the tool ---
def run_detect_defects():
    """Execute detection on the current wafer; return a result dict for the LLM.

    Returns a plain dict (JSON-serializable) describing the findings in words
    the LLM can reason about -- not raw tensors.
    """
    if _current_wafer is None:
        return {"error": "No wafer is currently loaded."}

    result = detect_defects(_current_wafer)

    # Shape the output for the LLM: the defects found, and confidences as
    # readable percentages, highest first.
    conf_sorted = sorted(result["confidences"].items(),
                         key=lambda kv: kv[1], reverse=True)
    confidences_pct = {name: f"{p*100:.1f}%" for name, p in conf_sorted}

    return {
        "defects_present": result["defects"],
        "any_defect": result["any_defect"],
        "confidence_by_type": confidences_pct,
    }


# Registry the orchestrator uses to look up the implementation by name.
TOOL_IMPLEMENTATIONS = {"detect_defects": run_detect_defects}


# ----------------------------------------------------------------------
# Self-test: load a real test wafer, call the tool directly (bypassing the
# LLM), and print what the LLM would receive.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import config

    test = np.load(os.path.join(config.PROCESSED_DIR, "test.npz"))
    set_current_wafer(test["maps"][2])        # sample 2 (a clean match earlier)

    result = run_detect_defects()
    print("Tool returns to the LLM:")
    for key, val in result.items():
        print(f"  {key}: {val}")