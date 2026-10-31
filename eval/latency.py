"""
eval/latency.py -- measure end-to-end LLM response latency.

Times how long assistant.ask() takes on the first/tool-calling question
("What defects does this wafer have?") across several test wafers. The first
run is a warm-up and discarded (cold model load). Reports average, min, max
of the warm runs -- an honest, hardware-specific latency figure.

Run on the server:
    CUDA_VISIBLE_DEVICES=0 python eval/latency.py
"""

import os
import sys
import time

import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import config
from assistant.orchestrator import WaferAssistant

N_WAFERS = 10
QUESTION = "What defects does this wafer have?"
BAR_FORMAT = "{desc}: [{bar}] {percentage:3.0f}% | {n_fmt}/{total_fmt} | ETA {remaining}"


def time_one(wafer):
    """Return seconds taken for one full ask() on a fresh assistant."""
    assistant = WaferAssistant(wafer)
    t0 = time.time()
    assistant.ask(QUESTION)
    return time.time() - t0


def main():
    test = np.load(os.path.join(config.PROCESSED_DIR, "test.npz"))
    maps = test["maps"]

    # Warm-up run (discarded): absorbs the cold model-load into GPU memory.
    print("Warm-up run (discarded)...")
    warm = time_one(maps[0])
    print(f"  warm-up took {warm:.2f}s (model load + first query)\n")

    # Timed runs.
    times = []
    bar = tqdm(range(N_WAFERS), bar_format=BAR_FORMAT, ascii=" %",
               desc="Timing", leave=False)
    for i in bar:
        times.append(time_one(maps[i]))

    times = np.array(times)
    print(f"\nLLM response latency over {N_WAFERS} queries "
          f"(first-question, tool-calling path):")
    print(f"  average: {times.mean():.2f}s")
    print(f"  min:     {times.min():.2f}s")
    print(f"  max:     {times.max():.2f}s")
    print(f"  (single RTX 6000 Ada, qwen3.5:9b)")


if __name__ == "__main__":
    main()
