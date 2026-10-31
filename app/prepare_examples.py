"""
app/prepare_examples.py -- pick a few diverse example wafers for the demo
gallery and save them as viewable PNG thumbnails + a small index.

Chooses examples with varied defect patterns (clean-ish, single-defect,
multi-defect, and the tricky edge cases) so the demo shows the system
handling variety. Saves to app/examples/.

Run from project root:
    python app/prepare_examples.py
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import config
from detector.infer import LABEL_NAMES

EXAMPLES_DIR = os.path.join(ROOT_DIR, "app", "examples")


def main():
    test = np.load(os.path.join(config.PROCESSED_DIR, "test.npz"))
    maps, labels = test["maps"], test["labels"]

    # Pick diverse examples by their defect-count, to show variety.
    # defect count per sample:
    counts = labels.sum(axis=1)

    chosen = []
    # one single-defect, some 2-defect, some 3-defect, one 4-defect if present
    for target_count in [1, 2, 2, 3, 3, 4]:
        candidates = np.where(counts == target_count)[0]
        # skip any already chosen
        candidates = [c for c in candidates if c not in chosen]
        if len(candidates) > 0:
            chosen.append(int(candidates[0]))
    chosen = chosen[:6]

    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    index = []
    for n, idx in enumerate(chosen):
        wafer = maps[idx]
        truth = [LABEL_NAMES[i] for i, v in enumerate(labels[idx]) if v == 1]

        # Save the raw array (so the app can run detection on it) ...
        npy_path = os.path.join(EXAMPLES_DIR, f"example_{n}.npy")
        np.save(npy_path, wafer)

        # ... and a PNG thumbnail (so the gallery can show it).
        png_path = os.path.join(EXAMPLES_DIR, f"example_{n}.png")
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.imshow(wafer, cmap="viridis", vmin=0, vmax=2)
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(png_path, dpi=80, bbox_inches="tight")
        plt.close(fig)

        index.append({"id": n, "npy": f"example_{n}.npy",
                      "png": f"example_{n}.png", "true_defects": truth})
        print(f"example_{n}: test idx {idx}, defects {truth}")

    with open(os.path.join(EXAMPLES_DIR, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nSaved {len(index)} examples to {EXAMPLES_DIR}")


if __name__ == "__main__":
    main()