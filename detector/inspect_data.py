"""
detector/inspect_data.py -- read-only inspection of the MixedWM38 dataset.

Opens the raw .npz and reports what is inside, then digs into the pixel-value
question: the documentation says maps use values 0/1/2, but the file also
contains 3. This script counts how often each value occurs and saves sample
maps that contain a 3, so its meaning can be determined by looking rather
than guessing. Writes only a figure into outputs/figures; changes no data.

Run from the project root:
    python detector/inspect_data.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display on the server; render straight to file
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_PATH = os.path.join(ROOT_DIR, "data", "raw", "Wafer_Map_Datasets.npz")
FIG_DIR = os.path.join(ROOT_DIR, "outputs", "figures")


def main():
    print("Loading:", NPZ_PATH)
    data = np.load(NPZ_PATH)
    maps = data["arr_0"]     # (N, 52, 52) pixel grids
    labels = data["arr_1"]   # (N, 8) multi-hot labels

    print("\nmaps  :", maps.shape, maps.dtype)
    print("labels:", labels.shape, labels.dtype)

    # --- Pixel-value frequency: how common is each of 0/1/2/3? ---
    values, counts = np.unique(maps, return_counts=True)
    total = maps.size
    print("\nPixel-value counts (across all maps):")
    for v, c in zip(values, counts):
        print(f"  value {v}: {c:>12,d}  ({100.0 * c / total:5.2f} %)")

    # --- How many maps contain at least one '3'? ---
    has_three = (maps == 3).any(axis=(1, 2))   # bool per wafer map
    print(f"\nMaps containing at least one '3': {has_three.sum():,} "
          f"of {maps.shape[0]:,} "
          f"({100.0 * has_three.sum() / maps.shape[0]:.2f} %)")

    # --- Save a few sample maps that contain a '3', so we can see it ---
    os.makedirs(FIG_DIR, exist_ok=True)
    idxs = np.where(has_three)[0][:6]   # first six such maps
    if idxs.size > 0:
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        for ax, i in zip(axes.ravel(), idxs):
            # Distinct colors per value so 0/1/2/3 are visually separable.
            ax.imshow(maps[i], cmap="viridis", vmin=0, vmax=3)
            ax.set_title(f"idx {i}  label {labels[i]}", fontsize=7)
            ax.axis("off")
        fig.suptitle("Sample wafer maps containing pixel value 3")
        out = os.path.join(FIG_DIR, "pixel_value_3_samples.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print("\nSaved sample figure to:", out)
    else:
        print("\nNo maps contain a 3 (unexpected given max==3).")


if __name__ == "__main__":
    main()