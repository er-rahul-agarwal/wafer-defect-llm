"""
detector/inspect_data.py -- read-only inspection of the MixedWM38 dataset.

Opens the raw .npz and reports what is actually inside: array names, shapes,
dtypes, value ranges, sample count, and label structure. Writes nothing and
changes nothing. Its purpose is to replace assumptions with measured facts
before any config value or model code depends on them.

Run from the project root:
    python detector/inspect_data.py
"""

import os
import numpy as np

# Build the path from this file's location so it works regardless of
# where the script is launched from.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_PATH = os.path.join(ROOT_DIR, "data", "raw", "Wafer_Map_Datasets.npz")


def main():
    print("Loading:", NPZ_PATH)
    # allow_pickle stays False (the default) -- this data is plain arrays,
    # and keeping pickle off is the safe choice for any downloaded file.
    data = np.load(NPZ_PATH)

    # The archive stores named arrays. Report every key it contains.
    print("\nArrays in the archive:", list(data.keys()))

    for key in data.keys():
        arr = data[key]
        print(f"\n--- '{key}' ---")
        print("  shape:", arr.shape)
        print("  dtype:", arr.dtype)
        print("  min / max:", arr.min(), "/", arr.max())
        # Unique values only when there are few of them (labels, pixel codes),
        # so we do not dump thousands of floats.
        uniq = np.unique(arr)
        if uniq.size <= 20:
            print("  unique values:", uniq)
        else:
            print("  unique values: (", uniq.size, "distinct -- too many to list )")


if __name__ == "__main__":
    main()