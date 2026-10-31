"""
detector/data_prep.py -- turn the raw MixedWM38 archive into clean, split,
model-ready arrays.

Reads data/raw/Wafer_Map_Datasets.npz (never modifies it), applies the logged
cleaning rule (clip pixel value 3 -> 2), splits 70/15/15 into train/val/test
with multi-label-aware stratification, and saves each split to
data/processed/. Training code reads only from data/processed, so raw stays
read-only (project rule, Section 4).

Run from the project root:
    python detector/data_prep.py
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

import config  # single source of truth for paths and constants

RAW_PATH = os.path.join(config.RAW_DIR, "Wafer_Map_Datasets.npz")


def main():
    print("Loading raw data:", RAW_PATH)
    data = np.load(RAW_PATH)
    maps = data["arr_0"]       # (38015, 52, 52) int32, pixel codes 0/1/2 (and stray 3)
    labels = data["arr_1"]     # (38015, 8)       int32, multi-hot over 8 base defects

    # --- Cleaning: clip the stray GAN artifact (value 3) down to 2. ---
    # Logged decision (see PROJECT_LOG 2026-10-31). Affects 214 pixels total.
    n_threes = int((maps == 3).sum())
    maps = np.clip(maps, 0, config.PIXEL_CLIP_MAX)
    print(f"Clipped {n_threes} pixels of value 3 -> {config.PIXEL_CLIP_MAX}.")

    # --- Stratification key for multi-label data ---
    # Each wafer's 8-bit label is turned into a single integer (bit pattern),
    # so train_test_split can balance those combination-classes across splits.
    strat_key = np.packbits(labels.astype(np.uint8), axis=1).ravel()

    # --- First split: 70% train, 30% temp (to be halved into val/test) ---
    idx = np.arange(maps.shape[0])
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, random_state=config.SEED, stratify=strat_key
    )
    # --- Second split: halve temp into 15% val, 15% test ---
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=config.SEED,
        stratify=strat_key[temp_idx]
    )

    print(f"Split sizes -- train: {len(train_idx):,}  "
          f"val: {len(val_idx):,}  test: {len(test_idx):,}")

    # --- Save each split to data/processed (raw is never written to) ---
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    for name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        out = os.path.join(config.PROCESSED_DIR, f"{name}.npz")
        np.savez(out, maps=maps[split_idx], labels=labels[split_idx])
        print(f"Saved {name}: {out}")


if __name__ == "__main__":
    main()