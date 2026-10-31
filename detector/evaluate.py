"""
detector/evaluate.py -- final, unbiased evaluation on the held-out test set.

Loads the best model saved during training and evaluates it ONCE on
data/processed/test.npz -- data untouched until now (project rule, Section 4).
Reports overall macro-F1, per-label F1 for all 8 base defect types, and a
bootstrap 95% confidence interval on macro-F1 (Section 10).

Usage (from project root):
    CUDA_VISIBLE_DEVICES=0 python detector/evaluate.py
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import config
from detector.model import WaferViT

BAR_FORMAT = "{desc}: [{bar}] {percentage:3.0f}% | {n_fmt}/{total_fmt} | ETA {remaining}"
TQDM_ASCII = " %"

# The 8 base defect types, in label-column order (MixedWM38 convention).
LABEL_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
               "Loc", "Random", "Scratch", "Near-Full"]


def load_test():
    path = os.path.join(config.PROCESSED_DIR, "test.npz")
    data = np.load(path)
    maps = data["maps"].astype(np.float32) / config.NORMALIZE_DIV
    maps = maps[:, None, :, :]
    labels = data["labels"].astype(np.float32)
    return torch.from_numpy(maps), torch.from_numpy(labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- Load the held-out test set ---
    test_x, test_y = load_test()
    loader = DataLoader(TensorDataset(test_x, test_y),
                        batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Test set: {len(test_x):,} wafer maps")

    # --- Load the best trained model ---
    model = WaferViT().to(device)
    best_path = os.path.join(config.MODEL_DIR, "wafer_vit_best.pt")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    print("Loaded:", best_path)

    # --- Predict over the test set ---
    all_true, all_pred = [], []
    with torch.no_grad():
        bar = tqdm(loader, bar_format=BAR_FORMAT, ascii=TQDM_ASCII,
                   desc="Evaluating", leave=False)
        for x, y in bar:
            probs = torch.sigmoid(model(x.to(device)))
            all_pred.append((probs > 0.5).int().cpu().numpy())
            all_true.append(y.int().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    # --- Overall macro-F1 ---
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    print(f"\nTest macro-F1: {macro:.4f}")
    print(f"Test micro-F1: {micro:.4f}")

    # --- Per-label F1 (which defects are easy vs hard) ---
    per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
    print("\nPer-label F1:")
    for name, score in zip(LABEL_NAMES, per_label):
        support = int(y_true[:, LABEL_NAMES.index(name)].sum())
        print(f"  {name:<10s} {score:.4f}   (present in {support:,} test maps)")

    # --- Bootstrap 95% CI on macro-F1 (Section 10) ---
    rng = np.random.default_rng(config.SEED)
    n = len(y_true)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)              # resample with replacement
        boot.append(f1_score(y_true[idx], y_pred[idx],
                             average="macro", zero_division=0))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nMacro-F1 95% CI (bootstrap, 1000x): [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()