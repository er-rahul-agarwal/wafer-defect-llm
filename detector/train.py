"""
detector/train.py -- train the small ViT on MixedWM38 (multi-label).

Loads the processed train/val splits, trains WaferViT with BCE-with-logits
loss (correct for multi-label: each of the 8 defects is an independent
yes/no), shows a progress bar per epoch, evaluates on validation each epoch,
and saves the best model (by validation macro-F1) to outputs/models/.

Smoke-test first (Section 18): run with --smoke to train 1 epoch on a small
slice, confirm it runs end to end, and read the real per-epoch time before
launching a full run.

Usage (from project root), pinning to one GPU:
    CUDA_VISIBLE_DEVICES=0 python detector/train.py --smoke
    CUDA_VISIBLE_DEVICES=0 python detector/train.py
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import config
from detector.model import WaferViT

# Your requested progress-bar format: [%%%%....] 62% | 830/1331 | ETA 00:05
BAR_FORMAT = "{desc}: [{bar}] {percentage:3.0f}% | {n_fmt}/{total_fmt} | ETA {remaining}"
# Use % for filled and . for empty, matching the requested bar style.
TQDM_ASCII = "%."


def load_split(name):
    """Load one processed split as (maps, labels) float tensors.

    Maps are normalized from codes {0,1,2} to [0,1] by dividing by 2, and
    given a channel dimension -> (N, 1, 52, 52). Labels become floats for
    BCE loss.
    """
    path = os.path.join(config.PROCESSED_DIR, f"{name}.npz")
    data = np.load(path)
    maps = data["maps"].astype(np.float32) / config.NORMALIZE_DIV
    maps = maps[:, None, :, :]                       # add channel dim
    labels = data["labels"].astype(np.float32)
    return torch.from_numpy(maps), torch.from_numpy(labels)


def evaluate(model, loader, device):
    """Run the model over a loader, return macro-F1 across the 8 labels.

    A prediction is 'present' when its sigmoid probability exceeds 0.5.
    Macro-F1 averages the per-label F1 equally, so rare defects count as
    much as common ones -- the honest metric for imbalanced multi-label.
    """
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x))
            preds = (probs > 0.5).int().cpu().numpy()
            all_pred.append(preds)
            all_true.append(y.int().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch on a small slice, to test and time.")
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device,
          "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")

    # --- Data ---
    train_x, train_y = load_split("train")
    val_x, val_y = load_split("val")

    if args.smoke:
        # Tiny slice + a single epoch: just enough to prove it runs and time it.
        train_x, train_y = train_x[:1024], train_y[:1024]
        val_x, val_y = val_x[:512], val_y[:512]
        epochs = 1
        print(f"SMOKE TEST: train {len(train_x)}, val {len(val_x)}, 1 epoch.")
    else:
        epochs = config.NUM_EPOCHS

    train_loader = DataLoader(TensorDataset(train_x, train_y),
                              batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y),
                            batch_size=config.BATCH_SIZE, shuffle=False)

    # --- Model, loss, optimizer ---
    model = WaferViT().to(device)
    criterion = nn.BCEWithLogitsLoss()               # multi-label loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    best_f1 = 0.0
    best_path = os.path.join(config.MODEL_DIR, "wafer_vit_best.pt")

    # --- Training loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        bar = tqdm(train_loader, bar_format=BAR_FORMAT, ascii=TQDM_ASCII,
                   desc=f"Epoch {epoch}/{epochs}")
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()

        val_f1 = evaluate(model, val_loader, device)
        dt = time.time() - t0
        avg_loss = running / len(train_loader)
        print(f"Epoch {epoch}: loss {avg_loss:.4f} | val macro-F1 {val_f1:.4f} "
              f"| {dt:.1f}s")

        # Save the best model by validation F1.
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"  new best (F1 {best_f1:.4f}) saved -> {best_path}")

    print(f"\nDone. Best val macro-F1: {best_f1:.4f}")
    if args.smoke:
        print("Smoke test OK. Read the per-epoch seconds above to size the full run.")


if __name__ == "__main__":
    main()