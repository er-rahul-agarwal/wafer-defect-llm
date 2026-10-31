"""
detector/infer.py -- clean inference for the trained wafer detector.

One job: take a single wafer map (a 52x52 array of codes 0/1/2) and return
which of the 8 base defect types are present, with confidences. This is the
function the LLM's tool will call -- the bridge between the vision subsystem
and the language subsystem.

The model is loaded once and cached, so repeated calls are fast.

Self-test (Section 8): run directly to predict on a real test-set sample.
    python detector/infer.py
"""

import os
import sys

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import config
from detector.model import WaferViT

# The 8 base defect types, in label-column order (same as evaluate.py).
LABEL_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
               "Loc", "Random", "Scratch", "Near-Full"]

# Module-level cache so the model loads from disk only once.
_model = None
_device = None


def _get_model():
    """Load the trained model once and reuse it on later calls."""
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = WaferViT().to(_device)
        best_path = os.path.join(config.MODEL_DIR, "wafer_vit_best.pt")
        _model.load_state_dict(torch.load(best_path, map_location=_device))
        _model.eval()
    return _model, _device


def detect_defects(wafer_map, threshold=0.5):
    """Predict defect types present on one wafer map.

    Args:
        wafer_map: a 52x52 array-like of pixel codes (0 blank, 1 pass, 2 fail).
                   Stray value 3, if present, is clipped to 2 (same rule as
                   training preprocessing).
        threshold: probability above which a defect is reported as present.

    Returns:
        dict with:
          'defects'      : list of defect-type names predicted present
          'confidences'  : {defect_name: probability} for ALL 8 types
          'any_defect'   : True if at least one type is present
    """
    model, device = _get_model()

    # Prepare the input exactly as training did: clip, normalize, add dims.
    arr = np.asarray(wafer_map, dtype=np.float32)
    arr = np.clip(arr, 0, config.PIXEL_CLIP_MAX)       # same clip rule
    arr = arr / config.NORMALIZE_DIV                   # codes -> [0,1]
    tensor = torch.from_numpy(arr)[None, None, :, :].to(device)  # (1,1,52,52)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor))[0].cpu().numpy()   # 8 probabilities

    confidences = {name: float(p) for name, p in zip(LABEL_NAMES, probs)}
    defects = [name for name, p in confidences.items() if p >= threshold]
    return {
        "defects": defects,
        "confidences": confidences,
        "any_defect": len(defects) > 0,
    }


# ----------------------------------------------------------------------
# Self-test: run inference on a real test-set sample and print the result.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Load one real wafer map from the processed test split.
    test = np.load(os.path.join(config.PROCESSED_DIR, "test.npz"))
    sample_map = test["maps"][0]          # raw 52x52 codes
    true_label = test["labels"][0]        # ground-truth multi-hot

    result = detect_defects(sample_map)

    print("Predicted defects:", result["defects"])
    print("Any defect:", result["any_defect"])
    print("\nConfidences:")
    for name, conf in result["confidences"].items():
        print(f"  {name:<10s} {conf:.3f}")

    # Show ground truth for comparison.
    true_names = [LABEL_NAMES[i] for i, v in enumerate(true_label) if v == 1]
    print("\nGround-truth defects:", true_names)