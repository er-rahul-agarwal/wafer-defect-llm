"""
config.py -- single source of truth for the wafer-defect-llm project.

Every path, hyperparameter, and setting lives here. No hardcoded values
anywhere else. Values are tagged:
  [DATA]   set from a dataset finding (filled after data inspection)
  [DESIGN] an engineering choice, justified in a comment
  [GUESS]  a starting point that needs tuning
  TODO     not knowable yet; names the step that will set it
"""

import os

# ----------------------------------------------------------------------
# PATHS -- built from ROOT_DIR so the project runs on any machine
# ----------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")
PROMPT_DIR = os.path.join(ROOT_DIR, "prompts")

# ----------------------------------------------------------------------
# DETECTOR -- the vision model (subsystem 1)
# ----------------------------------------------------------------------
DATASET_NAME = "MixedWM38"          # [DESIGN] 38-class, mixed defects
# --- Values below measured by detector/inspect_data.py on 2026-10-31 ---
NUM_LABELS = 8                      # [DATA] arr_1 is 8-dim multi-hot (8 base defect types)
IMAGE_SIZE = 52                     # [DATA] arr_0 maps are 52 x 52
NUM_SAMPLES = 38015                 # [DATA] total wafer maps in the archive

# Pixel codes in the raw maps (arr_0):
PIXEL_BLANK = 0                     # [DATA] background / outside wafer
PIXEL_PASS = 1                      # [DATA] die passed electrical test
PIXEL_FAIL = 2                      # [DATA] die failed electrical test
# Value 3 appears in 214 pixels across 105 maps (0.0002% of pixels): a stray
# GAN edge artifact, not a documented code. Preprocessing clips 3 -> 2
# (nearest valid code). Applied identically to train and inference data.
PIXEL_CLIP_MAX = 2                  # [DESIGN] clip any value above this down to it
BASE_MODEL = None                   # TODO choose pretrained ViT tag in detector/model.py
BATCH_SIZE = None                   # [GUESS] set in detector/train once VRAM is checked
LEARNING_RATE = None                # [GUESS] set in detector/train
NUM_EPOCHS = None                   # [GUESS] set in detector/train

# ----------------------------------------------------------------------
# ASSISTANT -- the LLM layer (subsystem 2)
# ----------------------------------------------------------------------
LLM_MODEL = None                    # TODO set in assistant/llm.py (confirm exact Ollama tag at pull time)
LLM_TEMPERATURE = 0.0               # [DESIGN] low temp for consistent tool-calling
OLLAMA_HOST = "http://localhost:11434"   # [DESIGN] default Ollama endpoint on the server

# ----------------------------------------------------------------------
# RAG (stretch) -- retrieval settings
# ----------------------------------------------------------------------
EMBEDDING_MODEL = None              # TODO set when RAG is added
CHUNK_SIZE = None                   # [GUESS] set when RAG is added
TOP_K = None                        # [GUESS] set when RAG is added

# ----------------------------------------------------------------------
# EXECUTION
# ----------------------------------------------------------------------
DEVICE = "cuda"                     # [DESIGN] GPU server; pin a card with CUDA_VISIBLE_DEVICES
SEED = 42                           # [DESIGN] fixed seed where determinism is possible
