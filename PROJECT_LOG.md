# PROJECT LOG -- wafer-defect-llm

A running record of everything done, as it is done. Negative results, ties,
and caveats go here too -- not just the wins.

## 2026-10-31 -- Project setup
- Created local repo at D:\post_doc\wafer-defect-llm (git init).
- Set git identity (Rahul Agarwal / er.rahul.agarwal03@gmail.com).
- Added .gitignore (data, outputs, runs, weights, venv, secrets excluded).
- Created component-based folder structure: detector/, assistant/, app/,
  eval/, prompts/, data/, outputs/, notebooks/.
- Added config.py skeleton (single source of truth, values tagged / TODO).
- Decisions settled: ViT-Tiny on MixedWM38 (multi-label) as detector;
  Qwen via Ollama (8B dev, 30B for quality) as LLM; tool-calling as core;
  RAG and evaluation promoted to core (per job-description findings).

## 2026-10-31 -- Dataset acquired and inspected (MixedWM38)
- Downloaded Wafer_Map_Datasets.npz (394 MB) from the authors' Google Drive
  (Junliangwangdhu/WaferMap, corrected C7/C9 labels) into data/raw/.
- Inspected with detector/inspect_data.py (read-only). Measured facts:
  - 38,015 wafer maps, each 52 x 52 (arr_0, int32).
  - Labels arr_1: 8-dim multi-hot, values {0,1}. Confirms multi-label framing.
  - Pixel codes: 0 blank (23.7%), 1 pass (51.9%), 2 fail (24.4%).
  - FINDING: value 3 in 214 pixels across 105 maps (0.0002%). Visual check
    (outputs/figures/pixel_value_3_samples.png) shows isolated single pixels
    at the wafer edge -> stray GAN artifact, no signal.
  - DECISION: clip 3 -> 2 in preprocessing (nearest valid code), applied
    identically to train and inference. Negligible impact; logged for traceability.
- config.py updated with measured values (IMAGE_SIZE, NUM_LABELS, pixel codes,
  clip rule).

## 2026-10-31 -- Data prep: cleaned and split
- detector/data_prep.py: clipped 214 value-3 pixels -> 2 (as logged).
- Stratified 70/15/15 split (multi-label aware, via packed-bit key, SEED=42):
  train 26,610 / val 5,702 / test 5,703 = 38,015 total.
- Saved to data/processed/{train,val,test}.npz (gitignored, server-only).
- Raw data never modified.

## 2026-10-31 -- ViT detector trained (first full run)
- Small ViT from scratch (1,214,984 params), 30 epochs, ~13s/epoch on RTX 6000 Ada (card 0).
- Multi-label (8 base defects), BCEWithLogitsLoss, AdamW lr=3e-4, batch 128.
- Best VALIDATION macro-F1: 0.9891. Best model saved to outputs/models/wafer_vit_best.pt.
- CAVEAT: validation score (used for model selection). Test-set evaluation pending
  for the unbiased number. Per-label breakdown + CI still to compute.

## 2026-10-31 -- Detector evaluated on held-out test set (FINAL)
- Test macro-F1: 0.9850 | micro-F1: 0.9837 | 95% CI [0.9762, 0.9894] (bootstrap 1000x).
- Validation was 0.9891 -> small gap to test = good generalization, no overfitting.
- Per-label F1 all 0.97-0.997; strong even on rare classes (Random: 0.978 on 23 maps).
- Label-name order appears correct (rare classes Random/Near-Full match known
  MixedWM38 distribution); high-confidence, not formally certified.
- DETECTOR SUBSYSTEM COMPLETE. Model: outputs/models/wafer_vit_best.pt.

## 2026-10-31 -- LLM runtime installed (Ollama + Qwen)
- Ollama installed user-local (no sudo) to ~/.local, server running via nohup.
  Models stored on ~/storage/ollama_models. Client v0.40.13.
- Note: .tgz download URL now 404s; current packages are .tar.zst (zstd present on server).
- Pulled qwen3.5:9b (6.6 GB, Q4_K_M, tool-calling capable) as the dev model.
  Plan: scale to qwen3.8:27b later for a quality comparison.
- Verified it responds (thinking disabled via --think=false for speed/consistency).
- config.py: LLM_MODEL still None -> set to "qwen3.5:9b" next.

## 2026-10-31 -- Tool-calling loop working END TO END (core complete)
- Built the LLM layer: assistant/llm.py (Ollama client), assistant/tools.py
  (detect_defects tool: schema + impl, Layer 3), assistant/orchestrator.py
  (tool-calling loop, Layer 5), prompts/system_prompt.txt (versioned).
- detect_defects takes no image arg; orchestrator holds the wafer out-of-band
  and sets it on the tool layer. LLM only decides WHETHER to detect.
- Verified end to end: asked about test sample 2, Qwen called the tool, got
  {Center, Edge-Ring, Loc} + confidences, and explained them grounded in the
  real detector output (quoted the actual confidence scores). No hallucination.
- CORE SYSTEM COMPLETE: detector (0.985 F1) + Qwen + tool-calling connected.