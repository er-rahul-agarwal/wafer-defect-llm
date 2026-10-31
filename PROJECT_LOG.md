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