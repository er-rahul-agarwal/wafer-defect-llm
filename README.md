# wafer-defect-llm

A conversational wafer-defect analysis assistant: a fine-tuned Vision
Transformer detects defect patterns on semiconductor wafer maps, and a
locally-run open-source LLM reasons about the results through tool-calling,
so you can have a grounded conversation about a specific wafer.

## Status

Early development. Project structure and conventions in place; detector
subsystem next.

## Architecture (two subsystems + interface)

- detector/   Vision Transformer fine-tuned on MixedWM38 (image -> defect patterns)
- assistant/  LLM layer: Ollama client, the detect_defects tool, the tool-calling loop
- app/        FastAPI backend + Gradio UI
- eval/       evaluation: detector metrics, and LLM tool-call success / groundedness

## Setup

TODO once the environment is pinned.

## References

TODO add dataset and model citations.