"""Prompts for LightOnOCR-3. Only these two are in distribution (model card, D4)."""

from __future__ import annotations

# Exact string from https://huggingface.co/lightonai/LightOnOCR-3-0.8B (`prompt = "grounding"`).
GROUNDING_PROMPT = "grounding"
# Empty prompt = image only = plain markdown transcription.
TRANSCRIBE_PROMPT = ""
