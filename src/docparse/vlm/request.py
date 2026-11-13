"""OpenAI chat request building and response extraction. Raw model output is stored unparsed."""

from __future__ import annotations

from typing import Any

from docparse.vlm.prompts import GROUNDING_PROMPT

DEFAULT_MODEL = "lightonai/LightOnOCR-3-0.8B"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 4096


def to_request(
    row: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    prompt: str = GROUNDING_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Build the chat body for one page row (uses ``row["png_b64"]``)."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{row['png_b64']}"}}
    ]
    if prompt:
        content.append({"type": "text", "text": prompt})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }


def parse(row: dict[str, Any], resp: dict[str, Any]) -> dict[str, Any]:
    """Extract raw text, finish reason, token counts and model. No grammar parsing here."""
    choice = resp["choices"][0]
    usage = resp.get("usage") or {}
    return {
        "raw": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason") or "",
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "model": resp.get("model", ""),
    }
