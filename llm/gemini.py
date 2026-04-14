from __future__ import annotations

import os
from google import genai
from prompts.presets import PRESET_PROMPTS


DEFAULT_MODEL_NAME = "gemini-2.5-flash"


def _build_prompt(labels: list[str], preset: str) -> str:
    if preset not in PRESET_PROMPTS:
        raise ValueError(f"Unsupported preset: {preset}")
    labels_text = ", ".join(labels) if labels else "none"
    return PRESET_PROMPTS[preset].format(labels=labels_text)


def generate_feedback(labels: list[str], preset: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "GEMINI_API_KEY is not set. Add it to your environment or .env file to enable AI feedback."

    prompt = _build_prompt(labels, preset)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=DEFAULT_MODEL_NAME,
        contents=prompt,
    )
    if not response or not getattr(response, "text", None):
        return "No response returned by Gemini."
    return response.text
