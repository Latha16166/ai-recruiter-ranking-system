"""
Thin wrapper around the Google Gemini API.
Centralizes retries, JSON extraction, and error handling.
"""

import json
import re
import time

from google import genai

from src import config

_client = None


def get_client():
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Run `export GEMINI_API_KEY=...` before running."
            )
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    if not (text.startswith("{") or text.startswith("[")):
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            text = match.group(1)

    return json.loads(text)


def ask_gemini_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> dict:

    client = get_client()
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=f"{system_prompt}\n\n{user_prompt}",
                config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                },
            )

            return _extract_json(response.text)

        except Exception as e:
            last_error = e
            wait = min(2 ** attempt, 20)
            print(
                f"[Gemini API error, attempt {attempt}/{max_retries}: {e}. Retrying in {wait}s]"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Gemini call failed after {max_retries} attempts: {last_error}"
    )