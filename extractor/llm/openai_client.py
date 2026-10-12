"""
OpenAI-compatible LLM Client

Works with any server that speaks the OpenAI chat-completions API, e.g.
the hosted vLLM Qwen:

    POST {OPENAI_BASE_URL}/chat/completions
    {"model", "messages", "temperature", "max_tokens"}
"""

from __future__ import annotations

import re
import time

import requests

from config.settings import (
    MAX_RETRIES,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    REQUEST_TIMEOUT,
)
from extractor.llm.base import BaseLLMClient

SYSTEM_PROMPT = (
    "You are a JSON extraction engine.\n"
    "Return ONLY valid JSON.\n"
    "Never explain.\n"
    "Never summarize.\n"
    "Never use markdown.\n"
    "Output exactly one JSON object."
)


class OpenAICompatibleClient(BaseLLMClient):
    """
    OpenAI-compatible implementation of the BaseLLMClient.
    """

    def __init__(self) -> None:
        self.url = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
        self.session = requests.Session()

        if OPENAI_API_KEY:
            self.session.headers["Authorization"] = (
                f"Bearer {OPENAI_API_KEY}"
            )

    def generate(
        self,
        prompt: str,
    ) -> str:

        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": OPENAI_MAX_TOKENS,
        }

        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):

            try:
                response = self.session.post(
                    self.url,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()

                content = (
                    response.json()["choices"][0]["message"]["content"]
                )

                return self._strip_fences(content)

            except Exception as e:
                last_error = e

                print(
                    f"[LLM API] Attempt {attempt}/{MAX_RETRIES} "
                    f"failed: {e}",
                    flush=True,
                )

                if attempt < MAX_RETRIES:
                    time.sleep(2)

        raise RuntimeError(
            f"Failed to get a response from {self.url} "
            f"after {MAX_RETRIES} attempts."
        ) from last_error

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove ```json ... ``` wrappers the model sometimes adds."""
        text = (text or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()
