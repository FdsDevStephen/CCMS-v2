"""
Simple "/generate" LLM Client

For hosted inference servers with this API (the vLLM Qwen2.5-3B wrapper,
the Qwen3-14B "qwen13" and Gemma servers):

    POST {GENERATE_URL}   {"prompt", "max_tokens", "temperature"}
    ->                    {"text", "reasoning"?, "tokens_generated", ...}

These servers have no system-prompt field and no JSON mode, so the JSON
instructions are prepended to the prompt and markdown fences are stripped
from the reply.

Qwen3 "thinks" before answering by default, which made it ~15x slower
and sometimes ran out of tokens before the answer. GENERATE_NO_THINK
appends Qwen3's "/no_think" switch to turn that off.
"""

from __future__ import annotations

import re
import time

import requests

from config.settings import (
    GENERATE_MAX_TOKENS,
    GENERATE_NO_THINK,
    GENERATE_REPETITION_PENALTY,
    GENERATE_URL,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
)
from extractor.llm.base import BaseLLMClient

SYSTEM_INSTRUCTIONS = (
    "You are a JSON extraction engine.\n"
    "Return ONLY valid JSON.\n"
    "Never explain. Never summarize. Never use markdown.\n"
    "Output exactly one JSON object.\n\n"
)


class GenerateClient(BaseLLMClient):
    """
    "/generate" API implementation of the BaseLLMClient.
    """

    def __init__(self) -> None:
        self.url = GENERATE_URL
        self.session = requests.Session()

    def generate(
        self,
        prompt: str,
    ) -> str:

        prompt = SYSTEM_INSTRUCTIONS + prompt

        if GENERATE_NO_THINK:
            prompt += "\n\n/no_think"

        payload = {
            "prompt": prompt,
            "max_tokens": GENERATE_MAX_TOKENS,
            "temperature": 0,
            "repetition_penalty": GENERATE_REPETITION_PENALTY,
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

                return self._clean(
                    response.json()["text"]
                )

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
    def _clean(text: str) -> str:
        """Drop <think> blocks and ```json fences around the answer."""
        text = re.sub(
            r"<think>.*?(?:</think>|$)",
            "",
            text or "",
            flags=re.DOTALL,
        )
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()
