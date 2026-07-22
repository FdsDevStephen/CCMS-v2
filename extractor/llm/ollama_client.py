"""
Ollama LLM Client

Responsible only for communicating with the Ollama server.
"""

from __future__ import annotations

import time

import ollama

from app.config import (
    MODEL_NAME,
    OLLAMA_HOST,
    MAX_RETRIES,
)
from extractor.llm.base import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """
    Ollama implementation of the BaseLLMClient.
    """

    def __init__(self) -> None:
        self.client = ollama.Client(host=OLLAMA_HOST)

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to Ollama and return the raw response.

        Args:
            prompt: Prompt to send to the model.

        Returns:
            Raw response text from the model.

        Raises:
            RuntimeError: If all retry attempts fail.
        """

        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):

            try:
                response = self.client.chat(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a JSON extraction engine.\n"
                                "Return ONLY valid JSON.\n"
                                "Never explain.\n"
                                "Never summarize.\n"
                                "Never use markdown.\n"
                                "Output exactly one JSON object."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    options={
                        "temperature": 0,
                        "top_p": 1,
                    },
                    format="json",
                )

                return response["message"]["content"].strip()

            except Exception as e:
                last_error = e

                print(f"[Ollama] Attempt {attempt}/{MAX_RETRIES} failed: {e}")

                if attempt < MAX_RETRIES:
                    time.sleep(2)

        raise RuntimeError(
            f"Failed to communicate with Ollama after {MAX_RETRIES} attempts."
        ) from last_error
