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

        self.client = ollama.Client(
            host=OLLAMA_HOST
        )


    # ======================================================
    # GENERATE
    # ======================================================

    def generate(
        self,
        prompt: str,
    ) -> str:

        last_error = None


        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

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


                return (
                    response[
                        "message"
                    ][
                        "content"
                    ].strip()
                )


            except Exception as e:

                last_error = e

                error_text = str(
                    e
                ).lower()


                # ==========================================
                # CUDA / GPU OUT OF MEMORY
                #
                # DO NOT RETRY.
                # ==========================================

                if (
                    "cudaMalloc" in str(e)
                    or "out of memory"
                    in error_text
                    or "unable to allocate"
                    in error_text
                    or "cuda" in error_text
                    and "memory"
                    in error_text
                ):

                    print(
                        "[Ollama] GPU "
                        "out of memory.",
                        flush=True,
                    )

                    print(
                        "[Ollama] "
                        "BGE-M3 may still "
                        "be occupying GPU "
                        "memory.",
                        flush=True,
                    )

                    raise RuntimeError(
                        "Ollama could not "
                        "start because the "
                        "GPU is out of memory. "
                        "Release BGE-M3 GPU "
                        "memory before "
                        "calling Ollama."
                    ) from e


                # ==========================================
                # NORMAL FAILURE
                # ==========================================

                print(
                    f"[Ollama] Attempt "
                    f"{attempt}/"
                    f"{MAX_RETRIES} "
                    f"failed: {e}",
                    flush=True,
                )


                if (
                    attempt
                    < MAX_RETRIES
                ):

                    time.sleep(
                        2
                    )


        raise RuntimeError(
            "Failed to communicate "
            f"with Ollama after "
            f"{MAX_RETRIES} attempts."
        ) from last_error