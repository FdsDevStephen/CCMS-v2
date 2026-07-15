"""
LLM Client Factory

Returns the configured LLM client.
"""

from config import LLM_PROVIDER
from extractor.llm.base import BaseLLMClient
from extractor.llm.ollama_client import OllamaClient


def get_llm_client() -> BaseLLMClient:
    """
    Return the configured LLM client.

    Returns:
        BaseLLMClient
    """

    if LLM_PROVIDER.lower() == "ollama":
        return OllamaClient()

    raise ValueError(f"Unsupported LLM Provider: {LLM_PROVIDER}")