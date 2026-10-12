"""
LLM Client Factory

Returns the configured LLM client.
"""

from app.config import LLM_PROVIDER
from extractor.llm.base import BaseLLMClient
from extractor.llm.ollama_client import OllamaClient


def get_llm_client() -> BaseLLMClient:
    """
    Return the configured LLM client.

    Returns:
        BaseLLMClient
    """

    provider = LLM_PROVIDER.lower()

    if provider == "ollama":
        return OllamaClient()

    if provider == "openai":
        from extractor.llm.openai_client import OpenAICompatibleClient

        return OpenAICompatibleClient()

    if provider in ("generate", "gemma"):
        from extractor.llm.generate_client import GenerateClient

        return GenerateClient()

    raise ValueError(f"Unsupported LLM Provider: {LLM_PROVIDER}")