"""
Base interface for all LLM clients.
"""

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM clients.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: Prompt to send to the model.

        Returns:
            Raw response string from the model.
        """
        raise NotImplementedError