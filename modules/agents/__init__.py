"""Agent module for environment interaction."""

from .base_agent import BaseAgent
from .ollama_agent import OllamaAgent

__all__ = ["BaseAgent", "OllamaAgent"]
