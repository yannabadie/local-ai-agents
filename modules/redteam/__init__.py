"""
Red-teaming module for cloud LLM security testing.

This module provides tools for testing security guardrails of cloud LLM APIs
including OpenAI GPT-4, Anthropic Claude, and Google Gemini.

Components:
- CloudClient: Unified interface for cloud LLM APIs
- TAP: Tree of Attacks with Pruning implementation
- PyRIT wrapper: Microsoft's red-teaming toolkit integration
- Garak wrapper: NVIDIA's vulnerability scanner integration
"""

from .cloud_client import CloudClient, CloudProvider
from .tap import TAPOrchestrator, TAPConfig

__all__ = [
    "CloudClient",
    "CloudProvider",
    "TAPOrchestrator",
    "TAPConfig",
]
