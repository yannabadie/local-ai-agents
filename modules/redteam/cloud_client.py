"""
Unified client for cloud LLM APIs.

Provides a single interface for interacting with:
- OpenAI (GPT-4, GPT-4-turbo, GPT-4o)
- Anthropic (Claude 3 Opus, Sonnet, Haiku)
- Google (Gemini Pro, Ultra)
"""

import os
import time
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
import logging

# Conditional imports for cloud providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"  # Local fallback


@dataclass
class CloudResponse:
    """Standardized response from cloud APIs."""
    content: str
    model: str
    provider: CloudProvider
    tokens_used: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""
    raw_response: Any = None
    cost_usd: float = 0.0


@dataclass
class CloudConfig:
    """Configuration for cloud API connections."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"

    # Rate limiting
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True

    # Cost tracking
    track_costs: bool = True
    max_cost_usd: float = 10.0  # Safety limit

    @classmethod
    def from_env(cls) -> "CloudConfig":
        """Load configuration from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "CloudConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Resolve environment variables in YAML
        def resolve_env(value: str) -> str:
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.getenv(env_var, "")
            return value

        return cls(
            openai_api_key=resolve_env(data.get("openai", {}).get("api_key", "")),
            anthropic_api_key=resolve_env(data.get("anthropic", {}).get("api_key", "")),
            google_api_key=resolve_env(data.get("google", {}).get("api_key", "")),
        )


# Pricing per 1M tokens (January 2026 - verified)
PRICING = {
    # OpenAI models (January 2026)
    # GPT-5.2 series (released December 2025) - flagship models
    "gpt-5.2": {"input": 1.75, "output": 14.0},  # Official pricing
    "gpt-5.2-pro": {"input": 10.0, "output": 40.0},
    "gpt-5.2-codex": {"input": 1.75, "output": 14.0},
    # GPT-4o series (still available, cost-effective)
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # o-series reasoning models
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Claude models
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    # Google Gemini models (January 2026) - Gemini 3 series only
    "gemini-3-pro-preview": {"input": 1.25, "output": 5.0},
    "gemini-3-flash-preview": {"input": 0.075, "output": 0.30},  # Free tier available
}


class BaseCloudProvider(ABC):
    """Abstract base class for cloud providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> CloudResponse:
        """Generate a response from the model."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the API connection is working."""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models."""
        pass


class OpenAIProvider(BaseCloudProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")
        self.client = openai.OpenAI(api_key=api_key)
        # Updated January 2026: GPT-5.2 series (verified available via API)
        self.available_models = [
            "gpt-5.2",           # Flagship - $1.75/$14 per 1M tokens
            "gpt-5.2-pro",       # Maximum capability with xhigh reasoning
            "gpt-5.2-codex",     # Agentic coding optimized
            "gpt-4o",            # Fast & capable
            "gpt-4o-mini",       # Most cost-effective
            "o3-mini",           # Small reasoning model
        ]

    def generate(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> CloudResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        latency = (time.time() - start_time) * 1000

        tokens_used = response.usage.total_tokens if response.usage else 0
        cost = self._calculate_cost(model, response.usage) if response.usage else 0.0

        return CloudResponse(
            content=response.choices[0].message.content or "",
            model=model,
            provider=CloudProvider.OPENAI,
            tokens_used=tokens_used,
            latency_ms=latency,
            finish_reason=response.choices[0].finish_reason or "",
            raw_response=response,
            cost_usd=cost,
        )

    def _calculate_cost(self, model: str, usage) -> float:
        if model not in PRICING:
            return 0.0
        pricing = PRICING[model]
        input_cost = (usage.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def test_connection(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False

    def list_models(self) -> List[str]:
        return self.available_models


class AnthropicProvider(BaseCloudProvider):
    """Anthropic API provider."""

    def __init__(self, api_key: str):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.available_models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]

    def generate(
        self,
        prompt: str,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> CloudResponse:
        start_time = time.time()

        params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            params["system"] = system_prompt

        response = self.client.messages.create(**params)
        latency = (time.time() - start_time) * 1000

        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        cost = self._calculate_cost(model, response.usage)

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return CloudResponse(
            content=content,
            model=model,
            provider=CloudProvider.ANTHROPIC,
            tokens_used=tokens_used,
            latency_ms=latency,
            finish_reason=response.stop_reason or "",
            raw_response=response,
            cost_usd=cost,
        )

    def _calculate_cost(self, model: str, usage) -> float:
        # Map full model names to pricing keys
        model_key = None
        if "opus" in model:
            model_key = "claude-3-opus"
        elif "sonnet" in model:
            model_key = "claude-3-sonnet"
        elif "haiku" in model:
            model_key = "claude-3-haiku"

        if model_key not in PRICING:
            return 0.0

        pricing = PRICING[model_key]
        input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def test_connection(self) -> bool:
        try:
            # Simple test with minimal tokens
            self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic connection test failed: {e}")
            return False

    def list_models(self) -> List[str]:
        return self.available_models


class GoogleProvider(BaseCloudProvider):
    """Google Generative AI provider."""

    def __init__(self, api_key: str):
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        genai.configure(api_key=api_key)
        # Updated January 2026: Gemini 3 series (announced Jan 12, 2026)
        # Note: gemini-3-flash-preview has a free tier
        self.available_models = [
            "gemini-3-pro-preview",    # Reasoning-first, 1M context, thinking_level param
            "gemini-3-flash-preview",  # Fast frontier-class, FREE TIER available
        ]

    def generate(
        self,
        prompt: str,
        model: str = "gemini-pro",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> CloudResponse:
        start_time = time.time()

        model_instance = genai.GenerativeModel(model)
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        response = model_instance.generate_content(
            prompt,
            generation_config=generation_config,
        )
        latency = (time.time() - start_time) * 1000

        # Estimate tokens (Gemini doesn't always return exact counts)
        tokens_used = len(prompt.split()) + len(response.text.split()) if response.text else 0
        cost = self._calculate_cost(model, tokens_used)

        return CloudResponse(
            content=response.text or "",
            model=model,
            provider=CloudProvider.GOOGLE,
            tokens_used=tokens_used,
            latency_ms=latency,
            finish_reason="stop",
            raw_response=response,
            cost_usd=cost,
        )

    def _calculate_cost(self, model: str, tokens: int) -> float:
        if model not in PRICING:
            return 0.0
        pricing = PRICING[model]
        # Rough estimate: assume 50/50 input/output split
        return (tokens / 1_000_000) * (pricing["input"] + pricing["output"]) / 2

    def test_connection(self) -> bool:
        try:
            # Use gemini-3-flash-preview for testing (latest)
            model = genai.GenerativeModel("gemini-3-flash-preview")
            model.generate_content("Hi", generation_config=genai.types.GenerationConfig(max_output_tokens=10))
            return True
        except Exception as e:
            logger.error(f"Google connection test failed: {e}")
            return False

    def list_models(self) -> List[str]:
        return self.available_models


class CloudClient:
    """
    Unified client for cloud LLM APIs.

    Provides a single interface for interacting with multiple cloud providers
    with automatic retry logic, cost tracking, and rate limit handling.

    Example:
        client = CloudClient.from_env()
        response = client.generate(
            prompt="Hello, how are you?",
            provider=CloudProvider.OPENAI,
            model="gpt-4o-mini"
        )
        print(response.content)
    """

    def __init__(self, config: Optional[CloudConfig] = None):
        self.config = config or CloudConfig.from_env()
        self._providers: Dict[CloudProvider, BaseCloudProvider] = {}
        self._total_cost = 0.0
        self._init_providers()

    def _init_providers(self):
        """Initialize available providers based on API keys."""
        if self.config.openai_api_key:
            try:
                self._providers[CloudProvider.OPENAI] = OpenAIProvider(self.config.openai_api_key)
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")

        if self.config.anthropic_api_key:
            try:
                self._providers[CloudProvider.ANTHROPIC] = AnthropicProvider(self.config.anthropic_api_key)
                logger.info("Anthropic provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic: {e}")

        if self.config.google_api_key:
            try:
                self._providers[CloudProvider.GOOGLE] = GoogleProvider(self.config.google_api_key)
                logger.info("Google provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Google: {e}")

    @classmethod
    def from_env(cls) -> "CloudClient":
        """Create client from environment variables."""
        return cls(CloudConfig.from_env())

    @classmethod
    def from_yaml(cls, path: Path) -> "CloudClient":
        """Create client from YAML configuration file."""
        return cls(CloudConfig.from_yaml(path))

    def available_providers(self) -> List[CloudProvider]:
        """List providers with valid API keys configured."""
        return list(self._providers.keys())

    def test_connection(self, provider: CloudProvider) -> bool:
        """Test connection to a specific provider."""
        if provider not in self._providers:
            logger.error(f"Provider {provider.value} not configured")
            return False
        return self._providers[provider].test_connection()

    def test_all_connections(self) -> Dict[CloudProvider, bool]:
        """Test connections to all configured providers."""
        return {
            provider: self.test_connection(provider)
            for provider in self._providers
        }

    def list_models(self, provider: CloudProvider) -> List[str]:
        """List available models for a provider."""
        if provider not in self._providers:
            return []
        return self._providers[provider].list_models()

    def generate(
        self,
        prompt: str,
        provider: CloudProvider,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> CloudResponse:
        """
        Generate a response from a cloud LLM.

        Args:
            prompt: The user prompt to send
            provider: Which cloud provider to use
            model: Specific model to use (defaults to cheapest)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            CloudResponse with the model's response

        Raises:
            ValueError: If provider is not configured
            RuntimeError: If cost limit exceeded
        """
        if provider not in self._providers:
            raise ValueError(f"Provider {provider.value} not configured. Available: {self.available_providers()}")

        # Check cost limit
        if self.config.track_costs and self._total_cost >= self.config.max_cost_usd:
            raise RuntimeError(f"Cost limit exceeded: ${self._total_cost:.4f} >= ${self.config.max_cost_usd}")

        # Set default model if not specified
        if model is None:
            model = self._get_default_model(provider)

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._providers[provider].generate(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt,
                    **kwargs
                )

                # Track cost
                if self.config.track_costs:
                    self._total_cost += response.cost_usd
                    logger.debug(f"Request cost: ${response.cost_usd:.6f}, Total: ${self._total_cost:.4f}")

                return response

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay
                    if self.config.exponential_backoff:
                        delay *= (2 ** attempt)
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    time.sleep(delay)

        raise last_error

    def _get_default_model(self, provider: CloudProvider) -> str:
        """Get the default (cheapest) model for a provider."""
        defaults = {
            CloudProvider.OPENAI: "gpt-4o-mini",
            CloudProvider.ANTHROPIC: "claude-3-haiku-20240307",
            CloudProvider.GOOGLE: "gemini-3-flash-preview",  # Gemini 3 series (Jan 2026)
        }
        return defaults.get(provider, "")

    def get_total_cost(self) -> float:
        """Get total cost accumulated in this session."""
        return self._total_cost

    def reset_cost_tracker(self):
        """Reset the cost tracker to zero."""
        self._total_cost = 0.0


# Convenience function
def quick_generate(
    prompt: str,
    provider: str = "openai",
    model: Optional[str] = None,
) -> str:
    """
    Quick one-off generation without managing a client.

    Args:
        prompt: The prompt to send
        provider: Provider name (openai, anthropic, google)
        model: Optional specific model

    Returns:
        The response content string
    """
    client = CloudClient.from_env()
    provider_enum = CloudProvider(provider)
    response = client.generate(prompt=prompt, provider=provider_enum, model=model)
    return response.content
