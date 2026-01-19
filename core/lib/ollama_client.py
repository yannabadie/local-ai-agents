"""
Ollama client wrapper with retry logic and structured responses.
"""

import time
import json
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass
import requests
from requests.exceptions import RequestException, Timeout

from core.config import get_settings
from core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelResponse:
    """Structured response from model."""
    content: str
    model: str
    total_duration_ms: float
    prompt_eval_count: int
    eval_count: int
    tokens_per_second: float
    done: bool = True

    @classmethod
    def from_ollama_response(cls, data: Dict) -> "ModelResponse":
        total_duration = data.get("total_duration", 0) / 1_000_000  # ns to ms
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 1) / 1_000_000_000  # ns to s
        tps = eval_count / eval_duration if eval_duration > 0 else 0

        return cls(
            content=data.get("response", ""),
            model=data.get("model", "unknown"),
            total_duration_ms=total_duration,
            prompt_eval_count=data.get("prompt_eval_count", 0),
            eval_count=eval_count,
            tokens_per_second=round(tps, 2),
            done=data.get("done", True)
        )


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: Optional[str] = None):
        settings = get_settings()
        self.base_url = base_url or settings.ollama.base_url
        self.timeout = settings.ollama.timeout
        self.max_retries = settings.ollama.max_retries
        self.default_model = settings.model.default_model

        logger.info(f"OllamaClient initialized: {self.base_url}")

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault("timeout", self.timeout)

        for attempt in range(self.max_retries):
            try:
                response = requests.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt == self.max_retries - 1:
                    raise
            except RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)

        raise RuntimeError("Max retries exceeded")

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = self._request("GET", "/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[Dict]:
        """List available models."""
        response = self._request("GET", "/api/tags")
        data = response.json()
        return data.get("models", [])

    def get_model_info(self, model: str) -> Dict:
        """Get detailed model information."""
        response = self._request("POST", "/api/show", json={"name": model})
        return response.json()

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> ModelResponse:
        """Generate completion from model."""
        model = model or self.default_model

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if system:
            payload["system"] = system

        logger.debug(f"Generating with {model}: {prompt[:50]}...")
        start_time = time.time()

        response = self._request("POST", "/api/generate", json=payload)
        data = response.json()

        result = ModelResponse.from_ollama_response(data)
        logger.info(f"Generated {result.eval_count} tokens @ {result.tokens_per_second} t/s")

        return result

    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Generator[str, None, None]:
        """Generate completion with streaming."""
        model = model or self.default_model

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if system:
            payload["system"] = system

        url = f"{self.base_url}/api/generate"

        with requests.post(url, json=payload, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> ModelResponse:
        """Chat completion with message history."""
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        response = self._request("POST", "/api/chat", json=payload)
        data = response.json()

        # Adapt chat response to ModelResponse format
        return ModelResponse(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", model),
            total_duration_ms=data.get("total_duration", 0) / 1_000_000,
            prompt_eval_count=data.get("prompt_eval_count", 0),
            eval_count=data.get("eval_count", 0),
            tokens_per_second=0,  # Calculate if needed
            done=data.get("done", True)
        )

    def benchmark(self, model: str, prompt: str = "Hello, how are you?", runs: int = 3) -> Dict:
        """Benchmark model performance."""
        results = []

        for i in range(runs):
            logger.info(f"Benchmark run {i+1}/{runs} for {model}")
            response = self.generate(prompt, model=model)
            results.append({
                "run": i + 1,
                "tokens_per_second": response.tokens_per_second,
                "total_duration_ms": response.total_duration_ms,
                "eval_count": response.eval_count
            })

        avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
        avg_duration = sum(r["total_duration_ms"] for r in results) / len(results)

        return {
            "model": model,
            "runs": results,
            "average_tokens_per_second": round(avg_tps, 2),
            "average_duration_ms": round(avg_duration, 2)
        }
