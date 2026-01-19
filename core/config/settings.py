"""
Centralized settings for Local AI Agents project.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class OllamaSettings:
    """Ollama server configuration."""
    host: str = "http://localhost"
    port: int = 11434
    timeout: int = 120
    max_retries: int = 3

    @property
    def base_url(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class ModelSettings:
    """Default model settings."""
    default_model: str = "deephat"
    temperature: float = 0.7
    num_ctx: int = 4096
    num_threads: int = 12

    # Installed models
    available_models: List[str] = field(default_factory=lambda: [
        "deephat",
        "elbaz-olmo",
        "deepseek-r1",
        "gemma3:4b"
    ])


@dataclass
class PathSettings:
    """Project paths."""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    models_dir: Path = field(default_factory=lambda: Path.home() / "models")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "logs")
    evidence_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "tests" / "poc")

    def __post_init__(self):
        # Ensure directories exist
        self.logs_dir.mkdir(exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class JailbreakSettings:
    """Jailbreak testing settings."""
    max_tokens: int = 2048
    test_timeout: int = 180
    evidence_screenshots: bool = True
    log_all_responses: bool = True


@dataclass
class Settings:
    """Main settings container."""
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    jailbreak: JailbreakSettings = field(default_factory=JailbreakSettings)

    # Hardware profile
    hardware: Dict = field(default_factory=lambda: {
        "cpu": "Intel Core Ultra 5 135U",
        "cores": 12,
        "threads": 14,
        "ram_gb": 16,
        "has_npu": True,
        "gpu": "Intel Graphics (integrated)"
    })


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
