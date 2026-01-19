"""
Intel NPU Runtime Module - Working solution for Core Ultra processors.

Provides LLM inference on Intel NPU using IPEX-LLM's llama.cpp portable approach.
This is the recommended solution for Intel Core Ultra processors (Meteor Lake, Lunar Lake, Arrow Lake).

Based on: https://github.com/intel/ipex-llm

Supported Hardware:
- Intel Core Ultra Processors (Series 1) - Meteor Lake: 1xxH/U/HL/UL
- Intel Core Ultra Processors (Series 2) - Lunar Lake: 2xxV
- Intel Core Ultra Processors (Series 2) - Arrow Lake: 2xxK/2xxH

Requirements:
- Windows 10/11 (NPU support is Windows-only for now)
- NPU Driver: Recommended version 32.0.100.3104 or later
- IPEX-LLM llama.cpp NPU portable zip

Current Limitations:
- Max input tokens: 960
- Max sequence length (input + output): 1024
"""

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator

logger = logging.getLogger(__name__)

# IPEX-LLM NPU Portable Zip download info
IPEX_LLM_NPU_ZIP_URL = "https://github.com/intel/ipex-llm/releases/latest/download/llama-cpp-npu-windows.zip"
IPEX_LLM_RELEASE_PAGE = "https://github.com/intel/ipex-llm/releases"


@dataclass
class NPUConfig:
    """Configuration for NPU runtime."""
    # Paths
    portable_dir: Path = field(default_factory=lambda: Path.home() / ".ipex-llm" / "npu")
    models_dir: Path = field(default_factory=lambda: Path.home() / "models")

    # Generation defaults
    max_tokens: int = 512  # Conservative default (max 1024 total)
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1

    # NPU settings
    npu_group_size: int = 64  # Quantization group size


@dataclass
class NPUResponse:
    """Response from NPU inference."""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_time_seconds: float = 0.0
    tokens_per_second: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class NPURuntime:
    """
    Intel NPU Runtime for LLM inference.

    Uses IPEX-LLM's llama.cpp NPU portable approach for inference
    on Intel Core Ultra processors.

    Example:
        runtime = NPURuntime()
        runtime.setup()  # Download and configure portable zip

        response = runtime.generate(
            model="deepseek-r1-q6_k.gguf",
            prompt="What is artificial intelligence?"
        )
        print(response.content)
    """

    def __init__(self, config: Optional[NPUConfig] = None):
        """
        Initialize NPU runtime.

        Args:
            config: Runtime configuration
        """
        self.config = config or NPUConfig()
        self._device_info = self._detect_device()
        self._cli_path: Optional[Path] = None

    def _detect_device(self) -> Dict[str, Any]:
        """Detect Intel device type and set appropriate environment."""
        info = {
            "platform": platform.system(),
            "processor": platform.processor(),
            "npu_available": False,
            "device_type": None,  # meteor_lake, lunar_lake, arrow_lake
            "driver_version": None,
            "env_vars": {},
        }

        if platform.system() != "Windows":
            logger.warning("NPU support is currently Windows-only")
            return info

        # Detect NPU via WMI
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_pnpentity", "where",
                 "(name like '%Intel%' and (name like '%NPU%' or name like '%Neural%'))",
                 "get", "name,deviceid"],
                capture_output=True, text=True, timeout=10
            )

            if "Intel" in result.stdout and ("NPU" in result.stdout or "Neural" in result.stdout):
                info["npu_available"] = True
                logger.info("Intel NPU detected")

                # Try to get driver version
                driver_result = subprocess.run(
                    ["wmic", "path", "win32_pnpentity", "where",
                     "(name like '%Intel%' and (name like '%NPU%' or name like '%Neural%'))",
                     "get", "driverversion"],
                    capture_output=True, text=True, timeout=10
                )
                version_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', driver_result.stdout)
                if version_match:
                    info["driver_version"] = version_match.group(1)

        except Exception as e:
            logger.debug(f"NPU detection error: {e}")

        # Detect device type from processor name
        proc_lower = info["processor"].lower()
        if "ultra" in proc_lower:
            if any(x in proc_lower for x in ["258v", "256v", "255v"]):
                info["device_type"] = "lunar_lake"
                # No env vars needed for Lunar Lake 2xxV
            elif any(x in proc_lower for x in ["288k", "287k", "285k", "288h", "287h", "285h"]):
                info["device_type"] = "arrow_lake"
                info["env_vars"]["IPEX_LLM_NPU_ARL"] = "1"
            else:
                # Assume Meteor Lake for 1xx series
                info["device_type"] = "meteor_lake"
                info["env_vars"]["IPEX_LLM_NPU_MTL"] = "1"

        logger.info(f"Device type: {info['device_type']}, Driver: {info['driver_version']}")
        return info

    @property
    def is_available(self) -> bool:
        """Check if NPU runtime is available and configured."""
        return (
            self._device_info.get("npu_available", False) and
            self._cli_path is not None and
            self._cli_path.exists()
        )

    def setup(self, force_download: bool = False) -> bool:
        """
        Download and setup IPEX-LLM NPU portable.

        Args:
            force_download: Force re-download even if exists

        Returns:
            True if setup successful
        """
        if platform.system() != "Windows":
            logger.error("NPU setup is only supported on Windows")
            return False

        portable_dir = self.config.portable_dir
        cli_path = portable_dir / "llama-cli-npu.exe"

        if cli_path.exists() and not force_download:
            logger.info(f"NPU portable already installed at {portable_dir}")
            self._cli_path = cli_path
            return True

        logger.info("Downloading IPEX-LLM NPU portable zip...")
        logger.info(f"From: {IPEX_LLM_RELEASE_PAGE}")

        try:
            # Create directory
            portable_dir.mkdir(parents=True, exist_ok=True)

            # Download zip
            zip_path = portable_dir / "llama-cpp-npu.zip"

            # Use PowerShell for download (more reliable on Windows)
            download_cmd = [
                "powershell", "-Command",
                f"Invoke-WebRequest -Uri '{IPEX_LLM_NPU_ZIP_URL}' -OutFile '{zip_path}'"
            ]

            logger.info("Downloading... (this may take a few minutes)")
            result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=600)

            if not zip_path.exists():
                # Try alternative download method
                logger.info("Trying alternative download method...")
                urllib.request.urlretrieve(IPEX_LLM_NPU_ZIP_URL, str(zip_path))

            # Extract zip
            logger.info("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(portable_dir)

            # Clean up zip
            zip_path.unlink()

            # Find the CLI executable
            for root, dirs, files in os.walk(portable_dir):
                if "llama-cli-npu.exe" in files:
                    self._cli_path = Path(root) / "llama-cli-npu.exe"
                    break

            if self._cli_path and self._cli_path.exists():
                logger.info(f"NPU portable installed at: {self._cli_path.parent}")
                return True
            else:
                logger.error("llama-cli-npu.exe not found after extraction")
                return False

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> NPUResponse:
        """
        Generate text using NPU.

        Args:
            model: Path to GGUF model file
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Stream output (not fully supported yet)

        Returns:
            NPUResponse with generated content
        """
        if not self.is_available:
            raise RuntimeError("NPU runtime not available. Run setup() first.")

        model_path = self._resolve_model_path(model)
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        # Check token limits
        prompt_tokens = len(prompt.split())  # Rough estimate
        if prompt_tokens > 960:
            logger.warning(f"Prompt may be too long ({prompt_tokens} words). NPU limit is 960 tokens.")

        # Build command
        cmd = [
            str(self._cli_path),
            "-m", str(model_path),
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--top-p", str(self.config.top_p),
            "--repeat-penalty", str(self.config.repeat_penalty),
            "-p", prompt,
        ]

        # Set environment variables for device type
        env = os.environ.copy()
        for key, value in self._device_info.get("env_vars", {}).items():
            env[key] = value

        logger.debug(f"Running: {' '.join(cmd)}")

        import time
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env,
                cwd=str(self._cli_path.parent),
            )

            total_time = time.time() - start_time

            if result.returncode != 0:
                logger.error(f"NPU inference failed: {result.stderr}")
                return NPUResponse(
                    content=f"Error: {result.stderr}",
                    metadata={"error": True, "returncode": result.returncode}
                )

            # Parse output
            content = result.stdout.strip()

            # Remove the prompt from output if echoed
            if content.startswith(prompt):
                content = content[len(prompt):].strip()

            # Estimate tokens
            completion_tokens = len(content.split())
            tokens_per_second = completion_tokens / total_time if total_time > 0 else 0

            return NPUResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_time_seconds=total_time,
                tokens_per_second=tokens_per_second,
                metadata={
                    "model": model,
                    "device_type": self._device_info.get("device_type"),
                }
            )

        except subprocess.TimeoutExpired:
            logger.error("NPU inference timed out")
            return NPUResponse(
                content="Error: Inference timed out",
                metadata={"error": True, "timeout": True}
            )
        except Exception as e:
            logger.error(f"NPU inference error: {e}")
            return NPUResponse(
                content=f"Error: {e}",
                metadata={"error": True, "exception": str(e)}
            )

    def _resolve_model_path(self, model: str) -> Path:
        """Resolve model name to full path."""
        model_path = Path(model)

        # If already a full path
        if model_path.is_absolute() and model_path.exists():
            return model_path

        # Check in models directory
        for ext in ["", ".gguf"]:
            candidate = self.config.models_dir / f"{model}{ext}"
            if candidate.exists():
                return candidate

        # Check subdirectories
        for subdir in self.config.models_dir.iterdir():
            if subdir.is_dir():
                for ext in ["", ".gguf"]:
                    candidate = subdir / f"{model}{ext}"
                    if candidate.exists():
                        return candidate

        raise FileNotFoundError(f"Model not found: {model}")

    def list_models(self) -> List[Dict[str, Any]]:
        """List available GGUF models."""
        models = []

        for path in self.config.models_dir.rglob("*.gguf"):
            models.append({
                "name": path.stem,
                "path": str(path),
                "size_mb": path.stat().st_size / (1024 * 1024),
            })

        return models

    def get_device_info(self) -> Dict[str, Any]:
        """Get NPU device information."""
        return self._device_info.copy()


def check_npu_status() -> Dict[str, Any]:
    """
    Check NPU availability and status.

    Returns:
        Dictionary with NPU status information
    """
    runtime = NPURuntime()
    info = runtime.get_device_info()

    info["is_ready"] = runtime.is_available
    info["setup_required"] = not runtime.is_available

    if info.get("npu_available") and not runtime.is_available:
        info["next_steps"] = [
            "1. Run: python -m modules.hardware.npu_runtime --setup",
            "2. Download a GGUF model to ~/models/",
            "3. Test with: python -m modules.hardware.npu_runtime --test --model <model.gguf>",
        ]

    if info.get("driver_version"):
        recommended = "32.0.100.3104"
        if info["driver_version"] < recommended:
            info["driver_warning"] = f"Consider updating NPU driver to {recommended} or later"

    return info


def main():
    """CLI entry point for NPU runtime."""
    import argparse

    parser = argparse.ArgumentParser(description="Intel NPU Runtime for LLM Inference")
    parser.add_argument("--setup", action="store_true", help="Download and setup NPU portable")
    parser.add_argument("--status", action="store_true", help="Check NPU status")
    parser.add_argument("--test", action="store_true", help="Test NPU inference")
    parser.add_argument("--model", help="Model to use for testing")
    parser.add_argument("--prompt", default="What is artificial intelligence?", help="Test prompt")
    parser.add_argument("--list-models", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.status:
        status = check_npu_status()
        print(json.dumps(status, indent=2))
        return

    runtime = NPURuntime()

    if args.setup:
        success = runtime.setup()
        print("Setup successful!" if success else "Setup failed!")
        return

    if args.list_models:
        models = runtime.list_models()
        print(f"Found {len(models)} models:")
        for m in models:
            print(f"  - {m['name']} ({m['size_mb']:.1f} MB)")
        return

    if args.test:
        if not args.model:
            print("Error: --model required for testing")
            return

        if not runtime.is_available:
            print("NPU not ready. Run --setup first.")
            return

        print(f"Testing NPU inference with {args.model}...")
        print(f"Prompt: {args.prompt}")
        print("-" * 40)

        response = runtime.generate(args.model, args.prompt)
        print(f"Response: {response.content}")
        print("-" * 40)
        print(f"Time: {response.total_time_seconds:.2f}s")
        print(f"Speed: {response.tokens_per_second:.1f} tokens/s")


if __name__ == "__main__":
    main()
