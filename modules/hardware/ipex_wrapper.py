"""
IPEX-LLM Wrapper for Intel NPU acceleration.

IPEX-LLM (Intel Extension for PyTorch - LLM) provides optimized
inference for Large Language Models on Intel hardware.

GitHub: https://github.com/intel/ipex-llm

Note on NPU Performance (January 2026 research):
- INT4 quantization is often SLOWER on NPU than CPU for LLMs
- FP16 shows better energy efficiency (3-5x) but not always speed
- Best use cases: static shapes, vision models, batch inference
- For typical LLM chat inference, CPU may outperform NPU
"""

import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Check for IPEX-LLM availability
try:
    import ipex_llm
    from ipex_llm.transformers import AutoModelForCausalLM as IPEXAutoModel
    IPEX_AVAILABLE = True
    IPEX_VERSION = getattr(ipex_llm, "__version__", "unknown")
except ImportError:
    IPEX_AVAILABLE = False
    IPEX_VERSION = None
    logger.warning("IPEX-LLM not installed. Run: pip install ipex-llm[npu]")


@dataclass
class IPEXConfig:
    """Configuration for IPEX-LLM inference."""
    # Model loading
    load_in_low_bit: str = "fp16"  # fp16, int8, int4, nf4
    optimize_model: bool = True
    use_cache: bool = True

    # Device
    device: str = "auto"  # auto, cpu, xpu (includes NPU)

    # Generation
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9

    # Memory optimization
    cpu_embedding: bool = False  # Keep embeddings on CPU to save XPU memory


class IPEXWrapper:
    """
    Wrapper for IPEX-LLM accelerated inference.

    Provides a unified interface for loading and running models
    with Intel NPU/iGPU acceleration via IPEX-LLM.

    Example:
        wrapper = IPEXWrapper()
        wrapper.load_model("microsoft/phi-2")
        response = wrapper.generate("Hello, how are you?")
    """

    def __init__(self, config: Optional[IPEXConfig] = None):
        """
        Initialize IPEX wrapper.

        Args:
            config: IPEX configuration options
        """
        if not IPEX_AVAILABLE:
            raise ImportError(
                "IPEX-LLM is not installed.\n"
                "Install with: pip install ipex-llm[npu]\n"
                "See: https://github.com/intel/ipex-llm"
            )

        self.config = config or IPEXConfig()
        self.model = None
        self.tokenizer = None
        self._device_info = self._detect_devices()

    def _detect_devices(self) -> Dict[str, Any]:
        """Detect available Intel devices."""
        info = {
            "ipex_version": IPEX_VERSION,
            "platform": platform.platform(),
            "has_xpu": False,
            "xpu_devices": [],
        }

        try:
            import intel_extension_for_pytorch as ipex
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                info["has_xpu"] = True
                info["xpu_devices"] = [
                    torch.xpu.get_device_name(i)
                    for i in range(torch.xpu.device_count())
                ]
                logger.info(f"XPU devices available: {info['xpu_devices']}")

        except ImportError:
            logger.debug("Intel Extension for PyTorch not available")
        except Exception as e:
            logger.debug(f"XPU detection failed: {e}")

        return info

    @property
    def has_npu_acceleration(self) -> bool:
        """Check if NPU acceleration is available."""
        return self._device_info.get("has_xpu", False)

    def load_model(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
    ):
        """
        Load a model with IPEX optimization.

        Args:
            model_path: HuggingFace model ID or local path
            tokenizer_path: Optional separate tokenizer path
        """
        from transformers import AutoTokenizer

        logger.info(f"Loading model: {model_path}")
        logger.info(f"Precision: {self.config.load_in_low_bit}")
        logger.info(f"Device: {self.config.device}")

        # Load tokenizer
        tokenizer_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

        # Load model with IPEX optimization
        load_kwargs = {
            "trust_remote_code": True,
            "optimize_model": self.config.optimize_model,
            "use_cache": self.config.use_cache,
            "cpu_embedding": self.config.cpu_embedding,
        }

        # Handle quantization
        if self.config.load_in_low_bit != "fp16":
            load_kwargs["load_in_low_bit"] = self.config.load_in_low_bit

        self.model = IPEXAutoModel.from_pretrained(
            model_path,
            **load_kwargs
        )

        # Move to XPU if available and configured
        if self.config.device == "auto":
            if self.has_npu_acceleration:
                logger.info("Moving model to XPU (NPU/iGPU)")
                self.model = self.model.to("xpu")
            else:
                logger.info("Using CPU (XPU not available)")
        elif self.config.device == "xpu" and self.has_npu_acceleration:
            self.model = self.model.to("xpu")

        logger.info("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using the loaded model.

        Args:
            prompt: Input prompt
            max_new_tokens: Override config max tokens
            temperature: Override config temperature
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch

        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move to correct device
        if self.has_npu_acceleration and self.config.device in ("auto", "xpu"):
            inputs = {k: v.to("xpu") for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        # Remove the prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices."""
        return self._device_info.copy()

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clear GPU cache if using XPU
        try:
            import torch
            if hasattr(torch, "xpu"):
                torch.xpu.empty_cache()
        except Exception:
            pass

        logger.info("Model unloaded")


def check_ipex_installation() -> Dict[str, Any]:
    """
    Check IPEX-LLM installation status and capabilities.

    Returns:
        Dictionary with installation info
    """
    info = {
        "ipex_llm_installed": IPEX_AVAILABLE,
        "ipex_version": IPEX_VERSION,
        "xpu_available": False,
        "npu_detected": False,
        "recommendations": [],
    }

    if not IPEX_AVAILABLE:
        info["recommendations"].append(
            "Install IPEX-LLM: pip install ipex-llm[npu]"
        )
        return info

    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            info["xpu_available"] = True
            info["xpu_device_count"] = torch.xpu.device_count()

            for i in range(torch.xpu.device_count()):
                device_name = torch.xpu.get_device_name(i)
                if "NPU" in device_name or "Neural" in device_name:
                    info["npu_detected"] = True
                    break

    except Exception as e:
        info["error"] = str(e)

    # Add recommendations
    if info["xpu_available"] and not info["npu_detected"]:
        info["recommendations"].append(
            "XPU available but NPU not detected. Check Intel drivers."
        )

    if info["npu_detected"]:
        info["recommendations"].append(
            "NPU detected! Note: For LLM inference, FP16 may perform better than INT4."
        )
        info["recommendations"].append(
            "NPU is optimal for static shapes and vision models."
        )

    return info


def main():
    """CLI entry point for IPEX wrapper testing."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="IPEX-LLM Wrapper")
    parser.add_argument("--check", action="store_true", help="Check installation")
    parser.add_argument("--model", help="Model to load and test")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Test prompt")

    args = parser.parse_args()

    if args.check:
        info = check_ipex_installation()
        print(json.dumps(info, indent=2))
        return

    if args.model:
        wrapper = IPEXWrapper()
        wrapper.load_model(args.model)
        response = wrapper.generate(args.prompt)
        print(f"Prompt: {args.prompt}")
        print(f"Response: {response}")
        wrapper.unload_model()


if __name__ == "__main__":
    main()
