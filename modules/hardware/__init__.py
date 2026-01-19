"""
Hardware optimization module for Local AI Agents.

This module provides tools for LLM inference on Intel hardware accelerators.

Components:
- NPURuntime: Production-ready Intel NPU inference (recommended)
- NPUBenchmark: Benchmarking suite for performance comparison
- IPEXWrapper: Low-level IPEX-LLM integration

Intel Core Ultra Processor Support:
- Series 1 (Meteor Lake): 1xxH/U/HL/UL
- Series 2 (Lunar Lake): 2xxV
- Series 2 (Arrow Lake): 2xxK/2xxH

Quick Start:
    from modules.hardware import NPURuntime

    runtime = NPURuntime()
    runtime.setup()  # Downloads IPEX-LLM portable (once)

    response = runtime.generate(
        model="deepseek-r1-q6_k.gguf",
        prompt="What is AI?"
    )
    print(response.content)
"""

from .npu_runtime import NPURuntime, NPUConfig, NPUResponse, check_npu_status
from .npu_benchmark import NPUBenchmark, BenchmarkResult
from .ipex_wrapper import IPEXWrapper

__all__ = [
    # Recommended for production
    "NPURuntime",
    "NPUConfig",
    "NPUResponse",
    "check_npu_status",
    # Benchmarking
    "NPUBenchmark",
    "BenchmarkResult",
    # Low-level
    "IPEXWrapper",
]
