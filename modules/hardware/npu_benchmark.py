"""
NPU Benchmark Module for Intel Core Ultra processors.

Benchmarks LLM inference performance across different hardware configurations:
- CPU only
- CPU + NPU (via IPEX-LLM)
- CPU + iGPU (if available)

Hardware: Intel Core Ultra 5 135U
- NPU: Intel AI Boost (2 compute engines)
- iGPU: Intel Graphics
- CPU: 12 cores / 14 threads

Based on research (January 2026):
- NPU is NOT optimal for standard LLM inference
- INT4 is actually slower on NPU than CPU for most models
- FP16 can show 3-5x energy efficiency gains
- Best use case: static shape inference, vision models
"""

import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import statistics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    model_name: str
    backend: str  # cpu, npu, igpu
    precision: str  # fp32, fp16, int8, int4
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_time_seconds: float = 0.0
    first_token_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    memory_used_mb: float = 0.0
    power_draw_watts: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def time_to_first_token_seconds(self) -> float:
        return self.first_token_latency_ms / 1000.0


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    hardware_info: Dict[str, Any]
    results: List[BenchmarkResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def get_comparison(self, metric: str = "tokens_per_second") -> Dict[str, float]:
        """Compare results by backend."""
        comparison = {}
        for result in self.results:
            key = f"{result.backend}_{result.precision}"
            comparison[key] = getattr(result, metric, 0.0)
        return comparison

    def to_json(self) -> str:
        """Export results to JSON."""
        return json.dumps({
            "hardware_info": self.hardware_info,
            "timestamp": self.timestamp,
            "results": [
                {
                    "model_name": r.model_name,
                    "backend": r.backend,
                    "precision": r.precision,
                    "prompt_tokens": r.prompt_tokens,
                    "generated_tokens": r.generated_tokens,
                    "total_time_seconds": r.total_time_seconds,
                    "first_token_latency_ms": r.first_token_latency_ms,
                    "tokens_per_second": r.tokens_per_second,
                    "memory_used_mb": r.memory_used_mb,
                    "power_draw_watts": r.power_draw_watts,
                }
                for r in self.results
            ]
        }, indent=2)


class NPUBenchmark:
    """
    Intel NPU benchmark suite for LLM inference.

    Tests model performance across different hardware configurations
    to identify optimal settings for the Intel Core Ultra platform.

    Example:
        benchmark = NPUBenchmark()
        results = benchmark.run_full_benchmark("deephat")
        print(results.to_json())
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize NPU benchmark.

        Args:
            ollama_base_url: Ollama server URL
            output_dir: Directory for benchmark results
        """
        self.ollama_base_url = ollama_base_url
        self.output_dir = output_dir or Path("tests/benchmarks/npu")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.hardware_info = self._detect_hardware()

    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware."""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "npu_available": False,
            "igpu_available": False,
            "ipex_available": False,
        }

        # Check for Intel NPU
        if platform.system() == "Windows":
            try:
                # Check for NPU via WMI or driver
                result = subprocess.run(
                    ["wmic", "path", "win32_pnpentity", "where",
                     "name like '%Neural%' or name like '%NPU%'", "get", "name"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if "Neural" in result.stdout or "NPU" in result.stdout:
                    info["npu_available"] = True
                    logger.info("Intel NPU detected")
            except Exception as e:
                logger.debug(f"NPU detection failed: {e}")

        # Check for IPEX-LLM
        try:
            import ipex_llm
            info["ipex_available"] = True
            info["ipex_version"] = getattr(ipex_llm, "__version__", "unknown")
            logger.info(f"IPEX-LLM available: {info['ipex_version']}")
        except ImportError:
            logger.info("IPEX-LLM not installed")

        return info

    def benchmark_ollama(
        self,
        model: str,
        prompt: str = "Write a short poem about artificial intelligence.",
        num_runs: int = 3,
    ) -> BenchmarkResult:
        """
        Benchmark a model using Ollama (CPU baseline).

        Args:
            model: Model name in Ollama
            prompt: Test prompt
            num_runs: Number of runs for averaging

        Returns:
            BenchmarkResult with performance metrics
        """
        import requests

        logger.info(f"Benchmarking {model} via Ollama (CPU)...")

        times = []
        first_token_times = []
        tokens_generated = []

        for run in range(num_runs):
            start_time = time.time()
            first_token_time = None
            tokens = 0

            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": True,
                    },
                    stream=True,
                    timeout=300,
                )

                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        tokens += 1
                        if first_token_time is None:
                            first_token_time = time.time() - start_time

                        if data.get("done"):
                            break

                total_time = time.time() - start_time
                times.append(total_time)
                first_token_times.append(first_token_time or 0)
                tokens_generated.append(tokens)

                logger.debug(f"Run {run + 1}: {tokens} tokens in {total_time:.2f}s")

            except Exception as e:
                logger.error(f"Benchmark run failed: {e}")
                continue

        if not times:
            return BenchmarkResult(
                model_name=model,
                backend="cpu",
                precision="mixed",
                metadata={"error": "All runs failed"}
            )

        avg_time = statistics.mean(times)
        avg_first_token = statistics.mean(first_token_times)
        avg_tokens = statistics.mean(tokens_generated)

        return BenchmarkResult(
            model_name=model,
            backend="cpu",
            precision="mixed",  # Ollama uses mixed precision
            prompt_tokens=len(prompt.split()),
            generated_tokens=int(avg_tokens),
            total_time_seconds=avg_time,
            first_token_latency_ms=avg_first_token * 1000,
            tokens_per_second=avg_tokens / avg_time if avg_time > 0 else 0,
            metadata={
                "num_runs": num_runs,
                "times": times,
                "ollama_url": self.ollama_base_url,
            }
        )

    def benchmark_ipex_npu(
        self,
        model_path: Path,
        prompt: str = "Write a short poem about artificial intelligence.",
        num_runs: int = 3,
        precision: str = "fp16",
    ) -> BenchmarkResult:
        """
        Benchmark a model using IPEX-LLM on NPU.

        Args:
            model_path: Path to model files
            prompt: Test prompt
            num_runs: Number of runs
            precision: Model precision (fp16, int4)

        Returns:
            BenchmarkResult with performance metrics
        """
        if not self.hardware_info.get("ipex_available"):
            logger.warning("IPEX-LLM not available, skipping NPU benchmark")
            return BenchmarkResult(
                model_name=str(model_path),
                backend="npu",
                precision=precision,
                metadata={"error": "IPEX-LLM not installed"}
            )

        try:
            from ipex_llm.transformers import AutoModelForCausalLM
            from transformers import AutoTokenizer

            logger.info(f"Benchmarking {model_path} on NPU ({precision})...")

            # Load model with IPEX optimization
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                load_in_low_bit=precision if precision != "fp16" else None,
                trust_remote_code=True,
                optimize_model=True,
                use_cache=True,
            )

            # Move to NPU if available
            if self.hardware_info.get("npu_available"):
                model = model.to("xpu")  # Intel XPU includes NPU

            tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            times = []
            first_token_times = []
            tokens_generated = []

            for run in range(num_runs):
                inputs = tokenizer(prompt, return_tensors="pt")
                if self.hardware_info.get("npu_available"):
                    inputs = {k: v.to("xpu") for k, v in inputs.items()}

                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                )
                total_time = time.time() - start_time

                tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
                times.append(total_time)
                tokens_generated.append(tokens)

                # First token latency requires streaming, estimate here
                first_token_times.append(total_time / tokens * 5)  # Rough estimate

                logger.debug(f"Run {run + 1}: {tokens} tokens in {total_time:.2f}s")

            avg_time = statistics.mean(times)
            avg_first_token = statistics.mean(first_token_times)
            avg_tokens = statistics.mean(tokens_generated)

            return BenchmarkResult(
                model_name=str(model_path.name),
                backend="npu",
                precision=precision,
                prompt_tokens=inputs["input_ids"].shape[-1],
                generated_tokens=int(avg_tokens),
                total_time_seconds=avg_time,
                first_token_latency_ms=avg_first_token * 1000,
                tokens_per_second=avg_tokens / avg_time if avg_time > 0 else 0,
                metadata={
                    "num_runs": num_runs,
                    "times": times,
                    "ipex_version": self.hardware_info.get("ipex_version"),
                }
            )

        except Exception as e:
            logger.error(f"IPEX NPU benchmark failed: {e}")
            return BenchmarkResult(
                model_name=str(model_path),
                backend="npu",
                precision=precision,
                metadata={"error": str(e)}
            )

    def run_full_benchmark(
        self,
        model: str,
        model_path: Optional[Path] = None,
    ) -> BenchmarkSuite:
        """
        Run full benchmark suite comparing CPU, NPU, and other backends.

        Args:
            model: Model name for Ollama
            model_path: Optional path for IPEX benchmarking

        Returns:
            BenchmarkSuite with all results
        """
        suite = BenchmarkSuite(hardware_info=self.hardware_info)

        # CPU baseline via Ollama
        logger.info("Running CPU baseline benchmark...")
        cpu_result = self.benchmark_ollama(model)
        suite.add_result(cpu_result)

        # NPU benchmark if IPEX available and model path provided
        if self.hardware_info.get("ipex_available") and model_path:
            logger.info("Running NPU benchmark...")
            for precision in ["fp16", "int4"]:
                npu_result = self.benchmark_ipex_npu(model_path, precision=precision)
                suite.add_result(npu_result)

        # Save results
        output_file = self.output_dir / f"benchmark_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            f.write(suite.to_json())

        logger.info(f"Benchmark results saved to {output_file}")
        return suite

    def print_comparison(self, suite: BenchmarkSuite):
        """Print a formatted comparison of benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Hardware: {suite.hardware_info.get('processor', 'Unknown')}")
        print(f"NPU Available: {suite.hardware_info.get('npu_available', False)}")
        print(f"IPEX Available: {suite.hardware_info.get('ipex_available', False)}")
        print("=" * 60)
        print(f"{'Backend':<15} {'Precision':<10} {'Tokens/s':<12} {'TTFT (ms)':<12}")
        print("-" * 60)

        for result in suite.results:
            print(f"{result.backend:<15} {result.precision:<10} "
                  f"{result.tokens_per_second:<12.2f} {result.first_token_latency_ms:<12.1f}")

        print("=" * 60)

        # Calculate speedup if NPU results available
        cpu_result = next((r for r in suite.results if r.backend == "cpu"), None)
        npu_results = [r for r in suite.results if r.backend == "npu"]

        if cpu_result and npu_results:
            print("\nSpeedup vs CPU:")
            for npu in npu_results:
                if cpu_result.tokens_per_second > 0:
                    speedup = npu.tokens_per_second / cpu_result.tokens_per_second
                    print(f"  {npu.precision}: {speedup:.2f}x")


def main():
    """CLI entry point for NPU benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(description="NPU Benchmark for LLM Inference")
    parser.add_argument("--model", default="deephat", help="Model name for Ollama")
    parser.add_argument("--model-path", type=Path, help="Model path for IPEX")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--output", type=Path, help="Output directory")

    args = parser.parse_args()

    benchmark = NPUBenchmark(output_dir=args.output)

    print("Hardware Detection:")
    print(json.dumps(benchmark.hardware_info, indent=2))

    suite = benchmark.run_full_benchmark(args.model, args.model_path)
    benchmark.print_comparison(suite)


if __name__ == "__main__":
    main()
