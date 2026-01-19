#!/usr/bin/env python3
"""
Benchmark runner script for local AI models.
Run with: python scripts/run_benchmark.py
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.models.benchmark import ModelBenchmark, full_benchmark
from core.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark local LLM models"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="deephat,elbaz-olmo,deepseek-r1",
        help="Comma-separated list of models to benchmark"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark only"
    )
    parser.add_argument(
        "--compliance",
        action="store_true",
        help="Run compliance tests"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for results"
    )

    args = parser.parse_args()

    models = args.models.split(",")
    benchmark = ModelBenchmark()

    print(f"\n{'='*70}")
    print("LOCAL AI AGENTS - MODEL BENCHMARK")
    print(f"{'='*70}")
    print(f"Models: {', '.join(models)}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"{'='*70}\n")

    if args.quick:
        # Quick single test
        result = benchmark.run_single(
            model=models[0],
            prompt="Write a short poem about code.",
            category="quick_test"
        )
        print(f"\nQuick test result:")
        print(f"  Model: {result.model}")
        print(f"  Tokens/sec: {result.tokens_per_second:.2f}")
        print(f"  TTFT: {result.time_to_first_token_ms:.0f}ms")
    else:
        # Full benchmark
        stats = benchmark.run_multi_model(models=models)
        report = benchmark.generate_report(stats)
        print(report)

        # Compliance tests if requested
        if args.compliance:
            print("\n--- Compliance Tests ---\n")
            for model in models:
                print(f"\nModel: {model}")
                results = benchmark.run_compliance_test(model)

        # Save results
        filepath = benchmark.save_results(args.output)
        print(f"\nResults saved to: {filepath}")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
