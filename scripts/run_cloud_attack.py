#!/usr/bin/env python3
"""
Cloud Attack CLI - Run red-team attacks on cloud LLM APIs.

Usage:
    python scripts/run_cloud_attack.py --target gpt-4o --technique tap --request "explain lockpicking"
    python scripts/run_cloud_attack.py --target claude-3-haiku --technique pyrit --request "bypass security"
    python scripts/run_cloud_attack.py --list-models
    python scripts/run_cloud_attack.py --test-connection

Examples:
    # Run TAP attack on GPT-4o
    python scripts/run_cloud_attack.py -t gpt-4o -T tap -r "explain how to pick a lock"

    # Run with custom max queries
    python scripts/run_cloud_attack.py -t gpt-4o -T tap -r "target behavior" --max-queries 50

    # List available models
    python scripts/run_cloud_attack.py --list-models

    # Test API connections
    python scripts/run_cloud_attack.py --test-connection
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def list_models():
    """List available models from all configured providers."""
    from modules.redteam import CloudClient

    console.print("\n[bold cyan]Available Cloud Models[/bold cyan]\n")

    try:
        client = CloudClient.from_env()

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Provider", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Status", style="yellow")

        for provider in client.available_providers():
            models = client.list_models(provider)
            for model in models:
                table.add_row(provider.value, model, "[green]Available[/green]")

        if not client.available_providers():
            console.print("[yellow]No providers configured. Check your API keys in .env[/yellow]")
        else:
            console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")
        sys.exit(1)


def test_connection():
    """Test connections to all configured providers."""
    from modules.redteam import CloudClient

    console.print("\n[bold cyan]Testing API Connections[/bold cyan]\n")

    try:
        client = CloudClient.from_env()

        results = client.test_all_connections()

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="green")

        for provider, success in results.items():
            status = "[green]Connected[/green]" if success else "[red]Failed[/red]"
            table.add_row(provider.value, status)

        if not results:
            console.print("[yellow]No providers configured. Set API keys in .env file.[/yellow]")
        else:
            console.print(table)

    except Exception as e:
        console.print(f"[red]Error testing connections: {e}[/red]")
        sys.exit(1)


def run_tap_attack(target_model: str, target_behavior: str, max_queries: int, provider: str):
    """Run a TAP attack."""
    from modules.redteam import CloudClient, TAPOrchestrator, TAPConfig
    from modules.redteam.cloud_client import CloudProvider

    console.print(Panel(
        f"[bold]TAP Attack[/bold]\n"
        f"Target: {provider}/{target_model}\n"
        f"Behavior: {target_behavior}\n"
        f"Max Queries: {max_queries}",
        title="Attack Configuration",
        border_style="cyan"
    ))

    try:
        client = CloudClient.from_env()

        config = TAPConfig(
            max_queries=max_queries,
            attacker_model="gpt-4o-mini",
            evaluator_model="gpt-4o-mini",
            save_evidence=True,
        )

        orchestrator = TAPOrchestrator(client, config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running TAP attack...", total=None)

            result = orchestrator.attack(
                target_behavior=target_behavior,
                target_provider=provider,
                target_model=target_model,
            )

            progress.update(task, completed=True)

        # Display results
        console.print("\n")

        if result.success:
            console.print(Panel(
                f"[bold green]Attack Successful![/bold green]\n\n"
                f"Best Score: {result.best_score:.2f}\n"
                f"Queries Used: {result.total_queries}\n"
                f"Cost: ${result.total_cost_usd:.4f}\n"
                f"Duration: {result.duration_seconds:.1f}s\n\n"
                f"[bold]Best Prompt:[/bold]\n{result.best_prompt[:500]}...\n\n"
                f"[bold]Response:[/bold]\n{result.best_response[:500] if result.best_response else 'N/A'}...",
                title="Results",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[bold yellow]Attack Did Not Succeed[/bold yellow]\n\n"
                f"Best Score: {result.best_score:.2f}\n"
                f"Queries Used: {result.total_queries}\n"
                f"Cost: ${result.total_cost_usd:.4f}\n"
                f"Duration: {result.duration_seconds:.1f}s",
                title="Results",
                border_style="yellow"
            ))

        # Show pruning stats
        if result.pruning_stats:
            console.print(f"\nPruning Stats: {result.pruning_stats.pruned_candidates}/{result.pruning_stats.total_candidates} pruned")

        return result

    except Exception as e:
        console.print(f"[red]Attack failed: {e}[/red]")
        logger.exception("Attack error")
        sys.exit(1)


def run_pyrit_attack(target_model: str, target_behavior: str, provider: str):
    """Run a PyRIT-based attack."""
    from modules.redteam.pyrit_wrapper import PyRITWrapper, PyRITConfig, PyRITPromptGenerator

    console.print(Panel(
        f"[bold]PyRIT Attack[/bold]\n"
        f"Target: {provider}/{target_model}\n"
        f"Behavior: {target_behavior}",
        title="Attack Configuration",
        border_style="cyan"
    ))

    try:
        # Generate prompts using PyRIT techniques
        prompts = PyRITPromptGenerator.generate(
            target_behavior=target_behavior,
            num_prompts=20,
        )

        console.print(f"\n[cyan]Generated {len(prompts)} adversarial prompts[/cyan]\n")

        for i, prompt in enumerate(prompts[:5], 1):
            console.print(f"[dim]{i}. {prompt[:100]}...[/dim]")

        console.print(f"\n... and {len(prompts) - 5} more prompts")

        # Save prompts for manual review
        output_dir = Path("tests/cloud_attacks/pyrit_prompts")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"prompts_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump({
                "target_behavior": target_behavior,
                "prompts": prompts,
                "timestamp": timestamp,
            }, f, indent=2)

        console.print(f"\n[green]Prompts saved to {output_file}[/green]")

    except Exception as e:
        console.print(f"[red]PyRIT attack failed: {e}[/red]")
        sys.exit(1)


def run_garak_scan(target_model: str, probes: list, provider: str):
    """Run a Garak vulnerability scan."""
    from modules.redteam.garak_wrapper import GarakWrapper, GarakConfig

    console.print(Panel(
        f"[bold]Garak Vulnerability Scan[/bold]\n"
        f"Target: {provider}/{target_model}\n"
        f"Probes: {', '.join(probes)}",
        title="Scan Configuration",
        border_style="cyan"
    ))

    try:
        config = GarakConfig(
            generator=provider,
            model_name=target_model,
            probes=probes,
        )

        wrapper = GarakWrapper(config)

        # List available probes
        console.print("\n[cyan]Available Probes:[/cyan]")
        for name, desc in wrapper.list_available_probes().items():
            console.print(f"  [dim]{name}[/dim]: {desc}")

        # Run scan
        result = wrapper.scan(probes=probes)

        # Display results
        console.print(f"\n[bold]Scan Results:[/bold]")
        console.print(f"Total Probes: {result.total_probes}")
        console.print(f"Total Attempts: {result.total_attempts}")

        for probe_result in result.probe_results:
            console.print(f"\n[cyan]{probe_result.probe_name}[/cyan]")
            console.print(f"  Prompts: {len(probe_result.successful_prompts)}")
            for prompt in probe_result.successful_prompts[:3]:
                console.print(f"    [dim]- {prompt[:80]}...[/dim]")

    except Exception as e:
        console.print(f"[red]Garak scan failed: {e}[/red]")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Cloud Attack CLI - Red-team testing for cloud LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Target configuration
    parser.add_argument(
        "-t", "--target",
        help="Target model (e.g., gpt-4o, claude-3-haiku, gemini-2.0-flash)"
    )
    parser.add_argument(
        "-p", "--provider",
        choices=["openai", "anthropic", "google", "ollama"],
        default="openai",
        help="Cloud provider (default: openai)"
    )

    # Attack configuration
    parser.add_argument(
        "-T", "--technique",
        choices=["tap", "pyrit", "garak"],
        default="tap",
        help="Attack technique (default: tap)"
    )
    parser.add_argument(
        "-r", "--request",
        help="Target behavior / request to elicit"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=30,
        help="Maximum API queries (default: 30)"
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        default=["dan", "encoding", "misleading"],
        help="Garak probes to use (for garak technique)"
    )

    # Utility commands
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models from configured providers"
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test API connections"
    )

    args = parser.parse_args()

    # Print banner
    console.print(Panel(
        "[bold red]Cloud Attack CLI[/bold red]\n"
        "[dim]Red-team testing framework for cloud LLMs[/dim]",
        border_style="red"
    ))

    # Handle utility commands
    if args.list_models:
        list_models()
        return

    if args.test_connection:
        test_connection()
        return

    # Validate attack parameters
    if not args.target:
        console.print("[red]Error: --target is required for attacks[/red]")
        parser.print_help()
        sys.exit(1)

    if not args.request and args.technique != "garak":
        console.print("[red]Error: --request is required for TAP and PyRIT attacks[/red]")
        sys.exit(1)

    # Run attack based on technique
    if args.technique == "tap":
        run_tap_attack(
            target_model=args.target,
            target_behavior=args.request,
            max_queries=args.max_queries,
            provider=args.provider,
        )
    elif args.technique == "pyrit":
        run_pyrit_attack(
            target_model=args.target,
            target_behavior=args.request,
            provider=args.provider,
        )
    elif args.technique == "garak":
        run_garak_scan(
            target_model=args.target,
            probes=args.probes,
            provider=args.provider,
        )


if __name__ == "__main__":
    main()
