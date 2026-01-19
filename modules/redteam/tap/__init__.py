"""
TAP (Tree of Attacks with Pruning) Implementation.

Based on: "Tree of Attacks: Jailbreaking Black-Box LLMs with Automatic Adversarial Prompts"
Paper: NeurIPS 2024

TAP achieves >80% success rate on GPT-4 with ~28 queries on average.

Components:
- Attacker: LLM that generates jailbreak candidates
- Evaluator: LLM that judges if bypass was successful
- Pruner: Mechanism to eliminate ineffective attack branches
- Orchestrator: Coordinates the tree-of-attacks process
"""

from .orchestrator import TAPOrchestrator, TAPConfig
from .attacker import TAPAttacker
from .evaluator import TAPEvaluator
from .pruner import TAPPruner

__all__ = [
    "TAPOrchestrator",
    "TAPConfig",
    "TAPAttacker",
    "TAPEvaluator",
    "TAPPruner",
]
