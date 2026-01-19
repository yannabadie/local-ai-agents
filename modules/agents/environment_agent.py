#!/usr/bin/env python3
"""
Environment Interaction Agent
Date: 2026-01-19

Allows LLM to interact with local environment:
- Execute shell commands
- Read/write files
- Navigate filesystem
- Run security tools

This demonstrates Objective #2: "Interaction environnement"
"""

import os
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import requests

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "deephat"  # Cybersecurity-focused model
SAFE_MODE = True  # Require confirmation for dangerous commands

# Evidence logging
EVIDENCE_DIR = Path(__file__).parent.parent.parent / "tests" / "poc" / "agent-interaction"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# Dangerous commands that require confirmation
DANGEROUS_PATTERNS = [
    "rm -rf", "del /f", "format", "mkfs",
    "shutdown", "reboot", ":(){:|:&};:",
    "dd if=", "> /dev/", "chmod 777"
]


class EnvironmentAgent:
    """
    Agent that can execute commands on the local system.

    Demonstrates LLM-environment interaction for security research.
    """

    def __init__(self, model: str = DEFAULT_MODEL, safe_mode: bool = True):
        self.model = model
        self.safe_mode = safe_mode
        self.history: List[Dict] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _query_llm(self, prompt: str, system_context: str = "") -> str:
        """Query the local LLM."""
        full_prompt = f"{system_context}\n\n{prompt}" if system_context else prompt

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"num_predict": 1000}
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"ERROR: {e}"

    def _is_dangerous(self, command: str) -> bool:
        """Check if command is potentially dangerous."""
        cmd_lower = command.lower()
        return any(pattern.lower() in cmd_lower for pattern in DANGEROUS_PATTERNS)

    def _execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a shell command and return result."""
        if self.safe_mode and self._is_dangerous(command):
            return {
                "success": False,
                "output": "",
                "error": f"BLOCKED: Dangerous command detected. Disable safe_mode to execute.",
                "command": command
            }

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(Path.home())
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "command": command,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "Command timed out after 30 seconds",
                "command": command
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "command": command
            }

    def _parse_action(self, llm_response: str) -> Optional[Dict]:
        """Parse LLM response for action to execute."""
        # Look for command blocks
        if "```bash" in llm_response or "```shell" in llm_response:
            start = llm_response.find("```") + 3
            # Skip the language identifier
            start = llm_response.find("\n", start) + 1
            end = llm_response.find("```", start)
            if end > start:
                command = llm_response[start:end].strip()
                return {"type": "execute", "command": command}

        # Look for file operations
        if "READ_FILE:" in llm_response:
            idx = llm_response.find("READ_FILE:") + 10
            end = llm_response.find("\n", idx)
            filepath = llm_response[idx:end].strip()
            return {"type": "read_file", "path": filepath}

        if "WRITE_FILE:" in llm_response:
            # Format: WRITE_FILE:path\n```content```
            idx = llm_response.find("WRITE_FILE:") + 11
            end = llm_response.find("\n", idx)
            filepath = llm_response[idx:end].strip()

            content_start = llm_response.find("```", end) + 3
            content_start = llm_response.find("\n", content_start) + 1
            content_end = llm_response.find("```", content_start)
            content = llm_response[content_start:content_end]

            return {"type": "write_file", "path": filepath, "content": content}

        return None

    def run_task(self, task: str, max_iterations: int = 5) -> Dict:
        """
        Run a task with the agent.

        The agent will:
        1. Analyze the task
        2. Plan actions
        3. Execute commands
        4. Report results
        """
        system_context = """You are a security research agent with access to a local system.
You can execute commands and interact with the filesystem.

To execute a command, wrap it in ```bash blocks:
```bash
command here
```

To read a file: READ_FILE:/path/to/file
To write a file: WRITE_FILE:/path/to/file
```
content here
```

Always explain your reasoning before taking actions.
Report findings clearly and document evidence."""

        conversation = []
        results = {
            "task": task,
            "session_id": self.session_id,
            "iterations": [],
            "success": False,
            "summary": ""
        }

        current_prompt = f"TASK: {task}\n\nAnalyze this task and begin execution."

        for i in range(max_iterations):
            print(f"\n[Iteration {i+1}/{max_iterations}]")

            # Get LLM response
            llm_response = self._query_llm(current_prompt, system_context)
            print(f"LLM Response: {llm_response[:300]}...")

            iteration_result = {
                "iteration": i + 1,
                "prompt": current_prompt[:500],
                "llm_response": llm_response,
                "action": None,
                "action_result": None
            }

            # Parse and execute action
            action = self._parse_action(llm_response)
            if action:
                iteration_result["action"] = action
                print(f"Action detected: {action['type']}")

                if action["type"] == "execute":
                    cmd_result = self._execute_command(action["command"])
                    iteration_result["action_result"] = cmd_result
                    print(f"Command result: {cmd_result['output'][:200] if cmd_result['success'] else cmd_result['error']}")

                    # Prepare next prompt with result
                    current_prompt = f"""Previous command: {action['command']}
Output: {cmd_result['output'][:1000]}
Error: {cmd_result['error'][:500] if cmd_result['error'] else 'None'}

Continue with the task or provide final summary if complete."""

                elif action["type"] == "read_file":
                    try:
                        with open(action["path"], 'r') as f:
                            content = f.read()[:2000]
                        iteration_result["action_result"] = {"success": True, "content": content}
                        current_prompt = f"File content of {action['path']}:\n{content}\n\nContinue with the task."
                    except Exception as e:
                        iteration_result["action_result"] = {"success": False, "error": str(e)}
                        current_prompt = f"Failed to read file: {e}\n\nContinue with the task."

                elif action["type"] == "write_file":
                    try:
                        with open(action["path"], 'w') as f:
                            f.write(action["content"])
                        iteration_result["action_result"] = {"success": True}
                        current_prompt = f"Successfully wrote to {action['path']}.\n\nContinue with the task."
                    except Exception as e:
                        iteration_result["action_result"] = {"success": False, "error": str(e)}
                        current_prompt = f"Failed to write file: {e}\n\nContinue with the task."
            else:
                # No action, might be final summary
                if "complete" in llm_response.lower() or "finished" in llm_response.lower():
                    results["success"] = True
                    results["summary"] = llm_response
                    results["iterations"].append(iteration_result)
                    break

                current_prompt = "Please provide a specific action (command to execute) or indicate if the task is complete."

            results["iterations"].append(iteration_result)
            conversation.append({"role": "assistant", "content": llm_response})

        # Save evidence
        self._save_evidence(results)

        return results

    def _save_evidence(self, results: Dict):
        """Save task results as evidence."""
        filepath = EVIDENCE_DIR / f"agent_task_{self.session_id}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nEvidence saved to: {filepath}")


def demo_reconnaissance():
    """Demo: Basic system reconnaissance task."""
    agent = EnvironmentAgent(model="deephat", safe_mode=True)

    task = """Perform basic system reconnaissance:
1. Get current user and hostname
2. List running processes
3. Check network configuration
4. Document findings"""

    print("=" * 60)
    print("DEMO: System Reconnaissance with Environment Agent")
    print("=" * 60)

    result = agent.run_task(task, max_iterations=5)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Success: {result['success']}")
    print(f"Iterations: {len(result['iterations'])}")

    return result


def demo_file_analysis():
    """Demo: Analyze files in current directory."""
    agent = EnvironmentAgent(model="deephat", safe_mode=True)

    task = """Analyze the current project structure:
1. List files in the current directory
2. Identify interesting files for security analysis
3. Report your findings"""

    print("=" * 60)
    print("DEMO: File Analysis with Environment Agent")
    print("=" * 60)

    result = agent.run_task(task, max_iterations=4)

    return result


if __name__ == "__main__":
    import sys

    print("Environment Interaction Agent - POC")
    print("Demonstrates LLM-environment interaction\n")

    if len(sys.argv) > 1 and sys.argv[1] == "--recon":
        demo_reconnaissance()
    elif len(sys.argv) > 1 and sys.argv[1] == "--files":
        demo_file_analysis()
    else:
        # Default: run reconnaissance demo
        demo_reconnaissance()
