"""
Ollama-based agent for task execution.
"""

import re
import json
from typing import Optional, List, Dict
from pathlib import Path

from .base_agent import BaseAgent, AgentAction
from core.lib.ollama_client import OllamaClient
from core.utils.logger import get_logger

logger = get_logger(__name__)


AGENT_SYSTEM_PROMPT = """You are an AI agent that can interact with the local environment.
You have access to the following capabilities:
- Execute shell commands
- Read and write files
- List directory contents

When you need to perform an action, respond with a JSON action block:

```action
{
    "type": "shell" | "read" | "write" | "list",
    "command": "the command or filepath",
    "content": "content for write operations (optional)"
}
```

After the action is executed, you will receive the result and can continue.
Always explain what you're doing before and after actions.
If a task is complete, say "TASK COMPLETE" and summarize what was done.
"""


class OllamaAgent(BaseAgent):
    """Agent that uses Ollama for reasoning and task execution."""

    def __init__(
        self,
        model: str = "deephat",
        working_dir: Optional[Path] = None,
        max_iterations: int = 10
    ):
        super().__init__(model, working_dir)
        self.client = OllamaClient()
        self.max_iterations = max_iterations
        self.conversation_history: List[Dict[str, str]] = []

    def _parse_action(self, response: str) -> Optional[Dict]:
        """Extract action JSON from response."""
        # Look for action block
        pattern = r"```action\s*\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse action JSON: {e}")
                return None
        return None

    def _execute_action(self, action_dict: Dict) -> str:
        """Execute an action and return result."""
        action_type = action_dict.get("type", "").lower()
        command = action_dict.get("command", "")
        content = action_dict.get("content", "")

        if action_type == "shell":
            result = self.execute_shell(command)
        elif action_type == "read":
            result = self.read_file(command)
        elif action_type == "write":
            result = self.write_file(command, content)
        elif action_type == "list":
            result = self.list_directory(command or ".")
        else:
            return f"Unknown action type: {action_type}"

        if result.success:
            return f"SUCCESS:\n{result.result}"
        else:
            return f"FAILED: {result.error}"

    def chat(self, message: str) -> str:
        """Send a message and get response."""
        self.conversation_history.append({"role": "user", "content": message})

        response = self.client.chat(
            messages=[
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                *self.conversation_history
            ],
            model=self.model,
            temperature=0.7
        )

        assistant_message = response.content
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def process_task(self, task: str) -> str:
        """Process a task with multi-step reasoning and action execution."""
        self.start_session()
        logger.info(f"Processing task: {task}")

        # Initial prompt
        current_message = f"Task: {task}\n\nPlease analyze this task and execute the necessary actions."
        full_output = []

        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")

            response = self.chat(current_message)
            full_output.append(f"=== Agent Response ===\n{response}")

            # Check for task completion
            if "TASK COMPLETE" in response.upper():
                logger.info("Task completed")
                break

            # Check for action
            action = self._parse_action(response)
            if action:
                logger.info(f"Executing action: {action}")
                result = self._execute_action(action)
                full_output.append(f"=== Action Result ===\n{result}")
                current_message = f"Action result:\n{result}\n\nContinue with the task or indicate completion."
            else:
                # No action found, might need clarification
                current_message = "Please specify an action to take or indicate if the task is complete."

        return "\n\n".join(full_output)

    def interactive_mode(self):
        """Run agent in interactive mode."""
        print(f"\n=== OllamaAgent Interactive Mode ===")
        print(f"Model: {self.model}")
        print(f"Working directory: {self.working_dir}")
        print("Type 'exit' to quit, 'clear' to reset conversation\n")

        self.start_session()

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue
                if user_input.lower() == "exit":
                    print("Goodbye!")
                    break
                if user_input.lower() == "clear":
                    self.conversation_history = []
                    print("Conversation cleared.")
                    continue

                # Check if it's a task (starts with "task:")
                if user_input.lower().startswith("task:"):
                    task = user_input[5:].strip()
                    result = self.process_task(task)
                    print(f"\n{result}\n")
                else:
                    response = self.chat(user_input)

                    # Check for and execute any actions
                    action = self._parse_action(response)
                    if action:
                        print(f"\nAgent: {response}")
                        result = self._execute_action(action)
                        print(f"\n[Action Result: {result}]")

                        # Get follow-up response
                        follow_up = self.chat(f"Action result: {result}")
                        print(f"\nAgent: {follow_up}\n")
                    else:
                        print(f"\nAgent: {response}\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def quick_test():
    """Quick test of the agent."""
    agent = OllamaAgent(model="deephat")
    agent.start_session()

    # Test shell command
    result = agent.execute_shell("echo Hello from agent")
    print(f"Shell test: {result.result}")

    # Test directory listing
    result = agent.list_directory(".")
    print(f"Directory listing: {result.result[:200]}...")

    print("\nAgent quick test completed!")


if __name__ == "__main__":
    quick_test()
