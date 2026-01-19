"""
Base agent class for environment interaction.
"""

import subprocess
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentAction:
    """Represents an action taken by the agent."""
    action_type: str  # "shell", "file_read", "file_write", "api_call"
    command: str
    timestamp: datetime = field(default_factory=datetime.now)
    result: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


@dataclass
class AgentSession:
    """Tracks an agent session."""
    session_id: str
    model: str
    start_time: datetime = field(default_factory=datetime.now)
    actions: List[AgentAction] = field(default_factory=list)
    working_directory: Path = field(default_factory=Path.cwd)

    def add_action(self, action: AgentAction):
        self.actions.append(action)

    def get_history(self) -> List[Dict]:
        return [
            {
                "type": action.action_type,
                "command": action.command,
                "success": action.success,
                "timestamp": action.timestamp.isoformat()
            }
            for action in self.actions
        ]


class BaseAgent(ABC):
    """Abstract base class for agents."""

    def __init__(self, model: str = "deephat", working_dir: Optional[Path] = None):
        self.model = model
        self.working_dir = working_dir or Path.cwd()
        self.session: Optional[AgentSession] = None
        self.logger = get_logger(f"agent.{self.__class__.__name__}")

    def start_session(self) -> AgentSession:
        """Start a new agent session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session = AgentSession(
            session_id=session_id,
            model=self.model,
            working_directory=self.working_dir
        )
        self.logger.info(f"Session started: {session_id}")
        return self.session

    def execute_shell(self, command: str, timeout: int = 30) -> AgentAction:
        """Execute a shell command."""
        action = AgentAction(action_type="shell", command=command)

        try:
            self.logger.info(f"Executing: {command}")
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.working_dir)
            )
            action.result = result.stdout
            action.success = result.returncode == 0
            if result.stderr:
                action.error = result.stderr

            self.logger.info(f"Command {'succeeded' if action.success else 'failed'}")

        except subprocess.TimeoutExpired:
            action.error = f"Command timed out after {timeout}s"
            action.success = False
            self.logger.error(action.error)

        except Exception as e:
            action.error = str(e)
            action.success = False
            self.logger.error(f"Command failed: {e}")

        if self.session:
            self.session.add_action(action)

        return action

    def read_file(self, filepath: str) -> AgentAction:
        """Read a file."""
        action = AgentAction(action_type="file_read", command=f"read:{filepath}")

        try:
            path = Path(filepath)
            if not path.is_absolute():
                path = self.working_dir / path

            content = path.read_text(encoding="utf-8")
            action.result = content
            action.success = True
            self.logger.info(f"Read file: {filepath} ({len(content)} chars)")

        except Exception as e:
            action.error = str(e)
            action.success = False
            self.logger.error(f"Failed to read {filepath}: {e}")

        if self.session:
            self.session.add_action(action)

        return action

    def write_file(self, filepath: str, content: str) -> AgentAction:
        """Write to a file."""
        action = AgentAction(action_type="file_write", command=f"write:{filepath}")

        try:
            path = Path(filepath)
            if not path.is_absolute():
                path = self.working_dir / path

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            action.result = f"Written {len(content)} chars to {filepath}"
            action.success = True
            self.logger.info(action.result)

        except Exception as e:
            action.error = str(e)
            action.success = False
            self.logger.error(f"Failed to write {filepath}: {e}")

        if self.session:
            self.session.add_action(action)

        return action

    def list_directory(self, path: str = ".") -> AgentAction:
        """List directory contents."""
        action = AgentAction(action_type="file_read", command=f"ls:{path}")

        try:
            dir_path = Path(path)
            if not dir_path.is_absolute():
                dir_path = self.working_dir / dir_path

            items = list(dir_path.iterdir())
            action.result = "\n".join(str(item) for item in items)
            action.success = True
            self.logger.info(f"Listed {len(items)} items in {path}")

        except Exception as e:
            action.error = str(e)
            action.success = False
            self.logger.error(f"Failed to list {path}: {e}")

        if self.session:
            self.session.add_action(action)

        return action

    @abstractmethod
    def process_task(self, task: str) -> str:
        """Process a natural language task. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def chat(self, message: str) -> str:
        """Send a chat message and get response. Must be implemented by subclasses."""
        pass
