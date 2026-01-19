"""
Structured logging with Rich for Local AI Agents.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.theme import Theme
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Custom theme for security research logging
SECURITY_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "critical": "red bold reverse",
    "success": "green bold",
    "exploit": "magenta bold",
    "evidence": "blue bold",
})

# Global console instance
console = Console(theme=SECURITY_THEME) if RICH_AVAILABLE else None

# Log format
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    rich_console: bool = True
) -> None:
    """Setup logging configuration."""

    handlers = []

    # Console handler
    if RICH_AVAILABLE and rich_console:
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True
        )
        rich_handler.setLevel(level)
        handlers.append(rich_handler)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        stream_handler.setLevel(level)
        handlers.append(stream_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    logger = logging.getLogger(name)

    # Setup if not already configured
    if not logger.handlers and not logging.root.handlers:
        setup_logging()

    return logger


class EvidenceLogger:
    """
    Specialized logger for collecting PoC evidence.
    Logs to both console and evidence files.
    """

    def __init__(self, poc_name: str, evidence_dir: Optional[Path] = None):
        self.poc_name = poc_name
        self.evidence_dir = evidence_dir or Path("tests/poc") / poc_name / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.evidence_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.logger = get_logger(f"poc.{poc_name}")

        # Add file handler for evidence
        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        self.logger.addHandler(fh)

    def log_exploit_attempt(self, technique: str, target: str, payload: str):
        """Log an exploit attempt."""
        self.logger.info(f"[EXPLOIT] Technique: {technique} | Target: {target}")
        self.logger.debug(f"Payload: {payload[:200]}...")

    def log_success(self, description: str, evidence: Optional[str] = None):
        """Log successful exploitation."""
        self.logger.warning(f"[SUCCESS] {description}")
        if evidence:
            evidence_file = self.evidence_dir / f"success_{datetime.now().strftime('%H%M%S')}.txt"
            evidence_file.write_text(evidence, encoding="utf-8")
            self.logger.info(f"Evidence saved: {evidence_file}")

    def log_failure(self, description: str, reason: str):
        """Log failed attempt."""
        self.logger.info(f"[FAILED] {description} - Reason: {reason}")

    def log_model_response(self, model: str, prompt: str, response: str):
        """Log full model interaction for evidence."""
        timestamp = datetime.now().strftime('%H%M%S')
        response_file = self.evidence_dir / f"response_{model}_{timestamp}.txt"

        content = f"""=== MODEL RESPONSE LOG ===
Timestamp: {datetime.now().isoformat()}
Model: {model}

=== PROMPT ===
{prompt}

=== RESPONSE ===
{response}
"""
        response_file.write_text(content, encoding="utf-8")
        self.logger.debug(f"Response logged: {response_file}")
