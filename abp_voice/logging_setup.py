"""Shared logging + rich console setup."""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler

# Suppress noisy HuggingFace messages by default.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

console = Console()

_LEVEL = os.environ.get("ABP_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=_LEVEL,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
