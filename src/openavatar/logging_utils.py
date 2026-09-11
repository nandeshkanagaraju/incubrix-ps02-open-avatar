"""Structured logging. Human-readable on stderr, JSON lines to a run log."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


def configure(log_file: str | Path | None = None, verbose: bool = False) -> None:
    global _CONFIGURED
    root = logging.getLogger("openavatar")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not _CONFIGURED:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(levelname)-7s %(name)s | %(message)s"))
        root.addHandler(h)
        _CONFIGURED = True
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("openavatar") else f"openavatar.{name}")
