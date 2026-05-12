"""
stock_regime/src/logging_config.py
====================================
Configures structured logging for the Stock Regime Engine.

Two handlers are set up:
  1. Console (StreamHandler) — INFO level, concise format
  2. File (RotatingFileHandler) — DEBUG level, timestamped format

Call ``configure_logging()`` once at application startup.
The log file is written to ``output/logs/stock_regime_YYYY-MM-DD.log``.
"""

from __future__ import annotations

import logging
import logging.handlers
from datetime import date
from pathlib import Path


def configure_logging(
    log_dir: str | Path = "output/logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB per file
    backup_count: int = 7,
) -> logging.Logger:
    """
    Set up console and rotating file logging for the Stock Regime Engine.

    Parameters
    ----------
    log_dir :
        Directory where log files are written.  Created if absent.
    console_level :
        Log level for the console handler (default: INFO).
    file_level :
        Log level for the file handler (default: DEBUG).
    max_bytes :
        Maximum size before the log file rotates.
    backup_count :
        Number of rotated files to retain.

    Returns
    -------
    logging.Logger
        The configured root-level ``stock_regime`` logger.

    Example
    -------
    >>> from stock_regime.src.logging_config import configure_logging
    >>> logger = configure_logging()
    >>> logger.info("Engine starting.")
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"stock_regime_{date.today().strftime('%Y-%m-%d')}.log"

    root_logger = logging.getLogger("stock_regime")
    root_logger.setLevel(logging.DEBUG)   # handlers filter from here

    # Avoid adding duplicate handlers on repeated calls (e.g. in tests)
    if root_logger.handlers:
        return root_logger

    # ── Console handler ─────────────────────────────────────────────
    console_fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_fmt)

    # ── Rotating file handler ────────────────────────────────────────
    file_fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)-40s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_fmt)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    root_logger.info("Logging configured → '%s'.", log_file)
    return root_logger
