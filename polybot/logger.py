"""
polybot/logger.py
=================
Structured, colour-aware logger for PolyBot.
Creates both a rotating file handler and a rich console handler.
"""

import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from polybot.config import LOG_LEVEL

# ── Colour codes for terminal output (Windows-safe) ──────────────────────────
RESET  = "\033[0m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"

LEVEL_COLOURS = {
    "DEBUG":    CYAN,
    "INFO":     GREEN,
    "WARNING":  YELLOW,
    "ERROR":    RED,
    "CRITICAL": BOLD + RED,
}


class ColouredFormatter(logging.Formatter):
    """ANSI-colour formatter for console output."""

    def format(self, record: logging.LogRecord) -> str:
        colour = LEVEL_COLOURS.get(record.levelname, RESET)
        record.levelname = f"{colour}{record.levelname:<8}{RESET}"
        return super().format(record)


def get_logger(name: str = "polybot", log_file: str = "polybot.log") -> logging.Logger:
    """
    Build and return a logger with:
      - Console handler (coloured, human-readable)
      - Rotating file handler (plain text, 5 MB × 3 backups)
    """
    logger = logging.getLogger(name)
    if logger.handlers:                         # already configured
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Force UTF-8 on Windows to avoid UnicodeEncodeError
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    fmt_str   = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_fmt  = "%Y-%m-%d %H:%M:%S"

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(ColouredFormatter(fmt_str, datefmt=date_fmt))
    logger.addHandler(ch)

    # Rotating file handler – always plain text
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3,
        encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(fmt_str, datefmt=date_fmt))
    logger.addHandler(fh)

    return logger


# Module-level default logger
log = get_logger()
