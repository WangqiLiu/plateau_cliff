# experiment/plateau_cliff/src/logging_utils.py
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path


class BraceLogger:
    """Small logging adapter with loguru-style brace formatting."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("plateau_cliff")

    @staticmethod
    def _format(message: str, *args: object) -> str:
        if not args:
            return message
        try:
            return message.format(*args)
        except Exception:
            return " ".join([message, *[str(arg) for arg in args]])

    def info(self, message: str, *args: object) -> None:
        self._logger.info(self._format(message, *args))

    def warning(self, message: str, *args: object) -> None:
        self._logger.warning(self._format(message, *args))

    def error(self, message: str, *args: object) -> None:
        self._logger.error(self._format(message, *args))

    def debug(self, message: str, *args: object) -> None:
        self._logger.debug(self._format(message, *args))

    def success(self, message: str, *args: object) -> None:
        self.info(message, *args)


logger = BraceLogger()


def setup_logger(log_dir: Path, run_name: str) -> Path:
    """Configure console and file logging for one experiment run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_logger = logging.getLogger("plateau_cliff")
    raw_logger.setLevel(logging.INFO)
    raw_logger.handlers.clear()
    raw_logger.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"))
    raw_logger.addHandler(console)

    log_path = log_dir / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s")
    )
    raw_logger.addHandler(file_handler)
    logger.info("Log file: {}", log_path)
    return log_path
