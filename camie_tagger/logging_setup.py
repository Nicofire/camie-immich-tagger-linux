"""Console and rotating-file logging with secret redaction."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

LOGGER_NAME = "camie_tagger"
_MASK = "***"
_MIN_SECRET_LENGTH = 8


class SecretRedactor(logging.Filter):
    """Replaces known secret values anywhere in a formatted log message."""

    def __init__(self) -> None:
        super().__init__()
        self._secrets: set[str] = set()

    def add(self, *values: str | None) -> None:
        for value in values:
            if value and len(value) >= _MIN_SECRET_LENGTH:
                self._secrets.add(value)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        message = record.getMessage()
        redacted = message
        for secret in self._secrets:
            redacted = redacted.replace(secret, _MASK)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


redactor = SecretRedactor()


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backups: int = 5,
) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric_level)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    console.addFilter(redactor)
    logger.addHandler(console)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
            )
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
            )
            file_handler.addFilter(redactor)
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning("Could not open log file %s: %s", log_file, exc)

    return logger
