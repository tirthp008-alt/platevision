"""Logging setup with plate text redaction for security and privacy."""

import logging
import re
import sys
from typing import Any
from app.core.config import settings


def redact_plate_text(text: str) -> str:
    """Mask plate number characters to protect privacy in logs.
    e.g., 'GJ01AB1234' -> 'GJ01****34'
    """
    if not text or len(text) < 4:
        return "****"
    if len(text) <= 6:
        return text[:2] + "**" + text[-1:]
    return text[:4] + "*" * (len(text) - 6) + text[-2:]


class PrivacyRedactingFormatter(logging.Formatter):
    """Custom formatter that automatically redacts vehicle plate patterns."""

    # Matches Indian & general plate-like patterns in log messages
    PLATE_REGEX = re.compile(
        r"\b([A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}|[0-9]{2}BH[0-9]{4}[A-Z]{1,2})\b",
        re.IGNORECASE,
    )

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if settings.ENABLE_REDACTED_LOGS:
            msg = self.PLATE_REGEX.sub(
                lambda m: redact_plate_text(m.group(0)), msg
            )
        return msg


class EncodingSafeStreamHandler(logging.StreamHandler):
    """Escape unsupported console characters after privacy redaction."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        encoding = getattr(self.stream, "encoding", None)
        if encoding:
            return message.encode(encoding, errors="backslashreplace").decode(encoding)
        return message


def setup_logger(name: str = "platevision") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = EncodingSafeStreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = PrivacyRedactingFormatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()
