"""Lightweight logging utilities used across the ACP runtime."""

from __future__ import annotations

import logging


def _ensure_configured(level: int) -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger ensuring the root logger is configured."""

    _ensure_configured(level)
    logger = logging.getLogger(name)
    if logger.level == logging.NOTSET:
        logger.setLevel(level)
    return logger
