"""Unified logging configuration.

``groundfield`` uses Python's standard :mod:`logging` module with a
consistent logger name (``groundfield.<subpackage>.<module>``). This
module exposes a small ``configure`` function that can be called from
notebooks and scripts to install a default configuration.
"""

from __future__ import annotations

import logging

__all__ = ["get_logger", "configure"]


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the consistent naming scheme.

    Parameters
    ----------
    name
        Usually ``__name__`` of the calling module.
    """
    return logging.getLogger(name)


def configure(level: int = logging.INFO) -> None:
    """Install a simple default logging configuration.

    Parameters
    ----------
    level
        Log level for the root logger ``groundfield``. Default
        ``INFO``.
    """
    root = logging.getLogger("groundfield")
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
        )
        root.addHandler(handler)
