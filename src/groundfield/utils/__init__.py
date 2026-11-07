"""Helper utilities.

Cross-cutting helpers used by several subpackages:

- structured logging configuration (:func:`get_logger`, :func:`configure`),
- name suggestions for error messages (:func:`suggest`,
  :func:`did_you_mean`),
- warnings of the public API: deprecated names (:func:`warn_deprecated`,
  removal no earlier than :data:`REMOVAL_VERSION`) and settings that no
  backend reads (:func:`warn_inert`,
  :class:`InertConfigurationWarning`).
"""

from groundfield.utils.deprecation import (
    DEPRECATED_SINCE,
    REMOVAL_VERSION,
    InertConfigurationWarning,
    warn_deprecated,
    warn_inert,
)
from groundfield.utils.logging import configure, get_logger
from groundfield.utils.suggest import did_you_mean, suggest

__all__ = [
    "get_logger",
    "configure",
    "suggest",
    "did_you_mean",
    "warn_deprecated",
    "warn_inert",
    "InertConfigurationWarning",
    "REMOVAL_VERSION",
    "DEPRECATED_SINCE",
]
