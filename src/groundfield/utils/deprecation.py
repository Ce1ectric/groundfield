"""Warnings of the public API: deprecated names and inert configuration.

Deprecation policy (calendar versioning, since 2026.11): a deprecated
name keeps working, emits a :class:`DeprecationWarning` that names its
replacement, and is removed no earlier than two releases and six months
after the announcement — for the names deprecated in 2026.11 that is
release :data:`REMOVAL_VERSION` (``2027.5``) at the earliest.

Inert configuration: a public setting that is accepted and stored but
read by no backend warns with :class:`InertConfigurationWarning`, so a
user does not believe a setting took effect when it did not.

Who gets warned
---------------
A warning must be actionable for the person who reads it. Both helpers
therefore attribute the warning to the **first caller outside the
declaring module and pydantic** (the user's line), and stay silent when
that caller is groundfield's own code — e.g. a generator that still
builds a soil model with an old keyword: the user cannot change that
call, and the maintainers are tracked separately.
"""

from __future__ import annotations

import sys
import warnings
from types import FrameType

__all__ = [
    "REMOVAL_VERSION",
    "DEPRECATED_SINCE",
    "InertConfigurationWarning",
    "warn_deprecated",
    "warn_inert",
]

#: Release in which the names deprecated in :data:`DEPRECATED_SINCE` are
#: removed at the earliest (two releases and six months later).
REMOVAL_VERSION = "2027.5"

#: First calendar-versioned release; the deprecations listed in its
#: changelog start here.
DEPRECATED_SINCE = "2026.11"

_LIBRARY_PREFIXES = ("pydantic.", "pydantic_core.")
_LIBRARY_MODULES = ("pydantic", "pydantic_core")


class InertConfigurationWarning(UserWarning):
    """A public setting is stored, but no backend reads it.

    Examples are a non-default :class:`~groundfield.BoundaryConditions`
    field or a soil ``relative_permittivity``: every backend uses the
    quasi-static conduction-current formulation with remote-earth
    boundary conditions, so these values do not change any result.
    Filter it like any warning category, e.g.
    ``warnings.simplefilter("ignore", InertConfigurationWarning)``.
    """


def _module_of(frame: FrameType) -> str:
    return str(frame.f_globals.get("__name__", ""))


def _is_groundfield(module: str) -> bool:
    return module == "groundfield" or module.startswith("groundfield.")


def _is_library(module: str) -> bool:
    return module in _LIBRARY_MODULES or module.startswith(_LIBRARY_PREFIXES)


def _emit(message: str, category: type[Warning]) -> None:
    """Warn on behalf of the code that called ``warn_deprecated`` / ``warn_inert``.

    Frame layout: 0 = ``_emit``, 1 = the public helper, 2 = the code
    that asked for the warning (the *declaring* code). ``level`` is the
    ``stacklevel`` of a frame as seen from the ``warnings.warn`` call
    below.
    """
    frame: FrameType | None = sys._getframe(2)
    level = 3
    if frame is None:  # pragma: no cover - defensive
        warnings.warn(message, category, stacklevel=2)
        return
    declaring = _module_of(frame)
    if not _is_groundfield(declaring):
        # Called directly from user code: attribute it to that line.
        warnings.warn(message, category, stacklevel=level)
        return
    while frame is not None:
        module = _module_of(frame)
        if module != declaring and not _is_library(module):
            break
        frame = frame.f_back
        level += 1
    if frame is not None and _is_groundfield(_module_of(frame)):
        # groundfield's own code caused it: not actionable for the user.
        return
    warnings.warn(message, category, stacklevel=level if frame is not None else 3)


def warn_deprecated(old: str, new: str) -> None:
    """Emit a :class:`DeprecationWarning` for a renamed or replaced name.

    Parameters
    ----------
    old
        What the caller used, e.g. ``"HomogeneousSoil: 'resistivity'"``.
    new
        Its replacement, e.g. ``"'rho'"``.

    Notes
    -----
    The message states the replacement and the earliest removal
    release (:data:`REMOVAL_VERSION`). See the module docstring for who
    is warned.
    """
    _emit(
        f"{old} is deprecated since groundfield {DEPRECATED_SINCE}; use {new} "
        f"instead. The old name will be removed in groundfield "
        f"{REMOVAL_VERSION} at the earliest.",
        DeprecationWarning,
    )


def warn_inert(message: str) -> None:
    """Emit an :class:`InertConfigurationWarning` with ``message``.

    Parameters
    ----------
    message
        Complete sentence(s) naming the setting, its value and why it
        has no effect.
    """
    _emit(message, InertConfigurationWarning)
