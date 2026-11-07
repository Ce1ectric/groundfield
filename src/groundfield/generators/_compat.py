"""Deprecation helpers for renamed generator names (review finding 10.69).

Every renamed public name of :mod:`groundfield.generators` keeps working
and emits a :class:`DeprecationWarning` that names the replacement and the
earliest removal release (:data:`REMOVAL_RELEASE`). The helpers here keep
the wording of those warnings uniform:

* :func:`warn_renamed` — the warning itself;
* :func:`rename_input_keys` — a ``model_validator(mode="before")`` helper
  that maps deprecated field names of a Pydantic model to the new ones,
  so configurations written by earlier releases still load;
* :func:`take_renamed_kwargs` — the same for keyword arguments of methods;
* :class:`DeprecatedKeyDict` — a ``dict`` whose deprecated keys stay
  readable.

The module is private; nothing here is part of the public API.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

__all__ = [
    "REMOVAL_RELEASE",
    "DeprecatedKeyDict",
    "rename_input_keys",
    "take_renamed_kwargs",
    "warn_renamed",
]

REMOVAL_RELEASE = "2027.5"
"""Earliest release in which deprecated generator names are removed."""


def warn_renamed(old: str, new: str, *, stacklevel: int = 3) -> None:
    """Emit the standard :class:`DeprecationWarning` for a renamed name.

    Parameters
    ----------
    old, new
        Deprecated and replacement name, e.g. ``"OrtsnetzLayout"`` and
        ``"LvNetworkLayout"``.
    stacklevel
        Forwarded to :func:`warnings.warn`.
    """
    warnings.warn(
        f"{old} is deprecated; use {new} instead. The old name keeps working "
        f"until groundfield {REMOVAL_RELEASE} at the earliest.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def rename_input_keys(data: Any, renames: Mapping[str, str], owner: str) -> Any:
    """Map deprecated input keys of a model to their new names.

    Intended for ``@model_validator(mode="before")``. Non-mapping input
    (for example an already constructed model instance) is returned
    unchanged.

    Parameters
    ----------
    data
        Raw validator input.
    renames
        Mapping ``old_name -> new_name``.
    owner
        Class name used in the warning, e.g. ``"TnNetworkConfig"``.

    Returns
    -------
    object
        ``data`` with every deprecated key replaced by its new name.

    Raises
    ------
    ValueError
        If both the deprecated and the new name are given.
    """
    if not isinstance(data, Mapping):
        return data
    found = [old for old in renames if old in data]
    if not found:
        return data
    out = dict(data)
    for old in found:
        new = renames[old]
        if new in out:
            raise ValueError(
                f"{owner}: got both {old!r} (deprecated) and {new!r}; pass only {new!r}."
            )
        warn_renamed(f"{owner}.{old}", f"{owner}.{new}", stacklevel=5)
        out[new] = out.pop(old)
    return out


def take_renamed_kwargs(
    owner: str, extra: Mapping[str, Any], renames: Mapping[str, str]
) -> dict[str, Any]:
    """Translate deprecated keyword arguments of a method.

    Parameters
    ----------
    owner
        Qualified method name used in messages, e.g.
        ``"LvNetworkLayout.to_world"``.
    extra
        The ``**kwargs`` the method received besides its own parameters.
    renames
        Mapping ``old_name -> new_name`` of accepted deprecated keywords.

    Returns
    -------
    dict
        ``{new_name: value}`` for every deprecated keyword given.

    Raises
    ------
    TypeError
        For a keyword that is neither a parameter nor a deprecated alias,
        mirroring Python's own error for an unexpected keyword argument.
    """
    out: dict[str, Any] = {}
    for key, value in extra.items():
        if key not in renames:
            raise TypeError(f"{owner}() got an unexpected keyword argument {key!r}")
        warn_renamed(
            f"{owner}({key}=...)", f"{owner}({renames[key]}=...)", stacklevel=4
        )
        out[renames[key]] = value
    return out


class DeprecatedKeyDict(dict):
    """``dict`` that still answers deprecated keys, with a warning.

    Iteration, ``len`` and equality only see the current keys; reading a
    deprecated key through ``[]`` or :meth:`get` returns the value of its
    replacement and emits a :class:`DeprecationWarning`.

    Parameters
    ----------
    data
        The mapping with the current keys.
    aliases
        Mapping ``old_key -> new_key``.
    owner
        Name used in the warning, e.g. ``"verify_current_balance()"``.
    """

    def __init__(
        self, data: Mapping[str, Any], aliases: Mapping[str, str], owner: str
    ) -> None:
        super().__init__(data)
        self._aliases = dict(aliases)
        self._owner = owner

    def _translate(self, key: Any) -> Any:
        if (
            isinstance(key, str)
            and key in self._aliases
            and not super().__contains__(key)
        ):
            new = self._aliases[key]
            warn_renamed(
                f"{self._owner}[{key!r}]", f"{self._owner}[{new!r}]", stacklevel=4
            )
            return new
        return key

    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(self._translate(key))

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for ``key`` (deprecated keys are translated)."""
        return super().get(self._translate(key), default)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str) and key in self._aliases:
            return super().__contains__(self._aliases[key])
        return super().__contains__(key)
