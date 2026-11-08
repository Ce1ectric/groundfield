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
  readable;
* :class:`PickleUpgrade` and :func:`upgrade_pickled_state` — bring the
  pickled state of a model written by an earlier release up to date.

The module is private; nothing here is part of the public API.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, ClassVar

__all__ = [
    "REMOVAL_RELEASE",
    "DeprecatedKeyDict",
    "PickleUpgrade",
    "rename_input_keys",
    "take_renamed_kwargs",
    "upgrade_pickled_state",
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


def upgrade_pickled_state(model: type, state: Any) -> Any:
    """Bring the pickled state of a Pydantic model up to date.

    Pydantic restores a pickled model from its ``__dict__`` without
    running any validator, so the ``model_validator(mode="before")`` hooks
    that map renamed fields of an input mapping (:func:`rename_input_keys`)
    never see a pickle (audit A2G-07: a v0.15.0-pickled ``TnNetworkConfig``
    unpickled and then failed in ``build()`` with ``AttributeError``).
    This function applies the model's rename tables to the state instead:

    * keys listed in ``model._field_renames`` (``old -> new``) move to the
      new field name, also in the set of explicitly set fields (a value
      stored under the new name wins);
    * values listed in ``model._value_renames`` (``field -> {old: new}``)
      are replaced (e.g. ``kind="star_kvs"``);
    * fields missing from the state (added by a later release) get their
      defaults, and keys that are no fields any more are dropped.

    Parameters
    ----------
    model
        The Pydantic model class being unpickled.
    state
        The state handed to ``__setstate__``.

    Returns
    -------
    object
        The upgraded state, or ``state`` itself when it is already current.
    """
    if not isinstance(state, dict) or not isinstance(state.get("__dict__"), dict):
        return state
    values: dict = state["__dict__"]
    fields = model.model_fields
    renames: Mapping[str, str] = getattr(model, "_field_renames", {})
    value_renames: Mapping[str, Mapping[Any, Any]] = getattr(
        model, "_value_renames", {}
    )
    if values.keys() == fields.keys() and not any(
        _is_old_value(values[name], old) for name, old in value_renames.items()
    ):
        return state
    upgraded: dict[str, Any] = {}
    for key, value in values.items():
        name = renames.get(key, key)
        if name not in fields or (name != key and name in values):
            continue
        upgraded[name] = value
    for name, info in fields.items():
        if name not in upgraded and not info.is_required():
            upgraded[name] = info.get_default(call_default_factory=True)
    for name, old in value_renames.items():
        if name in upgraded and _is_old_value(upgraded[name], old):
            upgraded[name] = old[upgraded[name]]
    fields_set = {
        renames.get(key, key) for key in state.get("__pydantic_fields_set__") or ()
    } & fields.keys()
    ordered = {name: upgraded[name] for name in fields if name in upgraded}
    return {**state, "__dict__": ordered, "__pydantic_fields_set__": fields_set}


def _is_old_value(value: Any, old: Mapping[Any, Any]) -> bool:
    try:
        return value in old
    except TypeError:  # unhashable value: never a renamed one
        return False


class PickleUpgrade:
    """Mixin for Pydantic models: unpickle states of earlier releases.

    Put it before :class:`pydantic.BaseModel` in the bases and list the
    renames of the class in class variables; the first two feed the
    class's ``model_validator(mode="before")`` for input mappings as well:

    * ``_field_renames`` — ``{old_field: new_field}``;
    * ``_value_renames`` — ``{field: {old_value: new_value}}``;
    * ``_unpickle_as`` — on a deprecated alias class (e.g. ``KvsConfig``),
      the current class its pickled instances become (an alias instance
      would otherwise differ from the same configuration loaded from JSON
      and keep the alias name in every new pickle).

    ``__setstate__`` passes the state through
    :func:`upgrade_pickled_state` first; a current state is used as is.
    """

    _field_renames: ClassVar[Mapping[str, str]] = {}
    _value_renames: ClassVar[Mapping[str, Mapping[Any, Any]]] = {}

    def __setstate__(self, state: Any) -> None:
        current = type(self).__dict__.get("_unpickle_as")
        if current is not None:
            # Same fields and slots as the alias subclass it replaces.
            object.__setattr__(self, "__class__", current)
        super().__setstate__(upgrade_pickled_state(type(self), state))  # type: ignore[misc]
