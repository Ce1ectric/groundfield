"""Which impedance a post-processing helper reads from a result (private).

Review pass 10 (10.18, 10.19): ``sweep``, ``convergence_study`` and the
rho-f exporters take ``quantity=`` with three meanings,

``"input"`` (default)
    Driving-point impedance of a current source,
    :math:`Z_\\text{in} = (\\varphi_a - \\varphi_r) / I_s`, or of a
    voltage source, :math:`U / I_s` with the solved source current
    (:meth:`FieldResult.input_impedance`) — what a measurement
    instrument or a fault sees at the feeding point, and the
    :math:`T(f) = \\mathrm{EPR}/I_F` of a fault study.
``"cluster"``
    Cluster potential over the cluster's own soil leakage,
    :math:`Z_c = \\varphi_c / \\sum_{e \\in c} I_e`
    (:meth:`FieldResult.cluster_impedance`) — the local grounding
    impedance of one galvanic cluster; the ``sweep`` default up to
    v0.15.0.
``"electrode"``
    Potential of one electrode over that electrode's own leakage,
    :math:`Z_e = \\varphi_e / I_e` (:meth:`FieldResult.grounding_impedance`)
    — the rho-f exporters' quantity up to v0.15.0.

With one source returning through remote earth the input and the
cluster impedance are linked by the split factor,
:math:`Z_\\text{in} = s\\,Z_c` with :math:`s = \\sum_{e \\in c} I_e / I_s`
(:func:`groundfield.postprocess.current_balance.split_factor`); they
agree when no conductor carries current out of the feeding cluster.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.result import FieldResult

__all__: list[str] = []

Quantity = Literal["input", "cluster", "electrode"]
QUANTITIES: tuple[str, ...] = ("input", "cluster", "electrode")


def check_quantity(quantity: str) -> None:
    """Raise ``ValueError`` for an unknown ``quantity``."""
    if quantity not in QUANTITIES:
        raise ValueError(
            f"quantity must be one of {', '.join(map(repr, QUANTITIES))}, "
            f"got {quantity!r}."
        )


def _members(result: "FieldResult", name: str) -> list[str]:
    return list(result.clusters.get(name, [name]) or [name])


def select_current_source(
    result: "FieldResult",
    *,
    source: str | None,
    electrode_name: str | None,
    caller: str,
    sources: list[Any] | None = None,
) -> Any | None:
    """Pick the source an ``"input"`` / default helper refers to.

    Current and voltage sources are candidates (a voltage source since
    2026.11; its record carries the solved source current).

    Parameters
    ----------
    result
        Solved result; its source snapshot is used.
    source
        Explicit source name, or ``None``.
    electrode_name
        Optional electrode; with several current sources it selects the
        one whose feeding cluster contains this electrode.
    caller
        Name of the public helper, for error messages.
    sources
        Source records to choose from; defaults to ``result.sources``.

    Returns
    -------
    object or None
        The source record (``name``, ``kind``, ``attached_to``,
        ``return_to``, ...), or ``None`` when there is no current source
        at all.

    Raises
    ------
    KeyError
        If ``source`` names no current source.
    ValueError
        If several current sources match and ``source`` is omitted.
    """
    pool = list(result.sources if sources is None else sources)
    current = [s for s in pool if s.kind in ("current", "voltage")]
    if source is not None:
        for s in current:
            if s.name == source:
                return s
        known = ", ".join(s.name for s in current) or "none"
        raise KeyError(f"{caller}: no current source {source!r} (known: {known}).")
    if not current:
        return None
    if len(current) == 1:
        return current[0]
    if electrode_name is not None:
        feeding = [
            s for s in current if electrode_name in _members(result, s.attached_to)
        ]
        if len(feeding) == 1:
            return feeding[0]
    names = ", ".join(s.name for s in current)
    raise ValueError(
        f"{caller}: the result carries several current sources ({names}); "
        "pass source=<name>."
    )


def impedance_series(
    result: "FieldResult",
    quantity: str,
    *,
    electrode_name: str | None = None,
    source: str | None = None,
    caller: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Impedance and its denominator current per frequency.

    Parameters
    ----------
    result
        Solved :class:`FieldResult`.
    quantity
        ``"input"``, ``"cluster"`` or ``"electrode"`` (module docstring).
    electrode_name
        Required for ``"cluster"`` and ``"electrode"``. With ``"input"``
        it is optional and must belong to the feeding cluster of the
        selected source (it then also selects among several sources).
    source
        Source name for ``"input"``; may be omitted with one current
        source.
    caller
        Name of the public helper, for error messages.

    Returns
    -------
    Z : numpy.ndarray, complex, shape (n_f,)
        Impedance in Ω; ``nan`` where the denominator current is zero.
    I : numpy.ndarray, complex, shape (n_f,)
        Denominator current in A: the impressed source current
        (``"input"``), the cluster leakage (``"cluster"``) or the
        electrode leakage (``"electrode"``).

    Raises
    ------
    ValueError
        Unknown ``quantity``; ``"input"`` without source snapshot, without
        current source, or with an ``electrode_name`` that the source does
        not feed; ``"cluster"`` / ``"electrode"`` without
        ``electrode_name``.
    KeyError
        Unknown electrode or source.
    """
    check_quantity(quantity)
    n_f = len(result.frequencies)
    if quantity == "input":
        if not result.sources:
            raise ValueError(
                f"{caller}: quantity='input' needs the source snapshot of the "
                "solve (FieldResult.sources), which this result does not carry "
                "(hand-built result?). Pass quantity='electrode' (U/I of the "
                "named electrode, the v0.15.0 behaviour) or quantity='cluster'."
            )
        record = select_current_source(
            result, source=source, electrode_name=electrode_name, caller=caller
        )
        if record is None:
            raise ValueError(f"{caller}: the result carries no current source.")
        if electrode_name is not None and electrode_name not in _members(
            result, record.attached_to
        ):
            raise ValueError(
                f"{caller}: electrode {electrode_name!r} is not fed by source "
                f"{record.name!r} (attached to {record.attached_to!r}); the "
                "input impedance is defined at the feeding point. Pass "
                f"quantity='electrode' for U/I of {electrode_name!r} or "
                "quantity='cluster' for its cluster impedance."
            )
        z = np.asarray(result.input_impedance(record.name), dtype=complex)
        i = np.asarray(record.current, dtype=complex)
        if i.shape != (n_f,):
            raise ValueError(f"{caller}: source current does not match frequencies.")
        return z, i
    if electrode_name is None:
        raise ValueError(f"{caller}: quantity={quantity!r} needs electrode_name.")
    if electrode_name not in result.electrode_potentials:
        raise KeyError(
            f"electrode '{electrode_name}' not in result; "
            f"available: {list(result.electrode_potentials)}"
        )
    if electrode_name not in result.electrode_currents:
        raise KeyError(
            f"electrode '{electrode_name}' not in FieldResult currents; "
            f"available: {list(result.electrode_currents)}"
        )
    if quantity == "cluster":
        members = [
            m
            for m in _members(result, electrode_name)
            if m in result.electrode_currents
        ]
        i = np.sum(
            [np.asarray(result.electrode_currents[m], dtype=complex) for m in members],
            axis=0,
        )
        z = np.asarray(result.cluster_impedance(electrode_name), dtype=complex)
    else:
        u = np.asarray(result.electrode_potentials[electrode_name], dtype=complex)
        i = np.asarray(result.electrode_currents[electrode_name], dtype=complex)
        if u.shape != (n_f,) or i.shape != (n_f,):
            raise ValueError(
                f"{caller}: inconsistent shapes for electrode '{electrode_name}': "
                f"frequencies=({n_f},), potentials={u.shape}, currents={i.shape}."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(i != 0, u / np.where(i != 0, i, 1.0), complex("nan"))
    return z, i
