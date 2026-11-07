"""Shared solve flow of the image family (private).

Moved out of :mod:`groundfield.solver.image` (review pass 10, 10.75 /
10.79); every name stays importable from there.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

__all__: list[str] = []


def _weighted_node_potential(
    phi_per_freq: "Sequence[np.ndarray]",
    idxs: "Sequence[int]",
    seg_lengths: np.ndarray,
    n_freq: int,
) -> list[complex]:
    """Node potential of one owner as a length-weighted segment average.

    Physics / consistency
    ---------------------
    A galvanic node (electrode or conductor pseudo-node) is a single
    equipotential in the model. The discrete stand-in for that single
    potential is the **Galerkin (length-weighted) average** of the
    segment-midpoint potentials,

    .. math::
        \\varphi_e \\;=\\; \\frac{\\sum_{k \\in e} L_k\\,\\varphi_k}
                                {\\sum_{k \\in e} L_k},

    which is the pairing dual to the uniform-per-unit-length current
    ansatz $I_k = I_e L_k / \\sum_j L_j$: it is the average potential
    over the electrode's *surface*, not over its segment *list*, and
    it is exactly the reduction the multi-port matrix
    :math:`Z_{ij}` uses inside
    :func:`_solve_cluster_currents` (audit 2026-07-08, WP-B3).

    Reporting the plain ``np.mean`` here — the behaviour up to
    0.14.1 — meant the impedance the user reads back was **not** the
    impedance the solver enforced whenever an electrode carried
    segments of unequal length (mesh electrodes with
    ``dx/nx != dy/ny``, rods split at a soil-layer interface):
    measured -10.9 % on a 4.509 m rod split at ``h_1 = 5 m``
    (weighted 45.686 Ω vs. reported 40.701 Ω), and ideally bonded
    electrodes reported *different* potentials although the solve
    constrains them to one (audit 2026-07-28, F10/F11). For uniform
    per-electrode segment lengths — every plain rod, ring or strip —
    the two averages are identical.

    Parameters
    ----------
    phi_per_freq
        One segment-potential vector per frequency, each of shape
        ``(n_segments,)``.
    idxs
        Segment indices belonging to the owner.
    seg_lengths
        Per-segment length in m, shape ``(n_segments,)``.
    n_freq
        Number of frequencies to reduce.

    Returns
    -------
    list[complex]
        Length-weighted node potential per frequency, in V.
    """
    w = np.asarray(seg_lengths, dtype=float)[idxs]
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        # Defensive only: segment lengths are positive by construction.
        return [complex(np.mean(phi_per_freq[k][idxs])) for k in range(n_freq)]
    return [complex((w @ phi_per_freq[k][idxs]) / w_sum) for k in range(n_freq)]


def _warn_ignored_sources(world, backend: str) -> None:
    """Warn when a backend drops non-current sources.

    Every backend supports impressed current sources only; other
    source kinds are skipped during the injection assembly. Without
    this warning a world driven solely by voltage sources solves to
    all-zero with no diagnostic.
    """
    n_ignored = sum(1 for s in world.sources if s.kind != "current")
    if n_ignored:
        warnings.warn(
            f"{backend}: {n_ignored} non-current source(s) ignored — "
            "only CurrentSource is supported. A world driven solely "
            "by non-current sources solves to all-zero.",
            UserWarning,
            stacklevel=3,
        )


def _reject_concrete_shells(world, backend: str) -> None:
    """Reject worlds carrying concrete-shell corrections (ADR-0012).

    Only the image family consumes the per-segment shell coefficient
    on the reaction-matrix diagonal. Every other backend would
    silently solve the bare-metal problem, which the electrode
    documentation explicitly promises will fail loudly instead.

    Raises
    ------
    NotImplementedError
        If any electrode carries a non-zero
        ``concrete_shell_coefficient_ohm_m``.
    """
    shelled = [
        e.name
        for e in world.electrodes
        if float(getattr(e, "concrete_shell_coefficient_ohm_m", 0.0) or 0.0) != 0.0
    ]
    if shelled:
        raise NotImplementedError(
            f"Backend '{backend}' does not implement the concrete-shell "
            f"correction (ADR-0012) carried by electrode(s) {shelled}. "
            "Use backend='image' / 'image_2layer', or set "
            "concrete_shell_coefficient_ohm_m = 0."
        )
