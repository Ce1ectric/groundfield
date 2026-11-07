"""Inductive branch assembly of the distributed-conductor model (private).

Partial-inductance matrix (ADR-0004) and the per-frequency earth-return
correction closures (Carson series, ADR-0005; Sommerfeld, ADR-0006)
over the finite branches of the nodal-analysis system.

Moved out of :mod:`groundfield.solver.image` (review pass 10, 10.75 /
10.79); every name stays importable from there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver._discretize import _DistributedBranch

__all__: list[str] = []


def _assemble_inductance_matrix(
    distributed_branches: list[_DistributedBranch],
    *,
    n_lumped_branches: int,
    n_total_branches: int,
    earth_model: str = "perfect_mirror",
    sigma_earth: float | None = None,
    layered_earth: object = None,
) -> tuple[np.ndarray | None, bool, "_CarsonBuilder | None"]:
    """Build the partial-inductance matrix over all finite branches.

    The output covers *all* finite branches (lumped + distributed) in
    the same ordering used by the solver's ``finite_branches`` list:
    lumped finite-impedance branches come first (no inductance, all
    zeros), distributed branches come after. Inductive entries are
    only populated when at least one distributed conductor was
    created with ``inductance_model == "neumann"``; the rest of the
    matrix is left zero.

    The earth is treated as a perfect magnetic mirror by default
    (ADR-0004). When ``earth_model == "carson_series"`` (ADR-0005)
    the function additionally returns a closure that, given an
    angular frequency $\\omega$, produces the Carson correction
    matrix $\\Delta Z_\\text{Carson}(\\omega)$ over the same
    branch indices. The solver evaluates that closure once per
    frequency and adds the result to the per-branch impedance block:

    $$
    Z_b(\\omega) \\;=\\; R \\;+\\; j\\omega\\,L
        \\;+\\; \\Delta Z_\\text{Carson}(\\omega).
    $$

    Cross-conductor mutual inductance is included automatically
    because the assembly iterates over every active distributed
    branch.

    Parameters
    ----------
    distributed_branches
        Branches produced by :func:`_build_distributed_topology`. The
        ``inductive`` flag selects whether each branch contributes
        to the matrix.
    n_lumped_branches
        Number of lumped finite-impedance branches that come before
        the distributed ones in ``finite_branches``.
    n_total_branches
        Total number of finite branches in ``finite_branches``.
    earth_model
        - ``"perfect_mirror"`` (default, ADR-0004): system stays
          real, third return value is ``None``.
        - ``"carson_series"`` (ADR-0005): per-meter Carson
          correction × length, third return is a closure
          ``omega -> dZ_carson(omega)``.
        - ``"sommerfeld"`` (ADR-0006): geometric Sommerfeld kernel
          integration over the segment-pair geometry, with
          layered-earth support, third return is a closure
          ``omega -> dZ_sommerfeld(omega)``.
    sigma_earth
        Earth conductivity in S/m. Required when
        ``earth_model == "carson_series"``.
    layered_earth
        :class:`LayeredEarth` configuration for the Sommerfeld
        kernel. Required when ``earth_model == "sommerfeld"``;
        ignored otherwise.

    Returns
    -------
    L : np.ndarray | None
        ``(n_total_branches, n_total_branches)`` partial-inductance
        matrix in henries, or ``None`` when no distributed conductor
        carries an inductive model.
    has_inductance : bool
        Convenience flag matching ``L is not None``.
    carson_builder : _CarsonBuilder | None
        Closure that evaluates the Carson correction matrix at a
        given $\\omega$. ``None`` when ``earth_model ==
        "perfect_mirror"`` or no inductive branches are present.
    """
    from groundfield.coupling.inductance import (
        build_carson_correction_matrix,
        build_inductance_matrix,
    )

    inductive_branches = [db for db in distributed_branches if db.inductive]
    if not inductive_branches:
        return None, False, None

    # Map each inductive branch back to its offset inside
    # ``finite_branches`` — the lumped block sits at indices
    # [0, n_lumped_branches), so the distributed-conductor branches
    # start at ``n_lumped_branches`` in the order they were appended.
    inductive_offsets: list[int] = []
    inductive_endpoints: list[np.ndarray] = []
    inductive_radii: list[float] = []
    for offset, db in enumerate(distributed_branches):
        if not db.inductive:
            continue
        inductive_offsets.append(n_lumped_branches + offset)
        inductive_endpoints.append(np.stack([db.p_a, db.p_b], axis=0))
        inductive_radii.append(db.wire_radius)

    if not inductive_offsets:
        return None, False, None

    seg_endpoints = np.stack(inductive_endpoints, axis=0)
    radii = np.array(inductive_radii)
    L_sub = build_inductance_matrix(seg_endpoints, radii, use_image=True)

    L_full = np.zeros((n_total_branches, n_total_branches))
    idx = np.array(inductive_offsets, dtype=int)
    L_full[np.ix_(idx, idx)] = L_sub

    carson_builder: "_CarsonBuilder | None" = None
    if earth_model == "carson_series":
        if sigma_earth is None or sigma_earth <= 0.0:
            raise ValueError(
                "earth_model='carson_series' requires sigma_earth > 0; "
                f"got sigma_earth={sigma_earth!r}"
            )
        seg_endpoints_snapshot = seg_endpoints.copy()
        radii_snapshot = radii.copy()
        idx_snapshot = idx.copy()
        sigma_snapshot = float(sigma_earth)
        n_total_snapshot = n_total_branches

        def _carson_at(omega: float) -> np.ndarray:
            dZ_full = np.zeros(
                (n_total_snapshot, n_total_snapshot),
                dtype=complex,
            )
            if omega <= 0.0:
                return dZ_full
            dZ_sub = build_carson_correction_matrix(
                seg_endpoints_snapshot,
                radii_snapshot,
                omega=omega,
                sigma_earth=sigma_snapshot,
            )
            dZ_full[np.ix_(idx_snapshot, idx_snapshot)] = dZ_sub
            return dZ_full

        carson_builder = _carson_at
    elif earth_model == "sommerfeld":
        if layered_earth is None:
            raise ValueError(
                "earth_model='sommerfeld' requires layered_earth "
                "(LayeredEarth instance); got None."
            )
        from groundfield.coupling.sommerfeld_inductance import (
            build_sommerfeld_correction_matrix,
        )

        seg_endpoints_snapshot = seg_endpoints.copy()
        radii_snapshot = radii.copy()
        idx_snapshot = idx.copy()
        earth_snapshot = layered_earth
        n_total_snapshot = n_total_branches

        def _sommerfeld_at(omega: float) -> np.ndarray:
            dZ_full = np.zeros(
                (n_total_snapshot, n_total_snapshot),
                dtype=complex,
            )
            if omega <= 0.0:
                return dZ_full
            dZ_sub = build_sommerfeld_correction_matrix(
                seg_endpoints_snapshot,
                radii_snapshot,
                omega=omega,
                earth=earth_snapshot,
            )
            dZ_full[np.ix_(idx_snapshot, idx_snapshot)] = dZ_sub
            return dZ_full

        carson_builder = _sommerfeld_at

    return L_full, True, carson_builder


# Type alias used by callers (no runtime cost; just documentation).
_CarsonBuilder = "callable[[float], np.ndarray]"
