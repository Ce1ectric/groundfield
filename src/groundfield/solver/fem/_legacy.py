"""Names of the equivalent-hemisphere reduction, kept for compatibility (private).

Up to 2026.11 ``fem`` replaced every cluster by an equivalent hemisphere
(:func:`equivalent_hemisphere_radius`) and warned with
:class:`FemReductionWarning` where the reduction did not represent the
world. The conforming model replaced it (user decision Q37); both names
stay importable from :mod:`groundfield.solver.fem`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from groundfield.geometry.electrodes import (
    GridMeshElectrode,
    MeshElectrode,
    RingElectrode,
    RodElectrode,
    StripElectrode,
)
from groundfield.references import dwight1936 as dw

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.geometry.electrodes import _ElectrodeBase

__all__ = ["FemReductionWarning", "equivalent_hemisphere_radius"]


class FemReductionWarning(UserWarning):
    """The equivalent-hemisphere reduction did not represent the world.

    Emitted by ``fem`` up to 2026.11, when it reduced every cluster to an
    equivalent hemisphere. Since the conforming model (user decision Q37)
    ``fem`` solves the electrodes as they are and never emits it; the class
    is kept so that filters and ``pytest.warns`` written against it still
    import.
    """


def equivalent_hemisphere_radius(electrode: "_ElectrodeBase", rho_top: float) -> float:
    """Radius of the hemisphere with an electrode's closed-form resistance.

    .. deprecated:: 2026.11
        ``fem`` no longer reduces electrodes to a hemisphere (user decision
        Q37); the function is kept for scripts that use the closed forms.
        Compute $a_\\text{eq} = \\rho/(2\\pi R)$ from the formulas of
        :mod:`groundfield.references.dwight1936` instead. Removal no earlier
        than 2027.5.

    The hemisphere radius $a_\\text{eq} = \\rho/(2\\pi R)$ with $R$ the
    closed form of Dwight (1936) for a rod, a ring and a strip, Sverak's
    formula for a ``GridMeshElectrode`` and one strip along the diagonal for
    a ``MeshElectrode`` (the formulas of 0.15.0, unchanged, valid only for
    shallow electrodes: a strip down to half its length, a ring down to
    its diameter).

    Parameters
    ----------
    electrode
        Single electrode primitive.
    rho_top
        Resistivity in Ω·m used inside the formula.

    Returns
    -------
    float
        Hemisphere radius $a_\\text{eq}$ in m.

    Warns
    -----
    DeprecationWarning
        On every call.
    """
    from groundfield.solver.registry import DEPRECATION_REMOVAL, warn_external

    warn_external(
        "groundfield.solver.fem.equivalent_hemisphere_radius is deprecated: fem "
        "no longer reduces electrodes to an equivalent hemisphere since 2026.11 "
        "(user decision Q37; it solves rods and rings as they are). Compute "
        "a_eq = rho / (2 pi R) from groundfield.references.dwight1936 instead. "
        f"It will be removed no earlier than {DEPRECATION_REMOVAL}.",
        DeprecationWarning,
    )
    if isinstance(electrode, RodElectrode):
        R = dw.rod(rho=rho_top, length=electrode.length, radius=electrode.wire_radius)
    elif isinstance(electrode, RingElectrode):
        R = dw.buried_ring(
            rho=rho_top,
            ring_diameter=2.0 * electrode.radius,
            wire_diameter=2.0 * electrode.wire_radius,
            depth=max(electrode.center[2], 1e-3),
        )
    elif isinstance(electrode, StripElectrode):
        # Straight horizontal wire — Dwight's classic ``horizontal_wire``
        # formula expects the *half-length* L (total = 2 L).
        L = electrode.length
        depth = max(electrode.start[2], 1e-3)
        R = dw.horizontal_wire(
            rho=rho_top,
            length=L / 2.0,
            radius=electrode.wire_radius,
            depth=depth,
        )
    elif isinstance(electrode, GridMeshElectrode):
        # Schwarz / Sverak / IEEE Std 80 (Sverak 1981) formula for a
        # buried rectangular meshed grid:
        #
        #     R ≈ ρ / L_C + ρ / sqrt(20 A) · (1 + 1 / (1 + h sqrt(20/A)))
        #
        # with L_C the total buried wire length, A the grid footprint
        # area and h the burial depth. Captures the dependence on the
        # inner mesh density, which the simple strip-along-diagonal
        # approximation used for the legacy ``MeshElectrode`` misses.
        dx, dy = electrode.size
        depth = max(electrode.corner[2], 1e-3)
        A = dx * dy
        n_long = electrode.n_y + 1  # longitudinal wires (one per y-row)
        n_tran = electrode.n_x + 1  # transverse wires (one per x-column)
        L_C = n_long * dx + n_tran * dy
        R = rho_top / L_C + (rho_top / math.sqrt(20.0 * A)) * (
            1.0 + 1.0 / (1.0 + depth * math.sqrt(20.0 / A))
        )
    elif isinstance(electrode, MeshElectrode):
        # Use the strip approximation along the diagonal as a rough
        # proxy for a mesh ground electrode. The FEM result for a
        # mesh is dominated by its overall extent, not the inner
        # spacing, so this is acceptable for the cross-check role.
        dx, dy = electrode.size
        diag = float(np.hypot(dx, dy))
        # horizontal_strip expects half-length L (total length 2*L),
        # a strip cross-section width and thickness. We model the
        # mesh as one equivalent strip of width ≈ wire_radius and
        # thickness ≈ wire_radius / 9 (so the b<a/8 guard passes).
        a = max(2.0 * electrode.wire_radius, 0.01)
        b = a / 9.0
        R = dw.horizontal_strip(
            rho=rho_top,
            length=diag / 2.0,
            width=a,
            thickness=b,
            depth=max(electrode.corner[2], 1e-3),
        )
    else:
        raise TypeError(
            f"FEM backend cannot reduce {type(electrode).__name__} to a "
            "hemisphere — extend equivalent_hemisphere_radius()."
        )
    return float(rho_top / (2.0 * np.pi * R))
