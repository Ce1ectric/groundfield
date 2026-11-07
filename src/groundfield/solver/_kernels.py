"""Reaction-matrix and field kernels of the image family (private).

Moved out of :mod:`groundfield.solver.image` (review pass 10, 10.75 /
10.79); every name stays importable from there.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from groundfield.solver._guards import _MIN_DISTANCE, _require_image_separation

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

__all__: list[str] = []

# ---------------------------------------------------------------------
# Potential evaluation
# ---------------------------------------------------------------------


def _potential_kernel(
    field_points: np.ndarray,  # (M, 3)
    source_points: np.ndarray,  # (N, 3)
    currents: np.ndarray,  # (N,)
    rho: float,
) -> np.ndarray:
    """Vectorised image-charge sum — pure point-source evaluation.

    Used for **field points away from the sources** (plots, profiles).
    Singularities are clamped at ``_MIN_DISTANCE``.

    Returns
    -------
    phi : np.ndarray, shape (M,)
        Potential at every field point.
    """
    image_points = source_points.copy()
    image_points[:, 2] = -image_points[:, 2]

    diff_real = field_points[:, None, :] - source_points[None, :, :]
    diff_image = field_points[:, None, :] - image_points[None, :, :]
    r_real = np.linalg.norm(diff_real, axis=2)
    r_image = np.linalg.norm(diff_image, axis=2)
    np.maximum(r_real, _MIN_DISTANCE, out=r_real)
    np.maximum(r_image, _MIN_DISTANCE, out=r_image)

    kernel = (1.0 / r_real) + (1.0 / r_image)
    phi = (rho / (4.0 * np.pi)) * (kernel @ currents)
    return phi


def _self_corrected_kernel(
    seg_points: np.ndarray,  # (N, 3)
    seg_lengths: np.ndarray,  # (N,)
    wire_radii: np.ndarray,  # (N,)
    currents: np.ndarray,  # (N,)
    rho: float,
    *,
    owner_names: "Sequence[str] | None" = None,
) -> np.ndarray:
    """Evaluation **at the segment midpoints** with proper self-action.

    The diagonal (segment onto itself) uses the analytical line
    self-potential; off-diagonal entries fall back to the point-source
    approximation 1/r + 1/r_image. The diagonal *image* term is the
    point form $1/(2z)$, which is the $L \\ll 4 z$ limit of the exact
    image-line term $(2/L)\\,\\operatorname{arsinh}(L/(4z))$ — see
    :func:`_check_segment_depths` for the shallow-electrode regime
    where that distinction matters.

    This is the reaction-matrix kernel shared by the whole closed-form
    family: ``image``, ``image_2layer``, ``mom``, ``bem``, ``cim``,
    ``mom_sommerfeld`` and ``mutual`` all route their homogeneous
    (direct + free-surface-image) part through it, which is why both
    depth guards of :func:`_require_image_separation` are enforced
    here rather than in the individual ``solve_*`` entry points.

    Parameters
    ----------
    owner_names
        Optional per-segment owner labels (already formatted, e.g.
        ``"electrode 'g1'"``) used only to make the guard's error
        message name what the user wrote. ``solve_image`` and
        ``solve_image_2layer`` supply them through
        :func:`_check_segment_depths`; the sibling backends call this
        kernel positionally and get the segment index plus coordinates
        instead. Passing them never changes a returned number.

    Returns
    -------
    phi : np.ndarray, shape (N,)
        Potential at the segment midpoints.

    Raises
    ------
    ValueError
        If any segment midpoint is at or above the soil surface
        (``z <= 0``) — the kernel mirrors ``|z|``, so a conductor in
        air would otherwise be solved as if it were buried — or lies
        in the soil-surface plane (``2*|z| <= wire_radius``), where
        source and mirror image coincide and the self-image term
        diverges (:func:`_require_image_separation`).
    """
    n = seg_points.shape[0]
    image_points = seg_points.copy()
    image_points[:, 2] = -image_points[:, 2]

    diff_real = seg_points[:, None, :] - seg_points[None, :, :]
    diff_image = seg_points[:, None, :] - image_points[None, :, :]
    r_real = np.linalg.norm(diff_real, axis=2)
    r_image = np.linalg.norm(diff_image, axis=2)

    # The 1 mm clamp below is meant for *field* evaluations; inside
    # the reaction matrix a clamped off-diagonal pair means two
    # distinct segments (nearly) coincide and the mutual term is
    # silently saturated — surface that instead of hiding it.
    n_clamped = (int((r_real < _MIN_DISTANCE).sum()) - n) // 2  # diag is 0
    if n_clamped > 0:
        warnings.warn(
            f"_self_corrected_kernel: {n_clamped} off-diagonal segment "
            f"pair(s) are closer than the {_MIN_DISTANCE * 1e3:.0f} mm "
            "singularity clamp — distinct electrodes/segments overlap "
            "and their mutual coupling is saturated at the clamp. "
            "Check the geometry for coincident conductors.",
            UserWarning,
            stacklevel=2,
        )

    # Off-diagonal: point source, safely clamped.
    np.maximum(r_real, _MIN_DISTANCE, out=r_real)
    np.maximum(r_image, _MIN_DISTANCE, out=r_image)
    kernel = (1.0 / r_real) + (1.0 / r_image)

    # Replace the diagonal with the line self-potential of the direct
    # contribution.
    # phi_self_direct = rho · I / (2π·L) · ln(L/a)
    #   ⇒ K_ii (direct part) = 2·ln(L/a) / L
    diag_direct = 2.0 * np.log(seg_lengths / wire_radii) / seg_lengths

    # Self-image contribution: point source at (x, y, -z), distance 2·z.
    #
    # Guard (audit 2026-07-28, F26/F29): the point-image form diverges
    # as z -> 0 and used to be *clamped* at ``_MIN_DISTANCE``, which
    # injected an arbitrary 1000 1/m into the diagonal and returned a
    # mesh-dependent, far too large impedance without any diagnostic.
    # A segment whose mirror image touches the conductor is now a hard
    # error, consistent with the inductive path
    # (``build_inductance_matrix``). The ``z <= 0`` branch was added
    # here in 0.15.0 (audit 2026-07-29): it used to sit in
    # ``solve_image`` / ``solve_image_2layer`` only, so mom / bem / cim
    # / mom_sommerfeld / mutual happily mirrored an airborne electrode
    # into the soil (a ring at z = -100 m returned 5.5162 Ohm).
    _require_image_separation(
        seg_points,
        wire_radii,
        where="_self_corrected_kernel",
        owner_names=owner_names,
    )
    z_mid = seg_points[:, 2]
    diag_image = 1.0 / (2.0 * np.abs(z_mid))

    np.fill_diagonal(kernel, diag_direct + diag_image)

    phi = (rho / (4.0 * np.pi)) * (kernel @ currents)
    return phi
