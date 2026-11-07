"""Validity guards of the image-family discretisation (private).

Hard errors and advisories that protect the closed-form half-space
kernels from geometries they cannot represent:

* :func:`_require_thin_wire` — segment length at or below the wire
  radius (the thin-wire self-term is invalid there);
* :func:`_require_image_separation` — leakage segments at or above the
  soil surface, or so shallow that the mirror image touches the wire;
* :func:`_check_segment_depths` — the same checks on the full segment
  extent plus the :class:`ShallowSegmentWarning` advisory.

Moved out of :mod:`groundfield.solver.image` (review pass 10, 10.75 /
10.79); every name stays importable from there.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from groundfield.solver._discretize import _Segment

__all__ = [
    "ShallowSegmentWarning",
]

# Numerical cutoff: no field point may be closer to a source than
# ``_MIN_DISTANCE`` (in metres). Distances below the cutoff are clamped
# to it to suppress the 1/r singularity during visual evaluations.
_MIN_DISTANCE = 1e-3


class ShallowSegmentWarning(UserWarning):
    """A leakage segment is shallow compared with its own length.

    The diagonal self-image entry of the reaction matrix uses the
    *point* image at distance $2z$ instead of the image *line* of the
    segment, i.e. the $L \\ll 4 z$ limit

    .. math::
        \\frac{1}{2 z} \\;=\\; \\lim_{L \\to 0}
        \\frac{2}{L}\\,\\operatorname{arsinh}\\!\\frac{L}{4 z}.

    For $L \\gtrsim 4 z$ the point form overestimates that entry by
    more than 13 % (and unboundedly as $z \\to 0$), so the reported
    grounding impedance is biased high and keeps moving under mesh
    refinement. Typical triggers: a counterpoise or PEN conductor
    discretised with segments much longer than its burial depth,
    surface-near tapes, and rings or meshes whose arc/wire segments
    are longer than four times the burial depth (a ring of radius
    25 m at $z = 0.8$ m with ``segment_length = 5 m`` gives 4.909 m
    arc segments and warns "by 26 %") — the warning is *not*
    restricted to distributed conductors.

    Scope of the quantified bias
    ----------------------------
    The percentage quoted in the message is the **horizontal-segment**
    figure $\\bigl(L/(4z)\\bigr)/\\operatorname{arsinh}
    \\bigl(L/(4z)\\bigr) - 1$. That is the applicable figure for every
    segment that can still reach the warning: a segment inclined by
    more than $30°$ to the horizontal and satisfying $L > 4 z$ would
    have its upper end above $z = 0$, which
    :func:`_check_segment_depths` rejects with a ``ValueError`` first
    (proof: $L > 4 z$ together with $L\\,|e_z| \\le 2 z$ forces
    $|e_z| < 1/2$). The message states the inclination it assumed so
    the reader can check that reasoning.

    Emission scope
    --------------
    The warning is raised by :func:`solve_image` and
    :func:`~groundfield.solver.image_2layer.solve_image_2layer` only,
    even though ``mom``, ``bem``, ``cim``, ``mom_sommerfeld`` and
    ``mutual`` share the biased diagonal through
    :func:`_self_corrected_kernel`. See the "Emission scope" note in
    :func:`_check_segment_depths` for why the *hard* guard is shared
    but the advisory is not.

    Silence with
    ``warnings.simplefilter("ignore", ShallowSegmentWarning)`` once
    the bias has been accepted, or refine
    ``Engine.segment_length`` / ``Conductor.discretize_segment_length``
    until the impedance stabilises.
    """


def _require_thin_wire(seg_len: float, wire_radius: float, electrode_name: str) -> None:
    """Reject segment lengths at or below the wire radius.

    The analytical line self-term on the reaction-matrix diagonal is
    $2 \\ln(L/a) / L$; for $L \\le a$ the logarithm is non-positive,
    the diagonal loses dominance and the solve silently returns
    nonsense currents. This is a hard error because no downstream
    result can be salvaged. The softer thin-wire quality criterion
    ($L \\ge 5a$, ``MIN_THINWIRE_RATIO``) remains an advisory
    diagnostic in :func:`groundfield.diagnostics.check_segment_resolution`.

    Raises
    ------
    ValueError
        If ``seg_len <= wire_radius``.
    """
    if seg_len <= wire_radius:
        raise ValueError(
            f"Electrode '{electrode_name}': discretised segment length "
            f"{seg_len:.4g} m is <= wire_radius {wire_radius:.4g} m. "
            "The thin-wire self-term ln(L/a) is invalid there. Use a "
            "larger segment_length, a smaller wire radius, or a larger "
            "electrode."
        )


# Absolute tolerance (m) below which a depth is treated as "exactly at
# the soil surface" rather than above or below it. Only used to keep
# round-off in the discretisers from turning a legitimate surface-flush
# endpoint (a driven rod at ``position=(x, y, 0)``, the Dwight case)
# into a spurious "conductor in air" rejection.
_SURFACE_TOL = 1e-9


def _require_image_separation(
    seg_points: np.ndarray,
    wire_radii: np.ndarray,
    *,
    where: str,
    owner_names: "Sequence[str] | None" = None,
) -> None:
    """Reject leakage segments without a usable mirror-image separation.

    Two hard errors are raised here, in this order, for the *segment
    midpoints* handed to the half-space kernel:

    1. **$z \\le 0$** — the midpoint is at or above the soil surface,
       i.e. the point source sits in the air. The kernel only ever
       sees $|z|$, so such a segment used to be solved silently as if
       it were buried at $+|z|$ (a ring at $z = -100$ m returned
       5.5162 Ω instead of failing).
    2. **$2|z| \\le a$** — the mirror image touches the conductor.

    This function lives in the reaction-matrix path shared by
    ``image``, ``image_2layer``, ``mom``, ``bem``, ``cim``,
    ``mom_sommerfeld`` and ``mutual`` (via
    :func:`_self_corrected_kernel` and
    :func:`~groundfield.solver.image_2layer._build_phi_hom_per_source_rho`),
    so **all** of them reject both regimes. Before 0.15.0 only
    condition 2 was checked here while condition 1 was checked in
    ``solve_image`` / ``solve_image_2layer`` alone, which is why
    ``compare_engines`` on an airborne electrode raised for the image
    family and returned a number for ``mom`` / ``bem`` / ``cim`` /
    ``mom_sommerfeld`` (audit 2026-07-29).

    Only the *midpoint* can be checked here: the kernel receives no
    segment directions, so a segment that straddles $z = 0$ with a
    midpoint at $z > 0$ is caught one level up, in
    :func:`_check_segment_depths` (``image`` / ``image_2layer`` only).

    Physics
    -------
    The half-space Green's function of the galvanic problem is built
    from the source and its perfect mirror image across the insulating
    soil surface,
    $G = \\frac{\\rho}{4\\pi}\\bigl(1/r + 1/r'\\bigr)$ with
    $r' \\to 2 z$ on the diagonal (segment onto its own image). The
    construction presupposes a **finite image separation**
    $2 z \\gg a$: as $z \\to 0$ the image line merges with the
    conductor and the diagonal self-image term diverges — the
    point-image form $1/(2z)$ like $1/z$, the exact image-line form
    $(2/L)\\,\\operatorname{arsinh}\\!\\bigl(L/(4z)\\bigr)$ like
    $\\ln\\bigl(L/(2z)\\bigr)$. A conductor lying *in* the surface
    plane therefore has no image separation at all and the reaction
    matrix has no finite limit at fixed wire radius; the correct
    model of a surface-laid conductor is a *coincident* source and
    image, i.e. twice the free-space line self-potential, which is a
    different kernel and not what this backend assembles.

    Until 0.14.1 the diagonal image term was silently clamped at
    ``_MIN_DISTANCE = 1 mm``, so a ring or strip at $z = 0$ returned
    a grounding resistance that was mesh-dependent and up to ~15x too
    high with no diagnostic (audit 2026-07-28, F26/F29). This guard
    mirrors the treatment the inductive path already received in
    :func:`groundfield.coupling.inductance.build_inductance_matrix`
    (``2*|z_mid| <= wire_radius`` raises), so both couplings now fail
    loudly in the same geometric situation.

    Parameters
    ----------
    seg_points
        Segment midpoints, shape ``(N, 3)``; column 2 is the depth
        $z$ (positive downwards).
    wire_radii
        Per-segment wire radius $a$ in m, shape ``(N,)``.
    where
        Caller name, used verbatim in the error message.
    owner_names
        Optional per-segment owner labels (already formatted, e.g.
        ``"electrode 'g1'"`` / ``"conductor 'pen'"``) so the message
        can name what the *user* wrote. When omitted — every caller
        outside this module — the sentence subject degrades to
        "the model"; the segment index and its coordinates are
        reported either way.

    Raises
    ------
    ValueError
        If any segment midpoint satisfies ``z <= 0`` or
        ``2*|z| <= wire_radius``.
    """
    pts = np.asarray(seg_points, dtype=float)
    z = pts[:, 2]
    a = np.asarray(wire_radii, dtype=float)

    def _who(k: int) -> str:
        """Grammatical subject naming the offending segment's owner."""
        return str(owner_names[k]) if owner_names is not None else "the model"

    def _where_at(k: int) -> str:
        return (
            f"segment {k} at (x, y, z) = ({pts[k, 0]:.4g}, "
            f"{pts[k, 1]:.4g}, {pts[k, 2]:.4g}) m"
        )

    airborne = z <= 0.0
    if airborne.any():
        k = int(np.argmin(z))
        raise ValueError(
            f"{where}: {_who(k)} has a {_where_at(k)}, i.e. at or "
            "above the soil surface. The galvanic image kernel is "
            "defined for buried conductors (z > 0) only: it mirrors "
            "|z|, so a conductor in the air would be solved as if it "
            "were buried at the same depth, and a conductor exactly in "
            "the soil-surface plane coincides with its own mirror "
            "image, where the self-image term of the reaction matrix "
            "diverges. z is a depth, positive downwards — use z > 0, "
            "e.g. the customary 0.5-0.8 m burial depth."
        )

    degenerate = 2.0 * np.abs(z) <= a
    if not degenerate.any():
        return
    k = int(np.argmax(degenerate))
    raise ValueError(
        f"{where}: {_who(k)} has a {_where_at(k)}, i.e. in the "
        f"soil-surface plane (|z| = {abs(z[k]):.4g} m <= "
        f"wire_radius/2 = {0.5 * a[k]:.4g} m). A conductor at z = 0 "
        "coincides with "
        "its own mirror image, so the half-space Green's function "
        "1/r + 1/r' has no image separation and the self-image term "
        "of the reaction matrix diverges (it was silently clamped at "
        f"{_MIN_DISTANCE * 1e3:.0f} mm before 0.15.0, which returned "
        "a mesh-dependent, far too large grounding impedance). Bury "
        "the electrode (e.g. z = 0.7 m); note that z is a depth, "
        "positive downwards."
    )


# Largest segment-length-to-depth ratio L/z for which the *point*
# image at distance 2 z is still an acceptable stand-in for the image
# *line* of the segment: already at ``L = 4 z`` the point form
# overestimates the horizontal self-image kernel entry
# ``(2/L)·arsinh(L/(4z))`` by 13 %, and the error grows without bound
# as z -> 0 (audit 2026-07-28, F26/F29).
_MIN_IMAGE_SEPARATION_RATIO = 4.0


def _segment_owner_label(seg: "_Segment") -> str:
    """User-facing name of a segment's owner.

    An electrode segment is owned by the electrode the user named. A
    galvanically coupled distributed-conductor segment carries the
    reserved pseudo-node name ``__cond_<name>__seg_<k>`` in
    ``electrode_name``, which is an internal identifier the user never
    wrote — report the *conductor* name instead (audit 2026-07-29:
    the ``z <= 0`` error used to read ``electrode
    '__cond_pen__seg_0'`` for a conductor named ``pen``).
    """
    if seg.conductor_name is not None:
        return f"conductor '{seg.conductor_name}'"
    return f"electrode '{seg.electrode_name}'"


def _segment_z_extent(seg: "_Segment") -> tuple[float, float]:
    """Vertical extent ``(z_top, z_bottom)`` of a segment.

    Uses the unit tangent ``_Segment.direction`` set by the
    discretisers: the segment spans
    $z_\\text{mid} \\pm \\tfrac{1}{2} L\\,|e_z|$. For a segment whose
    direction was not recorded the extent collapses to the midpoint,
    i.e. the caller silently degrades to the historic midpoint-only
    check rather than guessing an orientation.
    """
    z_mid = float(seg.midpoint[2])
    if seg.direction is None:
        return z_mid, z_mid
    half = 0.5 * float(seg.length) * abs(float(seg.direction[2]))
    return z_mid - half, z_mid + half


def _check_segment_depths(
    all_segments: "list[_Segment]", *, where: str, stacklevel: int = 3
) -> None:
    """Validate the burial depth and extent of every leakage segment.

    Four regimes are distinguished (audit 2026-07-28, F26/F29;
    extended 2026-07-29):

    1. $z_\\text{mid} \\le 0$ — the segment *midpoint* is at or above
       the soil surface. The galvanic image kernel mirrors $|z|$, so a
       conductor in air was silently solved as if it were buried at
       $+|z|$ (a ring at $z = -100$ m returned 5.5162 Ω instead of
       failing). Hard error, raised by
       :func:`_require_image_separation` so that ``mom`` / ``bem`` /
       ``cim`` / ``mom_sommerfeld`` / ``mutual`` see it too.
    2. $2 |z_\\text{mid}| \\le a$ — the mirror image touches the
       conductor; also :func:`_require_image_separation`. Hard error.
    3. **Segment extent above the surface**: $z_\\text{top} < 0$ with
       $z_\\text{top} = z_\\text{mid} - \\tfrac{1}{2} L |e_z|$, i.e.
       part of the conductor is in the air even though its midpoint is
       buried. F26 asked for "every electrode segment has $z > 0$";
       checking midpoints only let
       ``RodElectrode(position=(0, 0, -0.5), length=1.2)`` solve to
       106.2444 Ω with nothing but a ``ShallowSegmentWarning``
       (audit 2026-07-29). Hard error.

       A segment whose upper end sits *exactly* in the surface plane
       ($z_\\text{top} = 0$) is **legal** and deliberately not
       rejected: that is the ordinary driven rod
       (``position=(x, y, 0)``), whose source-plus-image is the
       length-$2L$ line Dwight's closed form is derived from. The
       comparison therefore uses a small tolerance,
       ``_SURFACE_TOL``.

       Only two segment families can reach this regime at all:
       ``RodElectrode`` (strictly vertical) and a sloped galvanic
       ``Conductor``. Strips, polylines, stars, rings and meshes are
       validated as horizontal by their pydantic models, so their
       extent equals their midpoint depth.
    4. $L > 4 z_\\text{mid}$ — legal but shallow: the point-image
       stand-in for the image line carries a bias of
       $\\gtrsim 13\\,\\%$ on the diagonal image term, which is why
       refining the mesh still changes the answer noticeably in this
       regime. :class:`ShallowSegmentWarning`, quantifying the bias of
       the worst offender.

       Regimes 3 and 4 interlock: a segment inclined by more than
       $30°$ to the horizontal cannot reach regime 4, because
       $L > 4 z$ and $L |e_z| \\le 2 z$ together force
       $|e_z| < 1/2$. The quoted bias is the horizontal-segment
       figure, which is therefore the right formula for every segment
       that survives to the warning; the message names the assumed
       inclination. (Up to 0.15.0 the message quoted that same
       horizontal figure for *vertical* worst offenders, where the
       vertical closed form
       $(1/L)\\ln\\bigl((4z+L)/(4z-L)\\bigr)$ does not even exist —
       $4z < L$ makes the argument negative — and the point form
       *under*-estimates. Regime 3 removes that class from the
       warning's reach entirely.)

    Emission scope
    --------------
    Regimes 1 and 2 are hard errors on physically invalid input and
    live in the **shared** kernel guard, so every backend in the
    family rejects them identically. Regimes 3 and 4 are raised from
    ``solve_image`` / ``solve_image_2layer`` only, for two reasons:

    * Regime 3 needs the segment *directions*, which
      :func:`_self_corrected_kernel` does not receive — its signature
      is fixed by five sibling backends.
    * Regime 4 is an advisory about a bias all backends share, but the
      shared kernel is re-entered per frequency and per excitation
      inside the current-solving backends, so emitting it there would
      repeat the same message several times per solve with a
      ``stacklevel`` pointing into solver internals instead of user
      code. Moving it would also change the warning surface of
      backends whose regression tests are not part of this change.
      The honest consequence — cross-engine runs warn asymmetrically
      about a bias all of them carry — is documented in
      ``docs/engines/image.md``.

    Parameters
    ----------
    all_segments
        Discretised leakage segments (electrode segments plus
        galvanically coupled distributed-conductor segments).
    where
        Caller name, used verbatim in the messages.
    stacklevel
        ``stacklevel`` of the :class:`ShallowSegmentWarning`; the
        default points at the caller of the ``solve_*`` function that
        calls this guard directly.

    Raises
    ------
    ValueError
        For regimes 1, 2 and 3.

    Warns
    -----
    ShallowSegmentWarning
        For regime 4.
    """
    if not all_segments:
        return
    z = np.array([float(s.midpoint[2]) for s in all_segments])
    a = np.array([float(s.wire_radius) for s in all_segments])
    lengths = np.array([float(s.length) for s in all_segments])
    owners = [_segment_owner_label(s) for s in all_segments]
    # |e_z| of the segment tangent: 0 for a horizontal segment, 1 for a
    # vertical one. Unknown direction is treated as horizontal, which
    # makes the extent check a no-op for that segment.
    vertical_fraction = np.array(
        [
            0.0 if s.direction is None else abs(float(s.direction[2]))
            for s in all_segments
        ]
    )

    # Regimes 1 + 2 — midpoint checks, shared with every backend.
    _require_image_separation(
        np.array([s.midpoint for s in all_segments], dtype=float),
        a,
        where=where,
        owner_names=owners,
    )

    # Regime 3 — segment extent, not just the midpoint.
    z_top = z - 0.5 * lengths * vertical_fraction
    straddling = z_top < -_SURFACE_TOL
    if straddling.any():
        k = int(np.argmin(z_top))
        raise ValueError(
            f"{where}: {owners[k]} has a segment that crosses the soil "
            f"surface — its upper end is at z = {z_top[k]:.4g} m "
            f"(midpoint z = {z[k]:.4g} m, L = {lengths[k]:.4g} m, "
            f"inclination |e_z| = {vertical_fraction[k]:.3g}), so "
            f"{-z_top[k]:.4g} m of conductor is in the air while the "
            "midpoint is buried. The galvanic image kernel is defined "
            "for buried conductors only and mirrors |z|, so the part "
            "above the surface would be solved as if it were buried "
            "(a 1.2 m rod at position z = -0.5 m used to return "
            "106.2444 Ω with no error). z is a depth, positive "
            "downwards: bury the whole conductor, or start it exactly "
            "at the surface (z = 0), which is the ordinary driven-rod "
            "case and remains legal."
        )

    # Regime 4 — shallow but legal.
    shallow = lengths > _MIN_IMAGE_SEPARATION_RATIO * z
    if shallow.any():
        k = int(np.argmax(lengths / z))
        n_shallow = int(shallow.sum())
        # Relative overestimation of the diagonal self-image entry by
        # the point form 1/(2z) versus the *horizontal* image line
        # (2/L)·arsinh(L/(4z)) for the worst offender. Regime 3
        # guarantees |e_z| < 1/2 here, i.e. the segment is inclined by
        # less than 30 degrees to the horizontal, so this is the
        # applicable closed form (the vertical one,
        # (1/L)·ln((4z+L)/(4z-L)), has no real value for 4z < L).
        ratio = lengths[k] / (4.0 * z[k])
        bias = ratio / np.arcsinh(ratio) - 1.0
        warnings.warn(
            f"{where}: {n_shallow} leakage segment(s) are shallower "
            f"than L/{_MIN_IMAGE_SEPARATION_RATIO:.0f} — worst case "
            f"{owners[k]} with L = {lengths[k]:.4g} m at "
            f"z = {z[k]:.4g} m (inclination |e_z| = "
            f"{vertical_fraction[k]:.3g}). The self-image term on the "
            "reaction-matrix diagonal is a point image at distance 2z "
            "instead of the segment's image line. For a horizontal "
            "segment — which any segment reaching this warning is to "
            "within 30 degrees, steeper ones are rejected outright — "
            f"that overestimates the entry by {bias * 100.0:.0f} % "
            "here, so the grounding impedance is biased high and "
            "still mesh-dependent. The same bias is present in the "
            "mom / bem / cim / mom_sommerfeld backends, which do not "
            "emit this warning. Refine Engine.segment_length / "
            "Conductor.discretize_segment_length until the impedance "
            "stabilises, or silence this with "
            'warnings.simplefilter("ignore", ShallowSegmentWarning).',
            ShallowSegmentWarning,
            stacklevel=stacklevel,
        )
