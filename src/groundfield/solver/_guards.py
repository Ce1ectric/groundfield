"""Validity guards of the image-family discretisation (private).

Hard errors and advisories that protect the closed-form half-space
kernels from geometries they cannot represent:

* :func:`_require_thin_wire` — segment length at or below the wire
  radius (the thin-wire self-term is invalid there);
* :func:`_require_image_separation` — leakage segments at or above the
  soil surface, or so shallow that the mirror image touches the wire;
* :func:`_check_segment_depths` — the same checks on the full segment
  extent plus the :class:`ShallowSegmentWarning` advisory;
* :func:`_warn_rod_ends_near_interfaces` and
  :func:`_warn_wires_near_interfaces` — :class:`InterfaceContactWarning`
  where the thin-wire model deviates from the solid wire next to a layer
  interface (rod ends, audit A3S-05; horizontal wires, audits W4A-07 and
  W4B-02).

Moved out of :mod:`groundfield.solver.image` (review pass 10, 10.75 /
10.79); every name stays importable from there.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from groundfield.solver._discretize import _Segment

__all__ = [
    "InterfaceContactWarning",
    "PointKernelInterfaceWarning",
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
    The bias belongs to the *point* form of the self image. The warning
    is raised wherever a solve uses it: by the image family
    (``image``, ``image_2layer``; ``cim`` is an alias) with
    ``kernel="point"``, and by the point-matching family (``mom``,
    ``mom_sommerfeld``; ``bem`` is an alias), whose self image is always
    the point form. It is not raised for the exact line kernel
    (``kernel="line"``, the default of the image family since 2026.11),
    whose Galerkin self image has no such bias: a 10 m strip at 0.1 m
    depth gives 17.37863 Ω with 2 m segments and 17.37859 Ω with
    0.25 m segments, where the point kernel gives 21.57 Ω and 17.23 Ω
    (audit A1-numerics F08). ``solve_mutual_*`` assemble the same
    reaction matrix without this advisory.

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


def _check_segment_depths(
    all_segments: "list[_Segment]",
    *,
    where: str,
    stacklevel: int = 3,
    kernel: str | None = "point",
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
    family rejects them identically. Regimes 3 and 4 are checked here,
    once per solve, by the pipelines of the image family and of the
    point-matching family (``mom``, ``mom_sommerfeld``; review pass 10,
    10.75): regime 3 needs the segment *directions*, which the kernel
    guard does not receive. Regime 4 is an advisory about the *point*
    form of the self image; with the exact line kernel
    (``kernel="line"``) the self image is a Galerkin integral without
    that bias, and the advisory is not emitted (audit A1-numerics F08).

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
    kernel
        Source kernel of the solve (``"point"``, ``"line"``); regime 4
        applies to the point form only. ``None`` (unknown) is treated
        like ``"point"``.

    Raises
    ------
    ValueError
        For regimes 1, 2 and 3.

    Warns
    -----
    ShallowSegmentWarning
        For regime 4, unless ``kernel == "line"``.
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

    # Regime 4 — shallow but legal; a bias of the point-form self image
    # only (the line kernel integrates the image line exactly, F08).
    if kernel == "line":
        return
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
            f"{vertical_fraction[k]:.3g}). With the point kernel the "
            "self-image term on the reaction-matrix diagonal is a point "
            "image at distance 2z instead of the segment's image line. "
            "For a horizontal segment — which any segment reaching this "
            "warning is to within 30 degrees, steeper ones are rejected "
            f"outright — that overestimates the entry by {bias * 100.0:.0f} % "
            "here, so the grounding impedance is biased high and still "
            "mesh-dependent. Use the exact line kernel "
            "(Engine(kernel='line'), the default of the image family), "
            "refine Engine.segment_length / "
            "Conductor.discretize_segment_length until the impedance "
            "stabilises, or silence this with "
            'warnings.simplefilter("ignore", ShallowSegmentWarning).',
            ShallowSegmentWarning,
            stacklevel=stacklevel,
        )


# ---------------------------------------------------------------------
# Point form of the cross-layer path (ADR-0007): diagonal quadrature
# ---------------------------------------------------------------------

#: Depth offset, in segment lengths, of the outer nodes of the 3-point
#: Gauss–Legendre rule with which the point form of the cross-layer path
#: (ADR-0007; ``kernel="point"`` and the ``mom`` family) averages the
#: layered correction on the reaction-matrix diagonal: ``sqrt(3/5)/2``.
#: The rule runs along the depth axis whatever the segment's orientation.
CROSS_LAYER_DIAGONAL_SPREAD = 0.5 * math.sqrt(0.6)

#: A horizontal segment whose midpoint lies closer to a layer interface
#: than this many segment lengths draws :class:`PointKernelInterfaceWarning`
#: on the point-form cross-layer path.
CROSS_LAYER_WARN_DISTANCE = 0.6


class PointKernelInterfaceWarning(UserWarning):
    """A horizontal segment lies close to a layer interface (point form).

    The point form of the reaction matrix (``kernel="point"`` of the image
    family, the ``mom`` family and ``solve_mutual_*`` with the point kernel)
    cannot represent a horizontal wire next to a layer interface, on either
    side of it:

    * on the cross-layer path of a two-layer soil (an electrode reaches the
      interface :math:`h_1`) it averages the layered correction on the
      diagonal over a 3-point rule along the *depth* axis,
      :math:`\\pm 0.387 L` around the segment midpoint, whatever the
      segment's orientation;
    * with every segment above :math:`h_1` (the Tagg/Sunde series, and the
      Sommerfeld diagonal of ``mom_sommerfeld``) it takes the image of a
      segment in the interface as a point source :math:`2|z - h_1|` away,
      where the image *line* of a horizontal segment is much closer to the
      wire than that point (audit A3S-02).

    A horizontal segment within :math:`0.6 L` of :math:`h_1` draws this
    warning: measured −3 % … −17 % against the exact line kernel on the
    cross-layer path (a 20 m strip and a ring r = 3 m in 300/50 and
    100/1000 Ω·m), −2.5 % and −0.4 % above :math:`h_1` (a ring r = 3 m,
    ``segment_length = 0.5`` m, 0.2 m above :math:`h_1` = 5 m in 1000/30
    and 30/1000 Ω·m). Closer still (:math:`|z - h_1| < 0.387 L`), or where
    the 3-point rule reaches above the soil surface, the solve raises
    :class:`~groundfield.solver.registry.BackendCapabilityError` instead
    (negative impedances: −552 % on the cross-layer path; −1931 Ω instead
    of 4.18 Ω for the ring 1 µm above :math:`h_1`). The exact line kernel,
    the default of the image family, integrates these segments exactly.
    """


def _conductor_hint(segments: "Sequence[_Segment]", mask: np.ndarray) -> str:
    """Name the conductor knob when a flagged segment belongs to a conductor."""
    flagged = np.flatnonzero(mask)
    if any(getattr(segments[int(k)], "conductor_name", None) for k in flagged):
        return "; for a conductor piece refine Conductor.discretize_segment_length"
    return ""


def _check_point_cross_layer_diagonal(
    segments: "Sequence[_Segment]",
    points: np.ndarray,
    lengths: np.ndarray,
    directions: np.ndarray | None,
    h_1: float,
    *,
    backend: str,
    remedy: str,
) -> None:
    """Reject segments the point-form cross-layer diagonal cannot represent.

    The diagonal rule evaluates the layered correction at the depths
    :math:`z \\pm 0.387 L` (:data:`CROSS_LAYER_DIAGONAL_SPREAD`). It is
    the segment's own extent for a vertical segment (rods are cut at the
    interface, so their rule never leaves the segment's layer) but not
    for a horizontal one. Measured against the exact line kernel (docb and
    fix2 evidence: rod through :math:`h_1` plus a 20 m strip, 300/50 Ω·m;
    a ring r = 3 m below :math:`h_1`, 100/1000 Ω·m), with
    :math:`q = |z - h_1| / (0.387 L)`:

    * the rule reaches above the soil surface (:math:`z < 0.387 L`):
      ``layered_green`` raised a bare ``ValueError`` about negative depths;
    * the rule reaches across the interface (:math:`q < 1`): −14 % …
      −552 % (negative impedances, ``mom`` alike);
    * :math:`1 \\le q < 1.55` (:math:`|z - h_1| < 0.6 L`): −3 % … −17 %;
    * :math:`q \\ge 1.8`: −0.9 % … −2.1 %, the usual offset between
      point matching and the line kernel.

    The first two raise for every segment, the third warns for horizontal
    (non-vertical) segments.

    Parameters
    ----------
    segments
        The leakage segments (for the owner names in the message).
    points, lengths
        Midpoints in m, shape ``(N, 3)``, and lengths in m, shape ``(N,)``.
    directions
        Unit tangents, shape ``(N, 3)``; ``None`` treats every segment as
        horizontal.
    h_1
        Interface depth in m.
    backend
        Label of the solve for the message (e.g. ``"image_2layer
        (kernel='point')"``).
    remedy
        Sentence naming the alternatives (the line kernel, a finer
        segment length).

    Raises
    ------
    groundfield.solver.registry.BackendCapabilityError
        If the diagonal rule of a segment reaches above the soil surface or
        across the interface.

    Warns
    -----
    PointKernelInterfaceWarning
        If a horizontal segment lies within ``0.6 L`` of the interface.
    """
    z = np.asarray(points, dtype=float)[:, 2]
    L = np.asarray(lengths, dtype=float)
    spread = CROSS_LAYER_DIAGONAL_SPREAD * L
    if directions is None:
        horizontal = np.ones(z.shape, dtype=bool)
    else:
        horizontal = np.abs(np.asarray(directions, dtype=float)[:, 2]) < 1.0 - 1e-9
    gap = np.abs(z - h_1)
    in_air = z < spread
    across = gap < spread
    near = horizontal & ~across & (gap < CROSS_LAYER_WARN_DISTANCE * L)
    if not (in_air.any() or across.any() or near.any()):
        return
    finer = np.minimum(
        np.where(horizontal, z / CROSS_LAYER_DIAGONAL_SPREAD, np.inf),
        np.where(horizontal, gap / CROSS_LAYER_WARN_DISTANCE, np.inf),
    )

    def example(mask: np.ndarray) -> str:
        k = int(np.flatnonzero(mask)[0])
        return (
            f"{_segment_owner_label(segments[k])}, segment at z = {z[k]:.3g} m, "
            f"length {L[k]:.3g} m"
        )

    def counted(mask: np.ndarray) -> str:
        n = int(mask.sum())
        return f"{n} segment{'s' if n != 1 else ''}"

    hint = ""
    if np.isfinite(finer).any():
        hint = f" (here below {float(np.min(finer[np.isfinite(finer)])):.3g} m)"
    hint += _conductor_hint(segments, in_air | across | near)
    if in_air.any() or across.any():
        from groundfield.solver.registry import BackendCapabilityError

        parts = []
        if in_air.any():
            parts.append(
                f"{counted(in_air)} reach above the soil surface (e.g. "
                f"{example(in_air)})"
            )
        if across.any():
            parts.append(
                f"{counted(across)} reach across the layer interface at "
                f"h_1 = {h_1:.3g} m (e.g. {example(across)})"
            )
        raise BackendCapabilityError(
            f"Backend {backend} cannot solve this world: the point form of the "
            "cross-layer reaction matrix averages the layered correction on "
            "its diagonal over +-0.387 segment lengths in depth, and "
            + "; ".join(parts)
            + ". Measured against the exact line kernel: -14 % ... -552 % "
            "(negative impedances). " + remedy + hint + "."
        )
    from groundfield.solver.registry import warn_external

    warn_external(
        f"{backend}: {counted(near)} of horizontal wires lie closer to the "
        f"layer interface at h_1 = {h_1:.3g} m than 0.6 segment lengths (e.g. "
        f"{example(near)}); the point form of the cross-layer reaction matrix "
        "samples the layered correction next to the interface there "
        "(-3 % ... -17 % against the exact line kernel). " + remedy + hint + ".",
        PointKernelInterfaceWarning,
    )


def _check_point_interface_diagonal(
    segments: "Sequence[_Segment]",
    points: np.ndarray,
    lengths: np.ndarray,
    directions: np.ndarray | None,
    h_1: float,
    *,
    backend: str,
    remedy: str,
    cross_layer: bool | None = None,
) -> None:
    """Point-form guard for horizontal wires on either side of an interface.

    The point form of the reaction-matrix diagonal fails for a horizontal
    wire next to the layer interface :math:`h_1` on both of its paths:

    * **cross-layer path** (a segment at or below :math:`h_1`): the 3-point
      depth rule of the layered correction, see
      :func:`_check_point_cross_layer_diagonal`, which this function calls;
    * **every segment above** :math:`h_1` (the Tagg/Sunde series of
      ``image_2layer`` and ``mom``, the series diagonal of
      ``mom_sommerfeld`` with two layers and its Sommerfeld diagonal with
      three or more): the image of a segment in the interface enters the
      diagonal as a point source :math:`2|z - h_1|` away, while the image
      *line* of a horizontal segment runs parallel to the wire at that
      distance. A ring r = 3 m (``segment_length = 0.5`` m) 1 µm / 1 mm /
      1 cm / 5 cm / 0.1 m above :math:`h_1` = 5 m read −1931 / −945 /
      −57.9 / 21.2 / 31.6 Ω instead of 4.18 / 4.77 / 15.4 / 28.0 / 33.5 Ω
      (1000/30 Ω·m) and +853 % … +0.1 % in 30/1000 Ω·m (audit A3S-02);
      0.2 m above it −2.5 % and −0.4 %.

    The thresholds are those of the cross-layer path, with
    :math:`q = |z - h_1| / (0.387 L)`: :math:`q < 1` raises for every
    segment, :math:`|z - h_1| < 0.6 L` warns for horizontal (non-vertical)
    segments. Vertical segments never reach either: a rod ends at or above
    the interface, so its segment midpoints keep :math:`L/2` from it.

    Above the interface the error of the point form grows smoothly with
    :math:`L/|z - h_1|` and with a conductive lower layer, and depends on
    the gap in wire radii, so no single threshold separates good from bad
    values (rings of 3 m and 25 m, ``h_1`` = 2 m, ``segment_length`` 0.5 /
    2 / 5 m, three contrasts, against the line kernel at the same segment
    length): for :math:`q \\approx 0.25` −13 … −24 % (1000/30 Ω·m) and
    +0.5 … +2.6 % (lower layer resistive), for :math:`q` = 0.52 and a gap of
    1 cm −15 % (audit A3S-02), but for :math:`q` = 0.63 … 0.69 and gaps of
    0.5 … 1.2 m only −3.0 … +0.9 %. The error threshold of the cross-layer
    path is therefore conservative here: it also refuses such coarse but
    accurate cases (a ring r = 25 m 1.2 m above ``h_1`` with 5 m segments,
    +0.9 %); a finer ``segment_length`` (the message names one) or the line
    kernel solves them. In the warning band (:math:`1 \\le q < 1.55`) the
    measured deviation is −2.6 … 0 %, at most 1 % beyond the usual offset
    of the point kernel.

    Parameters
    ----------
    segments
        The leakage segments (for the owner names in the message).
    points, lengths
        Midpoints in m, shape ``(N, 3)``, and lengths in m, shape ``(N,)``.
    directions
        Unit tangents, shape ``(N, 3)``; ``None`` treats every segment as
        horizontal.
    h_1
        Depth of the interface in m (the shallowest one; every segment lies
        above it when ``cross_layer`` is false).
    backend
        Label of the solve for the message.
    remedy
        Sentence naming the alternatives.
    cross_layer
        Whether the solve runs on the cross-layer path; ``None`` decides by
        the deepest midpoint (``z_max >= h_1``).

    Raises
    ------
    groundfield.solver.registry.BackendCapabilityError
        If a segment midpoint lies within ``0.387 L`` of the interface (or,
        on the cross-layer path, the 3-point rule reaches above the soil
        surface).

    Warns
    -----
    PointKernelInterfaceWarning
        If a horizontal segment lies within ``0.6 L`` of the interface.
    """
    z = np.asarray(points, dtype=float)[:, 2]
    if cross_layer is None:
        cross_layer = bool(z.size and float(z.max()) >= h_1)
    if cross_layer:
        _check_point_cross_layer_diagonal(
            segments,
            points,
            lengths,
            directions,
            h_1,
            backend=backend,
            remedy=remedy,
        )
        return
    L = np.asarray(lengths, dtype=float)
    if directions is None:
        horizontal = np.ones(z.shape, dtype=bool)
    else:
        horizontal = np.abs(np.asarray(directions, dtype=float)[:, 2]) < 1.0 - 1e-9
    gap = np.abs(z - h_1)
    across = gap < CROSS_LAYER_DIAGONAL_SPREAD * L
    near = horizontal & ~across & (gap < CROSS_LAYER_WARN_DISTANCE * L)
    if not (across.any() or near.any()):
        return
    finer = np.where(horizontal, gap / CROSS_LAYER_WARN_DISTANCE, np.inf)
    hint = ""
    if np.isfinite(finer).any():
        hint = f" (here below {float(np.min(finer[np.isfinite(finer)])):.3g} m)"
    hint += _conductor_hint(segments, across | near)

    def example(mask: np.ndarray) -> str:
        k = int(np.flatnonzero(mask)[0])
        return (
            f"{_segment_owner_label(segments[k])}, segment at z = {z[k]:.4g} m, "
            f"length {L[k]:.3g} m"
        )

    def counted(mask: np.ndarray) -> str:
        n = int(mask.sum())
        return f"{n} segment{'s' if n != 1 else ''}"

    if across.any():
        from groundfield.solver.registry import BackendCapabilityError

        raise BackendCapabilityError(
            f"Backend {backend} cannot solve this world: the point form of the "
            "reaction matrix takes the image of every segment in the layer "
            "interface as a point source 2|z - h_1| away, and "
            f"{counted(across)} lie closer than 0.387 segment lengths to the "
            f"layer interface at h_1 = {h_1:.4g} m (e.g. {example(across)}), "
            "where the image line of a horizontal wire runs much closer to the "
            "wire than that point. Measured against the exact line kernel for a "
            "ring 1 um ... 0.1 m above h_1: -1931 instead of 4.18 Ohm ... -5.8 % "
            "(1000/30 Ohm m), +853 % ... +0.1 % (30/1000 Ohm m). " + remedy + hint + "."
        )
    from groundfield.solver.registry import warn_external

    warn_external(
        f"{backend}: {counted(near)} of horizontal wires lie closer to the "
        f"layer interface at h_1 = {h_1:.4g} m than 0.6 segment lengths (e.g. "
        f"{example(near)}); the point form takes the image of each segment in "
        "the interface as a point source there (-2.5 % and -0.4 % against the "
        "exact line kernel for a ring 0.2 m above h_1, 1000/30 and 30/1000 "
        "Ohm m). " + remedy + hint + ".",
        PointKernelInterfaceWarning,
    )


# ---------------------------------------------------------------------
# Rod ends at a layer interface: a limit of the thin-wire model
# ---------------------------------------------------------------------


class InterfaceContactWarning(UserWarning):
    """A wire lies too close to a layer interface for the thin-wire model.

    The thin-wire model carries the current on the wire axis. Two limits
    follow next to a layer interface:

    **Rod ends** (audit A3S-05). A rod head or tip within one wire radius of
    an interface: the thin wire has no end face, so a rod that ends exactly
    at the interface :math:`h_1` is a wire entirely on one side of it, and
    a rod that reaches past the interface by a few micrometres touches the
    other layer only through a part much shorter than its radius. The solid
    rod with a flat end face is continuous in between: its end face carries
    the contact. Against an axisymmetric finite-volume model of the solid
    rod (4.5 m, :math:`a` = 8 mm, 1000/30 Ω·m) the thin wire reads +21 %
    with its tip at the interface, +7 % 1 mm above it and −4.6 % 1 mm
    below it; beyond two radii it is within a few per cent, like anywhere
    else (+0.9 % in homogeneous soil). An end that meets the end of another
    rod has no free end face and is not reported.

    **Horizontal wires** (audits W4A-07, W4B-02). A horizontal wire (every
    electrode but a rod, and the leaking pieces of a galvanic distributed
    conductor) whose axis lies within the reach :func:`wire_contact_reach`
    of an interface: next to a more conductive layer the current crowds to
    the wire's near side, which a current on the axis cannot follow. Against
    the solid torus (the axisymmetric ``fem``, confirmed by a Hankel thin
    ring plus the exact 2-D proximity correction to 5.2e-4, to 2e-4 from two
    radii on) a ring (a = 5 mm)
    over a layer 33 times more conductive reads +0.59 % at 4 radii, +1.2 %
    at 3, +3.6 % at 2, +8.6 % at 1.5 and +32 % at 1.1 radii, and +89 % when
    the interface cuts the wire 0.99 radii below its axis; under a more
    resistive layer up to +2.6 % (a 55 mm wire on a 1 m ring at 1.01
    radii, contrast 1000). The reach is the largest distance at which the
    thin wire (converged in its segment length) is 1 % off, in five ring
    geometries (r0 / a = 18 ... 400: rings of 0.5 ... 5 m at 0.3 ... 2 m,
    a = 5 ... 55 mm; the more conductive layer below the wire, for the 2 m
    ring also above it; the more resistive layer below and above every
    one; the thick wire on the 1 m ring is the widest everywhere), rounded
    up to 0.1 a, plus 0.1 a for the contrasts between the nodes (audits
    W4R2A-04, W4R3A-02; up to W4R2A-04 four geometries, 0.2 ... 0.4 a short
    of the thick wire; up to W4R3A-02 the resistive side measured above
    the wire for the thin ring only, the thick wire 1 % off up to 0.74 a
    beyond it):

    ======================  =====  =====  =====  =====  =====  =====  =====  =====  =====
    contrast                1.4    2      3      5.5    10     18     33     100    1000
    more conductive beyond  1.0 a  1.3 a  1.9 a  2.7 a  3.3 a  3.7 a  3.9 a  4.1 a  4.3 a
    ======================  =====  =====  =====  =====  =====  =====  =====  =====  =====

    ======================  =====  =====  =====  =====  =====  =====
    contrast                1.4    3      10     33     100    1000
    more resistive beyond   1.0 a  1.3 a  1.9 a  2.2 a  2.2 a  2.3 a
    ======================  =====  =====  =====  =====  =====  =====

    (log-linear in the contrast between the nodes, held beyond the last
    and at 1.0 a down to the contrast 1.3; 1.0 a: only a wire that the
    interface cuts.) Checked against every distance measured at the nodes
    and between them (conductive 1.3 ... 1.9, 2.5, 4, 7, 14, 25, 50, 300;
    resistive 2, 5.5, 18, 300): no thin wire 1 % or more off lies outside
    the reach (2348 conductive points at one segment length, the largest
    outside +0.94 %; 2470 resistive points converged in it, +0.93 %). The
    thick wire's conductive 1 % distances, converged in the segment length,
    lie 0.004 ... 0.17 a inside the reach; the node at 1000 is 4.3 a (it
    was 4.2) for the 4.15 a at 300. Below a contrast of 1.3 nothing is
    reported: a cut wire reads at most +0.96 % off at 1.3 and +1.06 % at
    1.35 (the thick wire cut 0.5 a below its axis by a more resistive layer
    above it; up to audit W4R3A-02 the limit was 1.4, where it reads
    +1.16 %, and up to W4R2A-04 2, where a wire cut at a contrast of 1.9
    read -1.46 %). Pieces of a galvanic distributed
    conductor count as horizontal while their slope ``|e_z|`` is at most
    0.1 (5.7 deg): a steeper piece passes the reach of an interface over at
    most ``2 reach / |e_z|`` wire radii of its length, like a rod through an
    interface, and is not reported (a step in reporting at the cutoff,
    kept as documented). The image family, ``mom``, ``mom_sommerfeld`` and
    the three-dimensional ``fem`` share this limit (the 3-D model's smoothed
    line source lies between the thin wire and the solid torus); the
    axisymmetric ``fem`` resolves the wire and does not warn.

    Remedy: keep rod ends and horizontal wires a few wire radii from a
    contrasting interface; bracket a rod end by moving it by
    :math:`\\pm 2a`; check a ring with ``fem``'s axisymmetric model
    (``FemOptions.model="axisymmetric"``), which solves the wire as the
    solid torus it is.
    """


def _rod_ends_near_interfaces(
    electrodes: "Sequence[object]", interfaces: "Sequence[float] | None"
) -> list[tuple[str, str, float, float, float]]:
    """Rod heads and tips within one wire radius of an interface.

    Parameters
    ----------
    electrodes
        Electrodes of the world; only :class:`RodElectrode` has ends that
        are not horizontal.
    interfaces
        Physical interface depths in m (the canonical stack).

    Returns
    -------
    list of tuple
        ``(electrode, "head" | "tip", z_end, h, wire_radius)`` in m, one
        entry per end with ``|z_end - h| <= wire_radius`` (to round-off).
        An end that meets the end of another rod (a rod split at the
        interface into two pieces, the usual way to give each layer its
        own piece) has no free end face and is not reported.
    """
    from groundfield.geometry.electrodes import RodElectrode

    if not interfaces:
        return []
    rods = [e for e in electrodes if isinstance(e, RodElectrode)]
    ends = []
    for e in rods:
        x, y, head = (float(c) for c in e.position)
        a = float(e.wire_radius)
        ends.append((e.name, "head", x, y, head, a))
        ends.append((e.name, "tip", x, y, head + float(e.length), a))

    def _joined(k: int) -> bool:
        name, _, x, y, z, a = ends[k]
        for j, (other, _w, xo, yo, zo, ao) in enumerate(ends):
            if j == k or other == name:
                continue
            reach = max(a, ao) * (1.0 + 1e-9)
            if abs(z - zo) <= reach and math.hypot(x - xo, y - yo) <= reach:
                return True
        return False

    out = []
    for k, (name, which, _x, _y, z_end, a) in enumerate(ends):
        for h in interfaces:
            # Inclusive, to round-off: a tip placed exactly one radius
            # from h (h = z_tip - a) is within reach.
            if abs(z_end - float(h)) <= a * (1.0 + 1e-9) and not _joined(k):
                out.append((name, which, z_end, float(h), a))
    return out


def _warn_rod_ends_near_interfaces(
    electrodes: "Sequence[object]", interfaces: "Sequence[float] | None"
) -> None:
    """Emit :class:`InterfaceContactWarning` for rod ends at an interface."""
    hits = _rod_ends_near_interfaces(electrodes, interfaces)
    if not hits:
        return
    from groundfield.solver.registry import warn_external

    name, which, z_end, h, a = hits[0]
    n = len(hits)
    warn_external(
        f"{n} rod end{'s' if n != 1 else ''} lie{'' if n != 1 else 's'} within "
        f"one wire radius of a layer interface (e.g. electrode '{name}', {which} "
        f"at z = {z_end:.6g} m, interface at {h:.6g} m, wire radius "
        f"{a * 1e3:.3g} mm). The thin-wire model has no end face: a rod ending "
        "at the interface does not touch the layer beyond it, one reaching "
        "past it by a fraction of its radius touches it through a short part "
        "only. Against a solid rod with a flat end (4.5 m, a = 8 mm, "
        "1000/30 Ohm m): +21 % with the tip at the interface, +7 % 1 mm above "
        "it, -4.6 % 1 mm below it. Keep rod ends a few wire radii from an "
        "interface, or bracket the result by moving the end by +-2a.",
        InterfaceContactWarning,
    )


# ---------------------------------------------------------------------
# Horizontal wires next to a contrasting interface (audits W4A-07, W4B-02)
# ---------------------------------------------------------------------

#: Reach of an interface over a more conductive layer, in wire radii of the
#: horizontal wire's axis: ``(contrast, reach)`` nodes (see
#: :class:`InterfaceContactWarning` and :func:`wire_contact_reach`). The
#: largest distance at which the thin wire is 1 % off the solid torus in six
#: ring geometries (r0 / a = 18 ... 400), rounded up to 0.1 a, plus 0.1 a for
#: the contrasts between the nodes (audit W4R2A-04, evidence ``fw4r2a/e03``);
#: 4.3 at 1000 (was 4.2): the thin wire converged in its segment length is
#: 1 % off up to 4.15 a at 300 (audit W4R3A-02, evidence ``fw4r3a/e05``).
WIRE_CONTACT_REACH_CONDUCTIVE: tuple[tuple[float, float], ...] = (
    (1.4, 1.0),
    (2.0, 1.3),
    (3.0, 1.9),
    (5.5, 2.7),
    (10.0, 3.3),
    (18.0, 3.7),
    (33.0, 3.9),
    (100.0, 4.1),
    (1000.0, 4.3),
)

#: Reach of an interface over a more resistive layer, in wire radii (the same
#: rule, with the layer below and above the wire of five ring geometries and
#: the thin wire converged in its segment length; the 55 mm wire on the 1 m
#: ring under the layer sets every node; 1.0: only a wire that the interface
#: cuts; audit W4R3A-02, evidence ``fw4r3a/e01``, ``e02``).
WIRE_CONTACT_REACH_RESISTIVE: tuple[tuple[float, float], ...] = (
    (1.4, 1.0),
    (3.0, 1.3),
    (10.0, 1.9),
    (33.0, 2.2),
    (100.0, 2.2),
    (1000.0, 2.3),
)

#: Below this contrast of the two resistivities no horizontal wire is
#: reported: a wire that the interface cuts is at most +0.96 % off at 1.3 and
#: +1.06 % at 1.35 (the 55 mm wire on the 1 m ring, cut 0.5 a below its axis
#: by a more resistive layer above it; audit W4R3A-02; up to then the limit
#: was 1.4, where that wire reads +1.16 %, and up to audit W4R2A-04 2).
WIRE_CONTACT_MIN_CONTRAST = 1.3


def wire_contact_reach(rho_wire: float, rho_other: float) -> float:
    """Reach of an interface for a horizontal wire, in wire radii.

    Within this distance of the interface (wire axis to interface plane)
    the thin-wire model of a horizontal wire deviates from the solid wire
    by more than about 1 % (:class:`InterfaceContactWarning`).

    Parameters
    ----------
    rho_wire
        Resistivity in Ω·m of the layer that holds the wire axis.
    rho_other
        Resistivity in Ω·m of the layer beyond the interface.

    Returns
    -------
    float
        The reach in wire radii, linear in ``log10`` of the contrast
        between the nodes of :data:`WIRE_CONTACT_REACH_CONDUCTIVE`
        (``rho_other < rho_wire``) or :data:`WIRE_CONTACT_REACH_RESISTIVE`
        and held beyond the last node and below the first down to the
        contrast :data:`WIRE_CONTACT_MIN_CONTRAST` (1.3), ``0.0`` below it.
        Measured on rings with r0 / a from 18 to 400 (audits W4R2A-04,
        W4R3A-02); a thicker wire on a smaller ring is not covered.
    """
    lo, hi = sorted((float(rho_wire), float(rho_other)))
    contrast = hi / lo
    if not contrast >= WIRE_CONTACT_MIN_CONTRAST:
        return 0.0
    table = (
        WIRE_CONTACT_REACH_CONDUCTIVE
        if rho_other < rho_wire
        else WIRE_CONTACT_REACH_RESISTIVE
    )
    x = [math.log10(k) for k, _ in table]
    y = [r for _, r in table]
    return float(np.interp(math.log10(contrast), x, y))


def _wires_near_interfaces(
    world: "object",
) -> list[tuple[str, str, float, float, float, float, float, float]]:
    """Horizontal wires within the reach of a contrasting interface.

    Parameters
    ----------
    world
        The world; its soil gives the interfaces and resistivities.

    Returns
    -------
    list of tuple
        ``(name, kind, z, h, wire_radius, rho_wire, rho_other, reach)`` per
        electrode or galvanic distributed conductor and interface:
        ``kind`` is ``"electrode"`` or ``"conductor"``, ``z`` the depth in m
        of the wire part nearest to the interface ``h`` (m),
        ``wire_radius`` in m, the resistivities in Ω·m and ``reach`` in wire
        radii (:func:`wire_contact_reach`). Rods are not horizontal wires
        (their ends have their own rule, :func:`_rod_ends_near_interfaces`).
    """
    from groundfield.geometry.electrodes import RodElectrode
    from groundfield.solver._layered import as_layer_stack

    soil = getattr(world, "soil", None)
    if soil is None:
        return []
    stack = as_layer_stack(soil)
    if stack.n_layers < 2:
        return []
    rhos = [float(r) for r in stack.rhos]
    depths = [float(d) for d in np.cumsum(stack.h)]

    parts: list[tuple[str, str, float, float, float]] = []
    for e in world.electrodes:
        if isinstance(e, RodElectrode):
            continue
        lo, hi = e.bounds()
        parts.append(
            (e.name, "electrode", float(lo[2]), float(hi[2]), float(e.wire_radius))
        )
    for c in world.conductors:
        if not (
            getattr(c, "is_distributed", False)
            and getattr(c, "coupling_to_soil", "isolated") == "galvanic"
        ):
            continue
        from groundfield.solver._discretize import _discretize_conductor

        segments, _ = _discretize_conductor(c)
        for seg in segments:
            dz = 0.0 if seg.direction is None else abs(float(seg.direction[2]))
            if dz > 0.1:  # not horizontal
                continue
            z_mid = float(seg.midpoint[2])
            half = 0.5 * float(seg.length) * dz
            parts.append(
                (
                    c.name,
                    "conductor",
                    z_mid - half,
                    z_mid + half,
                    float(seg.wire_radius),
                )
            )

    out = []
    seen: set[tuple[str, int]] = set()
    for name, kind, z_lo, z_hi, a in parts:
        for j, h in enumerate(depths):
            if (name, j) in seen:
                continue
            if z_lo <= h <= z_hi:
                z, distance = h, 0.0
            else:
                z = z_lo if abs(z_lo - h) <= abs(z_hi - h) else z_hi
                distance = abs(z - h)
            rho_wire, rho_other = (
                (rhos[j], rhos[j + 1]) if z < h else (rhos[j + 1], rhos[j])
            )
            if distance == 0.0:  # on the interface: the larger contrast reach
                rho_wire, rho_other = (
                    max(rhos[j], rhos[j + 1]),
                    min(rhos[j], rhos[j + 1]),
                )
            reach = wire_contact_reach(rho_wire, rho_other)
            if reach > 0.0 and distance < reach * a:
                seen.add((name, j))
                out.append((name, kind, z, h, a, rho_wire, rho_other, reach))
    return out


def _warn_wires_near_interfaces(world: "object") -> None:
    """Emit :class:`InterfaceContactWarning` for horizontal wires (W4A-07)."""
    hits = _wires_near_interfaces(world)
    if not hits:
        return
    from groundfield.solver.registry import warn_external

    name, kind, z, h, a, rho_wire, rho_other, reach = hits[0]
    n = len({(hit[1], hit[0]) for hit in hits})  # a wire near two interfaces once
    contrast = max(rho_wire, rho_other) / min(rho_wire, rho_other)
    side = "conductive" if rho_other < rho_wire else "resistive"
    warn_external(
        f"{n} horizontal wire{'s' if n != 1 else ''} lie{'' if n != 1 else 's'} "
        "within reach of a contrasting layer interface, where the thin-wire "
        f"model deviates from the solid wire by more than about 1 % (e.g. {kind} "
        f"'{name}' at z = {z:.6g} m, interface at {h:.6g} m: "
        f"{abs(z - h) / a:.3g} wire radii from it, the layer beyond "
        f"{contrast:.3g} times more {side}; reach {reach:.3g} radii). The thin "
        "wire carries its current on its axis; next to a more conductive layer "
        "the current crowds to the wire's near side: against the solid wire a "
        "ring reads +3.6 % at 2 radii, +8.6 % at 1.5 and +32 % at 1.1 radii "
        "above a 33 times more conductive layer, +89 % when the interface cuts "
        "the wire. Keep horizontal wires a few radii from a contrasting "
        "interface, or check a ring with fem's axisymmetric model "
        "(FemOptions.model='axisymmetric'), which solves the solid torus.",
        InterfaceContactWarning,
    )
