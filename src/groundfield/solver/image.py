"""Image-charge backend for **homogeneous** soil.

Computes the potential field of an arbitrary grounding system in a
homogeneous half-space (resistivity $\\rho$, soil surface at
$z = 0$, $z$ axis pointing into the soil) using the
classical image-charge method.

Notes
-----
A point current source $I$ at $r_s = (x_s, y_s, z_s)$ with
$z_s > 0$ (inside the soil) produces, in a homogeneous half-space
with an insulating soil surface, the potential
$$
\\varphi(r) \\;=\\; \\frac{\\rho\\, I}{4\\pi}\\,
\\Big(\\frac{1}{|r - r_s|} + \\frac{1}{|r - r_s'|}\\Big),
$$
with the image $r_s' = (x_s, y_s, -z_s)$ mirrored at the soil
surface. An extended electrode is discretised into $N$ segments;
each segment carries one point current source at its midpoint. The
total current $I_e$ of an electrode is distributed **uniformly
per unit length** across its segments — a surprisingly good
approximation for wire electrodes at low frequencies (cf. Sunde 1968,
Tagg 1964).

The input impedance of an electrode is computed as the average of the
potential on its own segment midpoints (average-potential method). For
a single driven rod the backend reproduces the Sunde formula within a
few per cent.

Further properties of this backend:

* Frequency-independent: in the quasi-static range
  $f < 1\\,\\mathrm{kHz}$ the backend returns the same real
  solution per frequency. Complex extensions (Carson,
  frequency-dependent soil) come in later backends.
* Multiple electrodes: each electrode has its own total current
  (sum of the current sources attached to it). An electrode without a
  source carries zero current and acts purely as a passive observer.

References
----------
.. [1] E. D. Sunde, *Earth Conduction Effects in Transmission
       Systems*, Dover, 1968.
.. [2] G. F. Tagg, *Earth Resistances*, Pitman, 1964.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from groundfield.geometry.electrodes import _ElectrodeBase
    from groundfield.solver.engine import Engine
    from groundfield.world import World

from groundfield.geometry.electrodes import (
    GridMeshElectrode,
    MeshElectrode,
    PolylineElectrode,
    RingElectrode,
    RodElectrode,
    StarElectrode,
    StripElectrode,
)
from groundfield.soil.models import HomogeneousSoil
from groundfield.solver.result import FieldResult, PointSource

__all__ = ["solve_image", "ShallowSegmentWarning"]

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


def _require_thin_wire(
    seg_len: float, wire_radius: float, electrode_name: str
) -> None:
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


# Frequently reused unit tangents (vertical rods, axis-aligned grid
# wires). Kept immutable so the shared instances cannot be mutated
# through a ``_Segment``.
def _frozen(v: "list[float]") -> np.ndarray:
    arr = np.array(v, dtype=float)
    arr.flags.writeable = False
    return arr


_TANGENT_Z = _frozen([0.0, 0.0, 1.0])
_TANGENT_X = _frozen([1.0, 0.0, 0.0])
_TANGENT_Y = _frozen([0.0, 1.0, 0.0])


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
    all_segments: "list[_Segment]", *, where: str
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
    vertical_fraction = np.array([
        0.0 if s.direction is None else abs(float(s.direction[2]))
        for s in all_segments
    ])

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
            stacklevel=3,
        )


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
        return [
            complex(np.mean(phi_per_freq[k][idxs])) for k in range(n_freq)
        ]
    return [
        complex((w @ phi_per_freq[k][idxs]) / w_sum) for k in range(n_freq)
    ]


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
        e.name for e in world.electrodes
        if float(getattr(e, "concrete_shell_coefficient_ohm_m", 0.0) or 0.0)
        != 0.0
    ]
    if shelled:
        raise NotImplementedError(
            f"Backend '{backend}' does not implement the concrete-shell "
            f"correction (ADR-0012) carried by electrode(s) {shelled}. "
            "Use backend='image' / 'image_2layer', or set "
            "concrete_shell_coefficient_ohm_m = 0."
        )


@dataclass
class _Segment:
    """Discretisation segment of an electrode or distributed conductor.

    Attributes
    ----------
    midpoint
        Cartesian coordinates ``(x, y, z)`` of the segment midpoint
        in metres. Used as the point-source location in the
        Green's-function evaluation.
    length
        Segment length in metres.
    electrode_name
        Name of the *node* the segment leaks current into. For an
        electrode segment this is the electrode's name (which is
        also the cluster root for cluster-building). For a
        galvanic-conductor segment in the distributed-conductor
        model (ADR-0003) this is the pseudo-node name
        ``f"__cond_{conductor.name}__seg_{k}"`` and the segment
        forms its own cluster.
    wire_radius
        Wire radius in metres (for the analytical line self-action
        on the Z-matrix diagonal).
    conductor_name
        For conductor segments: the owning conductor's name. ``None``
        for electrode segments.
    direction
        Unit tangent ``(dx, dy, dz)`` of the segment axis, shape
        ``(3,)``. Only ``|dz|`` is consumed so far, by
        :func:`_check_segment_depths`, to reconstruct the segment's
        vertical *extent* ``z_mid ± L·|dz|/2`` — a midpoint-only depth
        check accepts a rod that pokes out of the ground (audit
        2026-07-29). ``None`` means "orientation not recorded"; the
        depth check then degrades to the midpoint rather than assuming
        one. Every discretiser in this module sets it.
    """

    midpoint: np.ndarray  # shape (3,)
    length: float
    electrode_name: str
    wire_radius: float
    conductor_name: str | None = None
    direction: np.ndarray | None = None
    # ADR-0007: which soil layer the segment lives in (0 = upper,
    # 1 = next, ..., n-1 = bottom semi-infinite). Set by the
    # discretiser when ``layer_interfaces`` is supplied; left at 0
    # for homogeneous-soil callers.
    layer_index: int = 0
    # ADR-0012 V2: per-segment radial shell coefficient
    # $C = \rho_c/(2\pi)\,\ln(r_b/r_a)$ in Ω·m. The MoM diagonal
    # entry for this segment is augmented by $C / \text{length}$
    # (the radial voltage drop through the concrete shell). Left
    # at ``0.0`` for segments that are not part of a concrete-
    # encased foundation electrode.
    concrete_shell_coefficient_ohm_m: float = 0.0


# ---------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------


def _split_interval_at_interfaces(
    t0: float, t1: float, interfaces,
) -> list[tuple[float, float]]:
    """Partition ``[t0, t1]`` at every interior interface value.

    ADR-0007 step 1 (audit 2026-07-08, WP-D2): segments must not
    straddle a soil-layer interface, otherwise the discretiser
    assigns the whole segment to the layer of its midpoint and the
    kernel evaluates up to ``ds/2`` of wire length with the wrong
    layer resistivity.
    """
    cuts = sorted(
        t for t in (interfaces or ())
        if t0 + 1e-12 < t < t1 - 1e-12
    )
    bounds = [t0, *cuts, t1]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def _discretize_rod(
    electrode: RodElectrode, ds: float, layer_interfaces=None,
) -> list[_Segment]:
    """Vertical driven rod into N equally long segments per layer.

    With ``layer_interfaces`` given, the rod's z-interval is first
    split at every interface it crosses; each part is then
    discretised independently, so no segment straddles an interface.
    """
    x0, y0, z0 = electrode.position
    segs: list[_Segment] = []
    parts = _split_interval_at_interfaces(
        z0, z0 + electrode.length, layer_interfaces,
    )
    for z_lo, z_hi in parts:
        part_len = z_hi - z_lo
        n = max(1, int(np.ceil(part_len / ds)))
        seg_len = part_len / n
        _require_thin_wire(seg_len, electrode.wire_radius, electrode.name)
        for k in range(n):
            zc = z_lo + (k + 0.5) * seg_len
            segs.append(
                _Segment(
                    midpoint=np.array([x0, y0, zc], dtype=float),
                    length=seg_len,
                    electrode_name=electrode.name,
                    wire_radius=electrode.wire_radius,
                    direction=_TANGENT_Z,
                )
            )
    return segs


def _discretize_ring(electrode: RingElectrode, ds: float) -> list[_Segment]:
    """Horizontal ring into N equally long arc segments."""
    perimeter = 2.0 * np.pi * electrode.radius
    n = max(8, int(np.ceil(perimeter / ds)))
    seg_len = perimeter / n
    _require_thin_wire(seg_len, electrode.wire_radius, electrode.name)
    cx, cy, cz = electrode.center
    segs: list[_Segment] = []
    for k in range(n):
        phi = 2.0 * np.pi * (k + 0.5) / n
        segs.append(
            _Segment(
                midpoint=np.array(
                    [cx + electrode.radius * np.cos(phi),
                     cy + electrode.radius * np.sin(phi),
                     cz],
                    dtype=float,
                ),
                length=seg_len,
                electrode_name=electrode.name,
                wire_radius=electrode.wire_radius,
                # Horizontal arc chord: |e_z| = 0.
                direction=np.array(
                    [-np.sin(phi), np.cos(phi), 0.0], dtype=float
                ),
            )
        )
    return segs


def _discretize_strip(
    electrode: StripElectrode, ds: float, layer_interfaces=None,
) -> list[_Segment]:
    """Straight strip into N equally long segments along its axis.

    A sloped strip whose z-range crosses a layer interface is split
    there first (ADR-0007 step 1) — horizontal strips are unaffected.
    """
    L = electrode.length
    p0 = np.array(electrode.start, dtype=float)
    p1 = np.array(electrode.end, dtype=float)
    direction = (p1 - p0) / L
    # Map interface depths to axis parameters t in [0, L].
    t_cuts: list[float] = []
    dz = p1[2] - p0[2]
    if layer_interfaces and abs(dz) > 1e-12:
        for h in layer_interfaces:
            t = (h - p0[2]) / dz * L
            t_cuts.append(t)
    segs: list[_Segment] = []
    # ADR-0012 V2: read the concrete-shell coefficient from the
    # electrode and propagate to every segment. ``0.0`` (default)
    # leaves the diagonal untouched.
    shell_coeff = float(electrode.concrete_shell_coefficient_ohm_m)
    for t_lo, t_hi in _split_interval_at_interfaces(0.0, L, t_cuts):
        part_len = t_hi - t_lo
        n = max(1, int(np.ceil(part_len / ds)))
        seg_len = part_len / n
        _require_thin_wire(seg_len, electrode.wire_radius, electrode.name)
        for k in range(n):
            midpoint = p0 + (t_lo + (k + 0.5) * seg_len) * direction
            segs.append(
                _Segment(
                    midpoint=midpoint,
                    length=seg_len,
                    electrode_name=electrode.name,
                    wire_radius=electrode.wire_radius,
                    concrete_shell_coefficient_ohm_m=shell_coeff,
                    direction=direction,
                )
            )
    return segs


def _discretize_polyline(
    electrode: "PolylineElectrode", ds: float, layer_interfaces=None,
) -> list[_Segment]:
    """Polyline (open or closed) — one strip-style chain per edge.

    Every edge is discretised exactly like :func:`_discretize_strip`
    (same segment-length convention, same thin-wire guard, same
    per-segment concrete-shell coefficient), so a closed rectangular
    polyline reproduces the four perimeter wires of the equivalent
    ``grid_mesh`` / strip-chain foundation.

    Degenerate edges (zero length, e.g. from duplicated polygon
    vertices) are skipped rather than raising — OSM footprints
    routinely carry repeated points.
    """
    shell_coeff = float(electrode.concrete_shell_coefficient_ohm_m)
    segs: list[_Segment] = []
    for start, end in electrode.edges:
        p0 = np.array(start, dtype=float)
        p1 = np.array(end, dtype=float)
        L = float(np.linalg.norm(p1 - p0))
        if L <= 1e-12:
            continue
        direction = (p1 - p0) / L
        n = max(1, int(np.ceil(L / ds)))
        seg_len = L / n
        _require_thin_wire(seg_len, electrode.wire_radius, electrode.name)
        for k in range(n):
            segs.append(
                _Segment(
                    midpoint=p0 + (k + 0.5) * seg_len * direction,
                    length=seg_len,
                    electrode_name=electrode.name,
                    wire_radius=electrode.wire_radius,
                    concrete_shell_coefficient_ohm_m=shell_coeff,
                    direction=direction,
                )
            )
    if not segs:
        raise ValueError(
            f"PolylineElectrode {electrode.name!r} discretised to zero "
            "segments — all edges are degenerate."
        )
    return segs


def _discretize_star(
    electrode: "StarElectrode", ds: float, layer_interfaces=None,
) -> list[_Segment]:
    """n-arm star (*Sternerder*) — one strip-style chain per radial arm.

    Every arm runs from the shared centre node to its tip and is
    discretised exactly like :func:`_discretize_strip` (same
    segment-length convention, thin-wire guard, per-segment concrete-
    shell coefficient). All arms belong to one electrode, so the shared
    centre makes them a single galvanic cluster.
    """
    shell_coeff = float(electrode.concrete_shell_coefficient_ohm_m)
    segs: list[_Segment] = []
    for start, end in electrode.edges:
        p0 = np.array(start, dtype=float)
        p1 = np.array(end, dtype=float)
        L = float(np.linalg.norm(p1 - p0))
        direction = (p1 - p0) / L
        n = max(1, int(np.ceil(L / ds)))
        seg_len = L / n
        _require_thin_wire(seg_len, electrode.wire_radius, electrode.name)
        for k in range(n):
            segs.append(
                _Segment(
                    midpoint=p0 + (k + 0.5) * seg_len * direction,
                    length=seg_len,
                    electrode_name=electrode.name,
                    wire_radius=electrode.wire_radius,
                    concrete_shell_coefficient_ohm_m=shell_coeff,
                    direction=direction,
                )
            )
    return segs


def _segments_per_wire(span: float, ds: float, n_spans: int) -> int:
    """Segments per grid wire, aligned to the crossing pattern.

    A wire of length ``span`` is crossed at ``n_spans + 1`` points (the
    perpendicular wires), dividing it into ``n_spans`` equal mesh spans.
    The requested target is ``ceil(span / ds)`` segments; this rounds it
    **up to the next multiple of** ``n_spans`` so every span holds a whole
    number of segments and the crossings coincide with segment endpoints,
    never with the midpoint point sources. With one interior span or none
    (``n_spans <= 1``) there is nothing to align and the plain count is
    returned. The result is never coarser than requested.
    """
    n = max(1, int(np.ceil(span / ds)))
    if n_spans <= 1:
        return n
    return int(np.ceil(n / n_spans)) * n_spans


def _grid_segments(
    *,
    cx: float,
    cy: float,
    cz: float,
    dx: float,
    dy: float,
    nx_wires: int,
    ny_wires: int,
    ds: float,
    electrode_name: str,
    wire_radius: float,
) -> list[_Segment]:
    """Common segment builder for the rectangular mesh / grid family.

    Parameters
    ----------
    nx_wires
        Number of longitudinal wires (each running from ``cx`` to
        ``cx + dx`` at one of ``nx_wires`` distinct ``y`` values).
    ny_wires
        Number of transverse wires (each running from ``cy`` to
        ``cy + dy`` at one of ``ny_wires`` distinct ``x`` values).
    """
    xs = np.linspace(cx, cx + dx, ny_wires)
    ys = np.linspace(cy, cy + dy, nx_wires)

    # Segments per wire, aligned so grid crossings fall on segment
    # *endpoints* rather than midpoints. Longitudinal wires (span dx) are
    # crossed by the ``ny_wires`` transverse wires, i.e. ``ny_wires - 1``
    # mesh spans; making the segment count a multiple of that span count
    # places one or more whole segments in each span, so no segment
    # midpoint (the point-source location) ever coincides with a crossing.
    # Coincident midpoints of a longitudinal and a transverse segment
    # would otherwise land within the singularity clamp and inflate the
    # grid resistance by ~2x for the uniform-current ``image`` backend (a
    # silent grid-discretisation trap; the current-solving backends are
    # far less sensitive but pay the same clamp warning).
    n_long = _segments_per_wire(dx, ds, ny_wires - 1)
    n_trans = _segments_per_wire(dy, ds, nx_wires - 1)

    segs: list[_Segment] = []
    # Longitudinal wires (along x for each y)
    seg_len = dx / n_long
    _require_thin_wire(seg_len, wire_radius, electrode_name)
    for y in ys:
        for k in range(n_long):
            xm = cx + (k + 0.5) * seg_len
            segs.append(
                _Segment(
                    midpoint=np.array([xm, y, cz], dtype=float),
                    length=seg_len,
                    electrode_name=electrode_name,
                    wire_radius=wire_radius,
                    direction=_TANGENT_X,
                )
            )
    # Transverse wires (along y for each x)
    seg_len = dy / n_trans
    _require_thin_wire(seg_len, wire_radius, electrode_name)
    for x in xs:
        for k in range(n_trans):
            ym = cy + (k + 0.5) * seg_len
            segs.append(
                _Segment(
                    midpoint=np.array([x, ym, cz], dtype=float),
                    length=seg_len,
                    electrode_name=electrode_name,
                    wire_radius=wire_radius,
                    direction=_TANGENT_Y,
                )
            )
    return segs


def _discretize_mesh(electrode: MeshElectrode, ds: float) -> list[_Segment]:
    """Mesh earth electrode (uniform spacing) as a grid of wires."""
    cx, cy, cz = electrode.corner
    dx, dy = electrode.size
    spacing = electrode.spacing
    nx = max(2, int(np.round(dx / spacing)) + 1)
    ny = max(2, int(np.round(dy / spacing)) + 1)
    return _grid_segments(
        cx=cx, cy=cy, cz=cz, dx=dx, dy=dy,
        nx_wires=ny, ny_wires=nx,
        ds=ds, electrode_name=electrode.name,
        wire_radius=electrode.wire_radius,
    )


def _discretize_grid_mesh(
    electrode: GridMeshElectrode, ds: float
) -> list[_Segment]:
    """Mesh earth electrode with explicit n_x × n_y meshes.

    The grid has ``n_x + 1`` transverse and ``n_y + 1`` longitudinal
    wires (one wire per cell boundary plus the outer perimeter).
    """
    cx, cy, cz = electrode.corner
    dx, dy = electrode.size
    return _grid_segments(
        cx=cx, cy=cy, cz=cz, dx=dx, dy=dy,
        nx_wires=electrode.n_y + 1,
        ny_wires=electrode.n_x + 1,
        ds=ds, electrode_name=electrode.name,
        wire_radius=electrode.wire_radius,
    )


def _discretize_electrode(
    electrode: "_ElectrodeBase", ds: float, layer_interfaces=None,
) -> list[_Segment]:
    """Dispatch to the per-primitive discretiser.

    Parameters
    ----------
    layer_interfaces
        Optional sequence of soil-layer interface depths in m
        (e.g. ``(h_1,)`` for a two-layer soil). Primitives whose
        geometry can cross an interface (rods, sloped strips) split
        their segments there so no segment straddles a layer boundary
        (ADR-0007 step 1, audit 2026-07-08 WP-D2). Horizontal
        primitives (rings, meshes) are unaffected.
    """
    if isinstance(electrode, RodElectrode):
        return _discretize_rod(electrode, ds, layer_interfaces)
    if isinstance(electrode, RingElectrode):
        return _discretize_ring(electrode, ds)
    if isinstance(electrode, StripElectrode):
        return _discretize_strip(electrode, ds, layer_interfaces)
    if isinstance(electrode, PolylineElectrode):
        return _discretize_polyline(electrode, ds, layer_interfaces)
    if isinstance(electrode, StarElectrode):
        return _discretize_star(electrode, ds, layer_interfaces)
    if isinstance(electrode, GridMeshElectrode):
        return _discretize_grid_mesh(electrode, ds)
    if isinstance(electrode, MeshElectrode):
        return _discretize_mesh(electrode, ds)
    raise TypeError(
        f"Image backend does not know {type(electrode).__name__}."
    )


# ---------------------------------------------------------------------
# Conductor discretisation (ADR-0003: distributed conductor model)
# ---------------------------------------------------------------------


@dataclass
class _DistributedBranch:
    """One longitudinal-segment branch of a distributed conductor.

    Carries the topology used by the nodal-analysis system
    (``node_a``, ``node_b``, ``R``) plus the geometric information
    required by the inductive-coupling assembly (ADR-0004): the
    branch endpoints in 3-D and the wire radius.
    """

    node_a: str
    node_b: str
    R: float
    p_a: np.ndarray  # shape (3,)
    p_b: np.ndarray  # shape (3,)
    wire_radius: float
    inductive: bool  # True if the parent conductor enables Neumann coupling


def _conductor_node_name(conductor_name: str, k: int) -> str:
    """Pseudo-node identifier for the *k*-th midpoint segment of a
    distributed conductor.

    The naming convention `__cond_{name}__seg_{k}` is reserved — the
    leading double underscore guarantees no collision with user
    electrode names.
    """
    return f"__cond_{conductor_name}__seg_{k}"


def _discretize_conductor(conductor) -> tuple[list[_Segment], list[_DistributedBranch]]:
    """Discretise a conductor into midpoint segments + longitudinal branches.

    Implements the distributed-conductor model documented in ADR-0003.
    The conductor is split into ``n = conductor.n_segments`` collinear
    sub-pieces of equal length. Each sub-piece produces:

    - one pseudo-electrode segment at its midpoint, contributing to
      the multi-port grounding matrix only when
      ``coupling_to_soil == "galvanic"``;
    - one longitudinal branch whose endpoints are the pseudo-node
      midpoints (or the conductor's start / end electrode cluster at
      the conductor ends) and whose resistance is
      $R^{(k)} = \\rho_\\text{mat}\\, \\ell_k / A$.

    Topology
    --------
    For a conductor with ``n`` segments and ``coupling_to_soil ==
    "galvanic"`` the longitudinal-branch chain reads

    ::

        start_cluster ──[R/2]── M_0 ──[R]── M_1 ──[R]── ... ──[R]── M_{n-1} ──[R/2]── end_cluster

    where each ``M_k`` is a pseudo-electrode node (single segment at
    the midpoint of sub-piece *k*). The half-resistance stubs at the
    two ends preserve the total series resistance
    $\\sum R^{(k)} = R_\\text{total}$. For
    ``coupling_to_soil == "isolated"`` the pseudo-electrode list is
    empty; the longitudinal chain becomes a series of
    finite-resistance branches between the two end clusters with
    interior nodes that carry no leakage.

    Parameters
    ----------
    conductor
        :class:`Conductor` instance with finite
        ``discretize_segment_length`` (i.e. ``is_distributed``).

    Returns
    -------
    segments : list[_Segment]
        Pseudo-electrode segments at the conductor midpoints. Empty
        for ``coupling_to_soil == "isolated"``.
    branches : list[(str, str, float)]
        Longitudinal-branch list ``(node_in, node_out, R)``. For an
        isolated conductor the chain runs through anonymous interior
        nodes ``f"__cond_{name}__node_{k}"`` (k=1..n-1).
    """
    n = conductor.n_segments
    L = conductor.length
    if not conductor.is_distributed or n <= 1:
        # Lumped fallback — caller handles via _build_finite_branches.
        return [], []

    seg_len = L / n
    R_total = conductor.series_resistance
    R_seg = R_total / n if R_total > 0.0 else 0.0
    p0 = np.array(conductor.start, dtype=float)
    p1 = np.array(conductor.end, dtype=float)
    direction = (p1 - p0) / L

    # End-cluster identifiers (must exist in the world by construction)
    start_node = conductor.start_electrode
    end_node = conductor.end_electrode
    if start_node is None or end_node is None:
        # Floating endpoints: bind them to anonymous "endpoint" nodes
        # that participate in KCL but not in any cluster.
        if start_node is None:
            start_node = f"__cond_{conductor.name}__endpoint_start"
        if end_node is None:
            end_node = f"__cond_{conductor.name}__endpoint_end"

    segments: list[_Segment] = []
    branches: list[_DistributedBranch] = []
    inductive = getattr(conductor, "inductance_model", None) is not None

    if conductor.coupling_to_soil == "galvanic":
        # Each sub-piece becomes a one-segment pseudo-electrode at its
        # midpoint; longitudinal branches link the consecutive nodes
        # with R/2 stubs at the two ends.
        midpoint_nodes: list[str] = []
        midpoints: list[np.ndarray] = []
        for k in range(n):
            midpoint = p0 + (k + 0.5) * seg_len * direction
            mid_node = _conductor_node_name(conductor.name, k)
            midpoint_nodes.append(mid_node)
            midpoints.append(midpoint)
            segments.append(
                _Segment(
                    midpoint=midpoint,
                    length=seg_len,
                    electrode_name=mid_node,
                    wire_radius=conductor.wire_radius,
                    conductor_name=conductor.name,
                    direction=direction,
                )
            )
        # Branch chain: start_cluster -[R/2]- M_0 -[R]- M_1 -...- M_{n-1} -[R/2]- end_cluster
        # Geometric endpoints of each branch (used for inductance
        # assembly): start at conductor.start, then run between
        # successive midpoints, then to conductor.end.
        branch_endpoints = [(p0, midpoints[0])]
        for k in range(n - 1):
            branch_endpoints.append((midpoints[k], midpoints[k + 1]))
        branch_endpoints.append((midpoints[-1], p1))
        Rs = [R_seg / 2.0] + [R_seg] * (n - 1) + [R_seg / 2.0]
        chain_nodes = [start_node] + midpoint_nodes + [end_node]
        for k, (pa, pb) in enumerate(branch_endpoints):
            branches.append(_DistributedBranch(
                node_a=chain_nodes[k],
                node_b=chain_nodes[k + 1],
                R=Rs[k],
                p_a=pa,
                p_b=pb,
                wire_radius=conductor.wire_radius,
                inductive=inductive,
            ))
    else:
        # ``isolated``: no leakage along the wire. Interior nodes are
        # anonymous; the chain is a series of n branches between the
        # two end clusters. The interior nodes do not appear in the
        # Z-matrix but participate in KCL (every interior node carries
        # zero leakage by construction).
        interior_nodes = [
            f"__cond_{conductor.name}__node_{k}" for k in range(1, n)
        ]
        nodes_chain = [start_node, *interior_nodes, end_node]
        # Geometric endpoints follow the conductor axis at the
        # sub-segment boundaries.
        node_positions = [p0 + (k * seg_len) * direction for k in range(n + 1)]
        for k in range(n):
            branches.append(_DistributedBranch(
                node_a=nodes_chain[k],
                node_b=nodes_chain[k + 1],
                R=R_seg,
                p_a=node_positions[k],
                p_b=node_positions[k + 1],
                wire_radius=conductor.wire_radius,
                inductive=inductive,
            ))

    return segments, branches


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
                (n_total_snapshot, n_total_snapshot), dtype=complex,
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
                (n_total_snapshot, n_total_snapshot), dtype=complex,
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


def _build_distributed_topology(
    conductors,
    cluster_id: dict[str, str],
) -> tuple[list[_Segment], list[_DistributedBranch], set[str]]:
    """Aggregate the per-conductor discretisations into one set.

    Parameters
    ----------
    conductors
        Iterable of :class:`Conductor` instances.
    cluster_id
        Cluster-root mapping from :func:`_build_clusters`. End-cluster
        identifiers in the resulting branch list are translated to
        their cluster roots so that ideal-galvanic shortcuts of the
        end electrodes are honoured automatically.

    Returns
    -------
    seg_list : list[_Segment]
        Conductor pseudo-electrode segments (galvanic case only).
    branches : list[_DistributedBranch]
        Longitudinal branches with cluster-root endpoints, plus the
        geometric data needed for the inductive-coupling assembly
        (per-branch endpoints in 3-D and wire radius). The
        topology-only triple ``(node_a, node_b, R)`` is recovered
        as ``(b.node_a, b.node_b, b.R)``.
    interior_nodes : set[str]
        Pseudo-node identifiers introduced by the distributed
        discretisation (midpoint nodes + isolated interior nodes).
        Used by the solver to know which nodes are *not* derived
        from real electrodes.
    """
    seg_list: list[_Segment] = []
    branches: list[_DistributedBranch] = []
    interior_nodes: set[str] = set()
    for c in conductors:
        if not getattr(c, "is_distributed", False):
            continue
        if c.start_electrode is None or c.end_electrode is None:
            # Floating endpoints — defer until the measurement-lead
            # work; for now the discretiser would still work, but the
            # solver needs an extra cluster row, which we leave out
            # of the first distributed-conductor release.
            raise ValueError(
                f"Distributed conductor '{c.name}' must have both "
                "start_electrode and end_electrode set; floating "
                "endpoints are not yet supported."
            )
        segs, brs = _discretize_conductor(c)
        for s in segs:
            seg_list.append(s)
            interior_nodes.add(s.electrode_name)
        # Translate end-cluster identifiers via cluster_id; interior
        # midpoint / isolated-node identifiers are passed through
        # unchanged.
        for db in brs:
            ra = cluster_id.get(db.node_a, db.node_a)
            rb = cluster_id.get(db.node_b, db.node_b)
            branches.append(_DistributedBranch(
                node_a=ra, node_b=rb,
                R=db.R, p_a=db.p_a, p_b=db.p_b,
                wire_radius=db.wire_radius, inductive=db.inductive,
            ))
            if db.node_a not in cluster_id:
                interior_nodes.add(db.node_a)
            if db.node_b not in cluster_id:
                interior_nodes.add(db.node_b)
    return seg_list, branches, interior_nodes


# ---------------------------------------------------------------------
# Potential evaluation
# ---------------------------------------------------------------------


def _potential_kernel(
    field_points: np.ndarray,  # (M, 3)
    source_points: np.ndarray,  # (N, 3)
    currents: np.ndarray,       # (N,)
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
    seg_points: np.ndarray,    # (N, 3)
    seg_lengths: np.ndarray,   # (N,)
    wire_radii: np.ndarray,    # (N,)
    currents: np.ndarray,      # (N,)
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


# ---------------------------------------------------------------------
# Cluster building and current sharing
# ---------------------------------------------------------------------


def _build_clusters(electrodes, conductors) -> dict[str, str]:
    """Union-find on electrodes: returns the cluster root per name.

    Only **ideal** conductors (``Conductor.is_ideal()`` returns
    ``True``, i.e. ``cross_section is None`` or the resulting series
    resistance is below the ideal-resistance threshold) are treated
    as galvanic shorts and merge their two end electrodes into one
    cluster.

    Finite-impedance conductors are *not* used for clustering — they
    enter the solver later as branches in the nodal-analysis system
    (see :func:`_solve_cluster_currents` and the module docstring).

    Conductors with purely geometric end-points (no
    ``start_electrode`` / ``end_electrode``) are ignored.
    """
    parent = {e.name: e.name for e in electrodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for c in conductors:
        a = getattr(c, "start_electrode", None)
        b = getattr(c, "end_electrode", None)
        if a is None or b is None:
            continue
        if a not in parent or b not in parent:
            continue
        # Only ideal conductors fuse clusters; finite-impedance
        # branches stay separate so that the nodal-analysis solver
        # sees them as proper branches.
        is_ideal = getattr(c, "is_ideal", None)
        if callable(is_ideal) and not is_ideal():
            continue
        union(a, b)

    return {n: find(n) for n in parent}


def _build_finite_branches(
    conductors,
    cluster_id: dict[str, str],
    *,
    distributed_as_lumped: bool = False,
) -> list[tuple[str, str, float]]:
    """Collect lumped finite-impedance branches as ``(node_a, node_b, R)``.

    Each finite, non-distributed conductor between two electrodes
    whose cluster roots differ becomes one branch in the
    nodal-analysis system. The branch resistance is
    :attr:`Conductor.series_resistance`.

    Conductors that are skipped here:

    - **Ideal** conductors — already used by :func:`_build_clusters`
      to merge clusters.
    - **Distributed** conductors (``is_distributed == True``) —
      normally handled by :func:`_build_distributed_topology`, which
      produces a per-segment branch chain. Backends that cannot
      consume the distributed topology (currently the FEM
      equivalent-hemisphere backend) can pass
      ``distributed_as_lumped=True`` to fall back to a single
      lumped branch with the conductor's full series resistance —
      the discretisation is then *ignored*.
    - Conductors whose two end electrodes already share a cluster
      (an ideal short loops over them) — the finite branch would be
      shorted out and carry no net current.
    """
    branches: list[tuple[str, str, float]] = []
    for c in conductors:
        a = getattr(c, "start_electrode", None)
        b = getattr(c, "end_electrode", None)
        if a is None or b is None:
            continue
        if a not in cluster_id or b not in cluster_id:
            continue
        is_ideal = getattr(c, "is_ideal", None)
        if not callable(is_ideal) or is_ideal():
            continue
        if (
            getattr(c, "is_distributed", False)
            and not distributed_as_lumped
            and getattr(c, "n_segments", 1) > 1
        ):
            # Routed through _build_distributed_topology instead.
            # A distributed conductor that degenerates to a single
            # sub-piece (n_segments == 1, i.e.
            # discretize_segment_length >= conductor length) is *not*
            # handled there — _discretize_conductor returns an empty
            # topology for n <= 1 — so it must fall through to the
            # lumped branch here. Previously it silently vanished
            # from the system (open circuit).
            continue
        node_a = cluster_id[a]
        node_b = cluster_id[b]
        if node_a == node_b:
            # Both endpoints already share an ideal cluster — the
            # finite branch is shorted by the ideal connection and
            # carries no net current. Skip it.
            continue
        R = float(getattr(c, "series_resistance", 0.0))
        if R <= 0.0:
            continue
        branches.append((node_a, node_b, R))
    return branches


def _solve_cluster_currents(
    *,
    electrodes,
    elec_input_current: dict[str, complex],
    cluster_id: dict[str, str],
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    elec_to_segidx: dict[str, list[int]],
    self_kernel,
    finite_branches: list[tuple[str, str, float]] | None = None,
    pseudo_owners: list[str] | None = None,
    omega: float = 0.0,
    inductance_matrix: np.ndarray | None = None,
    carson_correction: np.ndarray | None = None,
    shell_coefficients: np.ndarray | None = None,
    multiport_cache: dict | None = None,
) -> dict[str, complex]:
    """Distribute input currents through grounding and finite-conductor branches.

    Solves the augmented nodal-analysis system that couples the
    multi-port grounding matrix $Z$ with the resistive branches
    introduced by finite-impedance conductors.

    Notation
    --------
    Let:

    - $\\mathbf{I}_e \\in \\mathbb{C}^{N_a}$
      — total current leaked into the soil per active electrode,
    - $\\boldsymbol{\\varphi}_n \\in \\mathbb{C}^{K_a}$
      — common potential per active node (= cluster of electrodes
      fused by ideal conductors),
    - $\\mathbf{I}_b \\in \\mathbb{C}^{M_a}$
      — current through each active finite-impedance branch,
    - $C \\in \\{0,1\\}^{N_a \\times K_a}$ the electrode-to-node
      incidence matrix,
    - $B \\in \\{-1,0,+1\\}^{M_a \\times K_a}$ the branch-to-node
      incidence matrix (``+1`` at the branch start, ``-1`` at its
      end),
    - $R_b \\in \\mathbb{R}^{M_a \\times M_a}$ the diagonal matrix
      of branch resistances.

    The system

    $$
    \\begin{bmatrix} Z & -C & 0 \\\\
                    C^{\\top} & 0 & B^{\\top} \\\\
                    0 & B & -R_b \\end{bmatrix}
    \\begin{bmatrix} \\mathbf{I}_e \\\\
                    \\boldsymbol{\\varphi}_n \\\\
                    \\mathbf{I}_b \\end{bmatrix}
    \\;=\\;
    \\begin{bmatrix} \\mathbf{0} \\\\
                    \\mathbf{I}_{\\text{in}} \\\\
                    \\mathbf{0} \\end{bmatrix}
    $$

    expresses, in turn, (i) the multi-port relation
    $Z\\mathbf{I}_e = C\\boldsymbol{\\varphi}_n$ between leakage
    currents and node potentials, (ii) Kirchhoff's current law per
    node $C^{\\top}\\mathbf{I}_e + B^{\\top}\\mathbf{I}_b
    = \\mathbf{I}_{\\text{in}}$, and (iii) Ohm's law along each branch
    $B\\boldsymbol{\\varphi}_n - R_b\\mathbf{I}_b = \\mathbf{0}$
    (i.e. $\\varphi_a - \\varphi_b = R_b\\, I_b$, with positive
    branch direction $a \\to b$). For $M_a = 0$ (no finite
    branches) the system collapses to the classical cluster
    constraint with one common potential per cluster.

    Active set
    ----------
    The function automatically identifies the *active set* of nodes:
    every node that carries an input current, plus every node that is
    transitively connected to an active node by at least one finite
    branch. Passive nodes (with no input current and no finite branch
    to the active region) carry zero current and are excluded from
    the linear system to keep its size minimal.

    Parameters
    ----------
    self_kernel
        Callable with signature
        ``(seg_points, seg_lengths, wire_radii, currents) -> phi``;
        captures the soil-specific self-action (homogeneous, 2-layer,
        n-layer via CIM, ...).
    finite_branches
        List of ``(node_a, node_b, R)`` triples, where ``node_a`` and
        ``node_b`` are *cluster roots* and ``R`` the finite branch
        resistance in Ω. Default ``None`` reproduces the historic
        ideal-cluster behaviour exactly.
    pseudo_owners
        Names of conductor pseudo-nodes (introduced by the
        distributed-conductor model, ADR-0003) that participate in
        the linear system in addition to the real electrodes. Each
        pseudo-owner is treated as its own one-segment cluster:
        ``cluster_id[name] == name`` is expected, and
        ``elec_to_segidx[name]`` must already point at its conductor
        midpoint segment.
    multiport_cache
        Optional dict used to reuse the multi-port grounding matrix
        ``Z`` across calls with identical geometry (ADR-0010 Tier 0a:
        the frequency loop only changes the branch block, never
        ``Z``). Pass one empty dict per solve and share it across the
        per-frequency calls; the first call fills it, subsequent
        calls skip the kernel assembly entirely. ``None`` (default)
        disables caching.
    """
    if finite_branches is None:
        finite_branches = []
    if pseudo_owners is None:
        pseudo_owners = []

    # Cluster-level input currents (sum over electrodes of the cluster).
    cluster_input: dict[str, complex] = {}
    for ename, ic in elec_input_current.items():
        if ic == 0j:
            continue
        cluster_input.setdefault(cluster_id[ename], 0j)
        cluster_input[cluster_id[ename]] += ic

    elec_total = {e.name: 0j for e in electrodes}
    for pn in pseudo_owners:
        elec_total[pn] = 0j
    if not cluster_input and not finite_branches:
        return elec_total

    # ------------------------------------------------------------------
    # Active node set: source nodes plus every node transitively
    # reachable through finite branches.
    # ------------------------------------------------------------------
    active_clusters_set: set[str] = set(cluster_input.keys())
    if finite_branches:
        # Union-find over branch endpoints to propagate activity.
        parent: dict[str, str] = {}

        def find_b(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union_b(a: str, b: str) -> None:
            parent[find_b(a)] = find_b(b)

        for a, b, _R in finite_branches:
            union_b(a, b)
        active_roots = {find_b(c) for c in active_clusters_set}
        for a, b, _R in finite_branches:
            if find_b(a) in active_roots or find_b(b) in active_roots:
                active_clusters_set.add(a)
                active_clusters_set.add(b)

    if not active_clusters_set:
        return elec_total

    active_clusters = sorted(active_clusters_set)
    cluster_idx = {c: k for k, c in enumerate(active_clusters)}
    K_a = len(active_clusters)

    # Owners: real electrodes + conductor pseudo-nodes. Real electrodes
    # come first (so the historic ordering is preserved when there are
    # no pseudo-owners), pseudo-owners are appended in the order in
    # which the discretiser produced them.
    #
    # An owner only enters the Z-matrix block if it has at least one
    # segment in ``elec_to_segidx`` — anonymous interior nodes from an
    # *isolated* distributed conductor have no leakage and contribute
    # to the system only via KCL (Block 2), not via Block 1.
    all_owners = [e.name for e in electrodes] + list(pseudo_owners)
    active_elecs = [
        n for n in all_owners
        if cluster_id[n] in active_clusters_set
        and len(elec_to_segidx.get(n, [])) > 0
    ]
    N_a = len(active_elecs)

    # Active branches: only those entirely inside the active node set
    # (always true here, but kept explicit for safety). We also keep
    # the original index of each active branch so that the optional
    # inductance matrix — built in the same order as ``finite_branches``
    # — can be restricted to the active subset below.
    active_branch_indices: list[int] = []
    active_branches: list[tuple[str, str, float]] = []
    for idx, (a, b, R) in enumerate(finite_branches):
        if a in active_clusters_set and b in active_clusters_set:
            active_branch_indices.append(idx)
            active_branches.append((a, b, R))
    M_a = len(active_branches)

    # ------------------------------------------------------------------
    # Multi-port grounding matrix Z[i, j]: average potential at
    # electrode i for 1 A injected (uniform per unit length) at
    # electrode j.
    #
    # ADR-0010 Tier 1 (audit 2026-07-08, WP-F): all N_a excitation
    # columns are passed to the kernel in ONE call as an
    # ``(n_segments, N_a)`` matrix. The kernel's O(N²) geometry
    # tensors (pairwise differences, norms, image terms) are then
    # built exactly once instead of once per column — previously 97 %
    # of the wall time of AP1-sized solves. With ``multiport_cache``
    # the assembled Z is additionally reused across the per-frequency
    # calls (Z is frequency-independent; only the branch block of the
    # augmented system changes with ω).
    # ------------------------------------------------------------------
    n_segments = seg_points.shape[0]
    # ADR-0012 V2: precompute the per-segment radial shell resistance
    # $R_\text{shell,k} = C_k / \Delta s_k$. The diagonal augmentation
    # ``phi[k] += R_shell_k · I_seg_k`` is applied below, after the
    # bulk-soil ``self_kernel`` call. ``None`` (default) means no
    # shell anywhere — historic behaviour preserved bit-exact.
    shell_diag: np.ndarray | None = None
    if shell_coefficients is not None and np.any(shell_coefficients > 0.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            shell_diag = np.where(
                seg_lengths > 0.0,
                np.asarray(shell_coefficients, dtype=float) / seg_lengths,
                0.0,
            )
    if (
        multiport_cache is not None
        and multiport_cache.get("active_elecs") == active_elecs
    ):
        Z = multiport_cache["Z"]
    else:
        excitation = np.zeros((n_segments, N_a))
        for j, name_j in enumerate(active_elecs):
            idxs_j = elec_to_segidx[name_j]
            L_j = seg_lengths[idxs_j].sum()
            # uniform per unit length, Σ = 1 A
            excitation[idxs_j, j] = seg_lengths[idxs_j] / L_j
        phi_all = self_kernel(
            seg_points, seg_lengths, wire_radii, excitation
        )
        if shell_diag is not None:
            phi_all = phi_all + shell_diag[:, None] * excitation
        # Row reduction: **length-weighted** average of the segment
        # potentials (audit 2026-07-08, WP-B3). This is the Galerkin
        # pairing consistent with the length-weighted excitation
        # columns and restores exact reciprocity Z[i, j] == Z[j, i]
        # at the discrete level. The historic unweighted mean broke
        # symmetry whenever an electrode carried segments of unequal
        # length (mesh electrodes with dx/nx != dy/ny, rods split at
        # the layer interface); for uniform per-electrode segment
        # lengths — every rod/ring/strip — the two averages are
        # identical, so historic results are unchanged there.
        Z = np.empty((N_a, N_a))
        for i, name_i in enumerate(active_elecs):
            idxs_i = elec_to_segidx[name_i]
            w_i = seg_lengths[idxs_i]
            Z[i, :] = (w_i @ phi_all[idxs_i, :]) / w_i.sum()
        if multiport_cache is not None:
            multiport_cache["active_elecs"] = list(active_elecs)
            multiport_cache["Z"] = Z

    # ------------------------------------------------------------------
    # Augmented linear system
    #   [ Z      -C       0    ] [ I_e    ]   [ 0       ]
    #   [ C^T    0        B^T  ] [ phi_n  ] = [ I_in    ]
    #   [ 0      B        R_b  ] [ I_b    ]   [ 0       ]
    # ------------------------------------------------------------------
    n_unknowns = N_a + K_a + M_a
    A = np.zeros((n_unknowns, n_unknowns))

    # Block 1: Z · I_e − C · phi_n = 0
    A[:N_a, :N_a] = Z
    elec_cluster = [cluster_idx[cluster_id[name]] for name in active_elecs]
    for i, k in enumerate(elec_cluster):
        A[i, N_a + k] = -1.0  # −C

    # Block 2: KCL per node:  C^T · I_e  +  B^T · I_b  =  I_in
    for k, c in enumerate(active_clusters):
        for i, name in enumerate(active_elecs):
            if cluster_id[name] == c:
                A[N_a + k, i] = 1.0  # C^T
    for m, (a, b, _R) in enumerate(active_branches):
        ka = cluster_idx[a]
        kb = cluster_idx[b]
        # Branch convention: positive direction a → b. KCL at node a:
        # the branch leaves a with +I_b → contributes +1 to row a.
        # KCL at node b: enters with +I_b → −1 (re-arranged so that
        # +B^T appears on the LHS of  C^T I_e + B^T I_b = I_in).
        A[N_a + ka, N_a + K_a + m] = +1.0
        A[N_a + kb, N_a + K_a + m] = -1.0

    # Block 3: Ohm's law per branch:  phi_a − phi_b = R · I_b
    #   ⇔  +1·phi_a − 1·phi_b − R · I_b = 0
    # (positive branch direction is a → b, so a positive I_b means
    # current leaves node a and enters node b — consistent with the
    # KCL signs above.)
    for m, (a, b, R) in enumerate(active_branches):
        ka = cluster_idx[a]
        kb = cluster_idx[b]
        A[N_a + K_a + m, N_a + ka] = +1.0
        A[N_a + K_a + m, N_a + kb] = -1.0
        A[N_a + K_a + m, N_a + K_a + m] = -R

    # ------------------------------------------------------------------
    # Right-hand side and solve.
    #
    # Two paths:
    #   (a) DC / no inductive coupling — Z and R_b are real, so we
    #       solve the real and imaginary parts of a complex source
    #       independently with the same factorisation. This keeps
    #       the historic fast path unchanged.
    #   (b) Inductive coupling active (``inductance_matrix is not
    #       None`` and ``omega != 0``) — the branch block becomes
    #       complex (Z_b = R + jωL), so we solve one complex linear
    #       system. The off-diagonal mutual-inductance entries M_{ij}
    #       go into the same Block 3 row that already carries the
    #       diagonal R + jωL_self contribution.
    # ------------------------------------------------------------------
    b_re = np.zeros(n_unknowns)
    b_im = np.zeros(n_unknowns)
    for k, c in enumerate(active_clusters):
        ic = cluster_input.get(c, 0j)
        b_re[N_a + k] = ic.real
        b_im[N_a + k] = ic.imag

    use_inductive = (
        inductance_matrix is not None
        and omega != 0.0
        and M_a > 0
    )

    if not use_inductive:
        # Multi-RHS solve: one LU factorisation for both the real and
        # the imaginary part (previously two `solve` calls factorised
        # the same real matrix twice — ADR-0010 Tier 1).
        sol = np.linalg.solve(A, np.column_stack([b_re, b_im]))
        I_active = sol[:N_a, 0] + 1j * sol[:N_a, 1]
    else:
        # Restrict the (full-finite-branches) inductance matrix to the
        # active subset; both ``finite_branches`` and
        # ``inductance_matrix`` must use the same ordering.
        n_total_branches = inductance_matrix.shape[0]
        if inductance_matrix.shape != (n_total_branches, n_total_branches):
            raise ValueError(
                f"inductance_matrix must be square, got "
                f"{inductance_matrix.shape}."
            )
        if n_total_branches != len(finite_branches):
            raise ValueError(
                f"inductance_matrix size {n_total_branches} does not "
                f"match number of finite_branches ({len(finite_branches)})."
            )
        L_active = inductance_matrix[
            np.ix_(active_branch_indices, active_branch_indices)
        ]
        # ADR-0005: optional Carson correction. Pre-restricted to
        # the active subset by the same index permutation as L.
        dZ_carson_active: np.ndarray | None = None
        if carson_correction is not None:
            if carson_correction.shape != inductance_matrix.shape:
                raise ValueError(
                    f"carson_correction shape {carson_correction.shape} "
                    f"must match inductance_matrix {inductance_matrix.shape}."
                )
            dZ_carson_active = carson_correction[
                np.ix_(active_branch_indices, active_branch_indices)
            ]
        A_c = A.astype(complex)
        # Augment Block 3 with the inductive contribution. The
        # diagonal already carries −R; we add −jω·L_self to it and
        # plug −jω·L_{m,m'} into the off-diagonal positions of
        # Block 3 row m, branch-current column m'. With Carson on,
        # the Carson dZ matrix is already the *full* per-pair
        # impedance contribution (per-unit-length value times the
        # parallel-projection length), so it adds with sign −1.
        for m in range(M_a):
            row = N_a + K_a + m
            for m_prime in range(M_a):
                contrib = 1j * omega * L_active[m, m_prime]
                if dZ_carson_active is not None:
                    contrib += dZ_carson_active[m, m_prime]
                if m_prime == m:
                    A_c[row, N_a + K_a + m_prime] -= contrib
                else:
                    A_c[row, N_a + K_a + m_prime] = -contrib
        b_c = b_re + 1j * b_im
        sol = np.linalg.solve(A_c, b_c)
        I_active = sol[:N_a]

    for i, name in enumerate(active_elecs):
        elec_total[name] = complex(I_active[i])

    return elec_total


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_image(world: "World", engine: "Engine") -> FieldResult:
    """Image-charge solver for homogeneous soil.

    Parameters
    ----------
    world
        World whose ``soil`` must be a :class:`HomogeneousSoil`.
    engine
        Engine configuration; relevant fields are ``segment_length``
        and ``frequencies``.
    """
    if not isinstance(world.soil, HomogeneousSoil):
        raise TypeError(
            "Backend 'image' requires HomogeneousSoil. "
            f"Got: {type(world.soil).__name__}. "
            "For layered models pick backend='image_2layer' "
            "(Tagg/Sunde) or backend='mom' (planned)."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")

    rho = world.soil.resistivity
    ds = engine.segment_length

    # 1) Discretisation of the electrodes
    all_segments: list[_Segment] = []
    elec_to_segidx: dict[str, list[int]] = {}
    for e in world.electrodes:
        segs = _discretize_electrode(e, ds)
        elec_to_segidx[e.name] = list(range(len(all_segments),
                                            len(all_segments) + len(segs)))
        all_segments.extend(segs)

    # 2) Per-electrode input currents from the configured sources
    elec_input_current: dict[str, complex] = {
        e.name: 0j for e in world.electrodes
    }
    n_ignored_sources = 0
    for src in world.sources:
        if src.kind != "current":
            # Voltage sources are not supported by this backend.
            n_ignored_sources += 1
            continue
        i_complex = src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg))
        if src.attached_to in elec_input_current:
            elec_input_current[src.attached_to] += i_complex
    if n_ignored_sources:
        warnings.warn(
            f"solve_image: {n_ignored_sources} non-current source(s) "
            "ignored — only CurrentSource is supported. A world driven "
            "solely by voltage sources solves to all-zero.",
            UserWarning,
            stacklevel=2,
        )

    # 3) Cluster building: electrodes joined by an *ideal* conductor
    #    share a common potential. Conductors with a finite series
    #    resistance enter the linear system as branches of the
    #    nodal-analysis solver instead.
    cluster_id = _build_clusters(world.electrodes, world.conductors)
    finite_branches = _build_finite_branches(world.conductors, cluster_id)

    # 3b) Distributed-conductor topology (ADR-0003).
    #     Conductors with a finite ``discretize_segment_length`` are
    #     split into sub-pieces; ``coupling_to_soil="galvanic"`` adds
    #     midpoint pseudo-electrode segments to the Z-matrix, while
    #     the longitudinal-segment chain is appended to the branch
    #     list. Pseudo-nodes get one-element entries in
    #     ``elec_to_segidx`` and are flagged as their own clusters.
    cond_segs, distributed_branches_objs, interior_nodes = _build_distributed_topology(
        world.conductors, cluster_id
    )
    pseudo_owners: list[str] = []
    for s in cond_segs:
        pn = s.electrode_name
        elec_to_segidx[pn] = [len(all_segments)]
        all_segments.append(s)
        cluster_id[pn] = pn
        pseudo_owners.append(pn)
    # Anonymous interior nodes from isolated distributed conductors
    # (no leakage segment, only KCL participation) are also flagged
    # as standalone clusters.
    for n in interior_nodes:
        if n not in cluster_id:
            cluster_id[n] = n
            pseudo_owners.append(n)
            elec_to_segidx[n] = []  # no segment, no leakage
    n_lumped_branches = len(finite_branches)
    distributed_branch_tuples = [
        (db.node_a, db.node_b, db.R) for db in distributed_branches_objs
    ]
    finite_branches = list(finite_branches) + distributed_branch_tuples

    # ADR-0004 + ADR-0005: assemble the partial-inductance matrix
    # for the active distributed-conductor branches (lumped branches
    # stay purely resistive and contribute zero entries). When
    # ``engine.earth_inductive_model == "carson_series"`` we also
    # receive a closure that builds dZ_carson(omega) per frequency.
    earth_inductive_model = getattr(
        engine, "earth_inductive_model", "perfect_mirror"
    )
    sigma_earth_for_carson: float | None = None
    layered_earth_for_sommerfeld: object = None
    if earth_inductive_model == "carson_series":
        from groundfield.coupling import resolve_earth_conductivity

        sigma_earth_for_carson = resolve_earth_conductivity(world.soil)
    elif earth_inductive_model == "sommerfeld":
        from groundfield.coupling import resolve_earth_layers

        layered_earth_for_sommerfeld = resolve_earth_layers(world.soil)
    inductance_matrix_full, has_inductance, carson_builder = _assemble_inductance_matrix(
        distributed_branches_objs,
        n_lumped_branches=n_lumped_branches,
        n_total_branches=len(finite_branches),
        earth_model=earth_inductive_model,
        sigma_earth=sigma_earth_for_carson,
        layered_earth=layered_earth_for_sommerfeld,
    )

    n_segments = len(all_segments)
    seg_points = np.array([s.midpoint for s in all_segments])  # (N, 3)
    seg_lengths = np.array([s.length for s in all_segments])    # (N,)

    # 3c) Burial-depth validation (audit 2026-07-28, F26/F29): the
    #     galvanic image construction needs a finite image separation
    #     2 z >> a. Reject surface-laid / airborne conductors and warn
    #     for the shallow regime where the point-image stand-in biases
    #     the diagonal.
    _check_segment_depths(all_segments, where="solve_image")

    # 4) Current sharing within clusters via the multi-port matrix
    wire_radii = np.array([s.wire_radius for s in all_segments])
    # ADR-0012 V2: per-segment concrete-shell coefficient (zero for
    # every segment that does not belong to a concrete-encased
    # foundation electrode — historic case).
    seg_shell_coeffs = np.array(
        [s.concrete_shell_coefficient_ohm_m for s in all_segments],
        dtype=float,
    )

    def _homogeneous_self(seg_pts, seg_lens, wr, currents):
        """Closure: homogeneous self-action with fixed rho."""
        return _self_corrected_kernel(seg_pts, seg_lens, wr, currents, rho)

    n_freq = len(engine.frequencies)
    omegas = [2.0 * np.pi * float(f) for f in engine.frequencies]
    real_electrode_names = {e.name for e in world.electrodes}

    # ADR-0010 Tier 1 (WP-F): the multi-port grounding matrix Z is
    # frequency-independent — share it across the per-frequency calls.
    _mp_cache: dict = {}

    def _solve_at(omega: float) -> tuple[dict[str, complex], np.ndarray]:
        """Solve once at a given angular frequency.

        Returns
        -------
        elec_total : dict
            Per-owner total leakage current.
        seg_currents : np.ndarray
            Per-segment current distribution (uniform per unit length).
        """
        carson_dz = (
            carson_builder(omega) if (has_inductance and carson_builder is not None)
            else None
        )
        elec_total = _solve_cluster_currents(
            electrodes=world.electrodes,
            elec_input_current=elec_input_current,
            cluster_id=cluster_id,
            seg_points=seg_points,
            seg_lengths=seg_lengths,
            wire_radii=wire_radii,
            elec_to_segidx=elec_to_segidx,
            self_kernel=_homogeneous_self,
            finite_branches=finite_branches,
            pseudo_owners=pseudo_owners,
            omega=omega if has_inductance else 0.0,
            inductance_matrix=inductance_matrix_full if has_inductance else None,
            carson_correction=carson_dz,
            shell_coefficients=seg_shell_coeffs,
            multiport_cache=_mp_cache,
        )
        sc = np.zeros(n_segments, dtype=complex)
        for ename, idxs in elec_to_segidx.items():
            if not idxs:
                continue
            I_total = elec_total.get(ename, 0j)
            if I_total == 0j:
                continue
            L_total = seg_lengths[idxs].sum()
            sc[idxs] = I_total * seg_lengths[idxs] / L_total
        return elec_total, sc

    def _phi_batch(sc_list: list[np.ndarray]) -> list[np.ndarray]:
        """Segment-midpoint potentials for a list of current vectors.

        All real/imaginary parts are stacked into one
        ``(n_segments, 2·len(sc_list))`` excitation matrix so the
        kernel's O(N²) geometry tensors are built exactly once for
        the whole frequency set (ADR-0010 Tier 1).
        """
        k = len(sc_list)
        stacked = np.zeros((n_segments, 2 * k))
        for m, sc in enumerate(sc_list):
            stacked[:, m] = sc.real
            stacked[:, k + m] = sc.imag
        if not stacked.any():
            return [np.zeros(n_segments, dtype=complex)] * k
        phi = _self_corrected_kernel(
            seg_points, seg_lengths, wire_radii, stacked, rho,
        )
        # ADR-0012 V2: same shell augmentation as in
        # _solve_cluster_currents so electrode_potentials /
        # cluster_impedance reflect the concrete-shell drop.
        if np.any(seg_shell_coeffs > 0.0):
            with np.errstate(divide="ignore", invalid="ignore"):
                shell_diag = np.where(
                    seg_lengths > 0.0,
                    seg_shell_coeffs / seg_lengths,
                    0.0,
                )
            phi = phi + shell_diag[:, None] * stacked
        return [phi[:, m] + 1j * phi[:, k + m] for m in range(k)]

    # Frequency loop. With no inductive coupling the system is
    # frequency-independent, so we solve once and replicate.
    elec_per_freq: list[dict[str, complex]] = []
    sc_per_freq: list[np.ndarray] = []
    phi_per_freq: list[np.ndarray] = []
    if has_inductance:
        for omega in omegas:
            et, sc = _solve_at(omega)
            elec_per_freq.append(et)
            sc_per_freq.append(sc)
        phi_per_freq = _phi_batch(sc_per_freq)
    else:
        et, sc = _solve_at(0.0)
        ph = _phi_batch([sc])[0]
        elec_per_freq = [et] * n_freq
        sc_per_freq = [sc] * n_freq
        phi_per_freq = [ph] * n_freq

    # Build the FieldResult mappings.
    electrode_potentials: dict[str, list[complex]] = {}
    electrode_currents: dict[str, list[complex]] = {}
    conductor_currents: dict[str, list[complex]] = {}
    conductor_potentials: dict[str, list[complex]] = {}
    for ename, idxs in elec_to_segidx.items():
        if not idxs:
            continue
        u_list = _weighted_node_potential(
            phi_per_freq, idxs, seg_lengths, n_freq
        )
        i_list = [elec_per_freq[k][ename] for k in range(n_freq)]
        if ename in real_electrode_names:
            electrode_potentials[ename] = u_list
            electrode_currents[ename] = i_list
        else:
            conductor_potentials[ename] = u_list
            conductor_currents[ename] = i_list

    # 7) Point-source list for post-processing (plots, profiles)
    point_sources = [
        PointSource(
            position=tuple(seg_points[i].tolist()),
            current=[complex(sc_per_freq[k][i]) for k in range(n_freq)],
            electrode_name=all_segments[i].electrode_name,
            length=float(seg_lengths[i]),
        )
        for i in range(n_segments)
    ]

    # Cluster map: electrode_name -> sorted list of cluster members
    # (only real electrodes are surfaced).
    cluster_members: dict[str, list[str]] = {}
    for ename in real_electrode_names:
        cluster_members[ename] = sorted(
            n for n in cluster_id
            if cluster_id[n] == cluster_id[ename] and n in real_electrode_names
        )

    metadata: dict = {
        "world_name": world.name,
        "n_segments": n_segments,
        "segment_length": ds,
        "stub": False,
        "earth_inductive_model": earth_inductive_model,
    }
    # ADR-0005 §"Eindringtiefen-Diagnostik": expose the
    # electromagnetic skin depth in soil at every solved frequency,
    # so notebooks and benchmarks can answer "is my geometry small
    # or large compared to delta(omega)?" without re-deriving the
    # formula. Only active for engines that ran a frequency loop.
    if has_inductance and sigma_earth_for_carson is not None:
        from groundfield.coupling.carson import skin_depth

        metadata["penetration_depth"] = {
            float(f): skin_depth(2.0 * np.pi * f, sigma_earth_for_carson)
            for f in engine.frequencies
        }
    elif has_inductance and isinstance(world.soil, HomogeneousSoil):
        # No Carson active, but homogeneous soil — still useful as a
        # *reference* skin depth, even though the perfect-mirror
        # solver does not actually use it.
        from groundfield.coupling.carson import skin_depth

        sigma_ref = 1.0 / float(world.soil.resistivity)
        metadata["penetration_depth"] = {
            float(f): skin_depth(2.0 * np.pi * f, sigma_ref)
            for f in engine.frequencies
        }
    if conductor_currents:
        metadata["conductor_node_currents"] = conductor_currents
        metadata["conductor_node_potentials"] = conductor_potentials

    return FieldResult(
        backend="image",
        frequencies=list(engine.frequencies),
        electrode_potentials=electrode_potentials,
        electrode_currents=electrode_currents,
        point_sources=point_sources,
        soil_resistivity=float(rho),
        soil=world.soil,
        clusters=cluster_members,
        metadata=metadata,
    )
