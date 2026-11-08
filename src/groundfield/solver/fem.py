"""Finite-element backend (``fem``) — experimental.

Status (review pass 10, 10.23)
------------------------------
``fem`` solves the volume PDE for an **equivalent hemisphere** per
cluster, not for the electrodes. In homogeneous soil that reproduces
the Dwight resistance of a single electrode by construction (the
hemisphere radius is chosen from it), so the backend then checks the
:mod:`groundfield.references.dwight1936` formulas, not the integral
engines. When an electrode reaches a layer that its hemisphere does not
reach — a rod through or below the top layer, a wire on (or within its
radius above) an interface below the hemisphere — the reduction injects
the current into the wrong layer (+1092 % for a 3 m rod through
$h_1$ = 2 m; −85 % … +3138 % over the measured contrasts; −18 % …
+1008 % for a 2 m ring on $h_1$ = 2 m) and the backend raises
:class:`~groundfield.solver.registry.BackendCapabilityError` instead of
returning a number. The same holds for a cluster that lies wholly below
a top layer holding more than half of its hemisphere
($a_\\text{eq}/2 < h_1 \\le a_\\text{eq}$; +51 % … +130 % and
−81 % … −98 % for a 3 m rod, audit round 2), and — user decision Q23
of 2026-11-08 — for a layer that no electrode of the cluster touches,
that conducts better than every layer the electrodes touch and whose top
lies above the larger of the hemisphere radius and the electrodes'
horizontal half-extent: the hemisphere injects the current straight
into a layer it spans and reaches one below it much less than the
electrodes do, and the result is too low, not conservative (−97 % …
+5 % against ``image_2layer`` and ``mom`` for a spanned layer, −78 % …
+1 % for a layer below the hemisphere within the half-extent; −92 % for
a 10 m ring at 0.8 m over $h_1$ = 2 m in 1000/30 Ω·m, −47 % over
$h_1$ = 5 m). The rules are structural: near-equal layers
(100/120 and 120/100 Ω·m, −7 % … +5 %) raise as well. All of them ask
which layers the electrodes touch with one predicate, the contact of the
thin-wire model the integral engines solve: a horizontal wire touches
every layer within its wire radius of its axis, a rod the layers its
axis passes through (an end on an interface does not count; audit
D2A-01, D2A-02). The closed forms of the hemisphere are refused outside
their range as well: a strip deeper than half its length (+0.2 % …
+474 %), a ring deeper than its radius (+0.4 % … −28 %) and any
``MeshElectrode`` (+118 % … +193 %), audit D2A, O-1; electrodes at or
above the soil surface are refused like in every backend.
It is **not** a valid cross-check either when a
cluster has several electrodes (radii summed, −35 % for a 2 m ring with
three 4 m rods in homogeneous soil; −55 % … −75 % for two bonded rods of
which one reaches into a conductive top layer), when the world has
several clusters (each solved alone, transfer impedances 0) or when a
layer interface lies above the deepest of the electrode's lowest point,
the hemisphere radius and the electrode's horizontal half-extent without
falling under the rules above (−1 % … +48 % where the spanned layer
conducts no better than the electrodes', −44 % … −78 % for a rod that
reaches into a conductive top layer by 1 µm … 8 cm, −16 % … +16 % for a
more resistive layer below a 10 m ring within its radius); a
:class:`FemReductionWarning` names every such reason. An interface just
below a deep horizontal electrode but deeper than the hemisphere radius
and the half-extent is not covered by any rule (+34 % for a 2 m ring
0.1 m above $h_1$ = 2 m in 1000/30 Ω·m; see the fem engine page).
Conforming axisymmetric rod and ring models without the reduction are
the planned replacement.

Mathematical / physical model
-----------------------------
The other engines in the family solve the **integral form** of the
quasi-static current-flow problem (image charges, BEM, MoM with a
layered Green's function). This backend instead discretises the
**volume PDE** directly,
$$
-\\nabla \\cdot (\\sigma(\\mathbf{r})\\, \\nabla \\varphi) \\;=\\; q,
\\qquad \\sigma(\\mathbf{r}) = 1/\\rho(\\mathbf{r}),
$$
with Neumann boundary at the soil surface
($\\partial \\varphi / \\partial z = 0$ at $z = 0$,
electrically insulating air) and the far-field decay
$\\varphi \\to 0$ as $|\\mathbf{r}| \\to \\infty$,
truncated to a finite outer radius $r_{\\text{far}}$ that carries
the monopole DtN condition described below.
$q$ is the current source density.

For most reference electrodes that are essentially **axisymmetric**
around their connection point (a single rod, a ring, a hemisphere),
we exploit the symmetry and discretise the problem on
a 2-D $(s, z)$ mesh with cylindrical coordinates. The PDE
becomes
$$
-\\frac{1}{s} \\frac{\\partial}{\\partial s}\\!
\\left(s\\, \\sigma\\, \\frac{\\partial \\varphi}{\\partial s}\\right)
- \\frac{\\partial}{\\partial z}\\!
  \\left(\\sigma\\, \\frac{\\partial \\varphi}{\\partial z}\\right)
\\;=\\; q,
$$
solved on a triangular finite-element mesh with linear hat
functions. The weak form is assembled with a sparse COO-builder; the
linear system is solved with ``scipy.sparse.linalg.spsolve``.

On a straight-sided triangle the hat-function gradients are
constant and $\\int_T s\\, dA = \\bar s_T |T|$ holds exactly, so
the centroid-radius element rule used below is an exact quadrature
of the axisymmetric weak form — not a one-point approximation.

Mesh: boundary-conforming spherical shell
-----------------------------------------
The Dirichlet electrode of this backend is the **equivalent
hemisphere** $r = a_{\\text{eq}}$ (see below). A mesh that does
not resolve that surface geometrically turns the electrode into a
staircase whose shape — and hence whose capacitance — changes
discontinuously with the mesh, so refinement does not converge.
The mesh is therefore built in **spherical shell coordinates**
centred on the electrode,
$$
s = r \\sin\\vartheta, \\qquad z = r \\cos\\vartheta,
\\qquad r \\in [a_{\\text{eq}}, r_{\\text{far}}],
\\quad \\vartheta \\in [0, \\pi/2],
$$
with a geometric ladder in $r$ and a uniform ladder in
$\\vartheta$. Consequences:

- The inner boundary $r = a_{\\text{eq}}$ is an exact mesh line,
  i.e. the discrete electrode **is** the intended hemisphere for
  every resolution (a *conforming* boundary; no staircase, no
  cut cells, no immersed-boundary weights). Strictly, it is the
  polyline inscribed in that hemisphere, refined together with the
  angular resolution; a layer interface shallower than
  $a_{\\text{eq}}$ splits one of its chords and the inserted node
  joins the Dirichlet set, so the electrode stays closed (see
  :func:`_cut_triangles_at_depth`).
- $\\vartheta = 0$ is the downward symmetry axis and
  $\\vartheta = \\pi/2$ is the soil surface, so the natural
  (Neumann) boundary condition sits exactly on $z = 0$.
- Element aspect ratios of the *uncut* shell are $O(1)$ and
  scale-invariant: the radial step is
  $r \\ln(r_{\\text{far}}/a_{\\text{eq}}) / n_{\\text{radial}}$
  and the transverse step is $r \\pi / (2 n_{\\text{axial}})$,
  both proportional to $r$ (measured shape quality
  $\\max \\sum \\ell^2 / |T| = 10.7$ at the default resolution,
  against 2.31 for an equilateral triangle). The interface cut is
  the exception: where a layer boundary grazes a node it does
  produce slivers. Over 66 adversarial grazing depths the worst
  cut element reached $\\sum \\ell^2 / |T| = 1.1 \\cdot 10^5$ at
  an area of $6.7 \\cdot 10^{-12}\\,\\text{m}^2$. Because the
  plane is snapped onto nodes within a relative tolerance of
  $10^{-6}$ and truly degenerate elements carry no volume and are
  skipped in the assembly, the effect on the answer is negligible:
  at fixed $r_{\\text{far}}$ the same sweep stays inside
  $-0.23837 \\ldots -0.23816\\,\\%$ (a spread of
  $2 \\cdot 10^{-4}$ percentage points around the uncut
  $-0.23939\\,\\%$), with no zero-area element produced. So the
  guarantee is *bounded* quality away from the cut, not $O(1)$
  quality everywhere.
- The near-electrode resolution is decoupled from the outer
  truncation. At a fixed node count it would still depend on
  $r_{\\text{far}}$ through $\\ln r_{\\text{far}}$ (−1.6 % at
  $h_1$ = 10 km for $\\rho_2 = \\rho_1$ up to 0.15.0); since review pass
  10 (10.49) the number of radial lines grows with
  $\\ln(r_{\\text{far}} / a_{\\text{eq}})$ whenever a layer stack
  deeper than 1 m stretches the domain beyond the homogeneous one, so
  the radial step ratio, and with it the resolution, stays that of the
  homogeneous mesh (−0.0006 % at $h_1$ = 10 km).
- Refining $(n_{\\text{radial}}, n_{\\text{axial}})$ by
  $n \\mapsto 2n - 1$ produces *nested* FE spaces,
  $V_h \\subset V_{h/2} \\subset V$. The conductance is the
  **minimum** of the energy over the admissible set (Dirichlet
  principle), so a larger space can only lower that minimum:
  $G_h \\ge G_{h/2} \\ge G$ and therefore
  $R_h = 1/G_h$ **increases monotonically towards the exact
  value from below** — the convergence guarantee this backend
  previously lacked. Measured on the reference rod
  ($n \\mapsto 2n-1$ from 29 × 19):
  $G_h = 0.015640831 \\to 0.015516853 \\to 0.015485931 \\to
  0.015478205$ against $G = 0.0154757$, i.e.
  $R_h = 61.95 \\to 64.62\\,\\Omega$.

Layer model
-----------
Layer boundaries enter through the piecewise-constant conductivity
$\\sigma(z)$. A spherical shell mesh is not aligned with the
horizontal interfaces $z = \\sum_{i \\le k} h_i$, so every element
crossed by an interface is **cut along the interface** at mesh-build
time (a triangle split into two or three sub-triangles, with the cut
nodes shared between neighbours). The conductivity jump is therefore
mesh-aligned as well and each element carries a single conductivity,
the one of the layer its centroid sits in. The PDE handles arbitrary
horizontally stratified soils (any ``n``).

Far field: monopole DtN (Robin) boundary
----------------------------------------
Truncating the domain with $\\varphi = 0$ at $r_{\\text{far}}$
short-circuits the remaining half-space and removes the
$\\rho / (2 \\pi r_{\\text{far}})$ tail of the resistance — a
one-sided error of order $a_{\\text{eq}} / r_{\\text{far}}$. Since
the leading far-field term of *any* grounding electrode (also in a
layered soil, where only the amplitude changes) is the monopole
$\\varphi \\propto 1/r$, its exact Dirichlet-to-Neumann map on a
sphere is the Robin condition
$$
\\partial \\varphi / \\partial r = -\\varphi / r_{\\text{far}}
\\quad \\Longrightarrow \\quad
\\int_\\Omega \\sigma \\nabla \\varphi \\cdot \\nabla v \\, dV
+ \\oint_{r_{\\text{far}}} \\frac{\\sigma}{r_{\\text{far}}}
  \\varphi\\, v \\, dS = 0 .
$$
This is exact for the homogeneous half-space (where
$\\varphi = a_{\\text{eq}}/r$ satisfies it identically), so the
truncation error drops from $O(a_{\\text{eq}}/r_{\\text{far}})$ to
the multipole residual $O((a_{\\text{eq}}/r_{\\text{far}})^3)$.
The conductance is then the full bilinear form,
$G = 1/R = a(\\varphi, \\varphi)$ = dissipated power inside the
truncation sphere plus the power carried through it.

Scope
-----
- **Geometry coverage.** The axisymmetric formulation captures
  :class:`RodElectrode`, :class:`RingElectrode`,
  :class:`StripElectrode`, :class:`GridMeshElectrode` and
  :class:`MeshElectrode` *as effective hemispheres* — the
  equivalent-hemisphere radius is computed from the electrode's
  geometric parameters before the FEM run: an electrode is replaced by
  the hemisphere of radius $a_{\\text{eq}}$ that has the same DC
  resistance in homogeneous soil of the top-layer resistivity (Dwight
  1936; Sverak's formula for grids). In homogeneous soil the replacement
  therefore reproduces the Dwight value; in layered soil the hemisphere
  extends to the depth $a_{\\text{eq}}$, not to the electrode's depth,
  and the result describes a different geometry. Star and polyline
  electrodes are not reducible and raise; a driven ``MeshElectrode``
  raises as well (its formula, one strip along the diagonal, reads
  +118 % … +193 %; audit D2A, O-1), and so do a strip deeper than half
  its length and a ring deeper than its radius (see
  :func:`equivalent_hemisphere_radius`).
- **Multi-electrode.** The hemisphere radii of a cluster's electrodes
  are summed, i.e. the electrodes are put in parallel without their
  mutual resistance; the result is a lower bound of the cluster
  resistance, and the per-electrode current split ($\\propto$ radius,
  top-layer resistivity only) ignores the soil.
- **Several clusters.** Each cluster is solved alone; finite branches
  between clusters enter a nodal analysis on the cluster level, but the
  clusters see no mutual resistance and passive clusters report 0 V.
- **Frequency.** Quasi-static, frequency-independent.

What the discretisation guarantees: for homogeneous soil the solver
reproduces $R = \\rho / (2 \\pi a_{\\text{eq}})$ to $< 1\\,\\%$ at
the default resolution and converges to it at second order. What it
does not: a statement about the electrodes the hemisphere replaces
(see *Status* above).

References
----------
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission
  Systems*, Dover, ch. 2.1.
- Dwight, H. B. (1936). Calculation of resistances to ground.
- Güemes, J. A., & Hernando, F. E. (2004). Method for calculating
  the ground resistance of grounding grids using FEM. *IEEE PWRD*
  19(2).
- Givoli, D. (1992). *Numerical Methods for Problems in Infinite
  Domains*, Elsevier — Dirichlet-to-Neumann (DtN) truncation
  boundaries; the monopole DtN map used here is its lowest mode.
- Strang, G., & Fix, G. (2008). *An Analysis of the Finite Element
  Method*, 2nd ed. — the Dirichlet/minimum-energy principle behind
  the one-sided convergence of $R_h$.
"""

from __future__ import annotations

import math
import re
import warnings
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from groundfield.geometry.electrodes import (
    GridMeshElectrode,
    MeshElectrode,
    RingElectrode,
    RodElectrode,
    StripElectrode,
)
from groundfield.references import dwight1936 as dw
from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    TwoLayerSoil,
)
from groundfield.solver._layered import as_layer_stack
from groundfield.solver.image import (
    _build_clusters,
    _build_finite_branches,
    _reject_concrete_shells,
    _warn_ignored_sources,
)
from groundfield.solver.result import FieldResult, PointSource
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.geometry.electrodes import _ElectrodeBase
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = ["FemReductionWarning", "equivalent_hemisphere_radius", "solve_fem"]

_log = get_logger(__name__)


class FemReductionWarning(UserWarning):
    """The equivalent-hemisphere reduction does not represent this world.

    ``fem`` replaces every galvanic cluster by one hemisphere with the
    Dwight resistance of its electrodes in homogeneous top-layer soil
    and solves each cluster alone. Emitted once per solve, listing every
    reason, when the result is therefore not the resistance of the
    actual geometry (review pass 10, 10.23):

    * a cluster with several electrodes (the hemisphere radii are summed,
      i.e. the electrodes' mutual resistance is neglected and the
      resistance is biased low — −35 % for a 2 m ring with three 4 m
      rods in homogeneous soil; −55 % … −75 % for two bonded 3 m rods of
      which one reaches 10 cm … 1 mm into a conductive 30/1000 Ω·m top
      layer, where the cluster touches that layer and Q23 does not apply);
    * several clusters (each is solved alone: no mutual resistance
      between them, transfer impedances are exactly 0, passive clusters
      report 0 V);
    * a layer interface between the electrode's deepest point and the
      hemisphere radius, with the hemisphere reaching below it into a
      layer the electrode does not touch and that conducts no better than
      the electrode's layer (+23 % for a 10 m ring at 0.8 m over a more
      resistive layer at ``h_1`` = 2 m, 30/1000 Ω·m; −1 % … +48 % over
      the measured cases of such a spanned layer, against
      ``image_2layer`` and ``mom``). Over a more conductive layer it
      raises (user decision Q23: −92 % in 1000/30 Ω·m); the opposite case
      — the electrode reaches a layer its hemisphere does not (+1092 % for
      a 3 m rod through ``h_1`` = 2 m, +1008 % for a 2 m ring on
      ``h_1`` = 2 m) — raises as well (review pass 10, 10.23), both with
      :class:`~groundfield.solver.registry.BackendCapabilityError`;
    * a layer interface above the electrode's deepest point that the
      hemisphere, which spans the soil from the surface, crosses as well
      (+22 % for a 10 m strip and +3 % for a ring r = 5 m, both at
      0.8 m below a more resistive top layer of ``h_1`` = 0.5 m,
      1000/30 Ω·m; audit round 1, F01). It raises instead when the
      cluster lies wholly below such an interface at a depth above
      ``a_eq/2`` — more than half of the hemisphere in a layer the
      electrodes do not touch (audit round 2, A2E-2) — and when the top
      layer the electrodes do not touch conducts better than theirs (user
      decision Q23: −71 % and −48 % for the same strip and ring under
      30/1000 Ω·m). It only warns where an electrode touches the
      conductive layer partly: a rod reaching 1 µm … 8 cm into a
      conductive top layer of 0.2 ``a_eq`` (−44 % … −78 %; whether a
      share-based contact rule should raise is open, audit D2A-01) or a
      wire within its radius of the interface (−22 % … +23 %);
    * a layer interface below the electrode and its hemisphere but
      within the electrode's horizontal half-extent (ring radius, half
      the strip length or grid diagonal) that separates a layer
      conducting no better than the electrodes': the electrode's current
      reaches it, the compact hemisphere's much less (+9 % for a ring
      r = 10 m at 0.8 m over a more resistive layer at ``h_1`` = 5 m in
      30/1000 Ω·m; −16 % … +16 % over 30/1000, 100/1000 and 30/10000 Ω·m
      at ``h_1`` = ``a_eq`` … 10 m). Over a more conductive layer it
      raises since audit D2A-03 (−47 %, −20 % and −11 % at ``h_1`` = 5,
      7 and 9 m in 1000/30 Ω·m: Q23 extended to the half-extent, a
      coordinator decision flagged for the author's review); beyond the
      half-extent the rule is silent (−8.4 % at ``h_1`` = 10 m, −1.3 % at
      20 m against ``image_2layer``).

    The layer rules are structural, not error estimates: a vanishing,
    more resistive top layer (``h_1`` = 4 mm, +1.6 %) is reported as well.
    Where none applies, rings and strips measured in 1000/30, 100/1000 and
    100/400 Ω·m deviate from ``mom`` by −7.4 … +4.3 % and a
    20 m × 20 m grid by +8.3 … +13.4 %, of which +12.1 % is the error of
    its formula in homogeneous soil (ring, strip, rod: +0.8 … +1.6 %).

    Escalate with ``warnings.simplefilter("error", FemReductionWarning)``
    when ``fem`` serves as a cross-check.
    """


def _reject_electrodes_out_of_the_soil(world: "World") -> None:
    """Refuse electrodes at or above the soil surface, like every backend.

    The integral backends refuse a leakage segment whose mid-point lies at
    or above the surface, in the surface plane (``2 z <= wire_radius``: the
    mirror image lies inside the wire) or with a part in the air
    (:func:`groundfield.solver._guards._check_segment_depths`). ``fem``
    solved such worlds: its hemisphere formulas clamp the depth at 1 mm,
    so a ring 0.5 m in the air read like a ring 1 mm deep (audit D2A-06).
    The same three conditions, on the electrodes (every one, driven or
    passive, as the integral backends discretise every one): a rod whose
    head lies above the surface (to round-off — a head at ``z = 0`` is the
    ordinary driven rod), a horizontal electrode at ``z <= 0`` or with
    ``2 z <= wire_radius``.

    Raises
    ------
    ValueError
        For the first such electrode (no other backend solves the world
        either, so not a :class:`BackendCapabilityError`).
    """
    from groundfield.solver._guards import _SURFACE_TOL

    for e in world.electrodes:
        lower, upper = e.bounds()
        z_top, z_bottom = float(lower[2]), float(upper[2])
        a = float(e.wire_radius)
        if z_bottom > z_top:  # a rod: its head is its shallowest point
            if z_top >= -_SURFACE_TOL:
                continue
            what = f"reaches above the soil surface (its head is at z = {z_top:.4g} m)"
        elif z_top <= 0.0:
            what = f"lies at or above the soil surface (z = {z_top:.4g} m)"
        elif 2.0 * z_top <= a:
            what = (
                f"lies in the soil-surface plane (z = {z_top:.4g} m <= "
                f"wire_radius/2 = {0.5 * a:.4g} m), partly in the air"
            )
        else:
            continue
        raise ValueError(
            f"Backend 'fem': electrode '{e.name}' {what}. Every backend refuses "
            "an electrode out of the soil: z is a depth, positive downwards. "
            "Bury it (e.g. at z = 0.5 ... 0.8 m), or start a rod at the "
            "surface (z = 0), which is the ordinary driven rod."
        )


def _depth_range(electrode: "_ElectrodeBase") -> tuple[float, float]:
    """Shallowest and deepest point of a fem-reducible electrode, in m."""
    if isinstance(electrode, RodElectrode):
        z0 = float(electrode.position[2])
        return z0, z0 + float(electrode.length)
    if isinstance(electrode, RingElectrode):
        z = float(electrode.center[2])
        return z, z
    if isinstance(electrode, StripElectrode):
        za, zb = float(electrode.start[2]), float(electrode.end[2])
        return min(za, zb), max(za, zb)
    z = float(electrode.corner[2])  # GridMeshElectrode, MeshElectrode
    return z, z


def _horizontal_half_extent(electrode: _ElectrodeBase) -> float:
    """Horizontal half-extent of a fem-reducible electrode, in m (rod: 0)."""
    if isinstance(electrode, RodElectrode):
        return 0.0
    if isinstance(electrode, RingElectrode):
        return float(electrode.radius)
    if isinstance(electrode, StripElectrode):
        (xa, ya, _), (xb, yb, _) = electrode.start, electrode.end
        return 0.5 * math.hypot(xb - xa, yb - ya)
    dx, dy = electrode.size  # GridMeshElectrode, MeshElectrode
    return 0.5 * math.hypot(dx, dy)


def _cluster_label(clusters: dict[str, list[str]], root: str) -> str:
    """``cluster {'a', 'b', ...}`` with up to four member names."""
    names = sorted(clusters[root])
    shown = ", ".join(f"'{n}'" for n in names[:4])
    return f"cluster {{{shown}{', ...' if len(names) > 4 else ''}}}"


def _reduction_issues(
    world: "World",
    clusters: dict[str, list[str]],
    active_clusters: list[str],
    a_eq_per_cluster: dict[str, float],
) -> list[str]:
    """Reasons why the hemisphere reduction misrepresents ``world``.

    Every canonical interface above the deepest of the electrode's lowest
    point ``z_deep``, the hemisphere radius ``a_eq`` (both inclusive) and
    the horizontal half-extent (exclusive) is a reason, except those
    between ``a_eq`` and ``z_deep``, which raise
    (:func:`_unreachable_layers`). Interfaces are taken from the
    canonical layer stack (10.48), so an interface between layers of
    equal resistivity is no reason. Worlds that reach this check passed
    :func:`_unreachable_layers`: a layer reported here is one the
    electrodes touch or one that conducts no better than the layers they
    touch (user decision Q23); within the horizontal half-extent (the
    "wide" reason) that holds for every reported layer since audit
    D2A-03, a more conductive one raises there as well.
    """

    def label(root: str) -> str:
        return _cluster_label(clusters, root)

    reasons: list[str] = []
    for root in active_clusters:
        members = clusters[root]
        if len(members) > 1:
            reasons.append(
                f"{label(root)}: {len(members)} electrodes are reduced to "
                "one hemisphere whose radius is the sum of their radii, i.e. "
                "their mutual resistance is neglected and the resistance is "
                "biased low"
            )
    if len(clusters) > 1 and active_clusters:
        reasons.append(
            f"the world has {len(clusters)} clusters, and each is solved "
            "alone: the mutual resistance between clusters is zero, transfer "
            "impedances are exactly 0 and passive clusters report 0 V"
        )
    interfaces = [float(d) for d in np.cumsum(as_layer_stack(world.soil).h)]

    def depths(values: list[float]) -> str:
        return ", ".join(f"{d:.3g}" for d in values)

    for root in active_clusters:
        a_eq = a_eq_per_cluster[root]
        z_deep = max(_depth_range(world.get_electrode(e))[1] for e in clusters[root])
        # The hemisphere spans the soil from the surface down to a_eq, so
        # every interface above the deeper of z_deep and a_eq separates
        # layers that hemisphere and electrode share differently (audit
        # round 1, F01: only interfaces between z_deep and a_eq counted).
        # Interfaces between a_eq and z_deep raise, see _unreachable_layers.
        below = [d for d in interfaces if z_deep < d <= a_eq]
        above = [d for d in interfaces if 0.0 < d <= z_deep and not a_eq < d < z_deep]
        # An interface deeper than both but within the electrode's
        # horizontal extent is reached by the electrode's current and much
        # less by the compact hemisphere (-46 % for a 10 m ring over
        # h_1 = 5 m, 1000/30 Ohm m; -0.3 % over h_1 = 20 m; eng2 e07).
        r_ext = max(
            _horizontal_half_extent(world.get_electrode(e)) for e in clusters[root]
        )
        wide = [d for d in interfaces if max(z_deep, a_eq) < d < r_ext]
        if below:
            reasons.append(
                f"{label(root)}: the electrode reaches z = {z_deep:.3g} m "
                f"but its equivalent hemisphere has the radius a_eq = "
                f"{a_eq:.3g} m, and the layer interface at h = {depths(below)} m "
                "lies between the two, so the hemisphere injects part of its "
                "current into a layer the electrode does not touch"
            )
        if above:
            reasons.append(
                f"{label(root)}: the layer interface at h = {depths(above)} m "
                f"lies above the electrode's lowest point z = {z_deep:.3g} m, "
                "while the equivalent hemisphere spans the soil from the "
                f"surface down to a_eq = {a_eq:.3g} m, so it shares its current "
                "between the layers differently than the electrode"
            )
        if wide:
            reasons.append(
                f"{label(root)}: the layer interface at h = {depths(wide)} m lies "
                f"below the electrode (z = {z_deep:.3g} m) and its equivalent "
                f"hemisphere (a_eq = {a_eq:.3g} m) but within the electrode's "
                f"horizontal half-extent of {r_ext:.3g} m, so the electrode's "
                "current reaches that interface and the compact hemisphere's "
                "much less"
            )
    return reasons


class _Unreachable(NamedTuple):
    """One cluster the reduction cannot represent (:func:`_unreachable_layers`)."""

    #: ``"wrong_layer"`` (10.23 / A2E-2) or ``"conductive_layer"`` (user
    #: decision Q23).
    kind: str
    #: Description for the error message.
    text: str


def _layer_bounds(stack) -> list[tuple[float, float]]:
    """Top and bottom depth in m of every layer of ``stack`` (last: ``inf``)."""
    depths = [0.0, *(float(d) for d in np.cumsum(stack.h)), math.inf]
    return list(zip(depths[:-1], depths[1:], strict=True))


def _layer_text(top: float, bottom: float, rho: float) -> str:
    """Depth range and resistivity of a layer for a message."""
    span = (
        f"below {top:.3g} m" if math.isinf(bottom) else f"{top:.3g} ... {bottom:.3g} m"
    )
    return f"{span} (rho = {rho:g} Ohm m)"


#: Relative round-off allowance of the contact predicate, in units of the
#: wire radius (as in :func:`groundfield.solver._guards._rod_ends_near_interfaces`).
_CONTACT_ROUND_OFF = 1e-9


def _electrode_layers(
    electrode: "_ElectrodeBase", bounds: list[tuple[float, float]]
) -> set[int]:
    """Indices of the layers ``electrode`` touches: the contact predicate.

    The one predicate of every layer rule of the reduction (audit D2A-01,
    D2A-02), the contact of the thin-wire model the integral engines solve:

    * a **horizontal** electrode (constant depth $z$, wire radius $a$)
      touches every layer that the band $[z - a, z + a]$ meets (closed): a
      wire lying in or within one radius of an interface touches both
      layers, and round-off of an interface depth (``0.1 + 0.2``) does
      not decide;
    * a **non-horizontal** electrode (a rod) touches the layers its axis
      passes through, the open interval between its ends (to round-off):
      the thin-wire model has no end face, so an end lying on an interface
      does not touch the layer beyond it — a rod whose head lies on $h_1$,
      or within one radius below it, does not touch the top layer.

    Electrodes above the soil surface were refused before
    (:func:`_reject_electrodes_out_of_the_soil`).
    """
    z_top, z_bottom = _depth_range(electrode)
    a = float(electrode.wire_radius)
    if z_bottom > z_top:
        eps = _CONTACT_ROUND_OFF * a
        return {
            k
            for k, (top, bottom) in enumerate(bounds)
            if z_top < bottom - eps and z_bottom > top + eps
        }
    reach = a * (1.0 + _CONTACT_ROUND_OFF)
    return {
        k
        for k, (top, bottom) in enumerate(bounds)
        if z_top - reach <= bottom and z_top + reach >= top
    }


def _unreachable_layers(
    world: World,
    clusters: dict[str, list[str]],
    active_clusters: list[str],
    a_eq_per_cluster: dict[str, float],
) -> list[_Unreachable]:
    """Clusters whose electrodes and hemisphere sit in different layers.

    The equivalent hemisphere injects the cluster current from the soil
    surface down to the radius ``a_eq``. Three configurations raise:

    * an electrode touches a layer whose top lies below ``a_eq`` — a rod
      through or below the top layer, a wire on (or within its radius
      above) an interface below the hemisphere — and injects into a layer
      the hemisphere never touches. Against ``mom``, rods through or
      below a 1 … 2.5 m top layer are off by +633 % to +3138 % under a
      conductive lower layer (1000/30 Ω·m) and by −10 % to −85 % over a
      resistive one (100/300, 100/1000 Ω·m); only near-equal layers stay
      within a few percent (100/120 Ω·m: −1.5 % to −4.1 %; review pass
      10, 10.23). A 2 m ring on ``h_1`` = 2 m (``a_eq`` = 1.3 m) is
      +1008 % (1000/30 Ω·m) and −18 % (30/1000 Ω·m) against
      ``image_2layer`` (silent up to audit D2A-02);
    * every electrode of the cluster lies below an interface at depth
      ``h`` with ``a_eq/2 < h <= a_eq``: more than half of the hemisphere
      lies in a layer the electrodes do not touch (audit round 2, A2E-2).
      A 3 m rod (``a_eq`` = 0.44 m) 0.5 m below the surface under a top
      layer of ``h/a_eq`` = 0.55 … 0.8: +51 % … +130 % against
      ``image_2layer`` in 1000/30 and 10000/30 Ω·m, −81 % … −98 % in
      30/1000 and 30/10000 Ω·m. At ``h = a_eq`` the rule joins the first
      one;
    * a layer that no electrode of the cluster touches, that conducts
      better than every layer the electrodes touch and whose top lies
      above ``max(a_eq, r_ext)``, ``r_ext`` the largest horizontal
      half-extent of the cluster's electrodes (user decision Q23,
      2026-11-08; extended from ``a_eq`` to the half-extent by audit
      D2A-03): the hemisphere injects the current straight into a layer
      it spans and reaches one below it much less than the electrodes,
      and the result is too low, i.e. not conservative. Measured against
      ``image_2layer`` and ``mom`` for rods, strips and rings below a more
      conductive top layer of any thickness (from 22 mm) and rings above
      a more conductive lower layer: −97 % … +5 %, the lower the larger
      the contrast and the share of the hemisphere in that layer
      (30/10000 Ω·m: −80 % … −97 %; 30/1000 Ω·m: −35 % … −80 %;
      near-equal layers, 100/120 and 120/100 Ω·m: −7 % … +5 %); for a
      layer below the hemisphere within the half-extent −78 % … +1 %
      (rings, strips), −60 % … +13 % (a 20 m grid, whose formula reads
      +12 % high in homogeneous soil). The fem error is continuous across
      ``a_eq`` (−67.9 % on both sides for a 10 m ring over 1000/30 Ω·m);
      the rule stops at ``r_ext``, where it is −9.8 % … −0.8 % (rings),
      −1.4 % (10 m strip) and +9.9 % (grid). A layer that conducts no
      better keeps the :class:`FemReductionWarning` (−1 % … +48 % when
      the hemisphere spans it, −16 % … +16 % below it within the
      half-extent).

    Which layers an electrode touches is one predicate for all three
    rules, the contact of the thin-wire model of the integral engines
    (:func:`_electrode_layers`; audit D2A-01, D2A-02): a horizontal wire
    touches every layer within its wire radius of its axis, a rod the
    layers its axis passes through, an end on an interface not counting.
    The checks are structural, independent of the size of the contrast.
    Layers are taken from the canonical layer stack (10.48), so a
    declared interface between equal resistivities is none. One reason is
    reported per cluster, in the order above.

    Returns
    -------
    list[_Unreachable]
        One entry per offending cluster (empty if none).
    """
    stack = as_layer_stack(world.soil)
    bounds = _layer_bounds(stack)
    found: list[_Unreachable] = []
    for root in active_clusters:
        a_eq = a_eq_per_cluster[root]
        members = clusters[root]
        # One contact predicate for the three rules (audit D2A-01, D2A-02).
        layers = {e: _electrode_layers(world.get_electrode(e), bounds) for e in members}
        touched = set().union(*layers.values())
        # 10.23: an electrode touches a layer whose top lies below a_eq.
        unreached = {
            e: sorted(bounds[k][0] for k in ks if bounds[k][0] > a_eq)
            for e, ks in layers.items()
        }
        offenders = [e for e in members if unreached[e]]
        if offenders:
            deepest = max(
                offenders, key=lambda e: _depth_range(world.get_electrode(e))[1]
            )
            z_deep = _depth_range(world.get_electrode(deepest))[1]
            crossed = [b[0] for b in bounds[1:] if a_eq < b[0] < z_deep]
            if crossed:  # the interface named up to 563ead7
                where = (
                    f"electrode '{deepest}' reaches z = {z_deep:.3g} m, below the "
                    f"layer interface at h = {crossed[0]:.3g} m"
                )
            else:  # a horizontal wire within its radius above the interface
                h = unreached[deepest][0]
                where = (
                    f"electrode '{deepest}' lies at z = {z_deep:.4g} m, within its "
                    f"wire radius of {world.get_electrode(deepest).wire_radius:.3g} m "
                    f"of the layer interface at h = {h:.4g} m, so it touches the "
                    "layer below it"
                )
            found.append(
                _Unreachable(
                    "wrong_layer",
                    f"{where}, but the equivalent hemisphere of its cluster ends at "
                    f"a_eq = {a_eq:.3g} m above it",
                )
            )
            continue
        # A2E-2: the cluster lies below an interface in (a_eq/2, a_eq], i.e.
        # it touches no layer above that interface.
        z_top = min(_depth_range(world.get_electrode(e))[0] for e in members)
        covering = [
            bounds[k][0]
            for k in range(1, min(touched) + 1)
            if 0.5 * a_eq < bounds[k][0] <= a_eq
        ]
        if covering:
            h = covering[-1]
            names = sorted(clusters[root])
            what = (
                f"electrode '{names[0]}'"
                if len(names) == 1
                else f"every electrode of the cluster of '{names[0]}'"
            )
            found.append(
                _Unreachable(
                    "wrong_layer",
                    f"{what} lies below the layer interface at h = {h:.3g} m "
                    f"(top at z = {z_top:.3g} m), but {100.0 * h / a_eq:.0f} % of "
                    f"its equivalent hemisphere (a_eq = {a_eq:.3g} m) lies above "
                    "it, in a layer the electrodes do not touch",
                )
            )
            continue
        # user decision Q23: a spanned, untouched layer that conducts better
        # than every layer the electrodes touch takes the current the
        # electrodes would send through their own layer(s).
        if stack.n_layers < 2:
            continue
        rho_touched = min(float(stack.rhos[k]) for k in touched)
        # The domain of the "wide" rule as well (audit D2A-03): a layer below
        # the hemisphere but within the electrodes' horizontal half-extent
        # is reached by their current and much less by the hemisphere.
        r_ext = max(_horizontal_half_extent(world.get_electrode(e)) for e in members)
        better = [
            k
            for k, (top, _) in enumerate(bounds)
            if top < max(a_eq, r_ext)
            and k not in touched
            and float(stack.rhos[k]) < rho_touched
        ]
        if better:

            def listed(ks: list[int]) -> str:
                return " and the layer ".join(
                    _layer_text(*bounds[k], float(stack.rhos[k])) for k in ks
                )

            spanned = [k for k in better if bounds[k][0] < a_eq]
            reached = [k for k in better if bounds[k][0] >= a_eq]
            own = ", ".join(
                f"{r:g}" for r in sorted({float(stack.rhos[k]) for k in touched})
            )
            extent = (
                f"within its electrodes' horizontal half-extent of {r_ext:.3g} m, "
                "which their current reaches and the compact hemisphere's much less"
            )
            if not reached:
                what = (
                    f"its equivalent hemisphere (a_eq = {a_eq:.3g} m) spans the "
                    f"layer {listed(spanned)}, which none of its electrodes touches "
                    "and which conducts better than the layer(s) they touch"
                )
            elif not spanned:
                what = (
                    f"the layer {listed(reached)} lies below its equivalent "
                    f"hemisphere (a_eq = {a_eq:.3g} m) but {extent}; none of its "
                    "electrodes touches it and it conducts better than the "
                    "layer(s) they touch"
                )
            else:
                what = (
                    f"its equivalent hemisphere (a_eq = {a_eq:.3g} m) spans the "
                    f"layer {listed(spanned)}, and the layer {listed(reached)} lies "
                    f"below it but {extent}; none of its electrodes touches these "
                    "layers and they conduct better than the layer(s) they touch"
                )
            found.append(
                _Unreachable(
                    "conductive_layer",
                    f"{_cluster_label(clusters, root)}: {what} (rho = {own} Ohm m)",
                )
            )
    return found


#: Measured consequence of the Q23 configurations, for the error message
#: (evidence d2solver/q23_fem_layer_cases_base.out, reproduced in
#: f2solver/q23_spanned_unguarded.out, and f2solver/wide_cases_refs.out:
#: fem against image_2layer and mom, segment length 0.25 m).
_CONDUCTIVE_LAYER_EFFECT = (
    "The hemisphere injects the cluster current straight into a layer it spans "
    "and reaches a layer below it much less than the electrodes do, so fem "
    "reads too low (not conservative): -97 % to +5 % against 'image_2layer' "
    "and 'mom' in the measured cases of a spanned layer (rods, strips and "
    "rings; 30/10000 Ohm m: -80 % to -97 %, 30/1000 Ohm m: -35 % to -80 %, "
    "near-equal layers 100/120 and 120/100 Ohm m: -7 % to +5 %), -78 % to +1 % "
    "for a layer below the hemisphere within the half-extent (rings and "
    "strips; a 20 m grid -60 % to +13 %, its formula reading +12 % high in "
    "homogeneous soil)."
)


def _first_clause(text: str, backend: str) -> str:
    """The first sentence of a backend's error message, without its prefix."""
    if "cannot solve this world: " in text:
        text = text.split("cannot solve this world: ", 1)[1]
    for prefix in (f"{backend}: ", f"solve_{backend}: "):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = re.split(r"\.\s+(?=[A-Z])", text, maxsplit=1)[0].rstrip(".")
    return text if len(text) <= 240 else text[:237] + "..."


#: Prefix of a :func:`_backend_refusal` result that is a refusal by the
#: engine's resource limit, not by the world.
_LIMIT = "limit: "


def _backend_refusal(world: "World", engine: "Engine", backend: str) -> str | None:
    """Why ``backend`` would refuse ``world``, or ``None`` if it accepts it.

    A refusal by the engine's resource limit (``Engine.max_segments``) is
    a setting of the engine, not of the world: it is returned with the
    prefix :data:`_LIMIT`, and the later checks are not run.

    Runs the backend's own checks without solving (audit D2A-06): the
    capability check of the registry, the discretisation with its resource
    limit, the burial-depth guard of every segment (galvanic conductor
    segments included) and the geometric guards of the backend's reaction
    matrix — the point form at the first interface (``mom``,
    ``mom_sommerfeld``, ``image_2layer`` with ``kernel='point'``) and the
    cross-layer refusal of ``mom_sommerfeld`` for three or more layers. It
    mirrors ``prepare`` of those backends (``solver/mom.py``,
    ``solver/mom_sommerfeld.py``, ``solver/image_2layer.py``) and resolves
    the kernel like :meth:`Engine.solve <groundfield.solver.engine.Engine.solve>`.
    """
    from groundfield.diagnostics import ResourceLimitError
    from groundfield.solver._guards import (
        _check_point_interface_diagonal,
        _check_segment_depths,
    )
    from groundfield.solver._kernels import DEFAULT_KERNEL
    from groundfield.solver._pipeline import (
        SegmentArrays,
        build_topology,
        discretise,
    )
    from groundfield.solver.registry import (
        check_capabilities,
        effective_kernel,
        resolve_backend,
    )

    stack = as_layer_stack(world.soil)
    h_1 = float(stack.h[0]) if stack.n_layers >= 2 else None
    requested = engine.kernel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            resolved = resolve_backend(backend, world)
            check_capabilities(
                resolved, world, kernel=requested, kernel_explicit=requested is not None
            )
            kernel = effective_kernel(
                resolved.spec, requested if requested is not None else DEFAULT_KERNEL
            )
            disc = discretise(world, engine, None if h_1 is None else (h_1,))
            build_topology(world, disc)  # adds the galvanic conductor segments
            _check_segment_depths(disc.segments, where=backend, kernel=kernel)
        except ResourceLimitError as exc:
            return _LIMIT + _first_clause(str(exc), backend)
        except (ValueError, NotImplementedError) as exc:
            return _first_clause(str(exc), backend)
        if h_1 is None:
            return None
        arrays = SegmentArrays.from_segments(disc.segments)
        z_max = float(arrays.points[:, 2].max())
        if backend == "mom_sommerfeld" and stack.n_layers >= 3 and z_max >= h_1:
            return (
                f"a segment reaches z = {z_max:.3g} m, at or below the first "
                f"interface at h_1 = {h_1:.3g} m, and it solves three or more "
                "layers only with every segment in the top layer"
            )
        if backend != "image_2layer" or kernel != "line":
            try:
                _check_point_interface_diagonal(
                    disc.segments,
                    arrays.points,
                    arrays.lengths,
                    arrays.directions,
                    h_1,
                    backend=f"'{backend}'",
                    remedy="",
                    cross_layer=z_max >= h_1,
                )
            except (ValueError, NotImplementedError):
                return (
                    "its point form refuses segments closer than 0.387 segment "
                    f"lengths to the interface at h_1 = {h_1:.3g} m at "
                    f"segment_length = {engine.segment_length:g} m"
                )
    return None


def _remedy(world: "World", engine: "Engine") -> str:
    """The backend(s) that solve ``world`` per their own checks (D2A-06)."""
    n_layers = as_layer_stack(world.soil).n_layers
    candidates = {1: ("image", "mom"), 2: ("mom", "image_2layer")}.get(
        n_layers, ("mom_sommerfeld",)
    )
    refusals = {b: _backend_refusal(world, engine, b) for b in candidates}
    limited = {
        b: r[len(_LIMIT) :]
        for b, r in refusals.items()
        if r is not None and r.startswith(_LIMIT)
    }
    accepted = [b for b in candidates if refusals[b] is None or b in limited]
    by_reason: dict[str, list[str]] = {}
    for b, reason in refusals.items():
        if reason is not None and b not in limited:
            by_reason.setdefault(reason, []).append(f"'{b}'")
    refused = " ".join(
        f"{' and '.join(names)} refuse{'s' if len(names) == 1 else ''} this "
        f"world: {reason}."
        for reason, names in by_reason.items()
    )
    if not accepted:
        return f"No backend solves this world yet. {refused}"
    if accepted == ["mom_sommerfeld"]:
        text = (
            "Use backend='mom_sommerfeld', which solves three or more layers "
            "when every segment lies in the top layer, as here."
        )
    elif accepted == ["mom", "image_2layer"]:
        text = "Use backend='mom' (one current per segment) or 'image_2layer'."
    else:
        text = "Use backend=" + " or ".join(f"'{b}'" for b in accepted) + "."
    if limited:
        names = " and ".join(f"'{b}'" for b in limited)
        text += (
            f" This engine's resource limit refuses the world for {names} "
            f"({next(iter(limited.values()))}): raise Engine.max_segments or "
            "coarsen segment_length (their other checks were not run)."
        )
    return f"{text} {refused}" if refused else text


#: Measured consequence of a burial depth beyond the electrode's half-extent
#: (evidence f2solver/depth_validity.out: the formula against the exact
#: uniform-current wire and ring, which ``image`` reproduces to 1e-5).
_DEPTH_RANGE_EFFECT = (
    "The closed forms of the equivalent hemisphere assume a burial depth "
    "small against the electrode: Dwight's series of a wire and its image "
    "converges only above half the wire length, his ring formula needs a depth "
    "below the ring radius (s << D, s twice the depth). Deeper, fem reads "
    "+0.2 % (1.25 times half the length) to +474 % (5.3 times) for strips and "
    "+0.4 % (1.5 times the radius) to -28 % (20 times) for rings against the "
    "exact uniform-current value that 'image' computes."
)

#: Measured consequence of the MeshElectrode formula (evidence
#: f2solver/mesh_formula.out: fem against image and mom at 0.8 m).
_MESH_FORMULA_EFFECT = (
    "Its closed form, one strip along the diagonal, ignores the wires of the "
    "mesh: +118 % to +193 % against 'image' for 10 m and 20 m meshes at 0.8 m "
    "(spacing from the size down to 1 m), at every depth. fem reduces a "
    "GridMeshElectrode with Sverak's formula (-1 % to +21 % for the same "
    "wires)."
)


def _formula_range_issues(
    world: "World", clusters: dict[str, list[str]], active_clusters: list[str]
) -> tuple[list[str], list[str]]:
    """Driven electrodes whose hemisphere formula is used outside its range.

    Audit D2A, O-1: the closed forms of :func:`equivalent_hemisphere_radius`
    were used at any depth (a 1.88 m strip at 5 m read +369 % in homogeneous
    soil, silently). Structural limits, from the formulas:

    * strip: Dwight's wire formula is the series of the wire and its image in
      $x = s/l$ ($s = 2t$ the image distance, $l$ the length) and converges
      for $x < 1$ only — a burial depth $t$ up to half the length (formula
      error $\\le 0.1\\,\\%$ there);
    * ring: Dwight's formula needs $s \\ll D$; up to $t = r$ ($s = D$) its
      error stays within $+1.0\\,\\%$, beyond it the ring reads ever lower;
    * MeshElectrode: one strip along the diagonal, wrong at every depth
      (+118 % … +193 %), so a driven mesh is refused;
    * rod and grid: no depth limit (Dwight's rod ignores the head depth,
      +3.9 % at 0.5 m … +10 % at 10 m; Sverak's grid formula is bounded,
      +6 % … +21 %; both conservative).

    Returns
    -------
    tuple[list[str], list[str]]
        Descriptions of electrodes too deep for their formula, and of
        driven mesh electrodes.
    """
    deep: list[str] = []
    meshes: list[str] = []
    for root in active_clusters:
        for name in clusters[root]:
            e = world.get_electrode(name)
            if isinstance(e, MeshElectrode):
                meshes.append(f"electrode '{name}' is a MeshElectrode")
            elif isinstance(e, StripElectrode):
                t, half = float(e.start[2]), 0.5 * float(e.length)
                if t > half:
                    deep.append(
                        f"electrode '{name}' (strip, {e.length:.3g} m long) lies at "
                        f"z = {t:.3g} m, deeper than half its length ({half:.3g} m)"
                    )
            elif isinstance(e, RingElectrode):
                t, r = float(e.center[2]), float(e.radius)
                if t > r:
                    deep.append(
                        f"electrode '{name}' (ring) lies at z = {t:.3g} m, deeper "
                        f"than its radius ({r:.3g} m)"
                    )
    return deep, meshes


def _unreachable_message(
    world: "World", engine: "Engine", found: list[_Unreachable]
) -> str:
    """Error message for the clusters of :func:`_unreachable_layers`."""
    wrong = [f.text for f in found if f.kind == "wrong_layer"]
    conductive = [f.text for f in found if f.kind == "conductive_layer"]
    parts: list[str] = []
    if wrong:
        parts.append(
            "; ".join(wrong) + ". The equivalent-hemisphere reduction injects "
            "the cluster current from the soil surface down to a_eq, into other "
            "layers than the electrodes (rods through or below the top layer: "
            "-98 % to +3138 % against 'mom' and 'image_2layer' over the "
            "contrasts 10000/30 ... 30/10000 Ohm m; a 2 m ring on an interface "
            "below its hemisphere: -18 % to +1008 % against 'image_2layer')."
        )
    if conductive:
        parts.append("; ".join(conductive) + ". " + _CONDUCTIVE_LAYER_EFFECT)
    return (
        "Backend 'fem' cannot solve this world: "
        + " ".join(parts)
        + " "
        + _remedy(world, engine)
    )


# ---------------------------------------------------------------------
# Equivalent-hemisphere reduction
# ---------------------------------------------------------------------


#: Electrode classes with a closed-form equivalent hemisphere
#: (:func:`equivalent_hemisphere_radius`).
_REDUCIBLE_ELECTRODES = (
    RodElectrode,
    RingElectrode,
    StripElectrode,
    GridMeshElectrode,
    MeshElectrode,
)


def equivalent_hemisphere_radius(electrode: "_ElectrodeBase", rho_top: float) -> float:
    """Equivalent-hemisphere radius giving the same homogeneous-soil
    resistance as ``electrode``.

    Uses the closed-form Dwight 1936 formulas through
    :mod:`groundfield.references.dwight1936`. The hemisphere radius is
    $$
    a_{\\text{eq}} \\;=\\; \\frac{\\rho}{2 \\pi R_{\\text{Dwight}}}.
    $$
    Parameters
    ----------
    electrode
        Single electrode primitive.
    rho_top
        Resistivity used inside the Dwight formula. For layered soil
        the top-layer resistivity is the natural choice — the FEM
        then re-solves the actual layered problem on the equivalent
        hemisphere.

    Returns
    -------
    float
        Hemisphere radius $a_{\\text{eq}}$ in m.

    Notes
    -----
    Range of the closed forms in depth (audit D2A, O-1; evidence
    ``f2solver/depth_validity.out``, against the exact uniform-current wire
    and ring and ``image`` / ``mom``): the strip formula is Dwight's series
    of the wire and its image in $s/l$ ($s = 2t$, $l$ the length), which
    converges for a burial depth $t < l/2$ only (error within
    $\\pm 0.2\\,\\%$ there; +0.2 … +0.4 % at $t = 1.25\\,l/2$, +2.6 … +5.4 %
    at $t = l$, up to +474 % at $t = 5.3\\,l/2$); the ring formula needs
    $s \\ll D$ (+0.05 … +1.0 % up to $t = r$; −0.5 … −0.7 % at $t = 2r$,
    −3.8 … −6.1 % at $4r$, −17 … −28 % at $20r$); the grid formula
    (Sverak) stays bounded, +6 % … +21 %
    from 0.25 m to 20 m depth; the rod formula ignores the head depth
    (+3.9 % at 0.5 m … +10 % at 10 m); the ``MeshElectrode`` formula, one
    strip along the diagonal, misses the wires of the mesh (+118 % …
    +193 % at 0.8 m). :func:`solve_fem` refuses the strip and the ring
    beyond their range and every driven ``MeshElectrode``; this function
    returns the formula's value regardless.
    """
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


# ---------------------------------------------------------------------
# Triangular axisymmetric mesh + assembly
# ---------------------------------------------------------------------


class _AxiMesh(NamedTuple):
    """Boundary-conforming axisymmetric mesh of the soil half-space.

    The mesh covers the spherical shell
    $a_{\\text{eq}} \\le r \\le r_{\\text{far}}$, $z \\ge 0$ of the
    $(s, z)$ half-plane, i.e. the soil outside the equivalent
    hemisphere and inside the truncation sphere.

    Attributes
    ----------
    nodes : np.ndarray, shape (Nv, 2)
        Vertex coordinates $(s, z)$.
    triangles : np.ndarray, shape (Nt, 3), int
        Per-triangle vertex indices.
    layer_index : np.ndarray, shape (Nt,), int
        0-based soil-layer index of every triangle. Interfaces are
        mesh-conforming, so this is exact (no partially filled
        elements).
    inner_nodes : np.ndarray, shape (Ni,), int
        Nodes on the electrode surface $r = a_{\\text{eq}}$. Tracked
        by index rather than by a geometric predicate so that the
        Dirichlet set is exactly the conforming hemisphere. Nodes
        inserted *on* the electrode boundary by an interface cut (a
        layer shallower than $a_{\\text{eq}}$ crosses one chord of
        the electrode polyline) are added to this set by
        :func:`_cut_triangles_at_depth`, so the set stays the
        complete discrete electrode and no free node is left on it.
    outer_edges : np.ndarray, shape (Ne, 2), int
        Boundary edges on the truncation sphere
        $r = r_{\\text{far}}$, in the order needed for the far-field
        surface integral.
    r_far : float
        Truncation radius.
    """

    nodes: np.ndarray
    triangles: np.ndarray
    layer_index: np.ndarray
    inner_nodes: np.ndarray
    outer_edges: np.ndarray
    r_far: float
    #: Radial node lines actually used (``n_radial``, or more when the
    #: truncation sphere is stretched by a deep layer stack, 10.49).
    n_radial: int = 0


def _layer_of_depth(z: np.ndarray, h_layers: list[float]) -> np.ndarray:
    """0-based layer index of the depths ``z`` for a layer stack.

    A depth exactly on an interface is attributed to the layer *below*
    it, matching :class:`~groundfield.solver._layered.LayerStack`
    conventions.
    """
    if not h_layers:
        return np.zeros(np.shape(z), dtype=int)
    cum = np.cumsum(np.asarray(h_layers, dtype=float))
    return np.searchsorted(cum, np.asarray(z, dtype=float), side="right")


def _cut_triangles_at_depth(
    nodes: np.ndarray,
    triangles: np.ndarray,
    outer_edges: list[tuple[int, int]],
    z_cut: float,
    *,
    tol: float,
    inner_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Split every triangle crossed by the horizontal plane $z = z_c$.

    Makes a soil-layer interface an exact mesh line so that the
    conductivity jump never runs through the interior of an element
    (an element straddling the jump would otherwise smear the
    interface over one mesh width — a first-order geometric error).

    The split is *conforming*: the intersection point of an edge is
    keyed on the (ordered) pair of its end nodes, so the two triangles
    sharing that edge insert the same node and no hanging nodes
    appear. Nodes within ``tol`` of the plane are snapped onto it
    first — a per-node decision, hence identical for all triangles
    touching the node, which is what keeps the split free of slivers.

    The Dirichlet (electrode) node set is carried through the split.
    An interface shallower than $a_{\\text{eq}}$ crosses one *chord*
    of the electrode polyline, so a cut node is inserted **on the
    electrode boundary**; leaving it out of the Dirichlet set would
    turn it into a free node, i.e. a small insulating patch with a
    natural zero-flux condition in the middle of the energised
    electrode. A cut node belongs to the electrode exactly when both
    end nodes of the edge it splits do — the criterion is
    topological, not a radius tolerance, because on strongly
    anisotropic meshes ($n_{\\text{radial}}$ small,
    $n_{\\text{axial}}$ large) a *radial* mesh edge leaving the
    electrode can dip below $r = a_{\\text{eq}}$ as well, and its cut
    node must stay free.

    Returns
    -------
    nodes, triangles, outer_edges, inner_nodes
        Updated mesh arrays. Existing node indices are preserved;
        cut nodes are appended. ``outer_edges`` entries split by the
        plane are replaced by their two halves, and ``inner_nodes``
        gains every cut node that lies on the electrode boundary.
    """
    nodes = np.array(nodes, dtype=float, copy=True)
    inner_nodes = np.asarray(inner_nodes, dtype=int)
    close = np.abs(nodes[:, 1] - z_cut) <= tol
    nodes[close, 1] = z_cut
    d = nodes[:, 1] - z_cut

    dv = d[triangles]
    cut_mask = (dv > 0.0).any(axis=1) & (dv < 0.0).any(axis=1)
    if not cut_mask.any():
        return nodes, triangles, outer_edges, inner_nodes

    extra: list[tuple[float, float]] = []
    edge_node: dict[tuple[int, int], int] = {}
    n_base = nodes.shape[0]

    def _split_node(ia: int, ib: int) -> int:
        """Index of the node where edge (ia, ib) crosses the plane."""
        key = (ia, ib) if ia < ib else (ib, ia)
        hit = edge_node.get(key)
        if hit is not None:
            return hit
        ka, kb = key
        t = d[ka] / (d[ka] - d[kb])
        s_new = nodes[ka, 0] + t * (nodes[kb, 0] - nodes[ka, 0])
        idx = n_base + len(extra)
        extra.append((float(s_new), float(z_cut)))
        edge_node[key] = idx
        return idx

    new_tris: list[tuple[int, int, int]] = []
    for t_idx in np.nonzero(cut_mask)[0]:
        tri = triangles[t_idx]
        dd = d[tri]
        on_plane = np.nonzero(dd == 0.0)[0]
        if on_plane.size == 1:
            # One vertex on the plane, the other two on opposite sides:
            # a single cut on the opposite edge → two sub-triangles.
            i0 = int(on_plane[0])
            i1, i2 = (i0 + 1) % 3, (i0 + 2) % 3
            p = _split_node(int(tri[i1]), int(tri[i2]))
            new_tris.append((int(tri[i0]), int(tri[i1]), p))
            new_tris.append((int(tri[i0]), p, int(tri[i2])))
        else:
            # Generic case: one vertex alone on its side of the plane.
            # Triangle + convex quad → three sub-triangles.
            pos = dd > 0.0
            lone = int(np.nonzero(pos if pos.sum() == 1 else ~pos)[0][0])
            i1, i2 = (lone + 1) % 3, (lone + 2) % 3
            p1 = _split_node(int(tri[lone]), int(tri[i1]))
            p2 = _split_node(int(tri[lone]), int(tri[i2]))
            new_tris.append((int(tri[lone]), p1, p2))
            new_tris.append((p1, int(tri[i1]), int(tri[i2])))
            new_tris.append((p1, int(tri[i2]), p2))

    if extra:
        nodes = np.vstack([nodes, np.array(extra, dtype=float)])
    triangles = np.vstack(
        [triangles[~cut_mask], np.array(new_tris, dtype=int).reshape(-1, 3)]
    )

    # A straight plane cuts a straight edge at most once, so every
    # affected boundary edge simply becomes two.
    split_edges: list[tuple[int, int]] = []
    for ia, ib in outer_edges:
        key = (ia, ib) if ia < ib else (ib, ia)
        p_split = edge_node.get(key)
        if p_split is None:
            split_edges.append((ia, ib))
        else:
            split_edges.extend([(ia, p_split), (p_split, ib)])

    # A cut node splitting an edge whose *both* end nodes are on the
    # electrode lies on the electrode boundary itself and must inherit
    # the Dirichlet condition (see the docstring).
    inner_set = set(inner_nodes.tolist())
    on_electrode = [
        p for (ka, kb), p in edge_node.items() if ka in inner_set and kb in inner_set
    ]
    if on_electrode:
        inner_nodes = np.concatenate(
            [inner_nodes, np.array(sorted(on_electrode), dtype=int)]
        )
    return nodes, triangles, split_edges, inner_nodes


def _build_axisymmetric_mesh(
    a_eq: float,
    h_layers: list[float],
    *,
    r_far_factor: float = 30.0,
    z_far_factor: float | None = None,
    n_radial: int = 60,
    n_axial: int = 40,
) -> _AxiMesh:
    """Build the boundary-conforming spherical-shell mesh.

    The mesh discretises the soil between the equivalent hemisphere
    $r = a_{\\text{eq}}$ and the truncation sphere
    $r = r_{\\text{far}}$ in the $(s, z)$ half-plane, using the
    spherical shell coordinates
    $$
    s = r \\sin\\vartheta, \\qquad z = r \\cos\\vartheta,
    \\qquad \\vartheta \\in [0, \\pi/2],
    $$
    with a **geometric** ladder in $r$ (the natural grading for the
    $1/r$ monopole field: equal relative steps ⇒ equal error per
    element) and a **uniform** ladder in $\\vartheta$. Every soil
    layer interface is then cut into the element grid
    (:func:`_cut_triangles_at_depth`).

    Two properties matter physically and both come from the choice of
    coordinates rather than from any tolerance:

    * the electrode surface $r = a_{\\text{eq}}$ is an exact mesh
      line — the discrete electrode is the intended hemisphere at
      every resolution, so refinement converges instead of
      re-shaping the electrode (a staircase disc on an $(s, z)$
      tensor grid does the latter);
    * the near-electrode resolution is $\\propto a_{\\text{eq}}
      \\ln(r_{\\text{far}}/a_{\\text{eq}}) / n_{\\text{radial}}$,
      i.e. only logarithmically sensitive to the outer truncation, so
      declaring a thick soil layer no longer coarsens the electrode.

    Parameters
    ----------
    a_eq
        Equivalent-hemisphere radius; the inner (Dirichlet) boundary.
    h_layers
        Finite layer thicknesses. Every interface depth
        $\\sum_{i \\le k} h_i$ inside the mesh is cut in as an exact
        element boundary, so the conductivity step is mesh-aligned.
    r_far_factor
        Truncation radius as a multiple of the characteristic length
        $\\bar L = a_{\\text{eq}} + \\sum_i h_i$, i.e.
        $r_{\\text{far}} = f_r \\bar L$ (subject to the
        $\\ge 4 a_{\\text{eq}}$ floor below). This is the *only*
        knob that sets the domain size — the truncation of this mesh
        is a single sphere, not a box.
    z_far_factor
        Deprecated axial companion of ``r_far_factor``, kept so that
        callers written against the $(s, z)$ box mesh of v0.14.1 keep
        working. ``None`` (the default) ignores it. A number acts as
        a *lower bound* on the same radius,
        $r_{\\text{far}} = \\max(f_r, f_z)\\,\\bar L$, and a warning
        is logged whenever it actually overrides ``r_far_factor``, so
        the knob can never silently do nothing.
    n_radial
        Number of node lines in $r$ (radial resolution).
    n_axial
        Number of node lines in $\\vartheta$ (transverse resolution,
        from the downward axis to the soil surface). The historical
        name is kept: on the previous $(s, z)$ tensor grid this was
        the axial node count.

    Returns
    -------
    _AxiMesh
        Nodes, triangles, per-triangle layer index, the electrode
        boundary nodes, the truncation-sphere edges and
        $r_{\\text{far}}$.

    Notes
    -----
    Passing ``n_radial → 2 n_radial - 1`` and
    ``n_axial → 2 n_axial - 1`` refines every element into four and
    yields a *nested* finite-element space (for homogeneous soil,
    where no interface cut perturbs the pattern), which is what makes
    the resistance converge monotonically.
    """
    if not np.isfinite(a_eq) or a_eq <= 0.0:
        raise ValueError(f"a_eq must be a positive length, got {a_eq!r}.")
    n_radial = max(int(n_radial), 3)
    n_axial = max(int(n_axial), 3)

    base_length = a_eq + (sum(h_layers) if h_layers else 1.0)
    # ``r_far_factor`` alone sets the truncation sphere. Silently
    # taking max(r_far_factor, z_far_factor) made every
    # ``r_far_factor <= z_far_factor`` a dead knob (with the old
    # defaults 30/20 that was every value <= 20), so a truncation
    # study saw no effect at all.
    factor = float(r_far_factor)
    if z_far_factor is not None and float(z_far_factor) > factor:
        _log.warning(
            "fem: z_far_factor=%g overrides r_far_factor=%g — the "
            "truncation of the spherical-shell mesh is a single radius, "
            "so the larger factor wins. Pass r_far_factor alone (and "
            "leave z_far_factor at None) to control it.",
            float(z_far_factor),
            factor,
        )
        factor = float(z_far_factor)
    r_far = factor * base_length
    # The shell must have room for the graded ladder even if the
    # caller passes a tiny truncation factor.
    r_far = max(r_far, 4.0 * a_eq)

    # Review pass 10, 10.49: a deep layer stack enlarges r_far through
    # ``base_length``; at a fixed node count the geometric ladder then
    # stretches over more decades and the near-electrode resolution
    # falls (rho_2 = rho_1 declared as two layers: -1.6 % at
    # h_1 = 10 km). The radial step ratio is therefore capped at that of
    # the homogeneous domain r_ref = factor * (a_eq + 1 m): the number of
    # radial intervals grows with ln(r_far / a_eq). Homogeneous soil and
    # stacks with sum(h) <= 1 m keep exactly ``n_radial`` lines (and the
    # nested-refinement property documented above).
    r_ref = max(factor * (a_eq + 1.0), 4.0 * a_eq)
    if r_far > r_ref:
        stretch = math.log(r_far / a_eq) / math.log(r_ref / a_eq)
        n_radial = int(math.ceil(1.0 + (n_radial - 1) * stretch))

    r_lines = np.geomspace(a_eq, r_far, n_radial)
    r_lines[0] = a_eq
    r_lines[-1] = r_far
    theta = np.linspace(0.0, 0.5 * np.pi, n_axial)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    # Make the axis (s = 0) and the soil surface (z = 0) exact.
    sin_t[0], cos_t[0] = 0.0, 1.0
    sin_t[-1], cos_t[-1] = 1.0, 0.0

    nodes = np.stack(
        [np.outer(r_lines, sin_t).ravel(), np.outer(r_lines, cos_t).ravel()],
        axis=1,
    )

    # Structured quads → two triangles each (index = i_r * n_axial + k).
    idx = np.arange(n_radial * n_axial).reshape(n_radial, n_axial)
    n00 = idx[:-1, :-1].ravel()
    n10 = idx[1:, :-1].ravel()
    n11 = idx[1:, 1:].ravel()
    n01 = idx[:-1, 1:].ravel()
    triangles = np.concatenate(
        [np.stack([n00, n10, n11], axis=1), np.stack([n00, n11, n01], axis=1)]
    )

    inner_nodes = idx[0, :].copy()
    outer_edges: list[tuple[int, int]] = [
        (int(idx[-1, k]), int(idx[-1, k + 1])) for k in range(n_axial - 1)
    ]

    # Cut the layer interfaces into the shell mesh. ``inner_nodes`` is
    # carried through the cut: an interface shallower than a_eq splits
    # a chord of the electrode polyline and the new node has to join
    # the Dirichlet set, otherwise it stays free and punches an
    # insulating patch into the electrode surface.
    if h_layers:
        cum = 0.0
        for h in h_layers:
            cum += float(h)
            if cum <= 0.0 or cum >= r_far:
                # Interface below the truncation sphere — never
                # crossed inside the mesh.
                continue
            nodes, triangles, outer_edges, inner_nodes = _cut_triangles_at_depth(
                nodes,
                triangles,
                outer_edges,
                cum,
                tol=1e-6 * max(cum, a_eq),
                inner_nodes=inner_nodes,
            )

    layer_index = _layer_of_depth(nodes[triangles, 1].mean(axis=1), h_layers)

    return _AxiMesh(
        nodes=nodes,
        triangles=triangles,
        layer_index=layer_index,
        inner_nodes=np.asarray(inner_nodes, dtype=int),
        outer_edges=np.asarray(outer_edges, dtype=int).reshape(-1, 2),
        r_far=float(r_far),
        n_radial=int(n_radial),
    )


def _assemble_stiffness_axisymmetric(
    nodes: np.ndarray,
    triangles: np.ndarray,
    sigma_per_triangle: np.ndarray,
) -> csr_matrix:
    """Assemble the axisymmetric weak-form stiffness matrix.

    The bilinear form for the cylindrical Laplacian with linear hat
    functions on a 2-D triangle $T$ becomes
    $$
    a_{ij}^T \\;=\\; 2\\pi\\, \\sigma_T \\bar s_T\\,
                     \\bigl(\\nabla \\phi_i \\cdot \\nabla \\phi_j\\bigr)
                     \\, |T|,
    $$
    with $\\bar s_T$ the centroid radius and $|T|$ the
    area of the triangle in the $(s, z)$ plane. The gradients are
    constant on $T$ and $\\int_T s\\, dA = \\bar s_T |T|$ holds
    exactly, so this element rule is an exact quadrature of the
    axisymmetric weak form.

    Degenerate (zero-area) triangles — which the interface cut can
    produce when a layer boundary grazes a node — carry no volume and
    are skipped.
    """
    Nv = nodes.shape[0]
    v = nodes[triangles]  # (Nt, 3, 2)
    x = v[:, :, 0]
    y = v[:, :, 1]
    area2 = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (
        y[:, 1] - y[:, 0]
    )
    area = 0.5 * np.abs(area2)
    keep = area > 0.0
    if not keep.any():
        return coo_matrix(([], ([], [])), shape=(Nv, Nv)).tocsr()
    x, y, area = x[keep], y[keep], area[keep]
    tri = triangles[keep]
    inv = 1.0 / (2.0 * area)
    b = (
        np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
        * inv[:, None]
    )
    c = (
        np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
        * inv[:, None]
    )
    weight = (
        np.asarray(sigma_per_triangle, dtype=float)[keep]
        * area
        * 2.0
        * np.pi
        * x.mean(axis=1)
    )
    K_e = weight[:, None, None] * (
        b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]
    )
    rows = np.broadcast_to(tri[:, :, None], K_e.shape).ravel()
    cols = np.broadcast_to(tri[:, None, :], K_e.shape).ravel()
    K = coo_matrix((K_e.ravel(), (rows, cols)), shape=(Nv, Nv)).tocsr()
    return K


def _assemble_far_field_robin(
    nodes: np.ndarray,
    outer_edges: np.ndarray,
    sigma_per_edge: np.ndarray,
    r_far: float,
) -> csr_matrix:
    """Assemble the monopole-DtN (Robin) far-field boundary term.

    The truncation sphere carries the exact Dirichlet-to-Neumann map
    of the monopole, $\\partial \\varphi/\\partial r = -\\varphi/r$,
    whose weak form contributes the surface mass matrix
    $$
    M_{ij} \\;=\\; \\oint_{r_{\\text{far}}}
    \\frac{\\sigma}{r_{\\text{far}}}\\, \\phi_i \\phi_j \\, dS,
    \\qquad dS = 2\\pi s\\, d\\ell .
    $$
    With linear hat functions on a straight boundary edge of length
    $\\ell$ between radii $s_1, s_2$ the integrals are exact:
    $$
    M^e \\;=\\; \\frac{2\\pi \\sigma \\ell}{12\\, r_{\\text{far}}}
    \\begin{pmatrix} 3 s_1 + s_2 & s_1 + s_2 \\\\
                     s_1 + s_2 & s_1 + 3 s_2 \\end{pmatrix}.
    $$
    The term is symmetric positive semi-definite, so it also makes the
    system non-singular without any Dirichlet condition on the outer
    boundary.
    """
    Nv = nodes.shape[0]
    if outer_edges.size == 0:
        return coo_matrix(([], ([], [])), shape=(Nv, Nv)).tocsr()
    p1 = nodes[outer_edges[:, 0]]
    p2 = nodes[outer_edges[:, 1]]
    length = np.hypot(p2[:, 0] - p1[:, 0], p2[:, 1] - p1[:, 1])
    s1 = p1[:, 0]
    s2 = p2[:, 0]
    pref = (
        2.0 * np.pi * np.asarray(sigma_per_edge, dtype=float) * length / (12.0 * r_far)
    )
    M_e = np.empty((outer_edges.shape[0], 2, 2))
    M_e[:, 0, 0] = pref * (3.0 * s1 + s2)
    M_e[:, 1, 1] = pref * (s1 + 3.0 * s2)
    M_e[:, 0, 1] = pref * (s1 + s2)
    M_e[:, 1, 0] = M_e[:, 0, 1]
    rows = np.broadcast_to(outer_edges[:, :, None], M_e.shape).ravel()
    cols = np.broadcast_to(outer_edges[:, None, :], M_e.shape).ravel()
    return coo_matrix((M_e.ravel(), (rows, cols)), shape=(Nv, Nv)).tocsr()


def _solve_hemisphere_conductance(
    mesh: _AxiMesh,
    sigma_per_layer: np.ndarray,
    h_layers: list[float],
    *,
    far_field: str = "robin",
) -> float:
    """Conductance $G = 1/R$ of the conforming hemisphere on ``mesh``.

    Solves the unit-potential problem ($\\varphi = 1$ on
    $r = a_{\\text{eq}}$) and returns the full bilinear form
    $G = a(\\varphi, \\varphi)$, which is simultaneously

    * the dissipated power at 1 V excitation (plus, for
      ``far_field='robin'``, the power leaving through the truncation
      sphere), and
    * the current injected into the electrode.

    Because $a$ is symmetric positive definite, the discrete
    $\\varphi$ *minimises* $a(\\cdot, \\cdot)$ over the FE space, so
    $G_h \\ge G$ and $R_h \\le R$ for every mesh: the resistance
    converges from below and monotonically under nested refinement.
    """
    sigma_per_triangle = np.asarray(sigma_per_layer, dtype=float)[mesh.layer_index]
    A = _assemble_stiffness_axisymmetric(mesh.nodes, mesh.triangles, sigma_per_triangle)
    dirichlet = np.zeros(mesh.nodes.shape[0], dtype=bool)
    dirichlet[mesh.inner_nodes] = True
    if far_field == "robin":
        sigma_edge = np.asarray(sigma_per_layer, dtype=float)[
            _layer_of_depth(mesh.nodes[mesh.outer_edges, 1].mean(axis=1), h_layers)
        ]
        A = A + _assemble_far_field_robin(
            mesh.nodes, mesh.outer_edges, sigma_edge, mesh.r_far
        )
    elif far_field == "dirichlet":
        dirichlet[mesh.outer_edges.ravel()] = True
    else:  # pragma: no cover - guarded by solve_fem
        raise ValueError(
            f"far_field must be 'robin' or 'dirichlet', got {far_field!r}."
        )

    phi = np.zeros(mesh.nodes.shape[0])
    phi[mesh.inner_nodes] = 1.0  # unit potential on the hemisphere
    free = ~dirichlet
    if free.any():
        A_ff = A[free][:, free]
        rhs = -(A[free][:, dirichlet] @ phi[dirichlet])
        phi[free] = spsolve(A_ff.tocsc(), rhs)
    # G = a(phi, phi) — energy of the discrete solution.
    return float(phi @ (A @ phi))


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_fem(
    world: "World",
    engine: "Engine",
    *,
    n_radial: int = 60,
    n_axial: int = 40,
    r_far_factor: float = 30.0,
    z_far_factor: float | None = None,
    far_field: str = "robin",
) -> FieldResult:
    """Experimental axisymmetric finite-element solver (equivalent hemispheres).

    Reduces every cluster to its **equivalent hemisphere** at the
    cluster centroid, then solves the volume PDE on a 2-D
    axisymmetric triangular mesh. This is the only volume-PDE engine
    in the suite, but it solves the problem of the hemisphere, not of
    the electrodes: it is a valid cross-check only for one compact
    electrode per world with no layer interface above the deepest of
    its lowest point, its hemisphere radius and its horizontal
    half-extent; otherwise it raises (see *Raises*) or emits a
    :class:`FemReductionWarning` naming the reason (review pass 10, 10.23;
    audit round 1, F01; user decision Q23; audit D2A; see the module
    docstring).

    The mesh conforms to the hemisphere $r = a_{\\text{eq}}$ and the
    layer interfaces $z = \\sum_{i \\le k} h_i$ (see
    :func:`_build_axisymmetric_mesh`), and the domain is truncated
    with the monopole DtN condition
    $\\partial\\varphi/\\partial r = -\\varphi/r_{\\text{far}}$. For
    homogeneous soil the discrete problem therefore converges to the
    analytic hemisphere resistance
    $R = \\rho / (2 \\pi a_{\\text{eq}})$ monotonically from below
    (second order in the mesh size), which is what makes a mesh
    refinement a meaningful accuracy knob.

    Parameters
    ----------
    world
        World to evaluate. Several clusters are accepted but solved
        without mutual coupling (warning).
    engine
        Engine configuration.
    n_radial, n_axial
        Mesh resolution: node lines in the radial ($r$) and
        transverse ($\\vartheta$) direction of the spherical shell
        mesh. Refining as $n \\mapsto 2n - 1$ keeps the FE spaces
        nested and the convergence monotone (homogeneous soil). A layer
        stack deeper than 1 m adds radial lines so that the radial step
        ratio stays that of the homogeneous mesh (10.49);
        ``metadata['fem_mesh'][cluster]['n_radial']`` reports the count
        used.
    r_far_factor
        Far-field truncation radius as a multiple of the
        characteristic length $a_{\\text{eq}} + \\sum h_i$. The
        truncation is a single sphere, so this one number is the
        domain size; it has effect for every value (a truncation
        sweep is meaningful).
    z_far_factor
        Deprecated. ``None`` (default) ignores it; a number acts as a
        lower bound on the same radius and logs a warning when it
        overrides ``r_far_factor``. Kept only for callers written
        against the v0.14.1 $(s, z)$ box mesh, where the radial and
        axial extents were separate.
    far_field
        ``'robin'`` (default) applies the exact monopole
        Dirichlet-to-Neumann map on the truncation sphere;
        ``'dirichlet'`` grounds it ($\\varphi = 0$), which is the
        cruder classical truncation and biases $R$ low by
        $\\approx a_{\\text{eq}} / r_{\\text{far}}$. Provided for
        truncation-error studies.

    Returns
    -------
    FieldResult
        ``metadata['equivalent_hemisphere_radius']`` reports the
        reduction used on the cluster,
        ``metadata['fem_cluster_resistance']`` the solved
        hemisphere resistance and ``metadata['fem_mesh']`` the mesh
        size actually used per cluster.

    Raises
    ------
    BackendCapabilityError
        If an electrode of a driven cluster touches a layer whose top lies
        below the cluster's hemisphere radius ``a_eq`` (rod through or
        below the top layer, a wire on an interface below the hemisphere;
        review pass 10, 10.23); if every electrode of the cluster lies
        below an interface at a depth between ``a_eq/2`` and ``a_eq``
        (more than half of the hemisphere in a layer the electrodes do
        not touch, audit round 2, A2E-2); or if a layer that no electrode
        of the cluster touches, that conducts better than every layer
        they touch and whose top lies above ``max(a_eq, r_ext)`` (``r_ext``
        the electrodes' horizontal half-extent) exists (user decision Q23,
        extended to the half-extent by audit D2A-03: the result would be
        too low, −97 % … +13 % measured; independent of the size of the
        contrast). A horizontal wire touches every layer within its wire
        radius, a rod the layers its axis passes through (audit D2A-01,
        D2A-02). The reduction cannot represent these worlds; the message
        names the cluster, the layer and the backends that solve the
        world. Also if a driven cluster contains an
        electrode without a closed-form hemisphere radius (polyline and
        star electrodes; the generators build mesh foundations of
        polylines), or one whose closed form is used outside its range: a
        strip deeper than half its length, a ring deeper than its radius,
        any ``MeshElectrode`` (audit D2A, O-1; see
        :func:`equivalent_hemisphere_radius`). The message names the
        backends whose own checks accept the world, or says that none
        does (audit D2A-06). A subclass of ``ValueError`` and
        ``NotImplementedError``.
    ValueError
        If an electrode, driven or passive, lies at or above the soil
        surface, in the surface plane (``2 z <= wire_radius``) or reaches
        above it (a rod head above ``z = 0``) — what every other backend
        refuses as well (audit D2A-06). A rod with its head at ``z = 0``
        is valid.

    Warns
    -----
    FemReductionWarning
        Once per solve when the reduction misrepresents the world in a
        way the guard above does not reject (several electrodes in a
        cluster, several clusters, a layer interface above the deepest
        of the electrode, its hemisphere and its horizontal half-extent
        that separates the electrodes from a layer touched by them or
        conducting no better than theirs; see :class:`FemReductionWarning`
        for the measured errors).
    """
    if far_field not in ("robin", "dirichlet"):
        raise ValueError(
            "far_field must be 'robin' (monopole DtN) or 'dirichlet' "
            f"(grounded truncation sphere), got {far_field!r}."
        )
    if not isinstance(world.soil, (HomogeneousSoil, TwoLayerSoil, MultiLayerSoil)):
        raise TypeError(
            "Backend 'fem' supports HomogeneousSoil, TwoLayerSoil, "
            f"and MultiLayerSoil. Got: {type(world.soil).__name__}."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")
    _reject_electrodes_out_of_the_soil(world)  # audit D2A-06
    _reject_concrete_shells(world, "fem")
    _warn_ignored_sources(world, "fem")

    stack = as_layer_stack(world.soil, canonical=False)
    rho_1 = float(stack.rhos[0])
    # The mesh follows the declared soil description (every layer
    # interface is cut into the elements). Unlike the integral backends,
    # fem does *not* merge equal-ρ layers (10.48) — keeping the mesh
    # topology consistent across a ρ₂ sweep is the stronger guarantee
    # here. Since the radial line count follows ln(r_far / a_eq) (10.49)
    # the two choices differ by at most 1e-5 at ρ₂ = ρ₁ (measured for a
    # 1.5 m rod, h_1 = 1 m ... 10 km); the reduction guard below uses the
    # canonical (physical) interfaces.
    h_layers = stack.h.tolist()

    # 1) Cluster the electrodes; build per-cluster equivalent hemispheres.
    cluster_id = _build_clusters(world.electrodes, world.conductors)
    clusters: dict[str, list[str]] = {}
    for ename, root in cluster_id.items():
        clusters.setdefault(root, []).append(ename)
    # FEM cannot consume the distributed-conductor topology because
    # the equivalent-hemisphere reduction is not defined for the tiny
    # midpoint pseudo-electrodes a distributed conductor would
    # produce. Fall back to lumped branches and warn the user — for
    # quantitative distributed-conductor work pick one of the
    # integral-equation backends (image, mom, cim, bem).
    has_distributed = any(getattr(c, "is_distributed", False) for c in world.conductors)
    if has_distributed:
        _log.warning(
            "fem: distributed conductors detected — the FEM backend "
            "treats every conductor as lumped (single branch with the "
            "full series resistance). Use 'image', 'image_2layer', "
            "'mom', 'cim' or 'bem' for distributed-conductor results."
        )
    has_inductance = any(
        getattr(c, "inductance_model", None) is not None for c in world.conductors
    )
    if has_inductance:
        _log.warning(
            "fem: inductance_model is not supported by the FEM backend "
            "(equivalent-hemisphere reduction is DC only). The "
            "computation falls back to the resistive solution; switch "
            "to image / mom / cim / bem for inductive coupling."
        )
    if engine.earth_inductive_model == "carson_series":
        _log.warning(
            "fem: earth_inductive_model='carson_series' is ignored — "
            "the FEM backend has no inductive branch model. "
            "Switch to image / image_2layer / mom / cim / bem for "
            "Carson-corrected results (ADR-0005)."
        )
    finite_branches = _build_finite_branches(
        world.conductors,
        cluster_id,
        distributed_as_lumped=True,
    )

    # Per-cluster active current.
    elec_input_current: dict[str, complex] = {e.name: 0j for e in world.electrodes}
    for src in world.sources:
        if src.kind != "current":
            continue
        i_complex = src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg))
        if src.attached_to in elec_input_current:
            elec_input_current[src.attached_to] += i_complex
    cluster_current: dict[str, complex] = {root: 0j for root in clusters}
    for ename, ic in elec_input_current.items():
        cluster_current[cluster_id[ename]] += ic

    # Active set: every cluster with a non-zero source plus every
    # cluster transitively reachable through a finite branch.
    active_set: set[str] = {r for r, ic in cluster_current.items() if ic != 0j}
    if finite_branches:
        # Seed-and-propagate: any cluster connected to an active one
        # via a chain of finite branches is itself active (current
        # flows through the branches).
        changed = True
        while changed:
            changed = False
            for a, b, _R in finite_branches:
                if a in active_set and b not in active_set:
                    active_set.add(b)
                    changed = True
                elif b in active_set and a not in active_set:
                    active_set.add(a)
                    changed = True
    active_clusters = [r for r in clusters if r in active_set]

    n_freq = len(engine.frequencies)
    electrode_potentials: dict[str, list[complex]] = {
        e.name: [0j] * n_freq for e in world.electrodes
    }
    electrode_currents: dict[str, list[complex]] = {
        e.name: [0j] * n_freq for e in world.electrodes
    }

    a_eq_per_cluster: dict[str, float] = {}
    Z_per_cluster: dict[str, float] = {}
    mesh_per_cluster: dict[str, dict[str, float]] = {}

    # A driven electrode without a closed-form hemisphere radius (polyline,
    # star) is a limit of the reduction, reported like every other one
    # before any mesh is built; the generators build mesh foundations of
    # polyline wires since audit round 2 (A2G-04).
    irreducible = [
        ename
        for root in active_clusters
        for ename in clusters[root]
        if not isinstance(world.get_electrode(ename), _REDUCIBLE_ELECTRODES)
    ]
    if irreducible:
        from groundfield.solver.registry import BackendCapabilityError

        kinds = sorted({type(world.get_electrode(n)).__name__ for n in irreducible})
        shown = ", ".join(repr(n) for n in irreducible[:5])
        more = f" and {len(irreducible) - 5} more" if len(irreducible) > 5 else ""
        raise BackendCapabilityError(
            "Backend 'fem' cannot solve this world: the equivalent-hemisphere "
            f"reduction has no closed form for {', '.join(kinds)} (driven "
            f"electrode(s) {shown}{more}). Use backend='image' or 'mom'."
        )

    # Audit D2A, O-1: a hemisphere formula outside its range is a limit of
    # the reduction as well, reported before any mesh is built.
    deep, meshes = _formula_range_issues(world, clusters, active_clusters)
    if deep or meshes:
        from groundfield.solver.registry import BackendCapabilityError

        parts = []
        if deep:
            parts.append("; ".join(deep) + ". " + _DEPTH_RANGE_EFFECT)
        if meshes:
            parts.append("; ".join(meshes) + ". " + _MESH_FORMULA_EFFECT)
        raise BackendCapabilityError(
            "Backend 'fem' cannot solve this world: "
            + " ".join(parts)
            + " "
            + _remedy(world, engine)
        )

    for root in active_clusters:
        # Build a single equivalent hemisphere from the parallel
        # combination of the per-electrode hemispheres. The
        # hemisphere DC resistance is R = ρ / (2 π a), so two
        # hemispheres in parallel give 1/R_par = 2π(a_1 + a_2)/ρ
        # — i.e. the *radii* add. (Inverting them, as a parallel
        # resistor formula would suggest, is wrong here because the
        # geometric factor sits in the numerator of the conductance.)
        a_per_electrode = [
            equivalent_hemisphere_radius(world.get_electrode(ename), rho_top=rho_1)
            for ename in clusters[root]
        ]
        a_eq_per_cluster[root] = float(sum(a_per_electrode))

    # Review pass 10 (10.23) and user decision Q23: a reduction that is
    # known to fail by up to +3138 % (wrong layer) or to read far too low
    # (a better-conducting layer the electrodes do not touch) raises
    # instead of warning — before any mesh is built.
    unreachable = _unreachable_layers(
        world, clusters, active_clusters, a_eq_per_cluster
    )
    if unreachable:
        from groundfield.solver.registry import BackendCapabilityError

        raise BackendCapabilityError(_unreachable_message(world, engine, unreachable))

    for root in active_clusters:
        a_eq = a_eq_per_cluster[root]

        # 2) Build the boundary-conforming spherical-shell mesh: the
        #    electrode surface r = a_eq and every layer interface are
        #    exact mesh lines.
        mesh = _build_axisymmetric_mesh(
            a_eq,
            h_layers,
            r_far_factor=r_far_factor,
            z_far_factor=z_far_factor,
            n_radial=n_radial,
            n_axial=n_axial,
        )
        mesh_per_cluster[root] = {
            "n_nodes": int(mesh.nodes.shape[0]),
            "n_triangles": int(mesh.triangles.shape[0]),
            "n_electrode_nodes": int(mesh.inner_nodes.size),
            "r_far": float(mesh.r_far),
            "n_radial": int(mesh.n_radial),
        }

        # 3) Unit-potential problem on the conforming hemisphere plus
        #    the monopole-DtN far field, and 4) the conductance as the
        #    energy of the discrete solution (see
        #    _solve_hemisphere_conductance).
        G_cluster = _solve_hemisphere_conductance(
            mesh,
            1.0 / stack.rhos,
            h_layers,
            far_field=far_field,
        )
        R_cluster = 1.0 / G_cluster if G_cluster > 0.0 else float("inf")
        Z_per_cluster[root] = R_cluster
        # Hemisphere-radius distribution per electrode is reused below
        # in step 5 once the leakage current per cluster is known.

    # Review pass 10, 10.23: say when the reduction does not represent
    # the world, once per solve and with every reason.
    reasons = _reduction_issues(world, clusters, active_clusters, a_eq_per_cluster)
    if reasons:
        warnings.warn(
            "fem (experimental): the equivalent-hemisphere reduction is not "
            "a valid model of this world — "
            + "; ".join(reasons)
            + ". The result is the resistance of the equivalent "
            "hemisphere(s), not of the electrodes; use 'mom' or "
            "'mom_sommerfeld' for these configurations.",
            FemReductionWarning,
            stacklevel=3,
        )

    # ------------------------------------------------------------------
    # 5) Nodal analysis on the cluster level.
    #    With per-cluster self-resistance R_c (FEM) plus optional
    #    finite-impedance branches between clusters, solve
    #
    #        diag(1/R_c) · phi_n   +   B^T · I_b   =   I_in
    #        B · phi_n - R_b · I_b                  =   0
    #
    #    For finite_branches == [] this collapses to
    #        phi_n = R_c · I_in
    #    — i.e. the historic single-cluster behaviour.
    # ------------------------------------------------------------------
    cluster_idx_map = {root: k for k, root in enumerate(active_clusters)}
    K_a = len(active_clusters)
    active_branches = [
        (a, b, R)
        for (a, b, R) in finite_branches
        if a in cluster_idx_map and b in cluster_idx_map
    ]
    M_a = len(active_branches)
    n_unk = K_a + M_a
    A_mat = np.zeros((n_unk, n_unk))
    rhs_re = np.zeros(n_unk)
    rhs_im = np.zeros(n_unk)
    for k, root in enumerate(active_clusters):
        R_c = Z_per_cluster[root]
        A_mat[k, k] = 1.0 / R_c if np.isfinite(R_c) and R_c > 0.0 else 0.0
        ic = cluster_current[root]
        rhs_re[k] = ic.real
        rhs_im[k] = ic.imag
    for m, (a, b, R) in enumerate(active_branches):
        ka = cluster_idx_map[a]
        kb = cluster_idx_map[b]
        # KCL contributions of branch m at nodes a and b
        A_mat[ka, K_a + m] = +1.0
        A_mat[kb, K_a + m] = -1.0
        # Branch Ohm's law:  phi_a - phi_b = R · I_b
        # ⇔  +phi_a - phi_b - R · I_b = 0
        A_mat[K_a + m, ka] = +1.0
        A_mat[K_a + m, kb] = -1.0
        A_mat[K_a + m, K_a + m] = -R
    if n_unk > 0:
        # Multi-RHS: one LU factorisation for both right-hand sides
        # (ADR-0010 Tier 1).
        sol = np.linalg.solve(A_mat, np.column_stack([rhs_re, rhs_im]))
        sol_re, sol_im = sol[:, 0], sol[:, 1]
    else:
        sol_re = np.zeros(0)
        sol_im = np.zeros(0)
    phi_node = sol_re[:K_a] + 1j * sol_im[:K_a]
    branch_current = sol_re[K_a:] + 1j * sol_im[K_a:]

    # 6) Per-electrode currents and potentials.
    #    Within a cluster the leakage current is split proportionally
    #    to each member's hemisphere conductance (G ∝ a), exactly as
    #    in the historic single-cluster code path. The leakage of a
    #    cluster is its KCL balance: I_leak = I_in - Σ I_branch_out.
    for k, root in enumerate(active_clusters):
        I_in_c = cluster_current[root]
        I_branch_out = 0j
        for m, (a, b, _R) in enumerate(active_branches):
            if a == root:
                I_branch_out += branch_current[m]
            elif b == root:
                I_branch_out -= branch_current[m]
        I_leak = I_in_c - I_branch_out
        u_cluster = complex(phi_node[k])
        a_per_electrode_c = [
            equivalent_hemisphere_radius(world.get_electrode(ename), rho_top=rho_1)
            for ename in clusters[root]
        ]
        a_total = sum(a_per_electrode_c)
        for ename, a_e in zip(clusters[root], a_per_electrode_c, strict=True):
            share = a_e / a_total if a_total > 0.0 else 0.0
            electrode_currents[ename] = [I_leak * share] * n_freq
            electrode_potentials[ename] = [u_cluster] * n_freq

    # Inactive clusters: zero current, zero potential.
    for root, members in clusters.items():
        if root in active_clusters:
            continue
        for ename in members:
            electrode_currents[ename] = [0j] * n_freq
            electrode_potentials[ename] = [0j] * n_freq

    point_sources: list[PointSource] = []
    cluster_members_map: dict[str, list[str]] = {
        ename: sorted(clusters[root])
        for root, members in clusters.items()
        for ename in members
    }

    return FieldResult(
        backend="fem",
        frequencies=list(engine.frequencies),
        electrode_potentials=electrode_potentials,
        electrode_currents=electrode_currents,
        point_sources=point_sources,
        soil_resistivity=rho_1,
        soil=world.soil,
        clusters=cluster_members_map,
        metadata={
            "world_name": world.name,
            "n_layers": int(stack.n_layers),
            "rhos": stack.rhos.tolist(),
            "h": h_layers,
            "n_radial": int(n_radial),
            "n_axial": int(n_axial),
            "mesh": "conforming spherical shell (a_eq <= r <= r_far)",
            "far_field": far_field,
            "equivalent_hemisphere_radius": {
                k: float(v) for k, v in a_eq_per_cluster.items()
            },
            "fem_cluster_resistance": {k: float(v) for k, v in Z_per_cluster.items()},
            "fem_mesh": mesh_per_cluster,
            "stub": False,
            "approximation": "equivalent-hemisphere reduction per cluster",
        },
    )
