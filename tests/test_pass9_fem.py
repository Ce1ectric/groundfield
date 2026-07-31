"""Regression tests for the ``fem`` backend after review pass 9 (0.15.0).

Findings covered
----------------
* **F05 / F06** — the axial mesh was linearly spaced over
  $Z_{\\text{far}} = 20 (a_{\\text{eq}} + \\sum h_i)$, so the
  near-electrode resolution decayed *linearly* with the declared layer
  thickness. A physically homogeneous soil declared as
  ``TwoLayerSoil(rho_1=rho, rho_2=rho, h_1=X)`` therefore changed the
  answer by up to $-95\\,\\%$ as $X$ grew, and every layered result
  drifted away from ``image_2layer`` with $h_1$.
* **F24** — the Dirichlet electrode was selected by the node predicate
  $s^2 + z^2 \\le a_{\\text{eq}}^2$ on a mesh that had exactly one
  $z$-line inside $a_{\\text{eq}}$, i.e. it was a *flat disc*, not the
  intended hemisphere. Its discrete shape changed discontinuously with
  the mesh, so refining did not converge
  ($-6.1\\,\\% \\to -2.0\\,\\% \\to +20.1\\,\\%$).

Both are fixed by meshing the soil as a **boundary-conforming
spherical shell** $a_{\\text{eq}} \\le r \\le r_{\\text{far}}$ with a
geometric ladder in $r$, a uniform ladder in the polar angle, layer
interfaces cut into the elements, and the monopole
Dirichlet-to-Neumann (Robin) condition on the truncation sphere.

Follow-up defects of that rewrite, closed here as well
------------------------------------------------------
* **F24 leftover** — ``inner_nodes`` was captured *before* the layer
  interfaces were cut into the mesh, so an interface shallower than
  $a_{\\text{eq}}$ (i.e. **every** realistic top layer of a large
  mesh electrode) inserted a node on the electrode boundary and left
  it *free*: a small insulating patch, natural zero-flux instead of
  $\\varphi = 1$. See
  ``test_fem_electrode_boundary_is_closed_under_interface_cuts`` and
  the closure helper :func:`_assert_electrode_is_closed`, now also
  invoked from ``test_fem_mesh_is_conforming_and_closes_the_domain``
  — which previously checked areas and edge counts only, which is
  how the defect survived.
* **dead truncation knob** — ``r_far = max(r_far_factor,
  z_far_factor) * base_length`` made ``r_far_factor`` inert for every
  value $\\le$ ``z_far_factor`` (default 20). Same defect class as
  F14. See ``test_fem_r_far_factor_sets_the_truncation_radius``.
* **inverted minimum-energy argument** in the module docstring and
  ``docs/engines/fem.md`` ("$G_h$ increases … $R_h$ decreases …
  from below"). Pinned by
  ``test_fem_conductance_falls_under_nested_refinement``.
* **aspect-ratio overclaim** — $O(1)$ holds for the uncut shell, not
  for cut elements. Quantified by
  ``test_fem_interface_cut_slivers_are_bounded_and_harmless``.

Physics used as the reference
-----------------------------
The ``fem`` backend solves the volume PDE
$-\\nabla\\cdot(\\sigma\\nabla\\varphi) = 0$ on the *equivalent
hemisphere* of radius $a_{\\text{eq}}$ (Sunde 1968 ch. 2.1). For
homogeneous soil that reduced problem has the closed-form solution
$\\varphi = a_{\\text{eq}}/r$ and hence the exact resistance
$$
R \\;=\\; \\frac{\\rho}{2 \\pi a_{\\text{eq}}},
$$
which is what the discretisation must reproduce — independently of the
electrode type, of the declared (but physically absent) layering, and
of the truncation radius. Because the discrete conductance is a
*minimum* of the energy functional over the finite-element space,
$R_h \\le R$ always, and $R_h \\uparrow R$ under nested refinement:
the convergence is one-sided and monotone, which every test below
relies on.

Measured pre-fix values (v0.14.1, commit ``5ba740d``) are quoted next
to each assertion so the tests can be recognised as regressions rather
than as tolerance bookkeeping.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver.fem import solve_fem

# The mesh-level tests import the private mesh builder locally so that
# the public-API regressions above them still run (and fail loudly on
# v0.14.1) instead of being masked by a collection error.

RHO = 100.0
SEG = 0.05
# Reference rod: L = 1.5 m, r_w = 5 mm  ->  a_eq = 0.2463023 m,
# R = rho / (2 pi a_eq) = 64.61772 Ohm.
ROD_LENGTH = 1.5
ROD_RADIUS = 0.005


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _rod_world(soil) -> gf.World:
    """Single-rod world used as the reference geometry."""
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w, "rod", name="g1", position=(0.0, 0.0, 0.0),
        length=ROD_LENGTH, wire_radius=ROD_RADIUS,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _engine() -> gf.Engine:
    return gf.create_engine(backend="fem", segment_length=SEG, frequencies=[50.0])


def _fem(world, **mesh_kwargs) -> tuple[float, float]:
    """Return ``(R_cluster, R_analytic_hemisphere)`` for ``world``.

    ``R_analytic_hemisphere`` is $\\rho_1 / (2 \\pi a_{\\text{eq}})$
    with the very $a_{\\text{eq}}$ the backend reduced the cluster to,
    i.e. the exact answer of the problem the FEM actually solves in
    homogeneous soil.
    """
    res = solve_fem(world, _engine(), **mesh_kwargs)
    a_eq = res.metadata["equivalent_hemisphere_radius"]["g1"]
    R_h = res.metadata["fem_cluster_resistance"]["g1"]
    return float(R_h), float(RHO / (2.0 * math.pi * a_eq))


@pytest.fixture(scope="module")
def homogeneous_reference() -> float:
    """Cluster resistance of the reference rod in truly homogeneous soil."""
    R_h, _ = _fem(_rod_world(gf.HomogeneousSoil(resistivity=RHO)))
    return R_h


# ---------------------------------------------------------------------
# F24 — the discrete electrode is the hemisphere, and refining converges
# ---------------------------------------------------------------------


def test_fem_homogeneous_matches_analytic_hemisphere() -> None:
    """Default mesh must reproduce $R = \\rho/(2\\pi a_{eq})$ to < 0.5 %.

    This is the acceptance criterion for F24: the equivalent-hemisphere
    reduction fixes the geometry, so in homogeneous soil the *discrete*
    problem has an exact answer and any deviation is discretisation
    error alone.

    Pre-fix (v0.14.1): 60.677 Ω against 64.618 Ω — a $-6.1\\,\\%$
    staircase-disc bias.
    """
    R_h, R_exact = _fem(_rod_world(gf.HomogeneousSoil(resistivity=RHO)))
    rel = (R_h - R_exact) / R_exact
    assert abs(rel) < 5e-3, f"R_fem={R_h:.5f} vs {R_exact:.5f} ({rel:+.3%})"
    # One-sided: the FE conductance is an energy minimum, so R_h <= R.
    assert rel <= 1e-6, f"R_h must not exceed the exact value ({rel:+.3%})"


@pytest.mark.parametrize(
    "kind, kwargs",
    [
        ("rod", dict(position=(0.0, 0.0, 0.0), length=1.5, wire_radius=0.005)),
        ("ring", dict(center=(0.0, 0.0, -0.8), radius=5.0, wire_radius=0.006)),
        ("strip", dict(start=(0.0, 0.0, -0.7), end=(20.0, 0.0, -0.7),
                       wire_radius=0.005)),
        ("grid_mesh", dict(corner=(0.0, 0.0, -0.7), size=(20.0, 20.0),
                           n_x=3, n_y=3, wire_radius=0.005)),
    ],
)
def test_fem_analytic_hemisphere_across_geometries(kind: str, kwargs) -> None:
    """The hemisphere is resolved for every $a_{eq}$, not just the rod's.

    ``a_eq`` spans 0.246 m … 5.56 m over these four electrodes; the
    conforming shell mesh is built relative to it, so the
    discretisation error must stay at a fraction of a per cent for all
    of them.

    Pre-fix (v0.14.1): rod $-6.10\\,\\%$, ring $+6.00\\,\\%$, strip
    $+6.34\\,\\%$, grid $+7.66\\,\\%$ — note the *sign flip*, a
    signature of a staircase electrode whose effective radius depends
    on where the mesh lines happen to fall.
    """
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=RHO))
    gf.create_electrode(w, kind, name="g1", **kwargs)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    R_h, R_exact = _fem(w)
    rel = (R_h - R_exact) / R_exact
    assert abs(rel) < 5e-3, f"{kind}: {R_h:.5f} vs {R_exact:.5f} ({rel:+.3%})"
    assert rel <= 1e-6, f"{kind}: R_h above the exact value ({rel:+.3%})"


def test_fem_refinement_converges_monotonically_second_order() -> None:
    """Nested refinement: monotone from below at order ≈ 2.

    ``n -> 2n - 1`` halves every element edge and keeps the
    finite-element spaces nested, so Dirichlet's principle forces
    $R_h \\le R_{h/2} \\le R$ and the error to shrink by ~4 per level.

    Pre-fix (v0.14.1) the same three meshes gave 43.977 / 63.335 /
    77.588 Ω — errors of $-31.9\\,\\%$, $-2.0\\,\\%$, $+20.1\\,\\%$:
    refining tripled the error and flipped its sign.
    """
    world = _rod_world(gf.HomogeneousSoil(resistivity=RHO))
    errors: list[float] = []
    for n_r, n_a in [(29, 19), (57, 37), (113, 73)]:
        R_h, R_exact = _fem(world, n_radial=n_r, n_axial=n_a)
        errors.append((R_h - R_exact) / R_exact)
    # (a) one-sided from below at every level
    assert all(e < 0.0 for e in errors), errors
    # (b) strictly shrinking
    assert all(
        abs(errors[i + 1]) < abs(errors[i]) for i in range(len(errors) - 1)
    ), errors
    # (c) observed order of convergence ≈ 2 (measured 1.97 / 1.99)
    orders = [
        math.log(abs(errors[i]) / abs(errors[i + 1]), 2.0)
        for i in range(len(errors) - 1)
    ]
    assert min(orders) > 1.8, orders
    # (d) the finest mesh is essentially exact (measured -0.0665 %)
    assert abs(errors[-1]) < 1e-3, errors


def test_fem_electrode_boundary_is_a_conforming_hemisphere() -> None:
    """The Dirichlet node set *is* the hemisphere $r = a_{eq}$.

    Structural counterpart of F24. Pre-fix the inner node set was
    selected geometrically and contained only nodes at $z = 0$ (24 of
    them, at every ``n_axial``) — a flat disc of radius
    $a_{\\text{eq}}$, whose homogeneous-soil resistance
    $\\rho / (4 a_{\\text{eq}})$ is 57 % above the hemisphere's. Now
    the electrode is a mesh line: one node per angular line, all at
    exactly $r = a_{\\text{eq}}$, spanning the full quarter arc from
    the axis to the surface, and refining the angular direction
    refines the electrode.
    """
    from groundfield.solver.fem import _build_axisymmetric_mesh

    a_eq = 0.2463023
    n_prev = 0
    for n_axial in (19, 37, 73):
        mesh = _build_axisymmetric_mesh(a_eq, [], n_axial=n_axial)
        pts = mesh.nodes[mesh.inner_nodes]
        r = np.hypot(pts[:, 0], pts[:, 1])
        # every electrode node sits exactly on the sphere
        assert np.allclose(r, a_eq, rtol=1e-12, atol=0.0), r
        # one node per angular line, growing with the resolution
        assert mesh.inner_nodes.size == n_axial
        assert mesh.inner_nodes.size > n_prev
        n_prev = mesh.inner_nodes.size
        # and the arc spans axis (z = a_eq) to surface (z = 0)
        assert pts[:, 1].max() == pytest.approx(a_eq, rel=1e-12)
        assert pts[:, 1].min() == pytest.approx(0.0, abs=1e-12)
        assert pts[:, 0].max() == pytest.approx(a_eq, rel=1e-12)
        assert pts[:, 0].min() == pytest.approx(0.0, abs=1e-12)
        # more than one z-line inside a_eq — the disc degeneracy is gone
        assert np.unique(np.round(pts[:, 1], 9)).size == n_axial


# ---------------------------------------------------------------------
# The electrode stays closed when a layer interface cuts through it
# ---------------------------------------------------------------------


def _boundary_edges(triangles: np.ndarray) -> np.ndarray:
    """Edges belonging to exactly one triangle, as sorted index pairs."""
    e = np.sort(
        np.concatenate(
            [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
        ),
        axis=1,
    )
    uq, counts = np.unique(e, axis=0, return_counts=True)
    return uq[counts == 1]


def _assert_electrode_is_closed(mesh, a_eq: float) -> None:
    """Assert the Dirichlet set is the whole discrete electrode.

    The electrode is the polyline inscribed in $r = a_{\\text{eq}}$,
    so its nodes are exactly the mesh nodes with
    $r \\le a_{\\text{eq}}$ (up to the $10^{-6}$ relative snapping
    tolerance of the interface cut). Every one of them must carry the
    Dirichlet condition: a free node there is an insulating patch in
    the middle of the energised surface, with a natural zero-flux
    boundary condition instead of $\\varphi = 1$.
    """
    r = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    tol = 1e-6
    inner = set(int(i) for i in mesh.inner_nodes)
    assert len(inner) == mesh.inner_nodes.size, "duplicate Dirichlet nodes"

    # (a) the Dirichlet set never reaches outside the electrode sphere
    assert (r[mesh.inner_nodes] <= a_eq * (1.0 + tol)).all(), (
        r[mesh.inner_nodes].max() / a_eq
    )
    # (b) and no node inside the sphere is left out of it
    strictly_inside = np.nonzero(r < a_eq * (1.0 - tol))[0]
    missing = [int(i) for i in strictly_inside if int(i) not in inner]
    assert not missing, [
        (i, float(mesh.nodes[i, 0]), float(mesh.nodes[i, 1]), float(r[i] / a_eq))
        for i in missing
    ]

    # (c) every boundary edge lying on the electrode has *both*
    #     endpoints in the Dirichlet set — the assertion that F24's
    #     leftover violated.
    bnd = _boundary_edges(mesh.triangles)
    on_elec = bnd[
        (r[bnd[:, 0]] <= a_eq * (1.0 + tol))
        & (r[bnd[:, 1]] <= a_eq * (1.0 + tol))
    ]
    assert on_elec.shape[0] > 0
    bad = [
        (int(ia), int(ib)) for ia, ib in on_elec
        if int(ia) not in inner or int(ib) not in inner
    ]
    assert not bad, (
        f"{len(bad)} electrode boundary edges have a non-Dirichlet "
        f"endpoint: {bad[:5]}"
    )

    # (d) those edges form a single open path covering the whole
    #     Dirichlet set: axis -> ... -> soil surface, no gap, no branch.
    assert on_elec.shape[0] == mesh.inner_nodes.size - 1
    deg: dict[int, int] = {}
    for ia, ib in on_elec:
        deg[int(ia)] = deg.get(int(ia), 0) + 1
        deg[int(ib)] = deg.get(int(ib), 0) + 1
    assert set(deg) == inner, "Dirichlet node not on any electrode edge"
    ends = sorted(n for n, d in deg.items() if d == 1)
    assert len(ends) == 2 and max(deg.values()) == 2, sorted(deg.values())
    end_pts = mesh.nodes[ends]
    # one end on the downward symmetry axis, one on the soil surface
    assert min(abs(end_pts[0, 0]), abs(end_pts[1, 0])) < 1e-12 * a_eq
    assert min(abs(end_pts[0, 1]), abs(end_pts[1, 1])) < 1e-12 * a_eq


A_EQ_ROD = 0.2463023


@pytest.mark.parametrize(
    "h_layers",
    [
        [],                                  # no cut at all
        [0.05],                              # 0.20 a_eq — cuts a chord
        [0.1],                               # 0.41 a_eq — cuts a chord
        [0.5 * A_EQ_ROD],                    # exactly on a node depth
        [A_EQ_ROD],                          # exactly at the pole
        [2.0 * A_EQ_ROD],                    # below the electrode
        [0.05, 0.1],                         # two cuts inside a_eq
        [0.1, 0.5],                          # one inside, one below
        [0.02, 0.03, 0.05, 1.0],             # three inside
    ],
)
def test_fem_electrode_boundary_is_closed_under_interface_cuts(h_layers) -> None:
    """No free node on the electrode, for any interface depth.

    The leftover of F24 that survived the spherical-shell rewrite:
    ``inner_nodes`` was captured *before* the layer interfaces were
    cut into the mesh and never updated. For any cumulative interface
    depth $0 < \\sum h_i < a_{\\text{eq}}$ the cutting plane crosses a
    *chord* of the electrode polyline, so a node was inserted at
    $r/a_{\\text{eq}} = 0.999808$ (for $h_1 = 0.1$ m) and left
    **free** — a small insulating patch with a natural zero-flux
    condition in the middle of the energised electrode. Measured
    pre-fix: 1 node inside $r < a_{\\text{eq}}$ and 2 electrode
    boundary edges with a non-Dirichlet endpoint for every
    $h_1 \\in \\{0.05, 0.1\\}$; 0 for $h_1 \\in \\{0.5, 2.0\\}$
    (interface deeper than $a_{\\text{eq}}$), which is why the
    existing mesh tests — parametrised on ``[1e-3]`` and
    ``[0.2463023]``, both of which happen to miss — never saw it.
    """
    from groundfield.solver.fem import _build_axisymmetric_mesh

    mesh = _build_axisymmetric_mesh(A_EQ_ROD, list(h_layers))
    _assert_electrode_is_closed(mesh, A_EQ_ROD)


def test_fem_interface_cut_really_does_split_the_electrode() -> None:
    """Guard against the closure test above being vacuous.

    If no interface ever crossed the electrode polyline there would be
    nothing to close, so pin the mechanism itself: an interface at
    $h_1 = 0.1$ m ($0.41\\, a_{\\text{eq}}$) must add exactly one node
    to the electrode, and it must sit just inside the sphere on the
    chord it splits.
    """
    from groundfield.solver.fem import _build_axisymmetric_mesh

    base = _build_axisymmetric_mesh(A_EQ_ROD, [])
    cut = _build_axisymmetric_mesh(A_EQ_ROD, [0.1])
    assert cut.inner_nodes.size == base.inner_nodes.size + 1
    added = sorted(set(cut.inner_nodes.tolist()) - set(base.inner_nodes.tolist()))
    assert len(added) == 1
    s, z = cut.nodes[added[0]]
    assert z == pytest.approx(0.1, abs=1e-12)
    # on the chord, i.e. just inside the inscribed polyline
    assert 0.999 < math.hypot(s, z) / A_EQ_ROD < 1.0


@pytest.mark.parametrize("h_1", [1.0, 5.0, 9.0])
def test_fem_electrode_closed_for_a_large_mesh_electrode(h_1: float) -> None:
    """The same check on a realistic 100 × 100 m mesh grid.

    Not a toy regime: a 100 × 100 m mesh electrode reduces to
    $a_{\\text{eq}} \\approx 6.6$ m, so *every* top-layer thickness
    below that — the whole range a soil survey reports — crossed the
    electrode polyline and left a free node on it.
    """
    from groundfield.solver.fem import (
        _build_axisymmetric_mesh,
        equivalent_hemisphere_radius,
    )

    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=RHO))
    gf.create_electrode(
        w, "mesh", name="g1", corner=(0.0, 0.0, -0.7), size=(100.0, 100.0),
        wire_radius=0.005,
    )
    a_eq = equivalent_hemisphere_radius(w.get_electrode("g1"), rho_top=RHO)
    assert a_eq > 5.0, a_eq  # the regime this test is about
    mesh = _build_axisymmetric_mesh(a_eq, [h_1])
    _assert_electrode_is_closed(mesh, a_eq)


def test_fem_closed_electrode_keeps_the_answer_below_the_exact_value() -> None:
    """Physics side of the same fix, at a *fixed* truncation radius.

    Comparing a layered run against the homogeneous default is
    confounded, because $r_{\\text{far}} = 30 (a_{\\text{eq}} +
    \\sum h_i)$ differs and a smaller sphere resolves the electrode
    better. Holding $r_{\\text{far}}$ fixed makes the layered mesh a
    genuine *refinement* of the homogeneous one (each interface cut
    only subdivides triangles), so Dirichlet's principle applies
    directly: $G$ can only fall and $R$ can only rise, and both stay
    on the correct side of the exact value.

    Measured at 60 × 40, $r_{\\text{far}} = 37.389$ m, $\\rho_2 =
    \\rho_1$: uncut $R = 64.46303\\,\\Omega$ ($-0.23939\\,\\%$);
    with the insulating patch $64.46872\\,\\Omega$
    ($-0.23059\\,\\%$, a spurious $+0.0088$ pp from a physically
    absent interface); with the patch closed $64.46358\\,\\Omega$
    ($-0.23855\\,\\%$, $+0.00085$ pp) — a 10× smaller artefact.
    """
    from groundfield.solver.fem import (
        _build_axisymmetric_mesh,
        _solve_hemisphere_conductance,
    )

    R_exact = RHO / (2.0 * math.pi * A_EQ_ROD)
    r_far = 30.0 * (A_EQ_ROD + 1.0)

    def solve(h_layers: list[float]) -> tuple[float, float]:
        base = A_EQ_ROD + (sum(h_layers) if h_layers else 1.0)
        mesh = _build_axisymmetric_mesh(
            A_EQ_ROD, list(h_layers), r_far_factor=r_far / base
        )
        assert mesh.r_far == pytest.approx(r_far, rel=1e-12)
        G = _solve_hemisphere_conductance(
            mesh, np.full(len(h_layers) + 1, 1.0 / RHO), list(h_layers)
        )
        return G, 1.0 / G

    G0, R0 = solve([])
    for h_layers in ([0.05], [0.1], [0.1, 0.5], [0.02, 0.03, 0.05]):
        G, R = solve(h_layers)
        # nested refinement of the same domain: G falls, R rises,
        # and the exact value is still an upper bound for R.
        assert G <= G0 + 1e-15, (h_layers, G, G0)
        assert R0 - 1e-9 <= R < R_exact, (h_layers, R0, R, R_exact)
        # the physically absent interface must barely move the answer
        # (measured +1.01e-5 / +8.4e-6 / +1.80e-5 / +2.85e-5)
        assert abs(R - R0) / R0 < 5e-5, (h_layers, R, R0)


def test_fem_conductance_falls_under_nested_refinement() -> None:
    """$G_h \\ge G_{h/2} \\ge G$ — the minimum-energy direction.

    Pinned as a test because the module docstring and ``fem.md`` used
    to state it backwards ("$G_h$ then increases and $R_h$ decreases
    monotonically towards the exact value from below"), which is
    self-contradictory and inverts the very sign the argument is meant
    to establish. $G$ is the *minimum* of the energy over the
    admissible set, so enlarging the space can only lower it.
    Measured: 0.015640831 / 0.015516853 / 0.015485931 / 0.015478205
    against the exact 0.0154757.
    """
    world = _rod_world(gf.HomogeneousSoil(resistivity=RHO))
    G: list[float] = []
    for n_r, n_a in [(29, 19), (57, 37), (113, 73), (225, 145)]:
        R_h, R_exact = _fem(world, n_radial=n_r, n_axial=n_a)
        G.append(1.0 / R_h)
    G_exact = 1.0 / R_exact
    assert all(G[i] > G[i + 1] for i in range(len(G) - 1)), G
    assert all(g > G_exact for g in G), (G, G_exact)
    assert G[-1] == pytest.approx(G_exact, rel=2e-4)


# ---------------------------------------------------------------------
# F05 / F06 — resolution decoupled from the declared layer thickness
# ---------------------------------------------------------------------


@pytest.mark.parametrize("h_1", [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0])
def test_fem_uniform_rho_independent_of_declared_layer_thickness(
    h_1: float, homogeneous_reference: float
) -> None:
    """``TwoLayerSoil(rho, rho, h_1)`` is homogeneous soil for every $h_1$.

    $\\rho_2 = \\rho_1$ means the interface carries no physics
    ($K = 0$), so $h_1$ must not move the result at all. Pre-fix
    (v0.14.1) the error against the analytic hemisphere ran
    $+16.9\\,\\%$ ($h_1 = 0.5$), $-5.6\\,\\%$ (1 m), $-24.5\\,\\%$
    (2 m), $-53.0\\,\\%$ (5 m), $-70.3\\,\\%$ (10 m), $-86.5\\,\\%$
    (30 m), $-94.9\\,\\%$ (100 m) — monotone in a parameter that has
    no physical effect, because ``Z_far`` and hence the uniform axial
    spacing scaled with $h_1$.
    """
    soil = gf.TwoLayerSoil(rho_1=RHO, rho_2=RHO, h_1=h_1)
    R_h, R_exact = _fem(_rod_world(soil))
    rel_exact = (R_h - R_exact) / R_exact
    rel_hom = (R_h - homogeneous_reference) / homogeneous_reference
    assert abs(rel_exact) < 1.5e-2, (
        f"h_1={h_1}: {R_h:.5f} vs analytic {R_exact:.5f} ({rel_exact:+.3%})"
    )
    assert abs(rel_hom) < 1.0e-2, (
        f"h_1={h_1}: {R_h:.5f} vs homogeneous FEM "
        f"{homogeneous_reference:.5f} ({rel_hom:+.3%})"
    )


def test_fem_uniform_rho_sweep_is_flat_and_ordered() -> None:
    """Across three decades of $h_1$ the spread must stay ≤ 1 %.

    The residual drift is the *logarithmic* loss of radial resolution
    as $r_{\\text{far}} = 30 (a_{\\text{eq}} + h_1)$ grows at fixed
    node count — which is exactly the decoupling the fix promises: it
    replaces a linear degradation with a logarithmic one. Measured
    spread 0.66 % over $h_1 \\in [0.5, 100]$ m; pre-fix 112 %.
    """
    values = [
        _fem(_rod_world(gf.TwoLayerSoil(rho_1=RHO, rho_2=RHO, h_1=h)))[0]
        for h in (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0)
    ]
    spread = (max(values) - min(values)) / max(values)
    assert spread < 1.0e-2, [f"{v:.4f}" for v in values]
    # A larger declared thickness can only cost resolution, never gain
    # it, so the sequence is (weakly) decreasing.
    assert all(
        values[i + 1] <= values[i] + 1e-9 for i in range(len(values) - 1)
    ), [f"{v:.4f}" for v in values]


@pytest.mark.parametrize("h_1", [2.0, 5.0, 10.0])
@pytest.mark.parametrize("rho_2", [10.0, 400.0])
def test_fem_two_layer_agrees_with_image_2layer(h_1: float, rho_2: float) -> None:
    """Real layer contrast: FEM vs the closed-form image series ≤ 5 %.

    With $h_1 \\ge 8 a_{\\text{eq}}$ the equivalent hemisphere stays
    inside the top layer, so the reduction is faithful and the
    remaining difference is the two methods' own error. Pre-fix
    (v0.14.1) the deviation was $-23.5\\,\\%$ / $-22.5\\,\\%$ at
    $h_1 = 2$ m, $-53.5\\,\\%$ / $-50.4\\,\\%$ at 5 m and
    $-70.7\\,\\%$ / $-68.4\\,\\%$ at 10 m ($\\rho_2 = 10$ / 400) —
    i.e. it *grew* with the layer depth, and the FEM even returned a
    lower $R$ for a more resistive lower layer than for homogeneous
    soil.
    """
    soil = gf.TwoLayerSoil(rho_1=RHO, rho_2=rho_2, h_1=h_1)
    R_fem, _ = _fem(_rod_world(soil))
    R_ref = (
        gf.create_engine(backend="image_2layer", segment_length=SEG)
        .solve(_rod_world(soil))
        .cluster_impedance("g1")[0]
        .real
    )
    rel = (R_fem - R_ref) / R_ref
    assert abs(rel) < 5e-2, (
        f"h_1={h_1}, rho_2={rho_2}: fem={R_fem:.4f} vs "
        f"image_2layer={R_ref:.4f} ({rel:+.2%})"
    )


def test_fem_two_layer_contrast_is_ordered_around_homogeneous(
    homogeneous_reference: float,
) -> None:
    """$\\rho_2 < \\rho_1 \\Rightarrow R$ drops, $\\rho_2 > \\rho_1
    \\Rightarrow R$ rises — around the homogeneous value.

    Pre-fix this failed in the worst possible way: at $h_1 = 5$ m the
    FEM returned 32.98 Ω for $\\rho_2 = 400\\,\\Omega\\text{m}$
    against 60.68 Ω for homogeneous soil, i.e. a *more resistive*
    lower layer lowered the resistance by 46 %.
    """
    R_low, _ = _fem(_rod_world(gf.TwoLayerSoil(rho_1=RHO, rho_2=10.0, h_1=5.0)))
    R_high, _ = _fem(_rod_world(gf.TwoLayerSoil(rho_1=RHO, rho_2=400.0, h_1=5.0)))
    assert R_low < homogeneous_reference < R_high, (
        R_low, homogeneous_reference, R_high
    )


# ---------------------------------------------------------------------
# Truncation: resolution and domain size are separate knobs
# ---------------------------------------------------------------------


@pytest.mark.parametrize("r_far_factor", [10.0, 30.0, 100.0])
def test_fem_result_insensitive_to_truncation_radius(r_far_factor: float) -> None:
    """The monopole-DtN sphere removes the truncation bias.

    $\\partial\\varphi/\\partial r = -\\varphi/r_{\\text{far}}$ is
    satisfied identically by the exact solution
    $\\varphi = a_{\\text{eq}}/r$, so the answer must not depend on
    where the domain is cut. Pre-fix (grounded box whose node spacing
    was tied to its size): $+21.2\\,\\%$ / $-20.8\\,\\%$ /
    $-59.2\\,\\%$ at factors 10 / 30 / 100 — enlarging the domain
    moved the answer by 80 percentage points, the fingerprint of
    resolution and truncation being one and the same knob. Measured
    now: $-0.14\\,\\%$ / $-0.24\\,\\%$ / $-0.37\\,\\%$, the residual
    being the graded ladder stretched over more decades.
    """
    R_h, R_exact = _fem(
        _rod_world(gf.HomogeneousSoil(resistivity=RHO)),
        r_far_factor=r_far_factor,
        z_far_factor=r_far_factor,
    )
    rel = (R_h - R_exact) / R_exact
    assert abs(rel) < 1e-2, f"factor={r_far_factor}: {rel:+.3%}"


def test_fem_grounded_truncation_is_biased_low_and_selectable() -> None:
    """``far_field='dirichlet'`` reproduces the classical truncation.

    Grounding the truncation sphere removes the
    $\\rho/(2\\pi r_{\\text{far}})$ tail of the resistance, so it must
    be biased *low* by roughly $a_{\\text{eq}}/r_{\\text{far}}$ and
    must sit below the DtN answer. Keeping it selectable makes the
    truncation error measurable instead of implicit.
    """
    world = _rod_world(gf.HomogeneousSoil(resistivity=RHO))
    R_robin, R_exact = _fem(world, r_far_factor=10.0, z_far_factor=10.0)
    R_dir, _ = _fem(
        world, r_far_factor=10.0, z_far_factor=10.0, far_field="dirichlet"
    )
    assert R_dir < R_robin < R_exact
    # a_eq / r_far = 1/10 of the way; measured -2.1 % against -0.14 %.
    assert (R_exact - R_dir) / R_exact > 5.0 * (R_exact - R_robin) / R_exact
    with pytest.raises(ValueError, match="far_field"):
        solve_fem(world, _engine(), far_field="sommerfeld")


@pytest.mark.parametrize("factor", [5.0, 10.0, 20.0, 30.0, 100.0])
def test_fem_r_far_factor_sets_the_truncation_radius(factor: float) -> None:
    """``r_far_factor`` is a live knob for *every* value.

    Same defect class as review-pass-9 F14 (a public accuracy knob
    that silently does nothing). The first spherical-shell rewrite
    computed
    ``r_far = max(r_far_factor, z_far_factor) * base_length`` with
    defaults 30 / 20, so with the reference rod every
    ``r_far_factor <= 20`` returned the *same* $r_{\\text{far}} =
    24.926$ m: ``r_far_factor=10`` and ``=5`` were indistinguishable
    from ``=20``, and a truncation study swept a dead parameter with
    no warning. In v0.14.1 ``r_far_factor`` set the radial extent
    independently, so this was a silent regression.

    Measured now: 5 → 6.2315 m, 10 → 12.4630 m, 20 → 24.9260 m,
    30 → 37.3891 m, 100 → 124.6302 m, and the resistance moves
    monotonically with it ($-0.095\\,\\%$ … $-0.369\\,\\%$).
    """
    world = _rod_world(gf.HomogeneousSoil(resistivity=RHO))
    res = solve_fem(world, _engine(), r_far_factor=factor)
    a_eq = res.metadata["equivalent_hemisphere_radius"]["g1"]
    r_far = res.metadata["fem_mesh"]["g1"]["r_far"]
    assert r_far == pytest.approx(factor * (a_eq + 1.0), rel=1e-12)


def test_fem_r_far_factor_is_not_shadowed_by_z_far_factor() -> None:
    """Distinct ``r_far_factor`` values must give distinct domains.

    The regression test for the shadowing itself: three factors below
    the old ``z_far_factor`` default of 20 all collapsed onto
    $r_{\\text{far}} = 24.926$ m and returned bit-identical
    resistances (64.487453 Ω for 5, 10 and 20).
    """
    world = _rod_world(gf.HomogeneousSoil(resistivity=RHO))
    seen = {}
    for factor in (5.0, 10.0, 15.0, 20.0):
        res = solve_fem(world, _engine(), r_far_factor=factor)
        seen[factor] = (
            res.metadata["fem_mesh"]["g1"]["r_far"],
            res.metadata["fem_cluster_resistance"]["g1"],
        )
    r_fars = [v[0] for v in seen.values()]
    resistances = [v[1] for v in seen.values()]
    assert len(set(r_fars)) == 4, seen
    assert len(set(resistances)) == 4, seen
    # a larger domain at fixed node count costs radial resolution, so
    # the resistance error grows monotonically with r_far
    assert all(
        resistances[i] > resistances[i + 1] for i in range(len(resistances) - 1)
    ), seen


def test_fem_z_far_factor_override_is_announced(caplog) -> None:
    """The deprecated axial factor may still win, but not silently.

    ``z_far_factor`` is kept for callers written against the v0.14.1
    $(s, z)$ box mesh. It now defaults to ``None`` (ignored); when a
    caller passes a value that overrides ``r_far_factor`` the backend
    says so, so the knob can never be dead without a trace.
    """
    world = _rod_world(gf.HomogeneousSoil(resistivity=RHO))
    with caplog.at_level("WARNING"):
        res = solve_fem(world, _engine(), r_far_factor=10.0, z_far_factor=40.0)
    a_eq = res.metadata["equivalent_hemisphere_radius"]["g1"]
    assert res.metadata["fem_mesh"]["g1"]["r_far"] == pytest.approx(
        40.0 * (a_eq + 1.0), rel=1e-12
    )
    assert "z_far_factor" in caplog.text and "overrides" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        solve_fem(world, _engine(), r_far_factor=40.0, z_far_factor=10.0)
    assert "z_far_factor" not in caplog.text


# ---------------------------------------------------------------------
# Mesh invariants (validity of the new machinery, not regressions)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "h_layers",
    [
        [], [2.0], [0.2463023], [1e-3], [0.7, 12.0], [100.0], [5.0, 5.0, 5.0],
        # interface depths that cut *through* the electrode polyline
        [0.05], [0.1], [0.05, 0.1, 3.0],
    ],
)
def test_fem_mesh_is_conforming_and_closes_the_domain(h_layers) -> None:
    """The shell mesh triangulates the quarter annulus without gaps.

    Checks the four properties the assembly relies on: no edge is
    shared by more than two triangles (conforming — the interface cut
    inserts shared nodes, not hanging ones), every element has a
    positive area, the element areas add up to the analytic area
    of the quarter annulus
    $\\tfrac{\\pi}{4}(r_{\\text{far}}^2 - a_{\\text{eq}}^2)$ up to the
    chord defect of a straight-sided mesh ($O(\\Delta\\vartheta^2)$),
    and the Dirichlet set is the complete discrete electrode. The last
    one is the check this test used to be missing — which is how the
    free node left on the electrode by a shallow interface cut
    survived a full review pass. Note that neither ``[1e-3]`` nor
    ``[0.2463023]`` exercises it (too shallow to reach the first
    chord / exactly on the pole), so the shallow-cut depths above are
    part of the regression.
    """
    from groundfield.solver.fem import _build_axisymmetric_mesh

    a_eq = 0.2463023
    mesh = _build_axisymmetric_mesh(a_eq, list(h_layers))
    v = mesh.nodes[mesh.triangles]
    area = 0.5 * np.abs(
        (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
        - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1])
    )
    assert (area >= 0.0).all()
    exact_area = 0.25 * math.pi * (mesh.r_far**2 - a_eq**2)
    assert area.sum() == pytest.approx(exact_area, rel=1e-3)

    edges = np.sort(
        np.concatenate(
            [
                mesh.triangles[:, [0, 1]],
                mesh.triangles[:, [1, 2]],
                mesh.triangles[:, [2, 0]],
            ]
        ),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert counts.max() <= 2, "non-conforming mesh (edge in > 2 triangles)"

    # ... and the Dirichlet set is the whole discrete electrode.
    _assert_electrode_is_closed(mesh, a_eq)


def test_fem_interface_cut_slivers_are_bounded_and_harmless() -> None:
    """The interface cut *does* produce slivers — quantify them.

    Documentation claim under test: the shell mesh has $O(1)$ aspect
    ratios, which holds for the **uncut** mesh only (measured shape
    quality $\\max \\sum \\ell^2 / |T| = 10.7$, equilateral = 2.31).
    Where an interface grazes a node the cut yields genuine slivers,
    so the docs must not promise $O(1)$ everywhere. What the
    implementation does guarantee is that they are harmless: nodes
    within a relative $10^{-6}$ of the plane are snapped onto it, no
    zero-area element is produced, and the answer does not move.

    Measured over 66 adversarial depths (every node depth of the shell
    approached from both sides at relative offsets $10^{-9}$ and
    $10^{-5}$), at a *fixed* truncation radius so only the cut varies:
    worst $\\sum \\ell^2 / |T| = 1.1 \\cdot 10^5$ at an area of
    $6.7 \\cdot 10^{-12}\\,\\text{m}^2$, resistance inside
    $-0.23837 \\ldots -0.23816\\,\\%$ against the uncut
    $-0.23939\\,\\%$ — a spread of $2 \\cdot 10^{-4}$ percentage
    points.
    """
    from groundfield.solver.fem import (
        _build_axisymmetric_mesh,
        _solve_hemisphere_conductance,
    )

    a_eq = A_EQ_ROD
    R_exact = RHO / (2.0 * math.pi * a_eq)
    r_far = 30.0 * (a_eq + 1.0)

    def build(h_layers: list[float]):
        base = a_eq + (sum(h_layers) if h_layers else 1.0)
        return _build_axisymmetric_mesh(
            a_eq, list(h_layers), r_far_factor=r_far / base
        )

    def quality(mesh) -> tuple[float, float, int]:
        v = mesh.nodes[mesh.triangles]
        area = 0.5 * np.abs(
            (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
            - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1])
        )
        per = (
            (v[:, [1, 2, 0], 0] - v[:, [0, 1, 2], 0]) ** 2
            + (v[:, [1, 2, 0], 1] - v[:, [0, 1, 2], 1]) ** 2
        ).sum(axis=1)
        ok = area > 0.0
        return float((per[ok] / area[ok]).max()), float(area[ok].min()), int(
            (~ok).sum()
        )

    base_mesh = build([])
    q_uncut, _, _ = quality(base_mesh)
    assert q_uncut < 25.0, q_uncut  # O(1) — this part of the claim holds
    R_uncut = 1.0 / _solve_hemisphere_conductance(
        base_mesh, np.array([1.0 / RHO]), []
    )

    depths = [
        float(z + eps * max(z, a_eq))
        for z in np.unique(base_mesh.nodes[:, 1])[1:23]
        for eps in (-1e-5, 1e-9, 1e-5)
        if z > 1e-6
    ]
    assert len(depths) >= 60, len(depths)
    worst_q, worst_area, n_zero, rels = 0.0, np.inf, 0, []
    for d in depths:
        mesh = build([d])
        q, a_min, nz = quality(mesh)
        worst_q, worst_area, n_zero = max(worst_q, q), min(worst_area, a_min), n_zero + nz
        R = 1.0 / _solve_hemisphere_conductance(
            mesh, np.full(2, 1.0 / RHO), [d]
        )
        rels.append((R - R_exact) / R_exact)
    # the slivers are real (so the docs may not claim O(1) everywhere)
    assert worst_q > 100.0 * q_uncut, worst_q
    # but they never degenerate, and they never move the answer
    assert n_zero == 0
    assert worst_area > 0.0
    assert max(rels) < 0.0, max(rels)
    assert max(rels) - min(rels) < 1e-5, (min(rels), max(rels))
    assert abs(max(rels) - (R_uncut - R_exact) / R_exact) < 5e-5


@pytest.mark.parametrize("h_layers", [[2.0], [0.7, 12.0], [5.0, 5.0, 5.0]])
def test_fem_layer_interfaces_are_mesh_conforming(h_layers) -> None:
    """No element straddles a conductivity jump.

    The conductivity is piecewise constant in $z$; an element crossing
    an interface would smear the jump over one mesh width. After the
    interface cut every triangle lies entirely within one layer, so
    the per-element conductivity assignment is exact.
    """
    from groundfield.solver.fem import _build_axisymmetric_mesh, _layer_of_depth

    a_eq = 0.2463023
    mesh = _build_axisymmetric_mesh(a_eq, list(h_layers))
    z = mesh.nodes[mesh.triangles, 1]
    for depth in np.cumsum(h_layers):
        straddling = (z < depth - 1e-12).any(axis=1) & (z > depth + 1e-12).any(
            axis=1
        )
        assert not straddling.any(), (
            f"{int(straddling.sum())} triangles cross z = {depth}"
        )
    # ... and the layer index agrees with the centroid depth.
    expected = _layer_of_depth(z.mean(axis=1), list(h_layers))
    assert np.array_equal(mesh.layer_index, expected)


def test_fem_conductance_equals_injected_current() -> None:
    """$a(\\varphi, \\varphi) = a(\\varphi, \\mathbf{1})$ — conservation.

    At 1 V on the electrode the energy of the discrete solution equals
    the current it injects, because
    $a(\\varphi, w) = a(\\varphi, \\varphi)$ for every $w$ that is 1
    on the electrode. Taking $w \\equiv 1$ turns the identity into a
    Kirchhoff check on the assembled system (stiffness *and* the
    Robin far-field term), so any inconsistency between the energy
    recovery of $R$ and the boundary conditions shows up here.
    """
    from scipy.sparse.linalg import spsolve

    from groundfield.solver.fem import (
        _assemble_far_field_robin,
        _assemble_stiffness_axisymmetric,
        _build_axisymmetric_mesh,
    )

    a_eq = 0.2463023
    sigma = 1.0 / RHO
    mesh = _build_axisymmetric_mesh(a_eq, [])
    A = _assemble_stiffness_axisymmetric(
        mesh.nodes, mesh.triangles, np.full(mesh.triangles.shape[0], sigma)
    )
    A = A + _assemble_far_field_robin(
        mesh.nodes, mesh.outer_edges,
        np.full(mesh.outer_edges.shape[0], sigma), mesh.r_far,
    )
    dirichlet = np.zeros(mesh.nodes.shape[0], dtype=bool)
    dirichlet[mesh.inner_nodes] = True
    phi = np.zeros(mesh.nodes.shape[0])
    phi[mesh.inner_nodes] = 1.0
    free = ~dirichlet
    phi[free] = spsolve(
        A[free][:, free].tocsc(), -(A[free][:, dirichlet] @ phi[dirichlet])
    )
    energy = float(phi @ (A @ phi))
    current = float(np.ones(mesh.nodes.shape[0]) @ (A @ phi))
    assert energy == pytest.approx(current, rel=1e-10)
    # and the potential is the analytic monopole a_eq / r
    r = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    assert np.abs(phi - a_eq / r).max() < 1e-3


def test_fem_metadata_reports_the_mesh_actually_used() -> None:
    """Mesh size, truncation radius and far-field model are reported.

    The mesh is an accuracy knob now, so a result must carry enough
    provenance to reproduce it.
    """
    res = solve_fem(
        _rod_world(gf.HomogeneousSoil(resistivity=RHO)), _engine(),
        n_radial=29, n_axial=19,
    )
    assert res.metadata["far_field"] == "robin"
    assert res.metadata["n_radial"] == 29
    assert res.metadata["n_axial"] == 19
    info = res.metadata["fem_mesh"]["g1"]
    assert info["n_nodes"] == 29 * 19
    assert info["n_electrode_nodes"] == 19
    assert info["n_triangles"] == 2 * 28 * 18
    a_eq = res.metadata["equivalent_hemisphere_radius"]["g1"]
    assert info["r_far"] == pytest.approx(30.0 * (a_eq + 1.0))
