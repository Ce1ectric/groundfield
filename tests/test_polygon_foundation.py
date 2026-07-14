"""Polygon foundation ring (``PolylineElectrode``) — regression suite.

Motivation (AP1): a foundation ring placed on the *oriented bounding
rectangle* (OMBR) of a building footprint

* overestimates the electrode perimeter of L-/U-shaped buildings, and
* lets the rectangles of neighbouring buildings coincide or overlap,
  which drives distinct segments into the solver's 1 mm singularity
  clamp (saturated mutual coupling — silently wrong numbers).

Following the real polygon removes both artefacts. The tests below pin

1. the *equivalence* of a rectangular polygon with the historic
   ``GridMeshElectrode`` ring (no regression on the existing model),
2. the concrete-shell bookkeeping on the polygon perimeter,
3. the geometric correctness (perimeter, L-shape bias),
4. the degenerate-edge cleanup, and
5. clamp-freedom for two houses sharing a wall.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.generators.grounding import _clean_polygon

RECT_POLY = ((-5.0, -4.0), (5.0, -4.0), (5.0, 4.0), (-5.0, 4.0))
RECT_SIZE = (10.0, 8.0)


def _foundation_world(spec_kwargs, *, soil=None, site_xy=(0.0, 0.0), source=True):
    soil = soil or gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    spec = gf.GroundingSystemSpec(
        electrodes=[
            gf.FoundationElectrodeSpec(
                style="ring", depth_m=0.7, wire_radius_m=0.0075, **spec_kwargs
            )
        ]
    )
    anchor = spec.build_at(
        world, site_xy=site_xy, name_prefix="h", rng=np.random.default_rng(1)
    )
    if source:
        gf.create_source(world, attached_to=anchor, magnitude=1.0)
    return world, anchor


def _z(world, anchor, ds=0.5):
    eng = gf.create_engine(backend="image", segment_length=ds, frequencies=[50.0])
    return complex(world.solve(eng).cluster_impedance(anchor)[0])


# ---------------------------------------------------------------------
# 1) Equivalence with the historic rectangular ring
# ---------------------------------------------------------------------


def test_rectangular_polygon_matches_grid_mesh_ring_exactly():
    """A polygon that *is* the rectangle must reproduce the old ring."""
    w_rect, a_rect = _foundation_world({"size_xy_m": RECT_SIZE})
    w_poly, a_poly = _foundation_world({"polygon_xy_m": RECT_POLY})

    assert type(w_rect.electrodes[0]).__name__ == "GridMeshElectrode"
    assert type(w_poly.electrodes[0]).__name__ == "PolylineElectrode"

    z_rect, z_poly = _z(w_rect, a_rect), _z(w_poly, a_poly)
    assert abs(z_poly - z_rect) <= 1e-12 * abs(z_rect)


def test_polygon_ring_is_one_electrode_no_internal_bonds():
    """One primitive, one cluster — the strip chain needed 4 + 3 bonds."""
    world, _ = _foundation_world({"polygon_xy_m": RECT_POLY}, source=False)
    assert len(world.electrodes) == 1
    assert len(world.conductors) == 0


# ---------------------------------------------------------------------
# 2) Concrete shell (ADR-0012) on the polygon perimeter
# ---------------------------------------------------------------------


def test_lumped_shell_uses_polygon_perimeter():
    """R_shell = C / L_perim with L_perim of the *polygon*."""
    kw = dict(concrete_rho_ohm_m=100.0, concrete_thickness_m=0.05,
              concrete_model="lumped")
    w_rect, _ = _foundation_world({"size_xy_m": RECT_SIZE, **kw}, source=False)
    w_poly, _ = _foundation_world({"polygon_xy_m": RECT_POLY, **kw}, source=False)
    (r_rect,) = w_rect.concrete_shell_corrections.values()
    (r_poly,) = w_poly.concrete_shell_corrections.values()
    assert r_poly == pytest.approx(r_rect, rel=1e-12)

    # Analytic check: C / L, with C = rho_c/(2 pi) ln(r_b/r_a).
    coeff = 100.0 / (2.0 * math.pi) * math.log((0.0075 + 0.05) / 0.0075)
    assert r_poly == pytest.approx(coeff / 36.0, rel=1e-12)


def test_distributed_shell_rides_on_every_ring_segment():
    """A shell more resistive than the soil raises the spreading impedance."""
    kw = dict(concrete_rho_ohm_m=500.0, concrete_thickness_m=0.05,
              concrete_model="distributed")
    world, anchor = _foundation_world({"polygon_xy_m": RECT_POLY, **kw})
    e = world.electrodes[0]
    assert e.concrete_shell_coefficient_ohm_m > 0.0
    assert not world.concrete_shell_corrections  # subsumed by the per-segment path

    w_bare, a_bare = _foundation_world({"polygon_xy_m": RECT_POLY})
    assert _z(world, anchor).real > _z(w_bare, a_bare).real


def test_transparent_shell_recovers_the_bare_ring():
    r"""$\rho_c = \rho_\text{soil}$ — the concrete is indistinguishable
    from soil, so both shell paths must return the bare-wire result.

    * ``distributed`` — the inflated wire radius and the added radial
      diagonal $C/\Delta s$ cancel exactly.
    * ``lumped`` — the inflated radius alone *under*-states the
      spreading impedance; the missing part is exactly the lumped
      shell resistance the TN generator injects on the service drop.

    This pins the whole ADR-0012 bookkeeping on the polygon ring.
    """
    rho = 100.0
    soil = gf.HomogeneousSoil(resistivity=rho)
    w_bare, a_bare = _foundation_world({"polygon_xy_m": RECT_POLY}, soil=soil)
    z_bare = _z(w_bare, a_bare).real

    kw = dict(concrete_rho_ohm_m=rho, concrete_thickness_m=0.05)
    w_dist, a_dist = _foundation_world(
        {"polygon_xy_m": RECT_POLY, "concrete_model": "distributed", **kw}, soil=soil)
    assert _z(w_dist, a_dist).real == pytest.approx(z_bare, rel=1e-5)

    w_lump, a_lump = _foundation_world(
        {"polygon_xy_m": RECT_POLY, "concrete_model": "lumped", **kw}, soil=soil)
    (r_shell,) = w_lump.concrete_shell_corrections.values()
    assert _z(w_lump, a_lump).real + r_shell == pytest.approx(z_bare, rel=1e-3)


# ---------------------------------------------------------------------
# 3) Geometry
# ---------------------------------------------------------------------


def test_polyline_length_and_edges():
    world, _ = _foundation_world({"polygon_xy_m": RECT_POLY}, source=False)
    e = world.electrodes[0]
    assert len(e.edges) == 4                      # closing edge included
    assert e.length == pytest.approx(36.0)        # 2 * (10 + 8)
    assert all(v[2] == pytest.approx(0.7) for v in e.vertices)


def test_polygon_is_placed_relative_to_site_centre():
    world, _ = _foundation_world({"polygon_xy_m": RECT_POLY},
                                 site_xy=(100.0, -50.0), source=False)
    xs = [v[0] for v in world.electrodes[0].vertices]
    ys = [v[1] for v in world.electrodes[0].vertices]
    assert min(xs) == pytest.approx(95.0) and max(xs) == pytest.approx(105.0)
    assert min(ys) == pytest.approx(-54.0) and max(ys) == pytest.approx(-46.0)


def test_l_shaped_building_ombr_underestimates_resistance():
    """The OMBR of an L-shape is a *larger* electrode — lower R.

    Quantifies the systematic bias the polygon ring removes.
    """
    l_shape = ((-5.0, -4.0), (5.0, -4.0), (5.0, 0.0),
               (0.0, 0.0), (0.0, 4.0), (-5.0, 4.0))
    w_poly, a_poly = _foundation_world({"polygon_xy_m": l_shape})
    w_ombr, a_ombr = _foundation_world({"size_xy_m": RECT_SIZE})  # its OMBR
    r_poly, r_ombr = _z(w_poly, a_poly).real, _z(w_ombr, a_ombr).real
    assert r_ombr < r_poly
    assert 0.02 < (r_poly - r_ombr) / r_poly < 0.20


# ---------------------------------------------------------------------
# 4) Degenerate input
# ---------------------------------------------------------------------


def test_clean_polygon_merges_short_edges_and_drops_repeated_closing_vertex():
    noisy = ((-5.0, -4.0), (-4.999, -4.0), (5.0, -4.0), (5.0, 4.0),
             (5.0, 4.0), (-5.0, 4.0), (-5.0, -4.0))
    cleaned = _clean_polygon(noisy, min_edge_m=0.5)
    assert cleaned == [(-5.0, -4.0), (5.0, -4.0), (5.0, 4.0), (-5.0, 4.0)]


def test_short_edges_do_not_trip_the_thin_wire_guard():
    noisy = ((-5.0, -4.0), (-4.999, -4.0), (5.0, -4.0), (5.0, 4.0), (-5.0, 4.0))
    world, anchor = _foundation_world({"polygon_xy_m": noisy})
    assert world.electrodes[0].length == pytest.approx(36.0)
    assert _z(world, anchor).real > 0.0


def test_polygon_collapsing_to_a_point_raises():
    tiny = ((0.0, 0.0), (0.1, 0.0), (0.1, 0.1))
    with pytest.raises(ValueError, match="fewer than 3 vertices"):
        _foundation_world({"polygon_xy_m": tiny}, source=False)


def test_mesh_style_with_polygon_is_rejected():
    with pytest.raises(ValueError, match="style='ring'"):
        gf.FoundationElectrodeSpec(style="mesh", polygon_xy_m=RECT_POLY)


# ---------------------------------------------------------------------
# 5) The actual motivation: no clamp for two houses sharing a wall
# ---------------------------------------------------------------------


def test_shared_wall_neighbours_do_not_hit_the_singularity_clamp():
    """Two terraced houses: real outlines touch, OMBRs coincide.

    With the OMBR ring the party-wall edges fall on top of each other
    and the solver warns about segments inside the 1 mm clamp. The
    polygon ring keeps the two outlines distinct (they share the wall
    line only where the buildings really do), and — because each ring
    follows its own outline — no *distinct* segments end up coincident
    once the two footprints are offset by the wall thickness.
    """
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    left = ((-8.0, -4.0), (-0.15, -4.0), (-0.15, 4.0), (-8.0, 4.0))
    right = ((0.15, -4.0), (8.0, -4.0), (8.0, 4.0), (0.15, 4.0))
    for i, poly in enumerate((left, right)):
        spec = gf.GroundingSystemSpec(electrodes=[gf.FoundationElectrodeSpec(
            style="ring", polygon_xy_m=poly, depth_m=0.7, wire_radius_m=0.0075)])
        spec.build_at(world, site_xy=(0.0, 0.0), name_prefix=f"h{i}",
                      rng=np.random.default_rng(i))
    gf.create_source(world, attached_to=world.electrodes[0].name, magnitude=1.0)

    eng = gf.create_engine(backend="image", segment_length=0.5, frequencies=[50.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        world.solve(eng)
    assert not [w for w in caught if "clamp" in str(w.message)]
