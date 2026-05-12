"""Tests for the electrode-spec and grounding-system layers.

Validation programme of ADR-0009 (extended generator framework):

* every concrete electrode spec round-trips through JSON;
* presence_prob honoured (Bernoulli per realisation);
* a multi-electrode grounding system creates the right electrodes
  + a bonding chain that makes them one cluster;
* offset_xy_m correctly translates each electrode;
* rod_circle helper produces N rods on a circle of given radius.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pydantic import BaseModel, Field, ValidationError

import groundfield as gf
from groundfield.generators import (
    ElectrodeSpec,
    FoundationElectrodeSpec,
    GroundingSystemSpec,
    HomogeneousSoilSpec,
    RingElectrodeSpec,
    RodElectrodeSpec,
    StripElectrodeSpec,
    rod_circle,
)


# ---------------------------------------------------------------------
# JSON round-trip per electrode spec
# ---------------------------------------------------------------------


class _Wrapper(BaseModel):
    """Helper that puts an ElectrodeSpec inside the discriminated union."""

    e: ElectrodeSpec


@pytest.mark.parametrize(
    "spec",
    [
        RodElectrodeSpec(length_m=2.0, depth_m=0.0, offset_xy_m=(1.0, 0.5)),
        RingElectrodeSpec(radius_m=4.0, depth_m=0.6),
        StripElectrodeSpec(length_m=20.0, orientation_deg=45.0),
        FoundationElectrodeSpec(size_m=10.0, depth_m=0.8, n_x=3, n_y=3),
    ],
)
def test_electrode_spec_json_roundtrip(spec) -> None:
    payload = _Wrapper(e=spec).model_dump_json()
    restored = _Wrapper.model_validate_json(payload).e
    assert type(restored) is type(spec)
    assert restored.model_dump() == spec.model_dump()


def test_electrode_spec_discriminator_visible() -> None:
    payload = json.loads(_Wrapper(e=RodElectrodeSpec(length_m=1.5)).model_dump_json())
    assert payload["e"]["kind"] == "rod"


# ---------------------------------------------------------------------
# rod_circle
# ---------------------------------------------------------------------


def test_rod_circle_places_n_rods_on_circle() -> None:
    rods = rod_circle(n=4, radius_m=2.0, length_m=2.5)
    assert len(rods) == 4
    for rod in rods:
        assert isinstance(rod, RodElectrodeSpec)
        ox, oy = rod.offset_xy_m
        r = math.hypot(ox, oy)
        assert math.isclose(r, 2.0, rel_tol=1e-9)


def test_rod_circle_angles_evenly_spaced() -> None:
    rods = rod_circle(n=8, radius_m=3.0)
    angles = sorted(math.atan2(r.offset_xy_m[1], r.offset_xy_m[0]) % (2 * math.pi)
                    for r in rods)
    diffs = np.diff(angles)
    assert np.allclose(diffs, np.full_like(diffs, 2 * math.pi / 8), atol=1e-9)


def test_rod_circle_rejects_invalid_n() -> None:
    with pytest.raises(ValueError, match="n must be"):
        rod_circle(n=0, radius_m=1.0)


# ---------------------------------------------------------------------
# GroundingSystemSpec.build_at
# ---------------------------------------------------------------------


def _world_with_homogeneous_soil() -> "gf.World":
    soil = HomogeneousSoilSpec(resistivity=100.0).to_soil(np.random.default_rng(0))
    return gf.create_world(name="test", soil=soil)


def test_grounding_system_build_creates_electrodes_and_bonds() -> None:
    """A multi-electrode system creates each electrode and bonds them."""
    grounding = GroundingSystemSpec(
        electrodes=[
            RingElectrodeSpec(radius_m=4.0, depth_m=0.6),
            *rod_circle(n=4, radius_m=2.0, length_m=2.5),
        ],
    )
    world = _world_with_homogeneous_soil()
    rng = np.random.default_rng(0)
    anchor = grounding.build_at(world, site_xy=(0.0, 0.0),
                                name_prefix="trafo", rng=rng)
    assert anchor is not None
    # 1 ring + 4 rods = 5 electrodes
    assert len(world.electrodes) == 5
    # 4 bonds (anchor → each non-anchor)
    bonds = [c for c in world.conductors if "_bond_" in c.name]
    assert len(bonds) == 4
    # Anchor name should be the first present electrode
    assert anchor.startswith("trafo_ring")


def test_grounding_system_offset_translates_electrode() -> None:
    grounding = GroundingSystemSpec(
        electrodes=[
            RodElectrodeSpec(length_m=1.0, offset_xy_m=(3.0, -2.0)),
        ],
    )
    world = _world_with_homogeneous_soil()
    grounding.build_at(world, site_xy=(10.0, 20.0),
                       name_prefix="house_0", rng=np.random.default_rng(0))
    rod = next(e for e in world.electrodes if e.name.startswith("house_0_"))
    # Position = site + offset
    assert math.isclose(rod.position[0], 13.0, rel_tol=1e-9)
    assert math.isclose(rod.position[1], 18.0, rel_tol=1e-9)


def test_grounding_system_presence_prob_bernoulli() -> None:
    """presence_prob = 0.0 always rejects, presence_prob = 1.0 always keeps."""
    grounding = GroundingSystemSpec(
        electrodes=[
            RodElectrodeSpec(length_m=1.0, presence_prob=1.0),
            RodElectrodeSpec(length_m=1.0, presence_prob=0.0,
                             offset_xy_m=(5.0, 0.0)),
        ],
    )
    world = _world_with_homogeneous_soil()
    grounding.build_at(world, site_xy=(0.0, 0.0),
                       name_prefix="b0", rng=np.random.default_rng(0))
    # Only the present rod ends up in the world.
    assert len(world.electrodes) == 1


def test_grounding_system_returns_none_when_all_absent() -> None:
    grounding = GroundingSystemSpec(
        electrodes=[
            RodElectrodeSpec(length_m=1.0, presence_prob=0.0),
        ],
    )
    world = _world_with_homogeneous_soil()
    anchor = grounding.build_at(world, site_xy=(0.0, 0.0),
                                name_prefix="b0",
                                rng=np.random.default_rng(0))
    assert anchor is None
    assert len(world.electrodes) == 0


def test_grounding_system_strip_orientation() -> None:
    """A strip with orientation_deg=90 lies along the y axis."""
    grounding = GroundingSystemSpec(
        electrodes=[
            StripElectrodeSpec(length_m=10.0, orientation_deg=90.0,
                               offset_xy_m=(0.0, 0.0)),
        ],
    )
    world = _world_with_homogeneous_soil()
    grounding.build_at(world, site_xy=(0.0, 0.0),
                       name_prefix="b0", rng=np.random.default_rng(0))
    strip = world.electrodes[0]
    sx, sy, _ = strip.start
    ex, ey, _ = strip.end
    # Length stays 10 m
    assert math.isclose(math.hypot(ex - sx, ey - sy), 10.0, rel_tol=1e-6)
    # And the strip is along ±y (cos(90°)=0, sin(90°)=1)
    assert math.isclose(abs(ey - sy), 10.0, rel_tol=1e-6)
    assert abs(ex - sx) < 1e-6


def test_foundation_ring_style_uses_perimeter_only() -> None:
    """``style='ring'`` produces a GridMeshElectrode with n_x=n_y=1."""
    grounding = GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(size_m=10.0, style="ring",
                                    n_x=4, n_y=4),  # n_x/n_y must be ignored
        ],
    )
    world = _world_with_homogeneous_soil()
    grounding.build_at(world, site_xy=(0.0, 0.0),
                       name_prefix="house_0", rng=np.random.default_rng(0))
    e = world.electrodes[0]
    assert e.kind == "grid_mesh"
    assert e.n_x == 1
    assert e.n_y == 1


def test_foundation_mesh_style_honours_n_x_n_y() -> None:
    """``style='mesh'`` (default) keeps the configured n_x / n_y."""
    grounding = GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(size_m=10.0, style="mesh",
                                    n_x=3, n_y=4),
        ],
    )
    world = _world_with_homogeneous_soil()
    grounding.build_at(world, site_xy=(0.0, 0.0),
                       name_prefix="house_0", rng=np.random.default_rng(0))
    e = world.electrodes[0]
    assert e.n_x == 3
    assert e.n_y == 4


def test_foundation_default_style_is_mesh() -> None:
    """The default ``style`` is ``'mesh'`` (one inner cross-brace each)."""
    spec = FoundationElectrodeSpec(size_m=10.0)
    assert spec.style == "mesh"
    assert spec.n_x == 2 and spec.n_y == 2


def test_grounding_system_json_roundtrip() -> None:
    grounding = GroundingSystemSpec(
        electrodes=[
            RingElectrodeSpec(radius_m=4.0, depth_m=0.6),
            RodElectrodeSpec(length_m=2.0, offset_xy_m=(1.0, 0.0)),
            FoundationElectrodeSpec(size_m=10.0, depth_m=0.8, n_x=2, n_y=2),
        ],
    )
    payload = grounding.model_dump_json()
    restored = GroundingSystemSpec.model_validate_json(payload)
    assert len(restored.electrodes) == 3
    assert isinstance(restored.electrodes[0], RingElectrodeSpec)
    assert isinstance(restored.electrodes[1], RodElectrodeSpec)
    assert isinstance(restored.electrodes[2], FoundationElectrodeSpec)
