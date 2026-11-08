"""Regression tests for review-pass-9 finding **F16** (0.15.0).

F16: the ADR-0012 V1 concrete-shell registry was written by
:meth:`GroundingSystemSpec._build_one` under the *electrode's own* name
but read by the PEN builders under the **cluster anchor** name, so the
lumped shell resistance was silently dropped whenever the foundation
electrode was not the first surviving electrode of its site. Switching
concrete on then had no effect at all.

0.15.0 folded a site's per-electrode entries onto the anchor. Since
2026.11 (review pass 10, 10.12 / 10.13) the registry is gone from the
generators: the shell is the coefficient ``concrete_shell_coefficient_ohm_m``
on the encased foundation wire itself, so it cannot be looked up under
the wrong name, rods bonded to the foundation do not leak through it, and
several encased electrodes of one site each carry their own shell (they
combine in parallel by construction). The tests below pin that for both
electrode orders.

This finding spans the generator/solver group boundary of the 0.15.0
implementation split, which is why it lives in its own module rather than
in ``tests/test_pass9_generators.py`` or ``tests/test_pass9_image_core.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf
from groundfield.generators.electrode_specs import (
    FoundationElectrodeSpec,
    RodElectrodeSpec,
)
from groundfield.generators.grounding import GroundingSystemSpec
from groundfield.soil.models import HomogeneousSoil


def _build(electrodes, *, rho_c: float | None) -> tuple[gf.World, str | None]:
    """Materialise one site and return ``(world, anchor)``."""
    specs = []
    for spec in electrodes:
        if isinstance(spec, FoundationElectrodeSpec):
            spec = spec.model_copy(update={"concrete_rho_ohm_m": rho_c})
        specs.append(spec)
    world = gf.create_world(soil=HomogeneousSoil(resistivity=100.0))
    system = GroundingSystemSpec(electrodes=specs)
    anchor = system.build_at(
        world,
        site_xy=(0.0, 0.0),
        name_prefix="house",
        rng=np.random.default_rng(0),
    )
    return world, anchor


FOUNDATION = FoundationElectrodeSpec(
    size_m=10.0,
    style="ring",
    n_x=1,
    n_y=1,
    concrete_thickness_m=0.05,
)
ROD = RodElectrodeSpec(length_m=1.5, wire_radius_m=0.008)


def _shells(world: gf.World) -> dict[str, float]:
    return {
        e.name: e.concrete_shell_coefficient_ohm_m
        for e in world.electrodes
        if e.concrete_shell_coefficient_ohm_m > 0.0
    }


# ---------------------------------------------------------------------
# The defect: the shell must reach the solve whatever the electrode order
# ---------------------------------------------------------------------


@pytest.mark.finding("10.12")
@pytest.mark.parametrize("order", [[FOUNDATION, ROD], [ROD, FOUNDATION]])
def test_shell_sits_on_the_foundation_in_either_order(order):
    """The F16 case (foundation at index 1) and its control: the shell is
    on the foundation wires, never on the rod, and no registry is used. The
    ring is one strip per side since user decision Q38 (one grid mesh up to
    then): four wires, one coefficient."""
    world, anchor = _build(order, rho_c=150.0)
    assert anchor is not None
    assert world.concrete_shell_corrections == {}
    shells = _shells(world)
    assert len(shells) == 4
    assert all("foundation" in name and "rod" not in name for name in shells)
    assert len(set(shells.values())) == 1


def test_no_shell_without_concrete():
    """``concrete_rho_ohm_m=None`` keeps the historic bare-wire path."""
    world, _ = _build([ROD, FOUNDATION], rho_c=None)
    assert world.concrete_shell_corrections == {}
    assert _shells(world) == {}


@pytest.mark.finding("10.12")
def test_shells_of_several_encased_electrodes_combine_in_parallel():
    """Two encased foundations in one cluster combine in parallel.

    Each foundation leaks its share of the cluster current through its
    *own* shell, ``R_i = C / L_perim,i``; two independent radial paths
    from one equipotential conductor give ``R_eq = (sum_i 1/R_i)**-1`` —
    for two identical foundations ``R/2``, not ``2R`` (the first cut of the
    0.15.0 F16 fix summed them). With the shell on each electrode the
    parallel combination is built into the solve: both carry the same C.
    """
    world_two, _ = _build([FOUNDATION, FOUNDATION], rho_c=150.0)
    shells = _shells(world_two)
    # two foundations of four sides each (user decision Q38)
    assert len(shells) == 8
    (c,) = set(shells.values())
    assert c > 0.0


# ---------------------------------------------------------------------
# The consequence: rho_c must actually move the result
# ---------------------------------------------------------------------


@pytest.mark.finding("10.12")
@pytest.mark.parametrize("order", [[FOUNDATION, ROD], [ROD, FOUNDATION]])
def test_shell_scales_with_concrete_resistivity(order):
    """C is proportional to rho_c for both electrode orders.

    The Sunde shell coefficient is ``C = rho_c / (2*pi) * ln(r_b / r_a)``,
    i.e. linear in ``rho_c`` at fixed geometry. Before 0.15.0 the
    ``[ROD, FOUNDATION]`` order applied *no* shell at all, so a factor-33
    change in rho_c was invisible; the cluster impedance must rise too.
    """
    world_a, anchor_a = _build(order, rho_c=150.0)
    world_b, anchor_b = _build(order, rho_c=5000.0)
    (c_low,) = set(_shells(world_a).values())
    (c_high,) = set(_shells(world_b).values())
    assert c_high == pytest.approx(c_low * 5000.0 / 150.0, rel=1e-12)

    def z(world, anchor):
        gf.create_source(world, attached_to=anchor, magnitude=1.0)
        engine = gf.create_engine(backend="image", segment_length=1.0, kernel="point")
        return world.solve(engine).input_impedance()[0].real

    assert z(world_b, anchor_b) > z(world_a, anchor_a)


def test_distributed_model_writes_no_lumped_entry():
    """The V2 'distributed' path subsumes the lumped registry."""
    spec = FOUNDATION.model_copy(
        update={"concrete_rho_ohm_m": 150.0, "concrete_model": "distributed"}
    )
    world = gf.create_world(soil=HomogeneousSoil(resistivity=100.0))
    anchor = GroundingSystemSpec(electrodes=[ROD, spec]).build_at(
        world,
        site_xy=(0.0, 0.0),
        name_prefix="house",
        rng=np.random.default_rng(0),
    )
    assert anchor is not None
    assert world.concrete_shell_corrections == {}
