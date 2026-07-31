"""Regression tests for review-pass-9 finding **F16** (0.15.0).

F16: the ADR-0012 V1 concrete-shell registry was written by
:meth:`GroundingSystemSpec._build_one` under the *electrode's own* name
but read by the PEN builders under the **cluster anchor** name, so the
lumped shell resistance was silently dropped whenever the foundation
electrode was not the first surviving electrode of its site. Switching
concrete on then had no effect at all.

The fix folds a site's per-electrode entries onto the anchor inside
:meth:`GroundingSystemSpec.build_at`, once the bonds that define the
galvanic cluster are in place.

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
    size_m=10.0, style="ring", n_x=1, n_y=1, concrete_thickness_m=0.05,
)
ROD = RodElectrodeSpec(length_m=1.5, wire_radius_m=0.008)


# ---------------------------------------------------------------------
# The defect: registry key must be the anchor, not the electrode
# ---------------------------------------------------------------------


def test_shell_registered_when_foundation_is_first():
    """Control: this case worked before 0.15.0 and must keep working."""
    world, anchor = _build([FOUNDATION, ROD], rho_c=150.0)
    assert anchor is not None
    assert list(world.concrete_shell_corrections) == [anchor]
    assert world.concrete_shell_corrections[anchor] > 0.0


def test_shell_registered_when_foundation_is_not_first():
    """The F16 case: foundation at index 1 — key must still be the anchor.

    Before 0.15.0 the entry sat under ``house_foundation_1`` while the
    consumers looked up ``house_rod_0``, so the shell was dropped.
    """
    world, anchor = _build([ROD, FOUNDATION], rho_c=150.0)
    assert anchor is not None
    assert anchor.startswith("house_rod")
    assert list(world.concrete_shell_corrections) == [anchor], (
        "shell must be keyed by the cluster anchor, not by the "
        "foundation electrode's own name"
    )
    assert world.concrete_shell_corrections[anchor] > 0.0


def test_no_registry_entry_without_concrete():
    """``concrete_rho_ohm_m=None`` keeps the historic bare-wire path."""
    world, _ = _build([ROD, FOUNDATION], rho_c=None)
    assert world.concrete_shell_corrections == {}


def test_shells_of_several_encased_electrodes_combine_in_parallel():
    """Two encased foundations in one cluster combine in parallel.

    Each electrode's ``R_i = rho_c / (2*pi*L_perim,i) * ln(r_b/r_a)`` is
    already the parallel reduction of the radial shell over its own
    perimeter, and each leaks its share of the cluster current through
    its *own* shell into the soil. Two independent radial paths from one
    equipotential conductor to earth therefore give
    ``R_eq = (sum_i 1/R_i)**-1`` — for two identical foundations ``R/2``,
    not ``2R``. Summing them (the first cut of the 0.15.0 F16 fix, caught
    in adversarial review) overestimates a two-foundation building by a
    factor of four.
    """
    world_one, anchor_one = _build([FOUNDATION], rho_c=150.0)
    world_two, anchor_two = _build([FOUNDATION, FOUNDATION], rho_c=150.0)
    assert anchor_one is not None and anchor_two is not None
    single = world_one.concrete_shell_corrections[anchor_one]
    double = world_two.concrete_shell_corrections[anchor_two]
    assert list(world_two.concrete_shell_corrections) == [anchor_two]
    assert double == pytest.approx(0.5 * single, rel=1e-12)
    assert double < single, "a second parallel leakage path must lower R"


# ---------------------------------------------------------------------
# The consequence: rho_c must actually move the result
# ---------------------------------------------------------------------


@pytest.mark.parametrize("order", [[FOUNDATION, ROD], [ROD, FOUNDATION]])
def test_shell_scales_with_concrete_resistivity(order):
    """R_shell is proportional to rho_c for both electrode orders.

    The Sunde shell resistance is
    ``R = rho_c / (2*pi*L_perim) * ln(r_b / r_a)``, i.e. linear in
    ``rho_c`` at fixed geometry. Before 0.15.0 the ``[ROD, FOUNDATION]``
    order returned *no* entry at all, so a factor-33 change in rho_c was
    invisible.
    """
    _, anchor_a = _build(order, rho_c=150.0)
    world_a, _ = _build(order, rho_c=150.0)
    world_b, anchor_b = _build(order, rho_c=5000.0)
    assert anchor_a is not None and anchor_b is not None
    r_low = world_a.concrete_shell_corrections[anchor_a]
    r_high = world_b.concrete_shell_corrections[anchor_b]
    assert r_high == pytest.approx(r_low * 5000.0 / 150.0, rel=1e-9)


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
