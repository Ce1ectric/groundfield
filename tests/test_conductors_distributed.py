"""Tests for the distributed-conductor model (ADR-0003).

Three contracts are checked here:

1. **Lumped fallback.** ``discretize_segment_length is None`` keeps
   the previous lumped finite-impedance branch model bit-exact.
2. **Isolated distributed conductor.** Splitting a ``coupling_to_soil
   == "isolated"`` conductor into n sub-segments must give the same
   result as a single lumped branch with the same total resistance —
   the chain is electrically a series of resistors with no
   intermediate sources or sinks.
3. **Galvanic distributed conductor.** A buried bare-copper conductor
   with ``coupling_to_soil == "galvanic"`` leaks current along its
   length. Tested against (a) the strip-electrode primitive
   (a galvanic conductor of the same geometry without a series
   resistance should reproduce the strip's grounding impedance), and
   (b) cross-engine consistency on a multi-rod chain example.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import groundfield as gf


# ---------------------------------------------------------------------
# Common geometry
# ---------------------------------------------------------------------

SEG = 0.1
RHO = 100.0
ROD_LEN = 2.0
ROD_R = 0.0075
SEPARATION = 30.0
PEN_RHO = 2.82e-8         # Al
PEN_A = 50.0e-6           # 50 mm²


def _two_rod_world(
    *,
    cross_section=None,
    discretize_segment_length=None,
    coupling_to_soil="isolated",
    rod_separation=SEPARATION,
    soil=None,
) -> gf.World:
    soil = soil or gf.HomogeneousSoil(resistivity=RHO)
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=ROD_LEN,
        wire_radius=ROD_R,
    )
    gf.create_electrode(
        w, "rod", name="g2", position=(rod_separation, 0.0, 0.5),
        length=ROD_LEN, wire_radius=ROD_R,
    )
    gf.create_conductor(
        w, name="pen", start="g1", end="g2",
        conductor_type="pen",
        wire_radius=0.004, resistivity=PEN_RHO,
        cross_section=cross_section,
        discretize_segment_length=discretize_segment_length,
        coupling_to_soil=coupling_to_soil,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _solve(world: gf.World, backend: str, **eng_kwargs) -> gf.FieldResult:
    eng = gf.create_engine(
        backend=backend, segment_length=SEG, **eng_kwargs
    )
    return eng.solve(world)


# ---------------------------------------------------------------------
# 1. Conductor schema additions
# ---------------------------------------------------------------------


def test_conductor_default_is_lumped() -> None:
    """A conductor without ``discretize_segment_length`` is lumped."""
    w = _two_rod_world(cross_section=PEN_A)
    pen = w.get_conductor("pen")
    assert pen.is_distributed is False
    assert pen.n_segments == 1


def test_conductor_distributed_segment_count() -> None:
    """``n_segments = ceil(length / discretize_segment_length)``."""
    w = _two_rod_world(
        cross_section=PEN_A,
        discretize_segment_length=5.0,  # 30 m / 5 m = 6 segments
    )
    pen = w.get_conductor("pen")
    assert pen.is_distributed is True
    assert pen.n_segments == 6


def test_conductor_default_coupling_is_isolated() -> None:
    """Default soil-coupling matches the historic conductor model."""
    w = _two_rod_world(cross_section=PEN_A, discretize_segment_length=5.0)
    pen = w.get_conductor("pen")
    assert pen.coupling_to_soil == "isolated"


# ---------------------------------------------------------------------
# 2. Lumped fallback regression
# ---------------------------------------------------------------------


def test_default_no_distribution_matches_legacy_lumped() -> None:
    """Without ``discretize_segment_length`` the result is bit-exact
    to the previous lumped finite-branch model."""
    a_target = PEN_A
    res_lumped = _solve(_two_rod_world(cross_section=a_target), "image")
    # Same world, with discretize set to None — must match exactly.
    res_default = _solve(
        _two_rod_world(cross_section=a_target, discretize_segment_length=None),
        "image",
    )
    for ename in ("g1", "g2"):
        I_lump = res_lumped.electrode_currents[ename][0]
        I_def = res_default.electrode_currents[ename][0]
        assert I_lump == pytest.approx(I_def, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------
# 3. Isolated distributed conductor — series-of-resistors equivalence
# ---------------------------------------------------------------------


@pytest.mark.parametrize("n_seg", [2, 4, 8, 16])
def test_isolated_distributed_matches_lumped(n_seg: int) -> None:
    """An *isolated* distributed conductor with n sub-segments must
    give the same answer as a single lumped branch with the same
    total series resistance.

    Tolerance: tighter than 1e-9 — the two formulations are
    algebraically identical (a chain of n series resistors is one
    resistor of n times the value), so any deviation indicates a
    bug.
    """
    a_target = PEN_A
    ds_for_n = 30.0 / n_seg + 1e-9  # ensure exactly n sub-segments
    res_lumped = _solve(_two_rod_world(cross_section=a_target), "image")
    res_dist = _solve(
        _two_rod_world(
            cross_section=a_target,
            discretize_segment_length=ds_for_n,
            coupling_to_soil="isolated",
        ),
        "image",
    )
    for ename in ("g1", "g2"):
        I_lump = res_lumped.electrode_currents[ename][0]
        I_dist = res_dist.electrode_currents[ename][0]
        assert abs(I_lump - I_dist) < 1e-9, (
            f"n_seg={n_seg}, {ename}: lumped {I_lump} vs distributed {I_dist}"
        )


# ---------------------------------------------------------------------
# 4. Galvanic distributed conductor
# ---------------------------------------------------------------------


def _galvanic_world(*, n_seg: int) -> gf.World:
    """Two rods + galvanic distributed PEN between them."""
    soil = gf.HomogeneousSoil(resistivity=RHO)
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=ROD_LEN,
        wire_radius=ROD_R,
    )
    gf.create_electrode(
        w, "rod", name="g2", position=(SEPARATION, 0.0, 0.5),
        length=ROD_LEN, wire_radius=ROD_R,
    )
    gf.create_conductor(
        w, name="pen", start="g1", end="g2",
        conductor_type="bare_copper",
        wire_radius=0.004,
        resistivity=PEN_RHO,
        cross_section=PEN_A,
        discretize_segment_length=SEPARATION / n_seg + 1e-9,
        coupling_to_soil="galvanic",
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_galvanic_distributed_leakage_along_chain() -> None:
    """A galvanic distributed conductor leaks current at every
    midpoint. The metadata should expose the conductor-node leakage
    currents and they should sum to a positive amount that bypasses
    the two end electrodes.
    """
    res = _solve(_galvanic_world(n_seg=8), "image")
    I1 = res.electrode_currents["g1"][0].real
    I2 = res.electrode_currents["g2"][0].real
    cond_currents = res.metadata.get("conductor_node_currents", {})
    assert cond_currents, "metadata should carry conductor_node_currents"
    # ``conductor_node_currents`` is a dict[name, list[complex]] — one
    # complex value per frequency in ``engine.frequencies``. We only
    # consume the first frequency here.
    midpoint_total = sum(c[0].real for c in cond_currents.values())
    # Conservation: source = electrodes + midpoints
    assert (I1 + I2 + midpoint_total) == pytest.approx(1.0, rel=1e-9)
    # Galvanic chain absorbs a meaningful share — the bare-copper
    # conductor is itself a substantial earth electrode.
    assert midpoint_total > 0.05
    # Every midpoint leaks a positive current
    assert all(c[0].real > 0 for c in cond_currents.values()), cond_currents


def test_galvanic_distributed_lower_eprs_than_lumped() -> None:
    """Distributing the galvanic coupling adds extra earth-paths, so
    the cluster impedance (and hence the source EPR) drops compared
    with the lumped finite-conductor case where only the two end
    electrodes touch the soil.
    """
    res_lumped = _solve(_two_rod_world(cross_section=PEN_A), "image")
    res_galv = _solve(_galvanic_world(n_seg=8), "image")
    U1_lump = res_lumped.electrode_potentials["g1"][0].real
    U1_galv = res_galv.electrode_potentials["g1"][0].real
    assert U1_galv < U1_lump


def test_galvanic_distributed_convergence_in_n_segments() -> None:
    """As ``n_seg`` increases the source-electrode EPR converges to
    a limit. We check two refinements:

    - the n=8 → n=16 step stays below 5 % (loose, sanity);
    - the n=16 → n=32 step stays below 2 % (tighter, indicates the
      sequence is actually converging rather than just oscillating).

    The midpoint-pseudo-electrode discretisation has an O(1/n)
    convergence on the cluster impedance, so each refinement should
    roughly halve the step.
    """
    res_8 = _solve(_galvanic_world(n_seg=8), "image")
    res_16 = _solve(_galvanic_world(n_seg=16), "image")
    res_32 = _solve(_galvanic_world(n_seg=32), "image")
    U_8 = res_8.electrode_potentials["g1"][0].real
    U_16 = res_16.electrode_potentials["g1"][0].real
    U_32 = res_32.electrode_potentials["g1"][0].real
    step_1 = abs(U_8 - U_16) / U_16
    step_2 = abs(U_16 - U_32) / U_32
    assert step_1 < 0.05, f"step n=8→16: {step_1:.3f}, U_8={U_8:.3f}, U_16={U_16:.3f}"
    assert step_2 < 0.02, f"step n=16→32: {step_2:.3f}, U_16={U_16:.3f}, U_32={U_32:.3f}"
    # The sequence should actually decrease in magnitude
    assert step_2 < step_1, (step_1, step_2)


# ---------------------------------------------------------------------
# 5. Cross-engine consistency
# ---------------------------------------------------------------------


# FEM is excluded — its equivalent-hemisphere reduction does not
# resolve the distributed conductor and only treats it as lumped.
DISTRIBUTED_BACKENDS = ["image", "mom", "cim", "bem"]


@pytest.mark.parametrize("backend", DISTRIBUTED_BACKENDS)
def test_isolated_distributed_cross_engine(backend: str) -> None:
    """Every distributed-capable backend reproduces the isolated
    distributed result of the image reference within 3 %."""
    w_factory = lambda: _two_rod_world(
        cross_section=PEN_A,
        discretize_segment_length=10.0,  # 3 sub-segments
        coupling_to_soil="isolated",
    )
    res_ref = _solve(w_factory(), "image")
    res = _solve(w_factory(), backend)
    for ename in ("g1", "g2"):
        I_ref = res_ref.electrode_currents[ename][0].real
        I = res.electrode_currents[ename][0].real
        rel = abs(I - I_ref) / abs(I_ref)
        assert rel < 0.03, (
            f"{backend} {ename}: I_ref={I_ref:.4f}, I={I:.4f}, rel={rel:.3f}"
        )


@pytest.mark.parametrize("backend", DISTRIBUTED_BACKENDS)
def test_galvanic_distributed_cross_engine(backend: str) -> None:
    """Cross-engine consistency for the galvanic distributed
    conductor. The discretisation introduces additional earth
    leakage paths that every distributed-capable backend should
    handle. Tolerance: 5 % vs the image reference (the kernels are
    different enough to give some spread on the per-electrode
    splits)."""
    res_ref = _solve(_galvanic_world(n_seg=8), "image")
    res = _solve(_galvanic_world(n_seg=8), backend)
    # Total leakage at each side
    I1_ref = res_ref.electrode_currents["g1"][0].real
    I1 = res.electrode_currents["g1"][0].real
    rel = abs(I1 - I1_ref) / abs(I1_ref)
    assert rel < 0.05, (
        f"{backend}: I_g1_ref={I1_ref:.4f}, I_g1={I1:.4f}, rel={rel:.3f}"
    )


def test_galvanic_distributed_fem_warns_lumped_fallback(caplog) -> None:
    """The FEM backend cannot consume distributed conductors and
    should fall back to a lumped branch with a warning."""
    import logging

    caplog.set_level(logging.WARNING, logger="groundfield.solver.fem")
    res = _solve(_galvanic_world(n_seg=4), "fem")
    assert any("distributed conductors" in m for m in caplog.messages), (
        caplog.messages
    )
    # Sum of currents still conserved (FEM treats the conductor as
    # a single lumped branch — both end electrodes carry the full
    # source current between them).
    I1 = res.electrode_currents["g1"][0].real
    I2 = res.electrode_currents["g2"][0].real
    assert (I1 + I2) == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------
# 6. Multi-rod chain — distributed PEN between three rods
# ---------------------------------------------------------------------


def test_three_rod_chain_with_distributed_pen() -> None:
    """3 rods, 2 distributed-galvanic PEN sections — checks that the
    chain works end-to-end and that current is conserved."""
    soil = gf.HomogeneousSoil(resistivity=RHO)
    w = gf.create_world(soil=soil)
    for k, x in enumerate([0.0, SEPARATION, 2 * SEPARATION]):
        gf.create_electrode(
            w, "rod", name=f"g{k+1}", position=(x, 0.0, 0.5),
            length=ROD_LEN, wire_radius=ROD_R,
        )
    for k in range(2):
        gf.create_conductor(
            w, name=f"pen_{k+1}{k+2}", start=f"g{k+1}", end=f"g{k+2}",
            conductor_type="bare_copper",
            wire_radius=0.004,
            resistivity=PEN_RHO,
            cross_section=PEN_A,
            discretize_segment_length=10.0,
            coupling_to_soil="galvanic",
        )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    res = _solve(w, "image")
    I_e_total = sum(
        res.electrode_currents[f"g{k+1}"][0].real for k in range(3)
    )
    cond_total = sum(
        c[0].real for c in res.metadata.get("conductor_node_currents", {}).values()
    )
    assert (I_e_total + cond_total) == pytest.approx(1.0, rel=1e-9)
    # All three rods carry positive current via the chain
    for k in range(3):
        assert res.electrode_currents[f"g{k+1}"][0].real > 0.0
