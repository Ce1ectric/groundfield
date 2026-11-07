"""Solver-level behaviour of the inductive stack that lands in wave 2.

The coupling-side APIs exist since review pass 10 (stream COUP-1):
``image=`` conventions, ``touching="exact"``, multi-frequency builders,
``Conductor.resistance_model="skin"``. The solver switches to them in
wave 2 (stream COUP-2, on top of the new solver structure); until then
each test below is a strict ``xfail`` — it turns into a failure (XPASS)
the moment the solver change lands, so the marker has to be removed in
the same commit.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.conductors import Conductor

WAVE2 = pytest.mark.xfail(strict=True, reason="wave 2: solver integration")

RHO_SOIL = 100.0
RHO_AL = 2.82e-8
A_150 = 150e-6
A_RADIUS = math.sqrt(A_150 / math.pi)


def _pen_world(
    length: float,
    *,
    dsl: float | None,
    inductance_model: str | None = "neumann",
    resistance_model: str = "dc",
    end_z: float = 0.7,
) -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=RHO_SOIL))
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="g2", position=(length, 0.0, 0.5), length=3.0)
    w.add_conductor(
        Conductor(
            name="pen",
            start=(0.0, 0.0, 0.7),
            end=(length, 0.0, end_z),
            start_electrode="g1",
            end_electrode="g2",
            wire_radius=A_RADIUS,
            resistivity=RHO_AL,
            cross_section=A_150,
            discretize_segment_length=dsl,
            coupling_to_soil="isolated",
            inductance_model=inductance_model,
            resistance_model=resistance_model,
        )
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _series_impedance(w: gf.World, model: str, freqs: list[float]) -> np.ndarray:
    eng = gf.create_engine(
        backend="image",
        segment_length=0.5,
        frequencies=freqs,
        earth_inductive_model=model,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = w.solve(eng)
    v1 = np.asarray(res.electrode_potentials["g1"])
    v2 = np.asarray(res.electrode_potentials["g2"])
    i2 = np.asarray(res.electrode_currents["g2"])
    return (v1 - v2) / i2


@pytest.mark.finding("10.20", "F32")
@pytest.mark.parametrize("freq", [50.0, 1000.0])
def test_carson_series_reactance_matches_carson(freq: float) -> None:
    """1 km isolated PEN: X = l * X'_Carson(GMR = a e^{-1/4}) within 1 %.

    v0.15.0: 2.04x (50 Hz) and 2.19x (1 kHz) — Carson's correction on
    top of the additive image (the solver now assembles it on the
    anti-parallel image, Carson's own baseline).
    """
    from groundfield.references.earth_return import carson_self_impedance

    length = 1000.0
    z = _series_impedance(_pen_world(length, dsl=10.0 + 1e-9), "carson_series", [freq])[
        0
    ]
    ref = length * carson_self_impedance(RHO_SOIL, freq, gmr=A_RADIUS * math.exp(-0.25))
    assert z.imag == pytest.approx(ref.imag, rel=1e-2)


@WAVE2
@pytest.mark.finding("10.24")
@pytest.mark.parametrize("dsl", [None, 500.0], ids=["lumped", "single_segment"])
def test_neumann_is_honoured_on_lumped_and_single_segment_conductors(dsl) -> None:
    """300 m PEN at 1 kHz: v0.15.0 gives 0.0564 + 0j (inductance dropped)."""
    z_dist = _series_impedance(_pen_world(300.0, dsl=10.0), "sommerfeld", [1000.0])[0]
    z = _series_impedance(_pen_world(300.0, dsl=dsl), "sommerfeld", [1000.0])[0]
    assert z.imag == pytest.approx(z_dist.imag, rel=2e-2)


@WAVE2
@pytest.mark.finding("10.37")
def test_skin_model_raises_the_pen_resistance_at_1khz() -> None:
    """150 mm² Al: R_ac/R_dc = 1.5496 at 1 kHz (solid-round Bessel)."""
    kw = dict(dsl=10.0)
    z_dc = _series_impedance(_pen_world(300.0, **kw), "sommerfeld", [1000.0])[0]
    z_skin = _series_impedance(
        _pen_world(300.0, resistance_model="skin", **kw), "sommerfeld", [1000.0]
    )[0]
    r_dc = RHO_AL * 300.0 / A_150
    assert (z_skin - z_dc).real == pytest.approx(0.5496 * r_dc, rel=1e-2)


@pytest.mark.finding("10.60")
def test_sommerfeld_pair_work_does_not_grow_with_the_number_of_frequencies(
    monkeypatch,
) -> None:
    """The solver builds the Sommerfeld matrices for all frequencies at once."""
    from groundfield.coupling import sommerfeld_inductance as si

    calls = {"n": 0}
    original = si._pair_values_none

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(si, "_pair_values_none", counting)
    _series_impedance(_pen_world(60.0, dsl=10.0), "sommerfeld", [50.0])
    one = calls["n"]
    calls["n"] = 0
    _series_impedance(_pen_world(60.0, dsl=10.0), "sommerfeld", [50.0, 250.0, 1000.0])
    assert calls["n"] == one


@WAVE2
@pytest.mark.finding("10.46")
def test_inductance_is_refinement_invariant_for_a_straight_conductor() -> None:
    """Exact touching pairs: the chain sum does not drift with refinement
    (16x16 quadrature: about 8e-5 between 4 and 40 segments)."""
    z4 = _series_impedance(_pen_world(300.0, dsl=75.0), "perfect_mirror", [1000.0])[0]
    z40 = _series_impedance(_pen_world(300.0, dsl=7.5), "perfect_mirror", [1000.0])[0]
    assert z40.imag == pytest.approx(z4.imag, rel=1e-7)


@pytest.mark.finding("10.46", "F52")
def test_sommerfeld_accepts_a_surface_crossing_conductor() -> None:
    """With the exact (image-free) baseline a lead may leave the soil.

    With the additive image the crossing segment is singular and the
    solve raised; with ``image="none"`` the coupling side splits the
    segment at z = 0.
    """
    w = _pen_world(60.0, dsl=10.0, end_z=-1.0)
    z = _series_impedance(w, "sommerfeld", [1000.0])[0]
    assert math.isfinite(z.real) and z.imag > 0.0


@pytest.mark.finding("10.60")
@pytest.mark.parametrize("model", ["carson_series", "sommerfeld"])
def test_multifrequency_solve_equals_single_frequency_solves(model: str) -> None:
    """One assembly for all frequencies; every frequency is bit-identical."""
    w = _pen_world(60.0, dsl=10.0)
    freqs = [0.0, 50.0, 250.0, 1000.0]
    z_all = _series_impedance(w, model, freqs)
    for k, f in enumerate(freqs):
        z_one = _series_impedance(_pen_world(60.0, dsl=10.0), model, [f])[0]
        assert z_all[k] == z_one


@pytest.mark.finding("10.60")
def test_branch_correction_is_built_once_for_all_frequencies() -> None:
    """The correction object assembles the engine's frequency set at once."""
    from groundfield.solver import _pipeline

    w = _pen_world(60.0, dsl=10.0)
    eng = gf.create_engine(
        backend="image",
        segment_length=0.5,
        frequencies=[50.0, 250.0, 1000.0],
        earth_inductive_model="sommerfeld",
    )
    disc = _pipeline.discretise(w, eng)
    topo = _pipeline.build_topology(w, disc)
    branches = _pipeline.assemble_branches(w, eng, topo)
    correction = branches.earth_correction
    assert correction.n_assemblies == 0
    first = correction(2.0 * np.pi * 250.0)
    assert correction.n_assemblies == 3  # the whole frequency set at once
    assert correction(2.0 * np.pi * 250.0) is first
    assert not first.flags.writeable
    # a frequency outside the engine's set is built on demand
    correction(2.0 * np.pi * 60.0)
    assert correction.n_assemblies == 4
    assert not np.any(correction(0.0))
