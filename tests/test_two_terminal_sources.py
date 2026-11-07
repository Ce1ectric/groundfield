"""Two-terminal sources: ``Source.return_to`` is honoured (finding 10.16).

Up to 0.15.0 no backend read ``return_to``: a source always returned
its current through remote earth, the auxiliary electrode of a
simulated fall-of-potential measurement carried 0 A and the substation
potential did not depend on the auxiliary distance at all (F01/F30).
The image family now injects $+I$ at ``attached_to`` and $-I$ at
``return_to`` through the shared injection step, so Kirchhoff's current
law closes the loop and the driving-point impedance is
$(\\varphi_a - \\varphi_r)/I$.

Independent cross-checks used below: superposition of two one-terminal
solves (linearity), the two-port reduction
$Z_{11} - Z_{12} - Z_{21} + Z_{22}$, and the classical 61.8 % rule of
the fall-of-potential method in homogeneous soil.
"""

from __future__ import annotations

import cmath
import math
import warnings

import numpy as np
import pytest

import groundfield as gf

pytestmark = pytest.mark.finding("10.16")

SEG = 0.5


def _soil(rho: float = 100.0) -> gf.HomogeneousSoil:
    """Homogeneous soil; ``resistivity`` was the field name up to 0.15.0."""
    try:
        return gf.HomogeneousSoil(rho=rho)
    except Exception:  # noqa: BLE001 - v0.15.0 names the field ``resistivity``
        return gf.HomogeneousSoil(resistivity=rho)


def _two_rods(soil=None, *, d_aux: float = 40.0) -> gf.World:
    w = gf.create_world(soil=soil if soil is not None else _soil())
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_electrode(w, "rod", name="aux", position=(d_aux, 0.0, 0.0), length=1.0)
    return w


def _solve(world: gf.World, **engine_kw) -> gf.FieldResult:
    kw = {"segment_length": SEG, "frequencies": [50.0], **engine_kw}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return world.solve(gf.create_engine(**kw))


def _potentials(res: gf.FieldResult, k: int = 0) -> dict[str, complex]:
    return {n: complex(v[k]) for n, v in res.electrode_potentials.items()}


def _currents(res: gf.FieldResult, k: int = 0) -> dict[str, complex]:
    return {n: complex(v[k]) for n, v in res.electrode_currents.items()}


# ---------------------------------------------------------------------
# Kirchhoff and linearity
# ---------------------------------------------------------------------


def test_return_current_closes_the_loop() -> None:
    w = _two_rods()
    gf.create_source(w, attached_to="g1", return_to="aux", magnitude=1.0)
    cur = _currents(_solve(w))
    assert abs(cur["g1"] - 1.0) < 1e-12
    assert abs(cur["aux"] + 1.0) < 1e-12
    assert abs(sum(cur.values())) < 1e-12


def test_two_terminal_source_is_the_superposition_of_two_injections() -> None:
    w = _two_rods()
    gf.create_source(w, attached_to="g1", return_to="aux", magnitude=1.0)
    closed = _potentials(_solve(w))
    singles = []
    for node in ("g1", "aux"):
        ws = _two_rods()
        gf.create_source(ws, attached_to=node, magnitude=1.0)
        singles.append(_potentials(_solve(ws)))
    for e in ("g1", "aux"):
        expected = singles[0][e] - singles[1][e]
        assert abs(closed[e] - expected) <= 1e-12 * abs(expected)


def test_input_impedance_is_the_two_port_reduction() -> None:
    """Z_in = Z11 - Z12 - Z21 + Z22 of the open-circuit two-port."""
    w = _two_rods()
    gf.create_source(w, attached_to="g1", return_to="aux", magnitude=2.0)
    res = _solve(w)
    z = np.empty((2, 2), dtype=complex)
    for j, node in enumerate(("g1", "aux")):
        ws = _two_rods()
        gf.create_source(ws, attached_to=node, magnitude=1.0)
        p = _potentials(_solve(ws))
        z[0, j], z[1, j] = p["g1"], p["aux"]
    expected = z[0, 0] - z[0, 1] - z[1, 0] + z[1, 1]
    z_in = complex(res.input_impedance()[0])
    p = _potentials(res)
    assert abs(z_in - (p["g1"] - p["aux"]) / 2.0) <= 1e-12 * abs(z_in)
    assert abs(z_in - expected) <= 1e-12 * abs(expected)
    # reciprocity of the open-circuit two-port
    assert abs(z[0, 1] - z[1, 0]) <= 1e-12 * abs(z[0, 1])


def test_magnitude_and_phase_are_honoured_at_both_terminals() -> None:
    w = _two_rods()
    gf.create_source(
        w, attached_to="g1", return_to="aux", magnitude=3.0, phase_deg=30.0
    )
    cur = _currents(_solve(w))
    i_src = 3.0 * cmath.exp(1j * math.radians(30.0))
    assert abs(cur["g1"] - i_src) < 1e-12
    assert abs(cur["aux"] + i_src) < 1e-12


def test_return_to_one_member_of_a_bonded_cluster() -> None:
    """The return current spreads over the whole cluster of the node."""
    w = _two_rods()
    gf.create_electrode(w, "rod", name="aux2", position=(40.0, 3.0, 0.0), length=1.0)
    gf.create_conductor(w, name="bond", start="aux", end="aux2")  # ideal bond
    gf.create_source(w, attached_to="g1", return_to="aux2", magnitude=1.0)
    res = _solve(w)
    cur = _currents(res)
    assert abs(cur["aux"] + cur["aux2"] + 1.0) < 1e-12
    assert cur["aux"].real < -0.3 and cur["aux2"].real < -0.3
    # Both members of the return cluster share one potential.
    p = _potentials(res)
    assert abs(p["aux"] - p["aux2"]) <= 1e-12 * abs(p["aux"])


def test_metallic_return_path_carries_most_of_the_current() -> None:
    """A 0.017 Ω feed lead in parallel with the soil path (~45 Ω)."""
    w = _two_rods()
    gf.create_conductor(
        w,
        name="lead",
        start="g1",
        end="aux",
        conductor_type="bare_copper",
        cross_section=50e-6,
    )
    gf.create_source(w, attached_to="g1", return_to="aux", magnitude=1.0)
    cur = _currents(_solve(w))
    assert abs(sum(cur.values())) < 1e-12
    # The soil loop sees only the lead's voltage drop.
    assert abs(cur["g1"]) < 1e-3


# ---------------------------------------------------------------------
# layered soil, inductive conductors, line kernel
# ---------------------------------------------------------------------


def test_two_terminal_source_in_two_layer_soil() -> None:
    soil = gf.TwoLayerSoil(rho_1=300.0, rho_2=50.0, h_1=2.0)
    w = _two_rods(soil)
    gf.create_source(w, attached_to="g1", return_to="aux", magnitude=1.0)
    res = _solve(w, backend="image_2layer")
    cur = _currents(res)
    assert abs(cur["aux"] + 1.0) < 1e-12
    singles = []
    for node in ("g1", "aux"):
        ws = _two_rods(soil)
        gf.create_source(ws, attached_to=node, magnitude=1.0)
        singles.append(_potentials(_solve(ws, backend="image_2layer")))
    closed = _potentials(res)
    for e in ("g1", "aux"):
        expected = singles[0][e] - singles[1][e]
        assert abs(closed[e] - expected) <= 1e-11 * abs(expected)


def test_two_terminal_source_with_an_inductive_conductor_at_all_frequencies() -> None:
    w = _two_rods(d_aux=60.0)
    gf.create_electrode(w, "rod", name="h1", position=(30.0, 2.0, 0.5), length=1.5)
    gf.create_conductor(
        w,
        name="pen",
        start="g1",
        end="h1",
        conductor_type="pen",
        cross_section=95e-6,
        resistivity=2.82e-8,
        discretize_segment_length=5.0,
        coupling_to_soil="isolated",
        inductance_model="neumann",
    )
    gf.create_source(w, attached_to="g1", return_to="aux", magnitude=1.0)
    res = _solve(w, frequencies=[0.0, 50.0, 1000.0])
    for k in range(3):
        cur = _currents(res, k)
        assert abs(sum(cur.values())) < 1e-12
        assert abs(cur["aux"] + 1.0) < 1e-12
    z = res.input_impedance()
    assert abs(z[2].imag) > abs(z[1].imag) > 0.0  # the PEN loop is inductive


def test_two_terminal_source_with_the_line_kernel() -> None:
    w = _two_rods()
    gf.create_source(w, attached_to="g1", return_to="aux", magnitude=1.0)
    cur = _currents(_solve(w, kernel="line"))
    assert abs(sum(cur.values())) < 1e-12
    assert abs(cur["aux"] + 1.0) < 1e-12


# ---------------------------------------------------------------------
# the measurement the finding is about
# ---------------------------------------------------------------------


def _fall_of_potential(d_aux: float, frac: float = 0.618) -> tuple[float, float]:
    """(phi_station, Z_meas) of a ring station with probe at frac·d_aux."""
    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, "ring", name="st", center=(0.0, 0.0, 0.7), radius=2.0)
    gf.create_electrode(w, "rod", name="aux", position=(d_aux, 0.0, 0.0), length=1.5)
    gf.create_electrode(
        w, "rod", name="probe", position=(frac * d_aux, 0.0, 0.0), length=0.5
    )
    gf.create_source(w, attached_to="st", return_to="aux", magnitude=1.0)
    p = _potentials(_solve(w, segment_length=1.0))
    return p["st"].real, (p["st"] - p["probe"]).real


def test_station_potential_depends_on_the_auxiliary_distance() -> None:
    near, _ = _fall_of_potential(20.0)
    far, _ = _fall_of_potential(2000.0)
    assert abs(near - far) / far > 0.05


def test_fall_of_potential_follows_the_61_8_percent_rule() -> None:
    """Tagg's rule: in homogeneous soil a probe at 61.8 % of the
    auxiliary distance reads the remote impedance of a compact station,
    while a probe at 50 % reads low by an error that falls like 1/d."""
    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, "ring", name="st", center=(0.0, 0.0, 0.7), radius=2.0)
    gf.create_source(w, attached_to="st", magnitude=1.0)
    z_remote = _potentials(_solve(w, segment_length=1.0))["st"].real

    def err(d: float, frac: float) -> float:
        return (_fall_of_potential(d, frac)[1] - z_remote) / z_remote

    assert abs(err(200.0, 0.618)) < 1e-4
    e200, e400 = err(200.0, 0.5), err(400.0, 0.5)
    assert e200 < -0.005
    assert 0.45 < e400 / e200 < 0.55


# ---------------------------------------------------------------------
# capability flags and the migration from the manual counter-source
# ---------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["image", "image_2layer", "image_nlayer"])
def test_image_family_declares_and_honours_return_to(backend: str) -> None:
    from groundfield.solver.registry import BackendCapabilityWarning, get_backend

    assert get_backend(backend).capabilities.two_terminal_sources
    soil = (
        _soil()
        if backend == "image"
        else gf.TwoLayerSoil(rho_1=100.0, rho_2=100.0, h_1=5.0)
    )
    w = _two_rods(soil)
    gf.create_source(w, attached_to="g1", return_to="aux", magnitude=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", BackendCapabilityWarning)
        res = w.solve(gf.create_engine(backend=backend, segment_length=SEG))
    assert abs(res.electrode_currents["aux"][0] + 1.0) < 1e-12


def test_manual_counter_source_at_the_return_terminal_warns() -> None:
    """The 0.15.0 workaround (a 180 deg source at the auxiliary electrode)
    now double-counts the return current; the solve says so."""
    w = _two_rods()
    gf.create_source(w, name="s", attached_to="g1", return_to="aux", magnitude=1.0)
    gf.create_source(
        w, name="s_return", attached_to="aux", magnitude=1.0, phase_deg=180.0
    )
    with pytest.warns(UserWarning, match=r"(?s)s_return.*counter-source"):
        res = w.solve(gf.create_engine(segment_length=SEG))
    assert abs(res.electrode_currents["aux"][0] + 2.0) < 1e-12
