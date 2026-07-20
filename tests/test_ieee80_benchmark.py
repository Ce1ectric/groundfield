"""IEEE Std 80-2013 Annex H grounding-grid benchmark.

Two independent checks:

1. **Sverak reference formula** (IEEE Std 80 Eq. 57,
   :func:`groundfield.references.ieee80.grid_resistance_sverak`) — unit
   sanity plus agreement with a field solve.
2. **Annex H benchmark** (informative) — reproduces the standard's own
   benchmark grids and compares against the reference-software results
   (CDEGS column of Table H.5 / H.6 / H.7). The ``mom`` backend solves
   the actual current distribution, as CDEGS/WinIGS do, so it matches
   the grid resistance, GPR and touch voltage to ~1 % and the
   (discretisation-sensitive) step voltage to a few percent.

Annex H geometry (Figure H.3–H.5): a 70 m × 70 m square grid, 6 × 6
conductors on a 14 m pitch, 2/0 AWG copper, buried 0.5 m, grid current
744.8 A.

* **Grid 1** — the bare grid in uniform 140 Ω·m soil (Table H.5).
* **Grid 2** — Grid 1 plus twenty 7.5 m 5/8-in ground rods at the
  perimeter intersections (Table H.6).
* **Grid 3** — the Grid 2 geometry in a two-layer soil
  (ρ1 = 300 Ω·m, ρ2 = 100 Ω·m, interface h = 6.096 m); the 7.5 m rods
  reach 8.0 m and therefore **cross the layer interface** (Table H.7).

All three are reproduced here. Grid 3 exercises the cross-layer scalar
Green's function (ADR-0007); it is tractable in both time and memory
because the layered-correction Hankel contraction is blocked and
interpolated (see ``tests/test_layered_green_chunking.py``). The worked
side-by-side comparison with the CDEGS/ETAP/WinIGS columns is in
``notebooks/43_ieee80_benchmark.ipynb``.

Discretisation caveat
---------------------
The Sverak grid check keeps segment midpoints **off** the grid crossings
(where coincident point sources would trip the singularity clamp and
return a resistance ~2× too large — a known grid-discretisation trap).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.references import ieee80


# ---------------------------------------------------------------------
# 1. Reference formula (Sverak, IEEE Std 80 Eq. 57)
# ---------------------------------------------------------------------
def test_sverak_scales_linearly_with_rho() -> None:
    kw = dict(total_conductor_length=1540.0, area=4900.0, depth=0.5)
    assert ieee80.grid_resistance_sverak(rho=200.0, **kw) == pytest.approx(
        2.0 * ieee80.grid_resistance_sverak(rho=100.0, **kw)
    )


def test_sverak_decreases_with_area() -> None:
    small = ieee80.grid_resistance_sverak(100.0, 1540.0, 2500.0, 0.5)
    large = ieee80.grid_resistance_sverak(100.0, 1540.0, 4900.0, 0.5)
    assert large < small


def test_sverak_deeper_grid_has_lower_resistance() -> None:
    shallow = ieee80.grid_resistance_sverak(100.0, 1540.0, 4900.0, 0.5)
    deep = ieee80.grid_resistance_sverak(100.0, 1540.0, 4900.0, 2.5)
    assert deep < shallow


def test_sverak_rejects_nonpositive_input() -> None:
    with pytest.raises(ValueError):
        ieee80.grid_resistance_sverak(100.0, -1.0, 4900.0, 0.5)


def test_image_grid_resistance_matches_sverak() -> None:
    """A uniform-soil grid solve agrees with Sverak's estimate."""
    rho, side, ndiv, depth = 100.0, 70.0, 10, 0.5  # 7 m conductor spacing
    total_length = 2.0 * (ndiv + 1) * side
    area = side * side

    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=rho))
    gf.create_electrode(
        world,
        "grid_mesh",
        name="grid",
        corner=(-side / 2.0, -side / 2.0, depth),
        size=(side, side),
        n_x=ndiv,
        n_y=ndiv,
        wire_radius=0.005,
    )
    gf.create_source(world, attached_to="grid", magnitude=1.0)
    # ds = 1.75 m keeps segment midpoints off the 7 m crossings.
    engine = gf.create_engine(backend="image", segment_length=1.75, frequencies=[0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R_num = world.solve(engine).cluster_impedance("grid")[0].real

    R_ref = ieee80.grid_resistance_sverak(rho, total_length, area, depth)
    assert abs(R_num - R_ref) / R_ref < 0.05


# ---------------------------------------------------------------------
# 2. IEEE Std 80-2013 Annex H benchmark (Grid 1 & Grid 2)
# ---------------------------------------------------------------------
_I_G = 744.8  # A, grid current for all Annex H cases
_SIDE = 70.0  # m, square grid side
_NMESH = 5  # meshes per side -> 6 conductors, 14 m pitch
_DEPTH = 0.5  # m, burial depth
_RHO = 140.0  # Ω·m, uniform soil (Grid 1 & Grid 2)
_A_COND = 0.5 * 9.266e-3  # 2/0 AWG copper radius (m)
_A_ROD = 0.5 * 15.875e-3  # 5/8-in ground-rod radius (m)
_ROD_LEN = 7.5  # m
# Grid 3 two-layer soil (Table H.4 / H.7): the 7.5 m rods span
# 0.5 m … 8.0 m and cross the interface at h = 6.096 m.
_RHO1, _RHO2, _H1 = 300.0, 100.0, 6.096  # Ω·m, Ω·m, m

# CDEGS reference results (Table H.5 / H.6 / H.7): grid resistance
# R_g [Ω], GPR [V], touch voltage T1 at the corner-mesh centre [V], step
# voltage S1 one metre out on the diagonal [V].
_REF = {
    "grid1": dict(R_g=1.0, GPR=743.9, T1=194.9, S1=89.3),
    "grid2": dict(R_g=0.917, GPR=682.8, T1=145.4, S1=70.7),
    "grid3": dict(R_g=0.97, GPR=719.5, T1=261.0, S1=101.9),
}


def _build_annex_h_grid(
    with_rods: bool, soil: "gf.SoilModel | None" = None
) -> "gf.World":
    if soil is None:
        soil = gf.HomogeneousSoil(resistivity=_RHO)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world,
        "grid_mesh",
        name="grid",
        corner=(-_SIDE / 2.0, -_SIDE / 2.0, _DEPTH),
        size=(_SIDE, _SIDE),
        n_x=_NMESH,
        n_y=_NMESH,
        wire_radius=_A_COND,
    )
    if with_rods:
        coords = np.linspace(-_SIDE / 2.0, _SIDE / 2.0, _NMESH + 1)
        perimeter = [
            (x, y)
            for x in coords
            for y in coords
            if abs(x) == _SIDE / 2.0 or abs(y) == _SIDE / 2.0
        ]
        for i, (x, y) in enumerate(perimeter):
            gf.create_electrode(
                world,
                "rod",
                name=f"rod{i}",
                position=(x, y, _DEPTH),
                length=_ROD_LEN,
                wire_radius=_A_ROD,
            )
            gf.create_conductor(
                world, start="grid", end=f"rod{i}", coupling_to_soil="isolated"
            )
    gf.create_source(world, attached_to="grid", magnitude=_I_G)
    return world


def _grid_metrics(res) -> tuple[float, float, float, float]:
    R_g = res.cluster_impedance("grid")[0].real
    GPR = R_g * _I_G
    c = _SIDE / 2.0

    def phi(p: list[float]) -> float:
        return float(res.potential(np.array([p]), frequency_index=0)[0].real)

    # T1: touch voltage at the centre of the corner mesh (7 m in from
    # the corner along both axes).
    T1 = GPR - phi([c - 7.0, c - 7.0, 0.0])
    # S1: step voltage between the point over the corner and one metre
    # out along the diagonal.
    d = 1.0 / np.sqrt(2.0)
    S1 = abs(phi([c, c, 0.0]) - phi([c + d, c + d, 0.0]))
    return R_g, GPR, T1, S1


def _annex_h_soil(key: str):
    """Two-layer soil for Grid 3, uniform soil (default) otherwise."""
    if key == "grid3":
        return gf.TwoLayerSoil(rho_1=_RHO1, rho_2=_RHO2, h_1=_H1)
    return None


@pytest.mark.parametrize(
    "with_rods,key",
    [(False, "grid1"), (True, "grid2"), (True, "grid3")],
)
def test_ieee80_annex_h_grid_matches_reference(with_rods, key) -> None:
    """Reproduce IEEE Std 80-2013 Annex H benchmark Grid 1 / 2 / 3.

    The ``mom`` backend solves the actual current distribution, as the
    reference programs CDEGS/WinIGS do, and reproduces the CDEGS results
    (Table H.5 / H.6 / H.7) to about 1 % on resistance, GPR and touch
    voltage, and a few percent on the discretisation-sensitive step
    voltage. Grid 3 additionally exercises the two-layer cross-layer
    Green's function (the 7.5 m rods cross the 6.096 m interface).
    """
    world = _build_annex_h_grid(with_rods, soil=_annex_h_soil(key))
    engine = gf.create_engine(backend="mom", segment_length=1.0, frequencies=[0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = world.solve(engine)
    R_g, GPR, T1, S1 = _grid_metrics(res)
    ref = _REF[key]

    assert abs(R_g - ref["R_g"]) / ref["R_g"] < 0.03, f"R_g {R_g} vs {ref['R_g']}"
    assert abs(GPR - ref["GPR"]) / ref["GPR"] < 0.03, f"GPR {GPR} vs {ref['GPR']}"
    assert abs(T1 - ref["T1"]) / ref["T1"] < 0.05, f"T1 {T1} vs {ref['T1']}"
    # Step voltage is the most discretisation-sensitive metric and the
    # reference programs themselves spread ~12 % on it.
    assert abs(S1 - ref["S1"]) / ref["S1"] < 0.10, f"S1 {S1} vs {ref['S1']}"
