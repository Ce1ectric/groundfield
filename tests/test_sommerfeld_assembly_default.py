"""Tabulated Sommerfeld assembly as the engine default (10.60, user decision Q16).

``Engine.sommerfeld_assembly`` selects how the Sommerfeld earth-return
correction of inductive conductors is assembled: ``"exact"`` integrates
every segment pair with its own quadrature (24 interpolation nodes of
$S(\\rho)$ per pair), ``"tabulated"`` shares one table of $S(\\rho)$ per
pair of depth groups and frequency. The user decided (2026-11-08,
question Q16) that ``"tabulated"`` is the default: it is faster on
networks and closer to a converged reference than the per-pair rule
(evidence ``q16_*`` of stream ddefaults). The per-pair rule stays
available as ``"exact"``.

The tests check the default, that every path that assembles inductive
branches (``image``, ``image_2layer``, ``mom``, ``mom_sommerfeld``,
``solve_mutual_matrix`` / ``solve_mutual_field``) follows the setting,
and that the default solve is the explicit tabulated one bit for bit and
within 1e-5 of the per-pair rule. They fail on the base (default
``"exact"``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.conductors import Conductor

pytestmark = pytest.mark.finding("10.60")

FREQS = [50.0, 1000.0]


def _pen_world(soil: object | None = None, *, source: bool = True) -> gf.World:
    """Two rods joined by an 80 m Neumann PEN in 5 m segments, 0.7 m deep."""
    w = gf.create_world(soil=soil or gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="g2", position=(80.0, 0.0, 0.5), length=3.0)
    w.add_conductor(
        Conductor(
            name="pen",
            start=(0.0, 0.0, 0.7),
            end=(80.0, 0.0, 0.7),
            start_electrode="g1",
            end_electrode="g2",
            cross_section=95e-6,
            discretize_segment_length=5.0,
            inductance_model="neumann",
        )
    )
    if source:
        gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _solve(world: gf.World, **kw: object) -> gf.FieldResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gf.Engine(segment_length=1.0, frequencies=FREQS, **kw).solve(world)


def test_tabulated_is_the_default_assembly() -> None:
    # user decision Q16
    assert gf.Engine().sommerfeld_assembly == "tabulated"
    assert gf.create_engine().sommerfeld_assembly == "tabulated"
    assert gf.Engine(sommerfeld_assembly="exact").sommerfeld_assembly == "exact"
    # engine JSON written before the field existed (0.15.0) gets the default
    old = {
        k: v for k, v in gf.Engine().model_dump().items() if k != "sommerfeld_assembly"
    }
    assert gf.Engine.model_validate(old).sommerfeld_assembly == "tabulated"


_SOILS = {
    "homogeneous": gf.HomogeneousSoil(rho=100.0),
    "two_layer": gf.TwoLayerSoil(rho_1=300.0, rho_2=60.0, h_1=4.0),
    "three_layer": gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=300.0, h=5.0),
            gf.SoilLayer(rho=60.0, h=6.0),
            gf.SoilLayer(rho=500.0, h=None),
        ]
    ),
}

_PATHS = [
    ("image", "homogeneous"),
    ("image", "two_layer"),
    ("mom", "two_layer"),
    ("mom_sommerfeld", "three_layer"),
    ("solve_mutual_matrix", "two_layer"),
    ("solve_mutual_field", "homogeneous"),
]


@pytest.mark.parametrize(("path", "soil"), _PATHS, ids=[f"{p}-{s}" for p, s in _PATHS])
def test_every_inductive_path_follows_the_setting(
    monkeypatch: pytest.MonkeyPatch, path: str, soil: str
) -> None:
    """The assembly method reaching the coupling module, per entry point."""
    # user decision Q16
    from groundfield.coupling import sommerfeld_inductance as si

    methods: list[str] = []
    original = si.build_sommerfeld_correction_matrices

    def spy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        methods.append(kwargs.get("method", "exact"))
        return original(*args, **kwargs)

    monkeypatch.setattr(si, "build_sommerfeld_correction_matrices", spy)

    def run(**kw: object) -> None:
        engine_kw = dict(segment_length=1.0, frequencies=FREQS, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if path.startswith("solve_mutual"):
                world = _pen_world(_SOILS[soil], source=False)
                gf.create_electrode(
                    world, "rod", name="far", position=(40.0, 30.0, 0.5), length=3.0
                )
                engine = gf.Engine(**engine_kw)
                if path == "solve_mutual_matrix":
                    gf.solve_mutual_matrix(world, engine, ["g1", "far"])
                else:
                    points = np.array([[40.0, 5.0, 0.0]])
                    gf.solve_mutual_field(world, engine, ["g1", "far"], points)
            else:
                gf.Engine(backend=path, **engine_kw).solve(_pen_world(_SOILS[soil]))

    run()
    assert methods and set(methods) == {"tabulated"}
    methods.clear()
    run(sommerfeld_assembly="exact")
    assert methods and set(methods) == {"exact"}


def test_default_solve_is_the_tabulated_solve() -> None:
    """Bit for bit the explicit tabulated solve, within 1e-5 of the per-pair rule."""
    # user decision Q16
    default = _solve(_pen_world())
    tabulated = _solve(_pen_world(), sommerfeld_assembly="tabulated")
    exact = _solve(_pen_world(), sommerfeld_assembly="exact")
    assert default.metadata["provenance"]["engine"]["sommerfeld_assembly"] == (
        "tabulated"
    )
    for name in ("g1", "g2"):
        u_default = np.asarray(default.electrode_potentials[name])
        u_tab = np.asarray(tabulated.electrode_potentials[name])
        u_exact = np.asarray(exact.electrode_potentials[name])
        np.testing.assert_array_equal(u_default, u_tab)
        assert not np.array_equal(u_default, u_exact)
        np.testing.assert_allclose(u_default, u_exact, rtol=1e-5)
