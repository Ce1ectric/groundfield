"""The concrete shell as one mechanism: a per-length coefficient on every electrode.

Review pass 10, findings 10.12-10.14 (user decision: the solver consumes the
concrete shell only as the distributed coefficient
``concrete_shell_coefficient_ohm_m``; the lumped model converts its input).
Up to 0.15.0 only strips, polylines and stars carried the coefficient, and the
lumped registry ``World.concrete_shell_corrections`` never reached
``Engine.solve`` of a world without a PEN service drop nor
``solve_mutual_matrix`` / ``solve_mutual_field`` (AP1 foundation
self-impedance -28.6 / -15.2 / -2.7 % at rho_1 = 30 / 100 / 1000 Ohm m).

Under the uniform current per owner of the image family the shell adds exactly
``C / L`` to the potential of a single-owner electrode of wire length ``L``:
the distributed coefficient and the lumped resistance ``R = C / L`` are the
same model, which the conversion helpers express.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import groundfield as gf
from groundfield.geometry.electrodes import (
    concrete_shell_coefficient,
    shell_coefficient_from_resistance,
    shell_resistance_from_coefficient,
)

#: rho_c = 100 Ohm m, 5 cm of concrete around a 5 mm wire.
_C = 100.0 / (2.0 * math.pi) * math.log(0.055 / 0.005)

_KINDS = {
    "rod": dict(position=(0.0, 0.0, 0.6), length=3.0),
    "ring": dict(center=(0.0, 0.0, 0.7), radius=4.0),
    "mesh": dict(corner=(-4.0, -4.0, 0.7), size=(8.0, 6.0), spacing=3.0),
    "grid_mesh": dict(corner=(-4.0, -4.0, 0.7), size=(8.0, 6.0), n_x=2, n_y=2),
    "strip": dict(start=(-4.0, 0.0, 0.7), end=(4.0, 0.0, 0.7)),
    "polyline": dict(
        vertices=[(0.0, 0.0, 0.7), (12.0, 0.0, 0.7), (12.0, 7.0, 0.7), (0.0, 7.0, 0.7)]
    ),
    "star": dict(center=(0.0, 0.0, 0.7), n_arms=4, arm_length=3.0),
}


def _world(kind: str, coefficient: float, soil=None) -> gf.World:
    world = gf.create_world(soil=soil or gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(
        world,
        kind,
        name="e",
        concrete_shell_coefficient_ohm_m=coefficient,
        **_KINDS[kind],
    )
    return world


@pytest.mark.finding("10.12")
@pytest.mark.parametrize("kind", ["rod", "ring", "mesh", "grid_mesh"])
@pytest.mark.parametrize(
    "soil",
    [gf.HomogeneousSoil(rho=100.0), gf.TwoLayerSoil(rho_1=30.0, rho_2=200.0, h_1=10.0)],
    ids=["homogeneous", "two_layer"],
)
def test_every_electrode_kind_carries_the_shell(kind, soil) -> None:
    """The image family adds exactly ``C / L`` to a single-owner electrode."""
    engine = gf.create_engine(backend="image", segment_length=0.5)
    impedance = {}
    for coefficient in (0.0, _C):
        world = _world(kind, coefficient, soil)
        gf.create_source(world, attached_to="e", magnitude=1.0)
        impedance[coefficient] = engine.solve(world).cluster_impedance("e")[0]
    electrode = _world(kind, _C, soil).electrodes[0]
    shell = _C / electrode.total_length()
    assert impedance[_C].real - impedance[0.0].real == pytest.approx(shell, rel=1e-9)
    assert shell_resistance_from_coefficient(electrode) == pytest.approx(
        shell, rel=1e-15
    )


@pytest.mark.finding("10.12")
@pytest.mark.parametrize("kind", ["rod", "ring", "polyline"])
def test_mutual_matrix_consumes_the_shell(kind) -> None:
    """``solve_mutual_matrix`` sees the shell of every electrode kind.

    The diagonal grows by ``C / L``; with one owner per cluster the
    off-diagonal reaction does not change.
    """
    soil = gf.TwoLayerSoil(rho_1=30.0, rho_2=100.0, h_1=5.0)
    engine = gf.create_engine(backend="image", segment_length=1.0)
    Z = {}
    for coefficient in (0.0, _C):
        world = _world(kind, coefficient, soil)
        gf.create_electrode(
            world, "rod", name="far", position=(30.0, 0.0, 0.5), length=3.0
        )
        Z[coefficient] = gf.solve_mutual_matrix(world, engine, ["e", "far"])
    electrode = _world(kind, _C, soil).electrodes[0]
    assert Z[_C][0, 0].real - Z[0.0][0, 0].real == pytest.approx(
        _C / electrode.total_length(), rel=1e-9
    )
    np.testing.assert_allclose(Z[_C][0, 1], Z[0.0][0, 1], rtol=1e-12)
    np.testing.assert_allclose(Z[_C][1, 1], Z[0.0][1, 1], rtol=1e-12)


@pytest.mark.finding("10.12")
@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_lumped_and_distributed_conversion(kind) -> None:
    """Lumped shell resistance and coefficient convert through the wire length."""
    electrode = _world(kind, 0.0).electrodes[0]
    resistance = 0.75
    coefficient = shell_coefficient_from_resistance(electrode, resistance)
    assert coefficient == pytest.approx(
        resistance * electrode.total_length(), rel=1e-15
    )
    assert shell_resistance_from_coefficient(electrode, coefficient) == pytest.approx(
        resistance, rel=1e-15
    )
    with pytest.raises(ValueError, match="non-negative"):
        shell_coefficient_from_resistance(electrode, -1.0)


def test_coefficient_from_material() -> None:
    """``C = rho_c / (2 pi) ln((r_a + t) / r_a)`` in Ohm m."""
    assert concrete_shell_coefficient(100.0, 0.005, 0.05) == pytest.approx(
        _C, rel=1e-15
    )
    assert concrete_shell_coefficient(100.0, 0.005, 0.0) == 0.0
    with pytest.raises(ValueError):
        concrete_shell_coefficient(-1.0, 0.005, 0.05)
    with pytest.raises(ValueError):
        concrete_shell_coefficient(100.0, 0.0, 0.05)


@pytest.mark.finding("10.12")
@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_json_round_trip_and_unchanged_json_without_shell(kind) -> None:
    """The coefficient round-trips; without a shell the JSON of rods,
    rings and meshes carries no new key (files stay readable by 0.15.0)."""
    world = _world(kind, _C)
    loaded = gf.World.model_validate_json(world.model_dump_json())
    assert loaded.electrodes[0].concrete_shell_coefficient_ohm_m == _C
    bare = json.loads(_world(kind, 0.0).model_dump_json())["electrodes"][0]
    has_key = "concrete_shell_coefficient_ohm_m" in bare
    assert has_key == (kind in {"strip", "polyline", "star"})


def test_shell_coefficient_rejects_negative_and_non_finite() -> None:
    for bad in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            gf.create_electrode(
                gf.create_world(soil=gf.HomogeneousSoil(rho=100.0)),
                "rod",
                name="r",
                position=(0.0, 0.0, 0.5),
                length=3.0,
                concrete_shell_coefficient_ohm_m=bad,
            )
