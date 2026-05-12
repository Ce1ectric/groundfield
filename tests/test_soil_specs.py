"""Tests for soil-spec resolution and JSON round-trip."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from groundfield.generators import (
    HomogeneousSoilSpec,
    LogNormal,
    MultiLayerSoilSpec,
    SoilLayerSpec,
    SoilSpec,
    TwoLayerSoilSpec,
    Uniform,
    materialise_soil,
)
from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    TwoLayerSoil,
)


def test_homogeneous_to_soil_with_fixed_value() -> None:
    spec = HomogeneousSoilSpec(resistivity=200.0)
    soil = spec.to_soil(np.random.default_rng(0))
    assert isinstance(soil, HomogeneousSoil)
    assert soil.resistivity == 200.0


def test_two_layer_to_soil_with_fixed_values() -> None:
    spec = TwoLayerSoilSpec(rho_1=100.0, rho_2=50.0, h_1=5.0)
    soil = spec.to_soil(np.random.default_rng(0))
    assert isinstance(soil, TwoLayerSoil)
    assert soil.rho_1 == 100.0
    assert soil.rho_2 == 50.0
    assert soil.h_1 == 5.0


def test_two_layer_to_soil_with_distributions_is_reproducible() -> None:
    spec = TwoLayerSoilSpec(
        rho_1=LogNormal(mu=5.0, sigma=0.7),
        rho_2=Uniform(low=20.0, high=80.0),
        h_1=5.0,
    )
    a = spec.to_soil(np.random.default_rng(42))
    b = spec.to_soil(np.random.default_rng(42))
    assert a.rho_1 == b.rho_1
    assert a.rho_2 == b.rho_2


def test_multi_layer_three_layers() -> None:
    spec = MultiLayerSoilSpec(
        layers=[
            SoilLayerSpec(resistivity=300.0, thickness_m=2.0),
            SoilLayerSpec(resistivity=100.0, thickness_m=5.0),
            SoilLayerSpec(resistivity=50.0, thickness_m=None),
        ],
    )
    soil = spec.to_soil(np.random.default_rng(0))
    assert isinstance(soil, MultiLayerSoil)
    assert len(soil.layers) == 3
    assert soil.layers[0].resistivity == 300.0
    assert soil.layers[-1].thickness is None


def test_multi_layer_rejects_non_terminal_semi_infinite() -> None:
    with pytest.raises(ValidationError):
        MultiLayerSoilSpec(
            layers=[
                SoilLayerSpec(resistivity=100.0, thickness_m=None),
                SoilLayerSpec(resistivity=200.0, thickness_m=None),
            ],
        )


def test_multi_layer_rejects_finite_last_layer() -> None:
    with pytest.raises(ValidationError):
        MultiLayerSoilSpec(
            layers=[
                SoilLayerSpec(resistivity=100.0, thickness_m=2.0),
                SoilLayerSpec(resistivity=200.0, thickness_m=5.0),
            ],
        )


# ---------------------------------------------------------------------
# JSON round-trip via discriminated union
# ---------------------------------------------------------------------


class _Wrapper(BaseModel):
    s: SoilSpec


@pytest.mark.parametrize(
    "spec",
    [
        HomogeneousSoilSpec(resistivity=150.0),
        TwoLayerSoilSpec(rho_1=100.0, rho_2=30.0, h_1=2.0),
        MultiLayerSoilSpec(layers=[
            SoilLayerSpec(resistivity=300.0, thickness_m=2.0),
            SoilLayerSpec(resistivity=50.0, thickness_m=None),
        ]),
    ],
)
def test_soil_spec_json_roundtrip(spec) -> None:
    payload = _Wrapper(s=spec).model_dump_json()
    restored = _Wrapper.model_validate_json(payload).s
    assert type(restored) is type(spec)
    assert restored.model_dump() == spec.model_dump()


def test_materialise_soil_dispatcher() -> None:
    """materialise_soil dispatches on the spec type."""
    rng = np.random.default_rng(0)
    s = materialise_soil(HomogeneousSoilSpec(resistivity=42.0), rng)
    assert isinstance(s, HomogeneousSoil)
    s = materialise_soil(TwoLayerSoilSpec(rho_1=10.0, rho_2=20.0, h_1=1.0), rng)
    assert isinstance(s, TwoLayerSoil)
