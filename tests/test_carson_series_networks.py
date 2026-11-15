"""``carson_series`` on generated LV networks: the stated error (audit W3B-02).

Tiled over service drops shorter than the skin depth, Carson's infinite-line
correction overstates the earth return of a generated network against
``sommerfeld`` (converged to 4e-4 there). The earth-return explanation
stated the error at 1 kHz for the 100-house star and grid networks only
(X 9.3 ... 14.9 %); the 5 ... 30-house star and radial networks read X up to
21 % high at 1 kHz and 21 ... 44 % at 50 Hz, which no text stated.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

import groundfield as gf
from groundfield.generators import (
    HomogeneousSoilSpec,
    PenConfig,
    RadialTrunkTopology,
    TnNetworkConfig,
    TnNetworkGenerator,
)

ROOT = Path(__file__).resolve().parents[1]


def _radial_x_ratio() -> list[float]:
    cfg = TnNetworkConfig(
        soil=HomogeneousSoilSpec(rho=100.0),
        building_counts={"residential": 10},
        pen_topology=RadialTrunkTopology(),
        pen=PenConfig(inductance_model="neumann", segment_length_m=5.0),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    z = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model in ("carson_series", "sommerfeld"):
            eng = gf.create_engine(
                segment_length=2.0,
                frequencies=[50.0, 1000.0],
                earth_inductive_model=model,
            )
            z[model] = eng.solve(world).input_impedance()
    return [
        c.imag / s.imag
        for c, s in zip(z["carson_series"], z["sommerfeld"], strict=True)
    ]


# audit W3B-02 (the auditor's test, on the earth-return explanation)
def test_w3b_02_the_stated_carson_series_error_covers_a_10_house_radial_network() -> (
    None
):
    """Base 098bbbc: the explanation stated X 9.3 ... 14.9 % high at 1 kHz
    (no 50 Hz figure); the generated 10-house radial network reads +41 % at
    50 Hz and +20 % at 1 kHz (with its drops in a row, decision D9)."""
    text = (ROOT / "docs/explanation/earth-return.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*Networks of short cables\.\*\*.*?(?=\n\n)", text, re.S)
    assert m, "the paragraph on networks of short cables is gone"
    paragraph = m.group(0)
    assert "50 Hz" in paragraph
    stated = [float(v) for v in re.findall(r"(\d+(?:\.\d+)?)\s*%", paragraph)]
    ratio_50, ratio_1k = _radial_x_ratio()
    assert max(stated) >= 100.0 * (ratio_50 - 1.0)
    assert 100.0 * (ratio_50 - 1.0) == pytest.approx(41.4, abs=1.0)
    assert 100.0 * (ratio_1k - 1.0) == pytest.approx(20.2, abs=1.0)
