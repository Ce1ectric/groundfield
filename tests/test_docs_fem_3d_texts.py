"""Texts of the three-dimensional ``fem`` model after audit W4B.

The explanation page, ADR-0016 and the module docstrings of the 3-D model
must state its accuracy with the conditions it holds under (W4B-02), the
tested layer contrast (W4B-03), the inner conformal radius of the square
concrete prism (W4B-04), the unconverged junction numbers of user decision
Q40 (W4B-05), and no longer carry the stale statements the auditor listed
(W4B-10). The files are read as text, so the tests run without the
``fem3d`` extra.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEM_PAGE = ROOT / "docs" / "explanation" / "engines" / "fem.md"
ADR = ROOT / "docs" / "adr" / "0016-conforming-fem.md"
FEM_PKG = ROOT / "src" / "groundfield" / "solver" / "fem"


def _text(path: Path) -> str:
    """The file with every run of whitespace collapsed to one space."""
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def test_w4b_02_the_3d_accuracy_is_stated_with_its_conditions() -> None:
    # audit W4B-02
    page, adr = _text(FEM_PAGE), _text(ADR)
    for text in (page, adr):
        assert "bounds the deviation" not in text
        assert "away from interfaces and other wires" in text or (
            "away from layer interfaces and other wires" in text
        )
        assert "thin-wire error" in text
        assert "2.5" in text and "no bound" in text


def test_w4b_03_the_tested_contrast_of_the_3d_model_is_stated() -> None:
    # audit W4B-03
    page, adr = _text(FEM_PAGE), _text(ADR)
    assert "any number of horizontal layers, any contrast |" not in page
    for text in (page, adr):
        assert "about 300" in text
        assert "BackendCapabilityError" in text


def test_w4b_04_the_square_prism_is_described_by_both_conformal_radii() -> None:
    # audit W4B-04
    solid = _text(FEM_PKG / "_solid.py")
    for text in (_text(FEM_PAGE), _text(ADR), solid):
        assert "inner conformal radius" in text
        assert re.search(r"0\.91(4|39)", text)
        assert "3.8 % less resistive" in text


def test_w4b_05_the_junction_numbers_are_marked_as_not_converged() -> None:
    # audit W4B-05
    page, adr = _text(FEM_PAGE), _text(ADR)
    for text in (page, adr):
        assert "sign not settled" in text
        assert "about 1 % or less" in text
        assert "not converged" in text
    assert "a corner −1.5 cm" not in page
    assert "less than 0.6 %" not in page


def test_w4b_10_no_stale_texts_of_the_3d_model() -> None:
    # audit W4B-10
    solid = _text(FEM_PKG / "_solid.py")
    mesh = (FEM_PKG / "_mesh3d.py").read_text(encoding="utf-8")
    wires = _text(FEM_PKG / "_wires3d.py")
    page, adr = _text(FEM_PAGE), _text(ADR)
    assert "every layer interface a mesh plane" not in solid
    assert "layer_fractions" not in mesh
    assert "interface_ratio: float = 0.1" not in mesh
    assert "R_\\\\eta/2" not in mesh
    assert "0.086" not in wires and "0.084" in wires
    assert "Every other world is refused with" not in adr
    # B8 / B10 of the generated table: the stale exclusion is explained
    assert "re-measured with the validation page at the release" in page
    # the 'raise refinement' remedy holds for the axisymmetric model only
    assert "multiplies its nodes by eight" in page
