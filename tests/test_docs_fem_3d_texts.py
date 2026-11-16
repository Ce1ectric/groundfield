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
    # B8 / B10 of the generated table: fem is excluded for its cost, not
    # refused (stream fw4int; the 3-D model solves both worlds)
    assert "exclude `fem` from the benchmark run for its cost" in page
    assert "which the axisymmetric model refuses" not in page
    # the 'raise refinement' remedy holds for the axisymmetric model only
    assert "multiplies its nodes by eight" in page


# ---------------------------------------------------------------------
# Audit W4R2B (stream fw4r2b)
# ---------------------------------------------------------------------

TOLERANCES = ROOT / "tests" / "benchmarks" / "tolerances.py"


def _consequences(adr: str) -> str:
    return adr.split("## Consequences", 1)[1].split("## Alternatives", 1)[0]


def test_w4r2b_03_the_texts_give_the_distance_from_which_2e_3_holds() -> None:
    # audit W4R2B-03: "away from other wires by about R/0.8" was not true
    # (+4.35e-3 at 7.5 cm); the model keeps such wires 1.5 R apart
    page, adr = _text(FEM_PAGE), _text(ADR)
    solid, wires = _text(FEM_PKG / "_solid.py"), _text(FEM_PKG / "_wires3d.py")
    for text in (page, adr):
        assert "R/0.8" not in text
        assert "1.5\\,R" in text and "beside each other" in text
        assert "BackendCapabilityError" in text
    assert "beside each other" in solid and "beside each other" in wires


def test_w4r2b_05_the_b8_note_does_not_read_moms_coarse_transfer_impedance_as_fems_error() -> (
    None
):
    # audit W4R2B-05 (test of auditor W4R2B): mom's floating-port Z_BA at
    # ds = 0.5 m is not converged; fem's transfer resistance is right
    page, tol = _text(FEM_PAGE), _text(TOLERANCES)
    for text in (page, tol):
        assert "0.96 % below the floating-port" not in text
        assert "1/32 m" in text and "not converged" in text


def test_w4r2b_05_the_transfer_estimate_is_documented() -> None:
    # audit W4R2B-05: the estimate covered the diagonal entries only
    for text in (_text(FEM_PAGE), _text(ADR), _text(FEM_PKG / "_solid.py")):
        assert "fem_error_estimate_transfer" in text


def test_w4r2b_06_the_level_ratio_of_rods_is_stated_once() -> None:
    # audit W4R2B-06 (test of auditor W4R2B): f04 gives 3.3 ... 4.4 for rods
    # and 3.0 ... 3.4 for rings
    page, adr = _text(FEM_PAGE), _text(ADR)
    assert "4.2 … 5.2 for rods" not in page
    for text in (page, adr):
        assert "3.0 … 5.2" not in text and "3.7 … 5.2" not in text
        assert "3.3 … 4.4" in text and "3.0 … 3.4" in text


def test_w4r2b_06_the_consequences_qualify_the_accuracy_of_the_3d_model() -> None:
    # audit W4R2B-06: "about 2e-3" without the conditions of section 5
    consequences = _consequences(_text(ADR))
    assert "(about 2e-3, with the optional" not in consequences
    assert "1.5\\,R" in consequences and "interfaces" in consequences


def test_w4r2b_06_the_cg_refusal_does_not_blame_a_contrast_of_1() -> None:
    # audit W4R2B-06: the refusal named "a layer contrast of 1" for
    # homogeneous soil and model='auto' for worlds that are not axisymmetric
    solid = _text(FEM_PKG / "_solid.py")
    assert "_convergence_cause" in solid and "in homogeneous soil" in solid


def test_w4r2b_06_the_station_memory_is_the_re_measured_one_in_the_changelog() -> None:
    """The station's memory was re-measured to 4.1 … 4.3 GB (f15 of stream
    fw4fem3d); the CHANGELOG's Added entry of the 3-D model said 3.6 … 4.4 GB
    (audit W4R2B-06)."""
    log = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "3.6 … 4.4 GB" not in log
    assert "4.1 … 4.3 GB" in log


def test_w4r2b_06_the_junction_table_carries_the_re_measured_values() -> None:
    """concrete-shells.md kept the T-joint +0.2 / +3.2 cm and the crossing
    −6.0 / −2.8 cm measured before audit W4B-01; f12 of stream fw4fem3d
    re-measured them as +0.63 / +3.71 and −5.10 / −1.90 cm (audit
    W4R2B-06)."""
    page = (ROOT / "docs/explanation/concrete-shells.md").read_text(encoding="utf-8")
    assert "| crossing | −6.0 cm | −2.8 cm |" not in page
    assert "−5.1 cm" in page and "−1.9 cm" in page


# ---------------------------------------------------------------------
# Audit W4R3B (stream fw4r3b)
# ---------------------------------------------------------------------


def test_w4r3b_05_the_refusal_radius_is_that_of_the_thickest_bare_wire() -> None:
    # audit W4R3B-05: the limit is 2.2 radii of the thickest bare wire of
    # the world, not of the two wires (two 4 mm strips 4 cm apart are
    # refused next to a 20 mm rod elsewhere)
    page, adr = _text(FEM_PAGE), _text(ADR)
    solid = _text(FEM_PKG / "_solid.py")
    assert "(3.3 wire radii;" not in page
    for text in (page, adr, solid):
        assert "thickest bare wire" in text


def test_w4r3b_03_the_turn_rule_has_no_jump_and_counts_the_turn() -> None:
    # audit W4R3B-03: "a V whose turn crosses 120° jumps by 1.5e-4" held for
    # 4 m arms only (0.5 m arms -1.26e-3); the rule now blends in the turn
    # accumulated along the wire
    page, wires = _text(FEM_PAGE), _text(FEM_PKG / "_wires3d.py")
    assert "jumps by 1.5e-4 of its impedance" not in page
    assert "has turned by less than 120°" not in page
    for text in (page, wires):
        assert "90°" in text and "150°" in text


def test_w4r3b_01_02_stacked_rods_and_narrow_vs_are_described_as_handled() -> None:
    # audit W4R3B-01: "Left alone are ... stacked rods"; audit W4R3B-02: "the
    # arms of a V narrower than 30°" ran beside each other although wires
    # that meet were promised to be left alone
    page = _text(FEM_PAGE)
    assert "collinear pieces of a galvanic conductor and stacked rods" not in page
    assert "the arms of a V narrower than 30°, a galvanic" not in page
    assert "end to end" in page and "narrower than 30°" in page
    assert "1.5\\,R/\\sin\\theta" in page


def test_w4r3b_05_the_transfer_margin_is_a_range() -> None:
    # audit W4R3B-05: "6.3 times" is the B8 analogue only; the measured
    # margin where the model holds starts near 1
    page = _text(FEM_PAGE)
    assert "margin of" in page and "6.3" in page
    assert "TR_" not in page and "RING_" not in page


def test_w4r3b_decisions_the_texts_state_the_in_line_factor_the_angle_and_shells() -> (
    None
):
    # the coordinator's decisions on the open points of stream fw4r3b: wires
    # end to end keep 3 R, wires that meet at less than 10 degrees run beside
    # each other, encased wires meet by their shell radius
    page, adr = _text(FEM_PAGE), _text(ADR)
    wires, solid = _text(FEM_PKG / "_wires3d.py"), _text(FEM_PKG / "_solid.py")
    assert "It is no bound for two wires end to end closer than" not in page
    assert "Touching wires that run together are not one conductor" in page
    for text in (page, adr):
        assert "3\\,R" in text and "10°" in text and "r_b" in text
    for text in (wires, solid):
        assert "IN_LINE_FACTOR" in text and "10°" in text
