"""Unit tests of the golden-value harness itself."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tests.golden import harness

pytestmark = pytest.mark.golden


def _decl(**kw) -> harness.Declaration:
    base = dict(
        source="t[0]",
        cases=("c",),
        quantities=("*",),
        findings=("10.99",),
        reason="test",
    )
    base.update(kw)
    return harness.Declaration(**base)


def test_identical_and_within_tolerance() -> None:
    old = {"a": np.array([1.0, 2.0]), "b": np.array([1.0 + 1j])}
    new = {"a": np.array([1.0, 2.0]), "b": np.array([1.0 + 1j + 1e-12])}
    comps = {c.quantity: c for c in harness.compare_case("c", new, old, tol=1e-9)}
    assert comps["a"].status == "identical"
    assert comps["b"].status == "within_tol"
    assert harness.judge(list(comps.values()), [], strict_mode=False) == []
    # strict mode rejects anything that is not bit-identical
    assert harness.judge(list(comps.values()), [], strict_mode=True)


def test_undeclared_change_fails_and_declared_change_passes() -> None:
    old = {"x": np.array([10.0, 20.0])}
    new = {"x": np.array([10.0, 22.0])}
    comps = harness.compare_case("c", new, old, tol=1e-9)
    assert comps[0].status == "changed"
    assert comps[0].max_rel == pytest.approx(0.1)
    assert harness.judge(comps, [], strict_mode=False)
    assert harness.judge(comps, [_decl()], strict_mode=False) == []
    assert comps[0].findings == ["10.99"]


def test_bounds_of_a_declaration() -> None:
    old = {"x": np.array([10.0])}
    new = {"x": np.array([12.0])}
    comps = harness.compare_case("c", new, old, tol=1e-9)
    assert harness.judge(comps, [_decl(max_rel_change=0.1)], strict_mode=False)
    assert not harness.judge(comps, [_decl(max_rel_change=0.5)], strict_mode=False)
    # a declared change that does not happen is reported
    same = harness.compare_case("c", old, old, tol=1e-9)
    assert harness.judge(same, [_decl(min_rel_change=1e-3)], strict_mode=False)


def test_shape_and_string_changes_are_structural() -> None:
    old = {"p": np.zeros((3, 3)), "s": np.array("abc")}
    new = {"p": np.zeros((4, 3)), "s": np.array("abd")}
    comps = {c.quantity: c for c in harness.compare_case("c", new, old)}
    assert comps["p"].status == "changed" and math.isinf(comps["p"].max_rel)
    assert comps["s"].status == "changed" and "char 2" in comps["s"].detail
    bounded = _decl(max_rel_change=1.0)
    assert harness.judge(list(comps.values()), [bounded], strict_mode=False)
    shaped = _decl(max_rel_change=1.0, allow_shape_change=True)
    assert not harness.judge(list(comps.values()), [shaped], strict_mode=False)


def test_missing_and_new_quantities_need_a_declaration() -> None:
    comps = harness.compare_case(
        "c", {"new": np.ones(1)}, {"old": np.ones(1)}, tol=1e-9
    )
    assert {c.status for c in comps} == {"missing", "new"}
    assert len(harness.judge(comps, [], strict_mode=False)) == 2


def test_nan_pattern_is_compared() -> None:
    old = {"x": np.array([np.nan, 1.0])}
    assert harness.compare_case("c", old, old)[0].status == "identical"
    new = {"x": np.array([0.0, 1.0])}
    assert harness.compare_case("c", new, old)[0].status == "changed"


def test_declaration_files_are_validated(tmp_path: Path) -> None:
    (tmp_path / "s.toml").write_text(
        '[[change]]\ncases=["c"]\nquantities=["*"]\nreason="r"\ntypo=1\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown keys"):
        harness.load_declarations(tmp_path)


def test_repository_declarations_parse() -> None:
    for d in harness.load_declarations():
        assert d.reason.strip(), d.source
        assert d.cases and d.quantities, d.source


def test_noise_floor_absorbs_round_off_residuals() -> None:
    old = {"rms": np.array(1.3e-13)}
    new = {"rms": np.array(2.4e-10)}
    comps = harness.compare_case("fit_vector_fit_synthetic", new, old, tol=1e-9)
    assert comps[0].status == "within_tol"
    other = harness.compare_case("some_other_case", new, old, tol=1e-9)
    assert other[0].status == "changed"
