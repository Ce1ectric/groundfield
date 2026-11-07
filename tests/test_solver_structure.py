"""Structure of the image-family solver (review pass 10: 10.75, 10.79).

The image backend used to be a 2509-line module that doubled as the
private library of seven sibling backends. These tests pin the split
into private modules, the decomposition of the augmented nodal solve
into named steps, and the promise that every moved private name stays
importable from :mod:`groundfield.solver.image` for the sibling
backends, tests and notebooks that import it from there.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

import groundfield.solver.image as image_module

#: Every private name another module imported from
#: ``groundfield.solver.image`` at v0.15.0 (63 of 81 private
#: cross-module imports, R6-06), plus the public ones.
_IMAGE_REEXPORTS = (
    "ShallowSegmentWarning",
    "solve_image",
    "_MIN_DISTANCE",
    "_Segment",
    "_assemble_inductance_matrix",
    "_build_clusters",
    "_build_distributed_topology",
    "_build_finite_branches",
    "_check_segment_depths",
    "_discretize_electrode",
    "_reject_concrete_shells",
    "_require_image_separation",
    "_segments_per_wire",
    "_self_corrected_kernel",
    "_solve_cluster_currents",
    "_warn_ignored_sources",
    "_weighted_node_potential",
)


@pytest.mark.finding("10.75")
@pytest.mark.parametrize("name", _IMAGE_REEXPORTS)
def test_moved_names_stay_importable_from_image(name: str) -> None:
    """Sibling backends keep importing the helpers from ``solver.image``."""
    assert hasattr(image_module, name), name


@pytest.mark.finding("10.79")
def test_image_module_is_split() -> None:
    """``solver/image.py`` keeps the entry point only (was 2509 lines)."""
    n_lines = len(Path(image_module.__file__).read_text().splitlines())
    assert n_lines < 600, n_lines
    for private in ("_discretize", "_guards", "_kernels", "_topology", "_pipeline"):
        importlib.import_module(f"groundfield.solver.{private}")


@pytest.mark.finding("10.79")
def test_augmented_solve_is_decomposed() -> None:
    """``_solve_cluster_currents`` orchestrates named steps (was 387 lines, CC 40)."""
    from groundfield.solver import _topology

    src = inspect.getsource(_topology._solve_cluster_currents)
    doc = inspect.getdoc(_topology._solve_cluster_currents) or ""
    code_lines = len(src.splitlines()) - len(doc.splitlines())
    assert code_lines < 80, code_lines
    for step in (
        "_active_system",
        "_multiport_matrix",
        "_augmented_matrix",
        "_augmented_rhs",
        "_owner_currents",
    ):
        assert callable(getattr(_topology, step)), step


@pytest.mark.finding("10.81")
@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("groundfield.solver.image", "_potential_kernel"),
        ("groundfield.solver.image", "_segment_z_extent"),
        ("groundfield.solver.image_2layer", "_two_layer_potential_kernel"),
    ],
)
def test_dead_helpers_are_removed(module: str, name: str) -> None:
    """Unused kernels and helpers of v0.15.0 are gone (nothing imported them)."""
    assert not hasattr(importlib.import_module(module), name)


@pytest.mark.finding("10.81")
def test_empty_field_study_placeholder_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("groundfield.solver.field_study")


def test_geometry_primitives_is_no_longer_a_placeholder() -> None:
    """``geometry/primitives.py`` (empty in v0.15.0) now holds WireSegments."""
    from groundfield.geometry import primitives

    assert primitives.__all__ == ["WireSegments"]
