"""Top-level namespace: curated sub-namespaces, lazy imports, ``__all__``.

Finding 10.72 (review pass 10): the flat namespace holds 128 names
(36 % generator family), ``annotations`` leaked from
``from __future__ import annotations``, and ``import groundfield``
eagerly imported pandas and several scipy submodules. Since 2026.11 the
heavy parts load on first use (PEP 562), and the curated namespaces
``gf.plot``, ``gf.safety``, ``gf.fit`` and ``gf.io`` group the
post-processing API. Every flat name keeps working.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
import traceback
import types
from pathlib import Path

import pytest

import groundfield as gf

_NAMESPACES = ("fit", "io", "plot", "safety")
_SRC = Path(gf.__file__).resolve().parents[1]


def _run_fresh(code: str) -> str:
    """Run ``code`` in a fresh interpreter that imports this checkout."""
    env = {**os.environ, "PYTHONPATH": str(_SRC)}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=True,
    )
    return proc.stdout


# ---------------------------------------------------------------------
# __all__ / dir()
# ---------------------------------------------------------------------


@pytest.mark.finding("10.72")
def test_annotations_does_not_leak() -> None:
    assert "annotations" not in vars(gf)
    assert "annotations" not in dir(gf)
    assert "annotations" not in gf.__all__


@pytest.mark.finding("10.72")
def test_all_is_unique_and_every_name_resolves() -> None:
    assert len(gf.__all__) == len(set(gf.__all__))
    for name in gf.__all__:
        assert getattr(gf, name) is not None, name


@pytest.mark.finding("10.72")
def test_every_public_non_module_attribute_is_in_all() -> None:
    for name in gf.__all__:  # load everything lazy first
        getattr(gf, name)
    public = {
        name
        for name in dir(gf)
        if not name.startswith("_")
        and not isinstance(getattr(gf, name), types.ModuleType)
    }
    assert public <= set(gf.__all__), sorted(public - set(gf.__all__))


@pytest.mark.finding("10.72")
def test_dir_shows_the_curated_namespaces_and_all_names() -> None:
    listing = dir(gf)
    assert set(_NAMESPACES) <= set(listing)
    assert set(gf.__all__) <= set(listing)


# ---------------------------------------------------------------------
# curated namespaces
# ---------------------------------------------------------------------


@pytest.mark.finding("10.72")
@pytest.mark.parametrize("namespace", _NAMESPACES)
def test_namespace_is_an_importable_module(namespace: str) -> None:
    module = getattr(gf, namespace)
    assert isinstance(module, types.ModuleType)
    assert module is importlib.import_module(f"groundfield.{namespace}")
    assert module.__all__, namespace
    for name in module.__all__:
        assert hasattr(module, name), (namespace, name)


@pytest.mark.finding("10.72")
@pytest.mark.parametrize(
    "namespace, sources",
    [
        ("plot", ["plotting", "geometry_plot"]),
        ("safety", ["safety"]),
        ("fit", ["vector_fitting", "rho_f_standard"]),
    ],
)
def test_namespace_covers_its_source_modules(namespace: str, sources) -> None:
    """``gf.<namespace>`` offers every public name of its source modules,
    and the objects are the very same as the flat names."""
    module = getattr(gf, namespace)
    for src in sources:
        source = importlib.import_module(f"groundfield.postprocess.{src}")
        assert set(source.__all__) <= set(module.__all__), (src, namespace)
    for name in module.__all__:
        if name in gf.__all__:
            assert getattr(module, name) is getattr(gf, name), name


@pytest.mark.finding("10.72")
def test_namespace_contents() -> None:
    assert {"plot_world", "plot_surface_potential", "plot_sweep_lines"} <= set(
        gf.plot.__all__
    )
    assert {
        "touch_voltage",
        "step_voltage",
        "permissible_touch_voltage_en50522",
    } <= set(gf.safety.__all__)
    assert {"vector_fit", "fit_rho_f_standard", "rho_f_from_field_result"} <= set(
        gf.fit.__all__
    )
    assert gf.io.save_bustype_json is gf.save_bustype_json


# ---------------------------------------------------------------------
# lazy loading
# ---------------------------------------------------------------------


_HEAVY = ("pandas", "matplotlib", "sympy", "scipy.special", "scipy.integrate")


@pytest.mark.finding("10.72")
def test_import_does_not_load_heavy_dependencies() -> None:
    out = _run_fresh(
        "import json, sys, groundfield\n"
        f"heavy = {_HEAVY!r}\n"
        "print(json.dumps(sorted(m for m in heavy if m in sys.modules)))\n"
    )
    assert json.loads(out) == []


@pytest.mark.finding("10.72")
def test_core_workflow_does_not_load_heavy_dependencies() -> None:
    """Building and solving a homogeneous world stays lean as well."""
    out = _run_fresh(
        "import json, sys, groundfield as gf\n"
        "w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))\n"
        "gf.create_electrode(w, 'rod', name='g1', position=(0, 0, 0.5), length=2)\n"
        "gf.create_source(w, attached_to='g1')\n"
        "w.solve(gf.create_engine(segment_length=0.5))\n"
        "print(json.dumps(sorted(m for m in ('pandas', 'matplotlib', 'sympy') "
        "if m in sys.modules)))\n"
    )
    assert json.loads(out) == []


@pytest.mark.finding("10.72")
def test_lazy_name_loads_on_first_access_and_is_cached() -> None:
    out = _run_fresh(
        "import json, sys, groundfield as gf\n"
        "before = 'groundfield.postprocess.sweep' in sys.modules\n"
        "fn = gf.sweep\n"
        "after = 'groundfield.postprocess.sweep' in sys.modules\n"
        "cached = 'sweep' in vars(gf) and vars(gf)['sweep'] is fn\n"
        "print(json.dumps([before, after, cached, fn.__module__]))\n"
    )
    assert json.loads(out) == [False, True, True, "groundfield.postprocess.sweep"]


@pytest.mark.finding("10.72")
def test_subpackages_are_reachable_as_attributes() -> None:
    out = _run_fresh(
        "import groundfield as gf\n"
        "print(gf.generators.TnNetworkGenerator.__name__, gf.postprocess.__name__)\n"
    )
    assert out.split() == ["TnNetworkGenerator", "groundfield.postprocess"]


@pytest.mark.finding("10.72")
def test_unknown_attribute_raises_with_suggestion() -> None:
    with pytest.raises(AttributeError) as excinfo:
        gf.HomogenousSoil  # noqa: B018 (misspelled on purpose)
    assert excinfo.value.name == "HomogenousSoil"
    rendered = "".join(traceback.format_exception_only(excinfo.value))
    assert "Did you mean: 'HomogeneousSoil'" in rendered


@pytest.mark.finding("10.72")
def test_static_imports_match_the_lazy_table() -> None:
    """The ``TYPE_CHECKING`` imports (what IDEs and mypy see) and the lazy
    table (what runs) must name the same objects from the same modules."""
    tree = ast.parse(Path(gf.__file__).read_text(encoding="utf-8"))
    static: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test).endswith(
            "TYPE_CHECKING"
        ):
            for imp in node.body:
                assert isinstance(imp, ast.ImportFrom)
                for alias in imp.names:
                    static[alias.asname or alias.name] = imp.module
    lazy = {
        name: module for module, names in gf._LAZY_IMPORTS.items() for name in names
    }
    assert static == lazy
