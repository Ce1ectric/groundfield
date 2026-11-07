"""Packaging contract: what installing groundfield puts into an environment.

Findings of review pass 10:

* 10.02 - the v0.15.0 wheel installed a top-level ``scripts`` package and
  the console scripts ``release`` / ``generate-third-party-licenses``,
  which collide file-for-file with groundinsight 0.3.0; no ``py.typed``.
* 10.11 - ``polars`` was a hard dependency that is never imported,
  ``poetry-core`` was unbounded, the metadata used the legacy
  ``[tool.poetry]`` tables, and ``profile.out`` (local paths) was tracked.

``test_wheel_contents`` inspects a built wheel: the one named by the
``GF_WHEEL`` environment variable (CI builds it with ``poetry build``),
or one built here with the ``poetry-core`` backend if it is importable;
otherwise the test is skipped.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

import groundfield

REPO = Path(__file__).resolve().parents[1]
CODE_DIRS = ("src", "tests", "scripts")


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def _all_dependency_names(data: dict[str, Any]) -> set[str]:
    """Names of every direct dependency: runtime, extras and Poetry groups."""
    names: set[str] = set()
    project = data.get("project", {})
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    names.update(Requirement(r).name.lower() for r in requirements)
    poetry = data.get("tool", {}).get("poetry", {})
    names.update(k.lower() for k in poetry.get("dependencies", {}))
    for group in poetry.get("group", {}).values():
        names.update(k.lower() for k in group.get("dependencies", {}))
    names.discard("python")
    return names


_DYNAMIC_IMPORTERS = {"import_module", "__import__", "importorskip", "find_spec"}


def _imported_modules(path: Path) -> set[str]:
    """Top-level modules a file imports, including optional dynamic imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and node.args:
            # importlib.import_module("x"), __import__("x"),
            # pytest.importorskip("x"), importlib.util.find_spec("x")
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else None
            if isinstance(func, ast.Name):
                name = func.id
            first = node.args[0]
            if (
                name in _DYNAMIC_IMPORTERS
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
            ):
                modules.add(first.value.split(".")[0])
    return modules


# ---------------------------------------------------------------- pyproject


@pytest.mark.finding("10.02")
def test_no_scripts_package_and_no_console_scripts() -> None:
    data = _pyproject()
    poetry = data.get("tool", {}).get("poetry", {})
    project = data.get("project", {})
    assert [p["include"] for p in poetry.get("packages", [])] == ["groundfield"]
    assert "scripts" not in poetry, "[tool.poetry.scripts] installs commands"
    assert "scripts" not in project, "[project.scripts] installs commands"
    assert "gui-scripts" not in project


@pytest.mark.finding("10.02")
def test_py_typed_marker_exists() -> None:
    assert (REPO / "src" / "groundfield" / "py.typed").is_file()


@pytest.mark.finding("10.11")
def test_polars_is_neither_a_dependency_nor_imported() -> None:
    assert "polars" not in _all_dependency_names(_pyproject())
    importers = [
        path.relative_to(REPO).as_posix()
        for folder in CODE_DIRS
        for path in sorted((REPO / folder).rglob("*.py"))
        if "polars" in _imported_modules(path)
    ]
    assert importers == []


@pytest.mark.finding("10.11")
def test_build_backend_is_bounded() -> None:
    requires = [Requirement(r) for r in _pyproject()["build-system"]["requires"]]
    (core,) = [r for r in requires if r.name == "poetry-core"]
    assert core.specifier, "poetry-core must be pinned to a range"
    assert core.specifier.contains("2.4.0")
    assert not core.specifier.contains("3.0.0"), "upper bound below 3.0 missing"


@pytest.mark.finding("10.11")
def test_metadata_uses_the_project_table() -> None:
    data = _pyproject()
    project = data["project"]
    for key in ("name", "version", "description", "readme", "license", "authors"):
        assert key in project, f"[project] {key} missing"
    assert project["name"] == "groundfield"
    legacy = {
        "name",
        "version",
        "description",
        "authors",
        "license",
        "readme",
        "repository",
        "documentation",
        "homepage",
        "classifiers",
        "extras",
        "dependencies",
    }
    poetry = data["tool"]["poetry"]
    assert legacy.isdisjoint(poetry), sorted(legacy & set(poetry))


def test_classifiers_and_python_range() -> None:
    project = _pyproject()["project"]
    classifiers = set(project["classifiers"])
    versions = {
        c.rsplit(" :: ", 1)[1]
        for c in classifiers
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    }
    assert versions == {"3.12", "3.13", "3.14"}
    for expected in (
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Typing :: Typed",
    ):
        assert expected in classifiers
    python = Requirement(f"python{project['requires-python']}").specifier
    assert python.contains("3.12") and python.contains("3.14")
    assert not python.contains("3.11")


def test_license_report_covers_runtime_dependencies() -> None:
    """``scripts/generate_third_party_licenses.py`` reads the PEP 621 tables."""
    spec = importlib.util.spec_from_file_location(
        "gf_license_tool", REPO / "scripts" / "generate_third_party_licenses.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    names = module._runtime_dependency_names(REPO / "pyproject.toml")

    project = _pyproject()["project"]
    requirements = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        requirements.extend(extra)
    assert names == sorted({Requirement(r).name for r in requirements}, key=str.lower)
    assert {"numpy", "scipy", "pydantic", "groundinsight", "shapely"} <= set(names)


def test_version_sources_agree() -> None:
    pyproject = _pyproject()["project"]["version"]
    citation = re.search(
        r'^version:\s*"?([^"\s]+)"?\s*$',
        (REPO / "CITATION.cff").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert citation is not None
    assert pyproject == groundfield.__version__ == citation.group(1)
    assert str(Version(pyproject)) == pyproject, "version is not canonical PEP 440"


# ---------------------------------------------------------------- repository


@pytest.mark.finding("10.11")
def test_profile_out_is_not_tracked() -> None:
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = subprocess.run(
        ["git", "ls-files", "--", "profile.out"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert tracked == []
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", "profile.out"],
        cwd=REPO,
        check=False,
    )
    assert ignored.returncode == 0, "profile.out is not in .gitignore"


# ---------------------------------------------------------------- the wheel


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if os.environ.get("GF_WHEEL"):
        return Path(os.environ["GF_WHEEL"])
    backend = pytest.importorskip(
        "poetry.core.masonry.api", reason="set GF_WHEEL or install poetry-core"
    )
    out = tmp_path_factory.mktemp("wheel")
    cwd = Path.cwd()
    os.chdir(REPO)  # the PEP 517 backend reads ./pyproject.toml
    try:
        name = backend.build_wheel(str(out))
    finally:
        os.chdir(cwd)
    return out / name


@pytest.mark.finding("10.02", "10.11")
def test_wheel_contents(wheel: Path) -> None:
    name, version = wheel.name.split("-")[:2]
    assert name == "groundfield"
    dist_info = f"groundfield-{version}.dist-info"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = archive.read(f"{dist_info}/METADATA").decode("utf-8")
        entry_points = (
            archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
            if f"{dist_info}/entry_points.txt" in names
            else ""
        )
    assert {n.split("/")[0] for n in names} == {"groundfield", dist_info}
    assert "groundfield/py.typed" in names
    assert "console_scripts" not in entry_points
    requires = [
        Requirement(line.split(":", 1)[1].strip()).name.lower()
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist:")
    ]
    assert "polars" not in requires
    for core in ("numpy", "scipy", "pydantic", "sympy", "matplotlib", "pandas"):
        assert core in requires
    assert "Classifier: Programming Language :: Python :: 3.15" not in metadata
