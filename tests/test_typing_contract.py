"""Static typing of the documented public calls (audit A1-06, A1-09).

The docstrings and the changelog show calls such as
``Engine(backend="fem", options={"fem": {"refinement": 1}})`` and
``BackendCapabilities(soils={"homogeneous"})``. On the 2026.11 base both
were type errors for mypy and Pyright / Pylance (VS Code), because the
fields were annotated with the stored types (``dict[str, BackendOptions]``,
``frozenset[str]``) instead of the accepted ones (A1-06).

Under the project's own mypy configuration (pydantic plugin,
``init_forbid_extra``) a misspelled soil keyword such as
``HomogeneousSoil(rhoo=100.0)`` was *not* reported: the
``validation_alias=AliasChoices(...)`` of the renamed soil fields made the
plugin type ``__init__`` as ``**kwargs`` (A1-09).

Each check runs mypy twice: without a configuration (dataclass_transform,
which is what Pyright / Pylance use) and with the ``[tool.mypy]`` /
``[tool.pydantic-mypy]`` tables of ``pyproject.toml``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import groundfield

pytest.importorskip("mypy")

pytestmark = pytest.mark.slow

SRC = Path(groundfield.__file__).resolve().parent.parent
PYPROJECT = SRC.parent / "pyproject.toml"

#: Documented calls; the lines marked ``# error`` must be reported, every
#: other line must type-check.
USER_CODE = """\
import groundfield as gf
from groundfield.solver.registry import BackendCapabilities

engine = gf.Engine(backend="fem", options={"fem": {"refinement": 1}})
options = {"fem": {"refinement": 1}}
engine_from_variable = gf.Engine(backend="fem", options=options)
caps = BackendCapabilities(soils={"homogeneous"}, conductors={"lumped"}, kernels=set())
tabulated = gf.create_engine(sommerfeld_assembly="tabulated")
assembly = gf.Engine(sommerfeld_assembly="tabulated")
soil = gf.HomogeneousSoil(rho=100.0)
layer = gf.SoilLayer(rho=1.0, h=2.0)
two_layer = gf.TwoLayerSoil(rho_1=100.0, rho_2=30.0, h_1=5.0)
stack = gf.MultiLayerSoil(layers=[layer, gf.SoilLayer(rho=30.0)])
typo_rho = gf.HomogeneousSoil(rhoo=100.0)  # error
typo_h = gf.SoilLayer(rho=1.0, hh=2.0)  # error
"""


def _project_config(directory: Path) -> Path:
    """The repository's mypy tables as an ini file (without ``files``)."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    mypy = dict(data["tool"]["mypy"])
    mypy.pop("files", None)
    overrides = mypy.pop("overrides", [])
    lines = ["[mypy]"]
    for key, value in mypy.items():
        if isinstance(value, list):
            value = ", ".join(value)
        lines.append(
            f"{key} = {str(value).lower() if isinstance(value, bool) else value}"
        )
    lines.append("[pydantic-mypy]")
    for key, value in data["tool"]["pydantic-mypy"].items():
        lines.append(f"{key} = {str(value).lower()}")
    for override in overrides:
        for module in override["module"]:
            lines += [f"[mypy-{module}]", "ignore_missing_imports = true"]
    path = directory / "mypy.ini"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def mypy_run(tmp_path_factory):
    """Run mypy on modules; returns the output. One cache per configuration."""
    directory = tmp_path_factory.mktemp("typing")
    (directory / "user_code.py").write_text(USER_CODE, encoding="utf-8")
    configs = {"none": Path(os.devnull), "project": _project_config(directory)}

    def run(config: str, *modules: str) -> str:
        args = [
            sys.executable,
            "-m",
            "mypy",
            "--no-error-summary",
            "--show-error-codes",
        ]
        args += ["--follow-imports=silent", "--cache-dir", str(directory / config)]
        args += ["--config-file", str(configs[config])]
        for module in modules:
            args += ["-m", module]
        env = dict(os.environ, MYPYPATH=f"{directory}{os.pathsep}{SRC}")
        proc = subprocess.run(
            args, capture_output=True, text=True, env=env, cwd=directory, check=False
        )
        return proc.stdout + proc.stderr

    return run


def _error_lines(output: str) -> set[int]:
    return {int(m) for m in re.findall(r"user_code\.py:(\d+): error", output)}


def _marked_lines() -> set[int]:
    return {i for i, line in enumerate(USER_CODE.splitlines(), 1) if "# error" in line}


@pytest.mark.finding("A1-06", "A1-09")
@pytest.mark.parametrize("config", ["none", "project"])
def test_documented_calls_type_check_and_typos_are_reported(mypy_run, config) -> None:
    output = mypy_run(config, "user_code")
    assert _error_lines(output) == _marked_lines(), output
    assert "rhoo" in output and "hh" in output


@pytest.mark.finding("A1-09")
def test_soil_models_and_registry_type_check_under_the_project_config(mypy_run) -> None:
    """No ``[pydantic-alias]`` errors (three on the base) and no others."""
    output = mypy_run(
        "project", "groundfield.soil.models", "groundfield.solver.registry"
    )
    assert ": error:" not in output, output
