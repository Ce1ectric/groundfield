#!/usr/bin/env python
"""Generate the validation page of the documentation from the engine benchmarks.

The page ``docs/explanation/validation.md``, its figures in
``docs/explanation/validation/`` and the validation tables inside the
engine theory pages (``docs/explanation/engines/*.md``, between the
markers ``<!-- validation:begin <name> -->`` and
``<!-- validation:end <name> -->``) are generated from two sources and
contain no hand-written numbers:

* ``tests/benchmarks/tolerances.py`` — the benchmark cases, their
  references and the bound every benchmark test asserts;
* ``docs/explanation/validation/measurements.json`` — the values measured
  by running the engine benchmark suite B1–B13 (``pytest tests/benchmarks
  --run-benchmark`` with ``GF_BENCHMARK_REPORT``), plus the kernel
  checks of this script: rod, ring and strip of the benchmark worlds
  solved with the point and the line kernel of the image family and
  compared with the exact resistance of the uniform-current model
  (closed forms below, no groundfield solver involved).

Usage, from the repository root::

    python scripts/generate_validation_page.py --measure   # run, store, render
    python scripts/generate_validation_page.py             # render the stored values
    python scripts/generate_validation_page.py --check     # exit 1 if out of date

``tests/test_validation_page.py`` runs the check; with ``--run-benchmark``
it also re-measures and compares the fresh values with the stored ones.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # ``tests.benchmarks`` is imported from the checkout
    sys.path.insert(0, str(ROOT))

DOCS = ROOT / "docs" / "explanation"
PAGE = DOCS / "validation.md"
ASSETS = DOCS / "validation"
MEASUREMENTS = ASSETS / "measurements.json"
ENGINE_PAGES: dict[str, Path] = {
    name: DOCS / "engines" / f"{name}.md"
    for name in ("image", "image_2layer", "mom", "mom_sommerfeld", "fem")
}
FIGURES = (
    "benchmark_overview.svg",
    "kernel_exactness.svg",
    "two_layer_top_layer.svg",
)
SCHEMA_VERSION = 1

#: Backends with a section of their own; the deprecated names follow.
BACKEND_ORDER = ("image", "image_2layer", "mom", "mom_sommerfeld", "fem")
ALIASES = {"cim": "image", "bem": "mom"}

_BEGIN = "<!-- validation:begin {name} -->"
_END = "<!-- validation:end {name} -->"

# ---------------------------------------------------------------------
# Kernel checks: exact resistances of the uniform-current model
# ---------------------------------------------------------------------
#
# Every closed form below is the Galerkin (segment-averaged) resistance of
# a wire that carries a uniform line current, with the reduced kernel
# 1/sqrt(r^2 + a^2) (a = wire radius) the line kernel uses for every pair.


def galerkin_self(length: float, d: float) -> float:
    """``(1/L²) ∫∫ dl dl' / sqrt((l - l')² + d²)`` of two parallel lines, 1/m.

    Two collinear-parallel lines of length ``L`` at distance ``d``; with
    ``d = a`` it is the self term of a wire of radius ``a``.
    """
    q = d / length
    return 2.0 * (math.asinh(1.0 / q) - math.sqrt(1.0 + q * q) + q) / length


def rod_uniform_resistance(rho: float, length: float, radius: float) -> float:
    """Rod from the surface: rod and air image form one line of length ``2L``."""
    return rho / (2.0 * math.pi) * galerkin_self(2.0 * length, radius)


def strip_uniform_resistance(
    rho_1: float,
    length: float,
    radius: float,
    depth: float,
    rho_2: float | None = None,
    h_1: float | None = None,
) -> float:
    """Horizontal strip at ``depth``; homogeneous or two-layer (Tagg/Sunde).

    The images of a horizontal line are parallel lines, so every image
    term is :func:`galerkin_self` at the vertical image distance. The
    series is summed until ``|K|^n < 1e-17``.
    """

    def par(dz: float) -> float:
        return galerkin_self(length, math.sqrt(dz * dz + radius * radius))

    total = galerkin_self(length, radius) + par(2.0 * depth)
    if rho_2 is not None and h_1 is not None and rho_2 != rho_1:
        k = (rho_2 - rho_1) / (rho_2 + rho_1)
        n = 1
        while abs(k) ** n > 1e-17:
            for dz in (
                2 * n * h_1,
                abs(2 * n * h_1 - 2 * depth),
                2 * n * h_1,
                2 * n * h_1 + 2 * depth,
            ):
                total += k**n * par(dz)
            n += 1
    return rho_1 / (4.0 * math.pi) * total


@dataclass(frozen=True)
class KernelCase:
    """One world of the kernel check."""

    key: str
    label: str
    soil: tuple[float, ...]  # (rho,) or (rho_1, rho_2, h_1)
    electrode: dict[str, Any]
    segment_lengths: tuple[float, ...]


KERNEL_CASES = (
    KernelCase(
        "rod",
        "rod 3 m from the surface, a = 8 mm, 100 Ω·m (B2 world)",
        (100.0,),
        dict(kind="rod", position=(0.0, 0.0, 0.0), length=3.0, wire_radius=0.008),
        (0.5, 0.25, 0.125),
    ),
    KernelCase(
        "ring",
        "ring r = 2 m at 0.8 m, a = 8 mm, 100 Ω·m (B3 world)",
        (100.0,),
        dict(kind="ring", center=(0.0, 0.0, 0.8), radius=2.0, wire_radius=0.008),
        (0.5, 0.25, 0.125),
    ),
    KernelCase(
        "strip",
        "strip 10 m at 0.5 m, a = 5 mm, 100 Ω·m",
        (100.0,),
        dict(
            kind="strip",
            start=(0.0, 0.0, 0.5),
            end=(10.0, 0.0, 0.5),
            wire_radius=0.005,
        ),
        (2.0, 0.5, 0.125),
    ),
    KernelCase(
        "strip_2l",
        "strip 10 m at 0.5 m, a = 5 mm, two-layer 1000/30 Ω·m, h₁ = 5 m",
        (1000.0, 30.0, 5.0),
        dict(
            kind="strip",
            start=(0.0, 0.0, 0.5),
            end=(10.0, 0.0, 0.5),
            wire_radius=0.005,
        ),
        (2.0, 0.5, 0.125),
    ),
)


def kernel_reference(case: KernelCase) -> float:
    """Exact uniform-current resistance of a kernel case in Ω."""
    e = case.electrode
    if case.key == "rod":
        return rod_uniform_resistance(case.soil[0], e["length"], e["wire_radius"])
    if case.key == "ring":
        from tests.benchmarks.references import ring_thin_wire_resistance

        return ring_thin_wire_resistance(
            case.soil[0], e["radius"], e["wire_radius"], e["center"][2]
        )
    length = math.dist(e["start"], e["end"])
    if len(case.soil) == 1:
        return strip_uniform_resistance(
            case.soil[0], length, e["wire_radius"], e["start"][2]
        )
    rho_1, rho_2, h_1 = case.soil
    return strip_uniform_resistance(
        rho_1, length, e["wire_radius"], e["start"][2], rho_2, h_1
    )


def measure_kernels() -> list[dict[str, Any]]:
    """Solve every kernel case with both kernels; signed deviations."""
    import groundfield as gf
    from tests.benchmarks.references import rod_thin_wire_limit

    rows: list[dict[str, Any]] = []
    for case in KERNEL_CASES:
        reference = kernel_reference(case)
        thin_wire = None
        if case.key == "rod":
            e = case.electrode
            thin_wire = rod_thin_wire_limit(e["length"], e["wire_radius"], case.soil[0])
        for kernel in ("point", "line"):
            for ds in case.segment_lengths:
                if len(case.soil) == 1:
                    soil = gf.HomogeneousSoil(rho=case.soil[0])
                else:
                    soil = gf.TwoLayerSoil(
                        rho_1=case.soil[0], rho_2=case.soil[1], h_1=case.soil[2]
                    )
                world = gf.create_world(soil=soil)
                params = dict(case.electrode)
                gf.create_electrode(world, params.pop("kind"), name="g1", **params)
                gf.create_source(world, attached_to="g1", magnitude=1.0)
                engine = gf.create_engine(segment_length=ds, kernel=kernel)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    z = world.solve(engine).cluster_impedance("g1")[0].real
                row = {
                    "case": case.key,
                    "kernel": kernel,
                    "ds": ds,
                    "resistance": z,
                    "reference": reference,
                    "deviation": z / reference - 1.0,
                }
                if thin_wire is not None:
                    row["thin_wire_reference"] = thin_wire
                    row["thin_wire_deviation"] = z / thin_wire - 1.0
                rows.append(row)
    return rows


# ---------------------------------------------------------------------
# Field quantities: uniform-current ansatz against point matching
# ---------------------------------------------------------------------

#: Worlds of the field check: (label, electrode, point above the corner /
#: tip / wire, outward direction, centre of a mesh or None).
FIELD_CASES: dict[str, tuple[str, dict[str, Any], tuple, tuple, tuple | None]] = {
    "grid": (
        "grid 20 m × 20 m, 4 × 4 meshes, at 0.5 m; corner",
        dict(kind="grid_mesh", corner=(0.0, 0.0, 0.5), size=(20.0, 20.0), n_x=4, n_y=4),
        (0.0, 0.0),
        (-1.0, -1.0),
        (2.5, 2.5),
    ),
    "star": (
        "star, 4 arms of 10 m, at 0.5 m; arm tip",
        dict(kind="star", center=(0.0, 0.0, 0.5), n_arms=4, arm_length=10.0),
        (10.0, 0.0),
        (1.0, 0.0),
        None,
    ),
    "ring": (
        "ring r = 5 m at 0.5 m",
        dict(kind="ring", center=(0.0, 0.0, 0.5), radius=5.0),
        (5.0, 0.0),
        (1.0, 0.0),
        None,
    ),
}
FIELD_RUNS = (("image", "point"), ("image", "line"), ("mom", None))


def measure_field() -> list[dict[str, Any]]:
    """Impedance, touch, step and mesh voltage: image family against ``mom``.

    Homogeneous 100 Ω·m, 1 A, ds = 0.5 m. Touch voltage with the feet 1 m
    outside the corner / tip / wire (``anchor="footprint"``); step voltage
    from the surface point above the corner / tip / wire 1 m outwards; mesh
    voltage between the electrode and the surface above the centre of the
    corner mesh. All values in V per A (= Ω).
    """
    import numpy as np

    import groundfield as gf

    rows: list[dict[str, Any]] = []
    for key, (_, electrode, corner, out, mesh) in FIELD_CASES.items():
        e = np.array([out[0], out[1], 0.0]) / math.hypot(*out)
        for backend, kernel in FIELD_RUNS:
            world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
            params = dict(electrode)
            gf.create_electrode(world, params.pop("kind"), name="g1", **params)
            gf.create_source(world, attached_to="g1", magnitude=1.0)
            engine = gf.create_engine(
                backend=backend, segment_length=0.5, kernel=kernel
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = engine.solve(world)
                row = {
                    "case": key,
                    "backend": backend,
                    "kernel": res.metadata["kernel"],
                    "impedance": res.cluster_impedance("g1")[0].real,
                    "touch": gf.touch_voltage(
                        res, world, electrode="g1", distance=1.0, direction=tuple(e)
                    ).real,
                    "step": gf.step_voltage(
                        res,
                        position=(corner[0], corner[1], 0.0),
                        direction=tuple(e),
                        step=1.0,
                    ).real,
                    "mesh": None,
                }
                if mesh is not None:
                    phi = res.potential(np.array([[mesh[0], mesh[1], 0.0]]))[0].real
                    row["mesh"] = res.electrode_potentials["g1"][0].real - phi
            rows.append(row)
    return rows


# ---------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------

_SUMMARY = re.compile(r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?)")


def run_benchmark_suite() -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """Run ``pytest tests/benchmarks --run-benchmark``; rows, counts, exit code."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "measured.json"
        env = dict(os.environ)
        env["GF_BENCHMARK_REPORT"] = str(report)
        paths = [str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")]
        env["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(var, "1")
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/benchmarks",
            "--run-benchmark",
            "-q",
            "-p",
            "no:cacheprovider",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        if not report.exists():
            raise RuntimeError(
                "the benchmark suite wrote no measurements:\n"
                + proc.stdout[-4000:]
                + proc.stderr[-4000:]
            )
        rows = json.loads(report.read_text(encoding="utf-8"))
    last = [line for line in proc.stdout.splitlines() if " in " in line]
    counts: dict[str, int] = {}
    if last:
        for number, word in _SUMMARY.findall(last[-1]):
            counts["errors" if word.startswith("error") else word] = int(number)
    return rows, counts, proc.returncode


def environment() -> dict[str, str]:
    """Versions and platform of the measuring interpreter."""
    import numpy
    import scipy

    import groundfield

    try:
        import numba  # noqa: F401

        has_numba = "yes"
    except ImportError:  # pragma: no cover - depends on the environment
        has_numba = "no"
    return {
        "date": _dt.datetime.now(_dt.UTC).date().isoformat(),
        "groundfield": groundfield.__version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "numba": has_numba,
        "machine": f"{platform.system()} {platform.machine()}",
    }


def measure() -> dict[str, Any]:
    """Run the benchmark suite and the kernel checks."""
    rows, counts, code = run_benchmark_suite()
    return {
        "schema_version": SCHEMA_VERSION,
        "environment": environment(),
        "suite": {"counts": counts, "exit_code": code},
        "benchmarks": rows,
        "kernel": measure_kernels(),
        "field": measure_field(),
    }


def load_measurements(path: Path = MEASUREMENTS) -> dict[str, Any]:
    """Read the stored measurements (raises for an unknown schema)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unknown schema_version {data.get('schema_version')}")
    return data


def store_measurements(data: dict[str, Any], path: Path = MEASUREMENTS) -> None:
    """Write the measurements as indented UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", "utf-8")


def compare_measurements(
    fresh: dict[str, Any],
    stored: dict[str, Any],
    *,
    rel: float = 1e-3,
    abs_tol: float = 1e-12,
) -> list[str]:
    """Differences between two measurement sets (empty list: they agree).

    Values agree when ``|a - b| <= max(rel * |b|, abs_tol)``; rows are
    matched by benchmark, key and case (kernel rows by case, kernel and
    segment length).
    """

    def bench_key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (row["benchmark"], row["backend"], row["case"])

    def kernel_key(row: dict[str, Any]) -> tuple[str, str, float]:
        return (row["case"], row["kernel"], row["ds"])

    def field_key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (row["case"], row["backend"], row["kernel"])

    problems: list[str] = []
    for name, key, field in (
        ("benchmarks", bench_key, "value"),
        ("kernel", kernel_key, "deviation"),
        ("field", field_key, "impedance"),
        ("field", field_key, "touch"),
        ("field", field_key, "step"),
    ):
        a = {key(r): r[field] for r in fresh[name]}
        b = {key(r): r[field] for r in stored[name]}
        for k in sorted(set(a) ^ set(b), key=str):
            problems.append(f"{name} {k}: present in only one of the two sets")
        for k in sorted(set(a) & set(b), key=str):
            if abs(a[k] - b[k]) > max(rel * abs(b[k]), abs_tol):
                problems.append(f"{name} {k}: measured {a[k]:.6e}, stored {b[k]:.6e}")
    return problems


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------


def _sci(x: float, digits: int = 2) -> str:
    text = f"{x:.{digits - 1}e}"
    return re.sub(r"e([+-])0*(\d)", r"e\1\2", text).replace("e+", "e")


def fmt_value(x: float | None) -> str:
    """Deviation: percent from 0.1 %, otherwise scientific (2 digits)."""
    if x is None:
        return "—"
    if x == 0.0:
        return "0"
    if abs(x) >= 1e-3:
        text = f"{x * 100:#.2g}".rstrip(".")
        return f"{text} %"
    return _sci(x)


def fmt_signed(x: float) -> str:
    """Signed deviation with a typographic minus."""
    text = fmt_value(x)
    if text.startswith("-"):
        return "\u2212" + text[1:]
    return text if x == 0.0 else f"+{text}"


def fmt_bound(x: float | None) -> str:
    """Bound: percent from 0.1 %, otherwise scientific (1 digit)."""
    if x is None:
        return "excluded"
    if x >= 1e-3:
        return f"{x * 100:g} %"
    return _sci(x, 1)


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


# ---------------------------------------------------------------------
# Rows per backend
# ---------------------------------------------------------------------


@dataclass
class Entry:
    """One (benchmark, tolerance key) pair as seen from one backend."""

    benchmark: str
    key: str
    title: str
    case: str
    reference: str
    bound: Any  # tests.benchmarks.tolerances.Bound
    rows: list[dict[str, Any]]

    @property
    def worst(self) -> float | None:
        """Largest measured value over the cases, ``None`` if unmeasured."""
        values = [r["value"] for r in self.rows]
        return max(values) if values else None

    @property
    def status(self) -> str:
        """``pass``, ``FAIL``, ``excluded`` or ``not measured``."""
        if self.bound.bound is None:
            return "excluded"
        if not self.rows:
            return "not measured"
        return "pass" if self.worst <= self.bound.bound else "FAIL"


def _row_backend(row: dict[str, Any]) -> str:
    """Backend a measured row validates (B9 keys name a group, not a backend)."""
    if row["benchmark"] == "B9":
        return row["case"].split()[0]
    return row["backend"]


def _key_backends(bench: str, key: str, rows: Sequence[dict[str, Any]]) -> list[str]:
    if bench == "B9":
        return sorted(
            {
                _row_backend(r)
                for r in rows
                if r["benchmark"] == "B9" and r["backend"] == key
            }
        )
    if bench == "B13":
        return ["image", "mom"] if key == "image_vs_mom" else ["image"]
    return [key]


def entries_for(backend: str, data: dict[str, Any]) -> list[Entry]:
    """Every tolerance entry that validates ``backend``, with its rows."""
    from tests.benchmarks.tolerances import BENCHMARKS

    measured = [r for r in data["benchmarks"] if r["bound"] is not None]
    out = []
    for bench_id, bench in BENCHMARKS.items():
        for key, b in bench.bounds.items():
            if backend not in _key_backends(bench_id, key, measured):
                continue
            rows = [
                r
                for r in measured
                if r["benchmark"] == bench_id
                and r["backend"] == key
                and (bench_id != "B9" or _row_backend(r) == backend)
            ]
            out.append(
                Entry(bench_id, key, bench.title, bench.case, bench.reference, b, rows)
            )
    return out


def _note(entry: Entry, data: dict[str, Any] | None = None) -> str:
    b = entry.bound
    parts = []
    if entry.benchmark == "B1" and data is not None:
        order = [
            r["value"]
            for r in data["benchmarks"]
            if r["benchmark"] == "B1" and r["case"] == "observed order"
        ]
        if order:
            parts.append(f"observed order of convergence {order[0]:.2f}")
    if b.bound is None and b.measured:
        parts.append(b.measured)
    if b.note:
        parts.append(b.note)
    if b.xfail and entry.status == "pass":
        parts.append(f"bound met since the fix of {b.xfail}")
    return "; ".join(parts)


def backend_table(
    entries: Iterable[Entry],
    data: dict[str, Any] | None = None,
    *,
    with_case: bool = False,
) -> str:
    """Markdown table of the entries of one backend."""
    head = "| Benchmark | Measure | Bound | Worst measured | Cases | Status | Note |"
    if with_case:
        head = (
            "| Benchmark | Case | Measure | Bound | Worst measured | Cases "
            "| Status | Note |"
        )
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for e in entries:
        label = f"{e.benchmark} {e.title}"
        if e.key not in BACKEND_ORDER and e.key not in ALIASES:
            label += f" (`{e.key}`)"
        cells = [label]
        if with_case:
            cells.append(_cell(e.case))
        cells += [
            _cell(e.bound.measure),
            fmt_bound(e.bound.bound),
            fmt_value(e.worst),
            str(len(e.rows)) if e.rows else "—",
            e.status,
            _cell(_note(e, data)),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def kernel_table(rows: Sequence[dict[str, Any]], cases: Sequence[str]) -> str:
    """Signed deviation from the exact uniform-current resistance."""
    lines = [
        "| Case | Exact uniform-current R | Kernel | "
        + " | ".join(f"ds = {ds:g} m" for ds in (2.0, 0.5, 0.25, 0.125))
        + " |",
        "|---|---|---|---|---|---|---|",
    ]
    by_case = {c.key: c for c in KERNEL_CASES}
    for key in cases:
        case = by_case[key]
        for kernel in ("point", "line"):
            sel = {
                r["ds"]: r for r in rows if r["case"] == key and r["kernel"] == kernel
            }
            ref = next(iter(sel.values()))["reference"]
            cells = [
                case.label if kernel == "point" else "",
                f"{ref:.4f} Ω" if kernel == "point" else "",
                f"`{kernel}`",
            ]
            for ds in (2.0, 0.5, 0.25, 0.125):
                cells.append(fmt_signed(sel[ds]["deviation"]) if ds in sel else "")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def field_table(rows: Sequence[dict[str, Any]]) -> str:
    """Deviation of the image family from ``mom`` for near-field quantities."""
    lines = [
        "| Case | Backend (kernel) | Impedance | Touch voltage, 1 m outside "
        "| Step voltage, 1 m | Mesh voltage |",
        "|---|---|---|---|---|---|",
    ]
    for key, (label, *_rest) in FIELD_CASES.items():
        sel = [r for r in rows if r["case"] == key]
        ref = next(r for r in sel if r["backend"] == "mom")

        def dev(r: dict[str, Any], q: str, ref: dict[str, Any] = ref) -> str:
            if r[q] is None or ref[q] is None:
                return "—"
            return fmt_signed(r[q] / ref[q] - 1.0)

        cells = [
            label,
            "`mom` (reference)",
            f"{ref['impedance']:.4f} Ω",
            f"{ref['touch']:.4f} V/A",
            f"{ref['step']:.4f} V/A",
            "—" if ref["mesh"] is None else f"{ref['mesh']:.4f} V/A",
        ]
        lines.append("| " + " | ".join(cells) + " |")
        for r in sel:
            if r["backend"] == "mom":
                continue
            lines.append(
                f"| | `{r['backend']}` (`{r['kernel']}`) | {dev(r, 'impedance')} | "
                f"{dev(r, 'touch')} | {dev(r, 'step')} | {dev(r, 'mesh')} |"
            )
    return "\n".join(lines)


def _rod_thin_wire_sentence(rows: Sequence[dict[str, Any]]) -> str:
    line = [r for r in rows if r["case"] == "rod" and r["kernel"] == "line"]
    point = [r for r in rows if r["case"] == "rod" and r["kernel"] == "point"]
    ref = line[0]["thin_wire_reference"]
    lo = min(r["thin_wire_deviation"] for r in line)
    hi = max(r["thin_wire_deviation"] for r in line)
    p_lo = min(r["thin_wire_deviation"] for r in point)
    p_hi = max(r["thin_wire_deviation"] for r in point)

    def rng(a: float, b: float) -> str:
        return (
            fmt_signed(a)
            if fmt_value(a) == fmt_value(b)
            else (f"{fmt_signed(a)} … {fmt_signed(b)}")
        )

    return (
        f"Against the thin-wire solution of the same rod ({ref:.4f} Ω, the B2 "
        "reference, which lets the current vary along the rod) the line kernel "
        f"stays at {rng(lo, hi)} for every segment length: that is the error of "
        "the uniform-current *ansatz*, not of the kernel. The point kernel "
        f"lies at {rng(p_lo, p_hi)}, closer by accident — its kernel error "
        "has the opposite sign and partly cancels the ansatz error."
    )


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------


def _summary_counts(data: dict[str, Any]) -> str:
    counts = data["suite"]["counts"]
    order = ("passed", "failed", "errors", "xfailed", "xpassed", "skipped")
    return ", ".join(f"{counts[k]} {k}" for k in order if counts.get(k))


def _env_sentence(data: dict[str, Any]) -> str:
    env = data["environment"]
    return (
        f"measured on {env['date']} with groundfield {env['groundfield']} "
        f"(Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}, "
        f"numba installed: {env['numba']}; {env['machine']})"
    )


def render_page(data: dict[str, Any]) -> str:
    """Markdown of ``docs/explanation/validation.md``."""
    from tests.benchmarks.tolerances import BENCHMARKS

    out: list[str] = []
    w = out.append
    w("# Validation and benchmarks")
    w("")
    w(
        "<!-- Generated by scripts/generate_validation_page.py from "
        "tests/benchmarks/tolerances.py and docs/explanation/validation/"
        "measurements.json. Do not edit by hand. -->"
    )
    w("")
    w('!!! info "Generated page"')
    w("")
    w(
        "    Every number on this page comes from one run of the engine "
        "benchmark suite, "
        + _env_sentence(data)
        + ", and from the bounds in `tests/benchmarks/tolerances.py`. "
        "The page is written by `python scripts/generate_validation_page.py`; "
        "`tests/test_validation_page.py` checks that it matches the stored "
        "measurements and, with `--run-benchmark`, that a fresh run reproduces "
        "them."
    )
    w("")
    w(
        f"Suite result of that run: **{_summary_counts(data)}** "
        "(`pytest tests/benchmarks --run-benchmark`; the skipped tests are "
        "combinations a benchmark does not apply to)."
    )
    w("")
    w("## What is compared with what")
    w("")
    w(
        "Each benchmark puts a world with a known answer in front of the "
        "backends. The references do not use a groundfield solver: closed forms "
        "(hemisphere, uniformly loaded ring with elliptic integrals, uniform line "
        "currents), an independent analytic Galerkin solver for the thin-wire rod, "
        "the Tagg/Sunde series summed to $|K|^m < 10^{-17}$, `mpmath` quadrature "
        "of the three-layer kernel, and Pollaczek's line impedance. Where a "
        "benchmark compares two backends on the same discretisation (B6, B7, "
        "B10, B13), it checks the current model of one against the other, not "
        "the accuracy of either; the bound states which."
    )
    w("")
    w("| ID | Benchmark | Case | Reference |")
    w("|---|---|---|---|")
    for bench_id, bench in BENCHMARKS.items():
        w(
            f"| {bench_id} | {_cell(bench.title)} | {_cell(bench.case)} | "
            f"{_cell(bench.reference)} |"
        )
    w("")
    w(
        "Deviations are relative, $|Z/Z_\\text{ref} - 1|$, unless the measure "
        "column says otherwise; *worst measured* is the largest value over the "
        "cases of a row (segment lengths, soils, electrodes)."
    )
    w("")
    w(
        "![Worst measured deviation and bound per benchmark and backend]"
        "(validation/benchmark_overview.svg)"
    )
    w("")
    w("## Tolerance table per backend")
    w("")
    w(
        "The cross-checks between backends run the image family with "
        '`kernel="point"`, the kernel of the `mom` family, so that they compare '
        "current models on one kernel (B8 runs every backend with its default "
        "kernel). The default line kernel of the image family is checked "
        "separately [below](#kernel-of-the-image-family)."
    )
    w("")
    for backend in BACKEND_ORDER:
        w(f"### `{backend}`")
        w("")
        w(backend_table(entries_for(backend, data), data))
        w("")
    w("### Deprecated names `cim` and `bem`")
    w("")
    w(
        "`cim` runs the `image` auto-dispatch and `bem` runs `mom` (deprecated "
        "aliases since 2026.11). Their rows reproduce the numbers of the backend "
        "they name."
    )
    w("")
    for alias, target in ALIASES.items():
        w(f"`{alias}` (runs `{target}`):")
        w("")
        w(backend_table(entries_for(alias, data), data))
        w("")
    w("## Kernel of the image family")
    w("")
    w(
        "The image family assumes a uniform current per unit length on each "
        "electrode. For that model the resistance of a straight wire and of a "
        "ring has a closed form, so the error of the **kernel** can be separated "
        "from the error of the **ansatz**. Signed deviation from the exact "
        'uniform-current resistance (`kernel="line"` is the default since '
        '2026.11, `kernel="point"` reproduces 0.15.0):'
    )
    w("")
    w(kernel_table(data["kernel"], [c.key for c in KERNEL_CASES]))
    w("")
    w(_rod_thin_wire_sentence(data["kernel"]))
    w("")
    w(
        "![Kernel error against the exact uniform-current resistance]"
        "(validation/kernel_exactness.svg)"
    )
    w("")
    w("## Two-layer soil, electrodes in the top layer (B6)")
    w("")
    w(
        "`mom` on the same discretisation is the reference: `mom_sommerfeld` "
        "integrates the same layered Green's function numerically (a kernel "
        "check), `image_2layer` replaces the computed current distribution by a "
        "uniform one (an ansatz check), `fem` solves an equivalent hemisphere "
        "(experimental)."
    )
    w("")
    w("![Deviation from mom per two-layer case](validation/two_layer_top_layer.svg)")
    w("")
    w("## Field quantities near the electrode")
    w("")
    w(
        "Impedances average the potential over the electrode, so the "
        "uniform-current ansatz of the image family errs little there; the "
        "surface potential next to a grid corner or a star tip depends on the "
        "local current density, which the ansatz gets wrong. `mom` on the same "
        "discretisation (homogeneous 100 Ω·m, 1 A, ds = 0.5 m) is the reference; "
        "touch voltage with the feet 1 m outside the corner, tip or wire "
        '(`anchor="footprint"`), step voltage from the surface point above the '
        "corner, tip or wire 1 m outwards, mesh voltage at the centre of the "
        "corner mesh. The `point` rows share the kernel of `mom` and isolate the "
        "ansatz:"
    )
    w("")
    w(field_table(data["field"]))
    w("")
    w(
        "The safety helpers emit `UniformCurrentFieldWarning` when a result of "
        "the image family is evaluated within 5 m of such a shape; see "
        "[the image engine]"
        "(engines/image.md#field-quantities-and-the-uniform-current-ansatz)."
    )
    w("")
    w("## All measured values")
    w("")
    w('??? note "Every value of the run (one row per benchmark case)"')
    w("")
    w("    | Benchmark | Key | Case | Measured | Bound |")
    w("    |---|---|---|---|---|")
    for r in data["benchmarks"]:
        value = f"{r['value']:.3g}" if r["bound"] is None else fmt_value(r["value"])
        w(
            f"    | {r['benchmark']} | `{r['backend']}` | {_cell(r['case'])} | "
            f"{value} | {fmt_bound(r['bound']) if r['bound'] is not None else '—'} |"
        )
    w("")
    w("## Regenerating this page")
    w("")
    w("```bash")
    w("python scripts/generate_validation_page.py --measure   # about 1–3 min")
    w("python -m pytest tests/test_validation_page.py         # page up to date")
    w("```")
    w("")
    w(
        "The benchmark tests themselves (`pytest tests/benchmarks "
        "--run-benchmark`) fail when a value leaves its bound; this page only "
        "reports them. A change of a bound or of a measured value therefore "
        "shows up here after `--measure`, and the test of the page fails until "
        "the page is regenerated."
    )
    w("")
    return "\n".join(out)


def render_engine_region(backend: str, data: dict[str, Any]) -> str:
    """Body of the generated validation region of one engine page."""
    out = [
        f"Generated from the benchmark run {_env_sentence(data)}; bounds from "
        "`tests/benchmarks/tolerances.py`. Details and all cases: "
        "[Validation and benchmarks](../validation.md).",
        "",
        backend_table(entries_for(backend, data), data),
    ]
    if backend == "image":
        out += [
            "",
            "Near-field quantities against `mom` on the same discretisation "
            "(uniform-current ansatz; homogeneous 100 Ω·m, ds = 0.5 m):",
            "",
            field_table(data["field"]),
        ]
    cases = {"image": ["rod", "ring", "strip"], "image_2layer": ["strip_2l"]}.get(
        backend
    )
    if cases:
        out += [
            "",
            "Kernel check — signed deviation from the exact uniform-current "
            "resistance:",
            "",
            kernel_table(data["kernel"], cases),
        ]
    return "\n".join(out)


def replace_region(text: str, name: str, body: str) -> str:
    """Replace the generated region ``name`` of a page by ``body``."""
    begin, end = _BEGIN.format(name=name), _END.format(name=name)
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"markers {begin} … {end} not found")
    return pattern.sub(lambda _: f"{begin}\n{body}\n{end}", text, count=1)


def render_all(data: dict[str, Any]) -> dict[Path, str]:
    """Every generated text file: the page and the engine pages."""
    files = {PAGE: render_page(data)}
    for backend, path in ENGINE_PAGES.items():
        if path.exists():
            files[path] = replace_region(
                path.read_text(encoding="utf-8"),
                backend,
                render_engine_region(backend, data),
            )
    return files


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

# Categorical slots of the reference palette (light surface), in order.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
_MARKERS = ("o", "s", "D", "^")
_INK, _INK2, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
_FLOOR = 1e-16


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "groundfield-validation",
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.edgecolor": _INK2,
            "axes.labelcolor": _INK,
            "axes.facecolor": _SURFACE,
            "figure.facecolor": _SURFACE,
            "xtick.color": _INK2,
            "ytick.color": _INK2,
            "axes.grid": True,
            "grid.color": _GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
        }
    )


def _save(fig: Any, name: str) -> None:
    fig.savefig(ASSETS / name, format="svg", metadata={"Date": None})


def figure_overview(data: dict[str, Any]) -> None:
    """Worst measured deviation and bound per benchmark and backend."""
    import matplotlib.pyplot as plt

    labels, worst, bounds = [], [], []
    for backend in (*BACKEND_ORDER,):
        for e in entries_for(backend, data):
            if e.bound.bound is None or not e.rows:
                continue
            labels.append(
                f"{e.benchmark} {backend}"
                + (f" ({e.key})" if e.key not in BACKEND_ORDER else "")
            )
            worst.append(max(e.worst, _FLOOR))
            bounds.append(e.bound.bound)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(7.2, 0.24 * n + 1.2))
    y = list(range(n))[::-1]
    for yi, b in zip(y, bounds, strict=True):
        ax.plot([_FLOOR, b], [yi, yi], color=_GRID, linewidth=1.0, zorder=1)
    ax.scatter(
        bounds, y, marker="|", s=90, color=_INK, linewidths=1.6, zorder=3, label="bound"
    )
    ax.scatter(
        worst,
        y,
        marker="o",
        s=26,
        color=_SERIES[0],
        zorder=4,
        edgecolors=_SURFACE,
        linewidths=0.8,
        label="worst measured",
    )
    ax.set_xscale("log")
    ax.set_xlim(_FLOOR / 3, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.8, n - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("deviation (relative; values below 1e-16 drawn at 1e-16)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2, fontsize=8)
    fig.suptitle(
        "Engine benchmarks: worst measured deviation and bound",
        x=0.01,
        ha="left",
        fontsize=10,
        color=_INK,
    )
    fig.tight_layout()
    _save(fig, "benchmark_overview.svg")
    plt.close(fig)


def figure_kernel(data: dict[str, Any]) -> None:
    """Point and line kernel against the exact uniform-current resistance."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, kernel in zip(axes, ("point", "line"), strict=True):
        for k, case in enumerate(KERNEL_CASES):
            rows = sorted(
                (
                    r
                    for r in data["kernel"]
                    if r["case"] == case.key and r["kernel"] == kernel
                ),
                key=lambda r: r["ds"],
            )
            ax.plot(
                [r["ds"] for r in rows],
                [max(abs(r["deviation"]), 1e-7) for r in rows],
                marker=_MARKERS[k],
                color=_SERIES[k],
                label=case.key.replace("_2l", " (two-layer)"),
            )
        ax.set_xscale("log")
        ax.set_xticks([0.125, 0.25, 0.5, 1.0, 2.0])
        ax.set_xticklabels(["0.125", "0.25", "0.5", "1", "2"])
        ax.minorticks_off()
        ax.set_yscale("log")
        ax.set_ylim(5e-8, 0.1)
        ax.set_xlabel("segment length ds in m")
        ax.set_title(f'kernel="{kernel}"', loc="left", fontsize=10, color=_INK)
    axes[0].set_ylabel("|R / R_exact − 1|")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, fontsize=8)
    fig.suptitle(
        "Image family against the exact uniform-current resistance",
        x=0.01,
        ha="left",
        fontsize=10,
        color=_INK,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "kernel_exactness.svg")
    plt.close(fig)


def figure_two_layer(data: dict[str, Any]) -> None:
    """B6: deviation from ``mom`` per two-layer case and backend."""
    import matplotlib.pyplot as plt

    rows = [r for r in data["benchmarks"] if r["benchmark"] == "B6"]
    cases = list(dict.fromkeys(r["case"] for r in rows))
    backends = ("image_2layer", "mom_sommerfeld", "fem")
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(cases) + 1.4))
    y = {c: len(cases) - 1 - i for i, c in enumerate(cases)}
    for k, backend in enumerate(backends):
        sel = [r for r in rows if r["backend"] == backend]
        ax.scatter(
            [max(r["value"], _FLOOR) for r in sel],
            [y[r["case"]] + (k - 1) * 0.18 for r in sel],
            marker=_MARKERS[k],
            s=26,
            color=_SERIES[k],
            edgecolors=_SURFACE,
            linewidths=0.6,
            label=backend,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlim(_FLOOR / 3, 0.2)
    ax.set_yticks(list(y.values()))
    ax.set_yticklabels(cases, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("|Z / Z_mom − 1| (values below 1e-16 drawn at 1e-16)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=3, fontsize=8)
    fig.suptitle(
        "B6: two-layer soil, electrodes in the top layer",
        x=0.01,
        ha="left",
        fontsize=10,
        color=_INK,
    )
    fig.tight_layout()
    _save(fig, "two_layer_top_layer.svg")
    plt.close(fig)


def render_figures(data: dict[str, Any]) -> None:
    """Write the SVG figures of the page."""
    _style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    figure_overview(data)
    figure_kernel(data)
    figure_two_layer(data)


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------


def write(data: dict[str, Any]) -> list[Path]:
    """Write every generated file; return the paths that changed."""
    changed = []
    for path, text in render_all(data).items():
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if old != text:
            path.write_text(text, encoding="utf-8")
            changed.append(path)
    render_figures(data)
    return changed


def check(data: dict[str, Any] | None = None) -> list[str]:
    """Out-of-date generated files (empty list: everything is current)."""
    data = load_measurements() if data is None else data
    problems = []
    for path, text in render_all(data).items():
        if not path.exists():
            problems.append(f"{path.relative_to(ROOT)} is missing")
        elif path.read_text(encoding="utf-8") != text:
            problems.append(f"{path.relative_to(ROOT)} is out of date")
    for name in FIGURES:
        if not (ASSETS / name).exists():
            problems.append(f"{(ASSETS / name).relative_to(ROOT)} is missing")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--measure", action="store_true", help="run the benchmarks, then render"
    )
    mode.add_argument(
        "--check", action="store_true", help="exit 1 if a generated file is stale"
    )
    args = parser.parse_args(argv)
    if args.check:
        problems = check()
        for p in problems:
            print(p)
        if problems:
            print("run: python scripts/generate_validation_page.py")
        return 1 if problems else 0
    if args.measure:
        data = measure()
        store_measurements(data)
        print(f"benchmark suite: {_summary_counts(data)}")
        if data["suite"]["exit_code"] != 0:
            print("warning: the benchmark suite failed; the page reports it")
    else:
        data = load_measurements()
    for path in write(data):
        print(f"updated {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
