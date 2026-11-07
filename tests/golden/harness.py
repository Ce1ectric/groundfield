"""Storage, comparison and reporting for the golden-value cases.

Command line (run from the repository root)::

    python -m tests.golden.harness generate            # store missing cases
    python -m tests.golden.harness generate --only NAME
    python -m tests.golden.harness generate --force --reason "new baseline"
    python -m tests.golden.harness report [--json PATH] [--markdown PATH]
    python -m tests.golden.harness check-manifest

Comparison rules
----------------
Numeric arrays
    Same shape required. The deviation is measured norm-wise,
    ``max|new - old| / max|old|`` (absolute if ``max|old| == 0``);
    NaN must stay NaN at the same positions.
Strings / integer / boolean arrays
    Exact equality.

Statuses
--------
``identical``  bit-for-bit equal.
``within_tol`` deviation <= rtol (default ``1e-9``; ``GF_GOLDEN_RTOL``).
``changed``    deviation > rtol, or a shape / dtype change (``inf``).
``missing``    stored quantity is no longer produced.
``new``        produced quantity has no stored value.

``GF_GOLDEN_STRICT=1`` turns ``within_tol`` into a failure. Use it to
prove that a performance refactor is bit-identical.

Expected changes
----------------
A stream that changes numbers on purpose declares it in
``tests/golden/expected_changes/<stream>.toml``::

    [[change]]
    cases = ["img2_rod_crosslayer"]      # fnmatch patterns
    quantities = ["*"]                   # fnmatch patterns
    findings = ["10.15"]
    reason = "Cross-layer rods: one owner per (electrode, layer)."
    max_rel_change = 10.0                # optional upper bound
    min_rel_change = 1e-3                # optional: the change must happen
    allow_shape_change = false           # optional

The stored golden values stay those of the last release for the
whole development cycle, so ``report`` always lists every number that
moved since that release, together with the declaring finding. That
list feeds the numbers-change block of the CHANGELOG.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).resolve().parent
DATA_DIR = GOLDEN_DIR / "data"
EXPECTED_DIR = GOLDEN_DIR / "expected_changes"
MANIFEST = DATA_DIR / "MANIFEST.json"
REPO_ROOT = GOLDEN_DIR.parent.parent

DEFAULT_RTOL = 1e-9

__all__ = [
    "Comparison",
    "Declaration",
    "compare_case",
    "load_case",
    "load_declarations",
    "load_manifest",
    "rtol",
    "strict",
]


# ---------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------


def rtol() -> float:
    return float(os.environ.get("GF_GOLDEN_RTOL", DEFAULT_RTOL))


def strict() -> bool:
    return os.environ.get("GF_GOLDEN_STRICT", "") not in ("", "0", "false", "no")


# ---------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------


def case_path(name: str) -> Path:
    return DATA_DIR / f"{name}.npz"


def _encode_key(key: str) -> str:
    # ``np.savez`` keys must be valid file names inside the zip.
    return key.replace("/", "__")


def _decode_key(key: str) -> str:
    return key.replace("__", "/")


def save_case(name: str, quantities: dict[str, np.ndarray]) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {_encode_key(k): np.asarray(v) for k, v in quantities.items()}
    for key, arr in payload.items():
        if arr.dtype == object:
            raise TypeError(f"{name}:{key}: object arrays are not storable")
    path = case_path(name)
    with path.open("wb") as fh:
        np.savez(fh, **payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(name: str) -> dict[str, np.ndarray]:
    with np.load(case_path(name), allow_pickle=False) as data:
        return {_decode_key(k): data[k] for k in data.files}


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {"cases": {}}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _write_manifest(manifest: dict) -> None:
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _provenance() -> dict:
    import scipy

    import groundfield

    # git state of the checkout the *imported* package comes from, so
    # values generated from a pristine release checkout (PYTHONPATH)
    # carry that commit and not the one of the working branch.
    src_root = Path(groundfield.__file__).resolve().parents[2]
    return {
        "groundfield": groundfield.__version__,
        "groundfield_path": str(Path(groundfield.__file__).resolve().parent),
        "git_commit": _git("rev-parse", "HEAD", cwd=src_root),
        "git_describe": _git("describe", "--tags", "--always", "--dirty", cwd=src_root),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------
# expected-change declarations
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Declaration:
    source: str
    cases: tuple[str, ...]
    quantities: tuple[str, ...]
    findings: tuple[str, ...]
    reason: str
    max_rel_change: float | None = None
    min_rel_change: float | None = None
    allow_shape_change: bool = False

    def matches(self, case: str, quantity: str) -> bool:
        return any(fnmatch.fnmatchcase(case, p) for p in self.cases) and any(
            fnmatch.fnmatchcase(quantity, p) for p in self.quantities
        )

    def label(self) -> str:
        ids = ", ".join(self.findings) or "no finding"
        return f"{self.source} ({ids})"


_ALLOWED_KEYS = {
    "cases",
    "quantities",
    "findings",
    "reason",
    "max_rel_change",
    "min_rel_change",
    "allow_shape_change",
}


def load_declarations(directory: Path = EXPECTED_DIR) -> list[Declaration]:
    decls: list[Declaration] = []
    for path in sorted(directory.glob("*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        unknown_top = set(data) - {"change"}
        if unknown_top:
            raise ValueError(f"{path.name}: unknown top-level keys {unknown_top}")
        for i, entry in enumerate(data.get("change", [])):
            unknown = set(entry) - _ALLOWED_KEYS
            if unknown:
                raise ValueError(f"{path.name}[{i}]: unknown keys {sorted(unknown)}")
            for required in ("cases", "quantities", "reason"):
                if required not in entry:
                    raise ValueError(f"{path.name}[{i}]: missing {required!r}")
            decls.append(
                Declaration(
                    source=f"{path.stem}[{i}]",
                    cases=tuple(entry["cases"]),
                    quantities=tuple(entry["quantities"]),
                    findings=tuple(entry.get("findings", ())),
                    reason=str(entry["reason"]),
                    max_rel_change=entry.get("max_rel_change"),
                    min_rel_change=entry.get("min_rel_change"),
                    allow_shape_change=bool(entry.get("allow_shape_change", False)),
                )
            )
    return decls


# ---------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------


@dataclass
class Comparison:
    case: str
    quantity: str
    status: str
    max_rel: float
    max_abs: float
    ref_scale: float
    detail: str = ""
    declared_by: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)


def _is_numeric(a: np.ndarray) -> bool:
    return a.dtype.kind in "fc"


def compare_arrays(
    old: np.ndarray, new: np.ndarray
) -> tuple[str, float, float, float, str]:
    """Return ``(status_hint, max_rel, max_abs, ref_scale, detail)``."""
    if old.shape != new.shape:
        return "shape", math.inf, math.inf, 0.0, f"shape {old.shape} -> {new.shape}"
    if _is_numeric(old) and _is_numeric(new):
        o = old.astype(complex)
        n = new.astype(complex)
        nan_o, nan_n = np.isnan(o), np.isnan(n)
        if not np.array_equal(nan_o, nan_n):
            return "nan", math.inf, math.inf, 0.0, "NaN pattern changed"
        mask = ~nan_o
        if not mask.any():
            return "identical", 0.0, 0.0, 0.0, ""
        if np.array_equal(o[mask], n[mask]):
            return "identical", 0.0, 0.0, float(np.max(np.abs(o[mask]))), ""
        diff = np.abs(n[mask] - o[mask])
        max_abs = float(np.max(diff))
        scale = float(np.max(np.abs(o[mask])))
        max_rel = max_abs / scale if scale > 0.0 else max_abs
        return "numeric", max_rel, max_abs, scale, ""
    if old.dtype.kind != new.dtype.kind:
        return "dtype", math.inf, math.inf, 0.0, f"dtype {old.dtype} -> {new.dtype}"
    if np.array_equal(old, new):
        return "identical", 0.0, 0.0, 0.0, ""
    detail = ""
    if old.dtype.kind == "U":
        a, b = old.ravel(), new.ravel()
        idx = int(np.flatnonzero(a != b)[0])
        sa, sb = str(a[idx]), str(b[idx])
        pos = next(
            (k for k, (x, y) in enumerate(zip(sa, sb, strict=False)) if x != y),
            min(len(sa), len(sb)),
        )
        detail = (
            f"element {idx}, char {pos}: "
            f"{sa[max(0, pos - 30) : pos + 30]!r} -> {sb[max(0, pos - 30) : pos + 30]!r}"
        )
    return "exact", math.inf, math.inf, 0.0, detail


def noise_floor(case: str, quantity: str) -> float:
    """Absolute noise floor of a quantity (``0.0`` if none is registered).

    Residual-type quantities (e.g. the rms error of an exact fit) are pure
    round-off; their *relative* change across numpy versions or BLAS
    builds is meaningless. ``tests.golden.cases.NOISE_FLOORS`` maps
    ``case -> {quantity pattern: absolute floor}``; a deviation below the
    floor counts as ``within_tol``.
    """
    from tests.golden.cases import NOISE_FLOORS

    floors = NOISE_FLOORS.get(case, {})
    return max(
        (v for pat, v in floors.items() if fnmatch.fnmatchcase(quantity, pat)),
        default=0.0,
    )


def compare_case(
    name: str,
    current: dict[str, np.ndarray],
    stored: dict[str, np.ndarray],
    *,
    tol: float | None = None,
) -> list[Comparison]:
    tol = rtol() if tol is None else tol
    out: list[Comparison] = []
    for key in sorted(set(stored) | set(current)):
        if key not in current:
            out.append(Comparison(name, key, "missing", math.inf, math.inf, 0.0))
            continue
        if key not in stored:
            out.append(Comparison(name, key, "new", math.inf, math.inf, 0.0))
            continue
        hint, max_rel, max_abs, scale, detail = compare_arrays(
            stored[key], current[key]
        )
        if hint == "identical":
            status = "identical"
        elif hint == "numeric" and (
            max_rel <= tol or max_abs <= noise_floor(name, key)
        ):
            status = "within_tol"
        else:
            status = "changed"
        out.append(Comparison(name, key, status, max_rel, max_abs, scale, detail))
    return out


def judge(
    comparisons: list[Comparison],
    declarations: list[Declaration],
    *,
    strict_mode: bool | None = None,
) -> list[str]:
    """Attach declarations and return the list of failure messages."""
    strict_mode = strict() if strict_mode is None else strict_mode
    failures: list[str] = []
    for c in comparisons:
        matching = [d for d in declarations if d.matches(c.case, c.quantity)]
        c.declared_by = [d.label() for d in matching]
        c.findings = sorted({f for d in matching for f in d.findings})
        moved = c.status in ("changed", "missing", "new") or (
            strict_mode and c.status == "within_tol"
        )
        if not moved:
            continue
        if not matching:
            failures.append(
                f"{c.case}:{c.quantity}: {c.status} "
                f"(max_rel={c.max_rel:.3e}, max_abs={c.max_abs:.3e}) "
                f"{c.detail} -- undeclared change; fix the regression or "
                f"declare it in tests/golden/expected_changes/<stream>.toml"
            )
            continue
        for d in matching:
            if c.status in ("missing", "new") or math.isinf(c.max_rel):
                if d.max_rel_change is not None and not d.allow_shape_change:
                    failures.append(
                        f"{c.case}:{c.quantity}: {c.status} {c.detail} but "
                        f"{d.label()} bounds the change (max_rel_change) and "
                        f"does not set allow_shape_change"
                    )
            elif d.max_rel_change is not None and c.max_rel > d.max_rel_change:
                failures.append(
                    f"{c.case}:{c.quantity}: max_rel={c.max_rel:.3e} exceeds "
                    f"{d.label()} max_rel_change={d.max_rel_change:g}"
                )
    # a declared change must actually happen where min_rel_change is set
    for d in declarations:
        if d.min_rel_change is None:
            continue
        hits = [c for c in comparisons if d.matches(c.case, c.quantity)]
        if not hits:
            continue  # declaration targets other cases
        biggest = max(c.max_rel for c in hits)
        if biggest < d.min_rel_change:
            failures.append(
                f"{hits[0].case}: declared change {d.label()} did not "
                f"materialise (largest max_rel={biggest:.3e} < "
                f"min_rel_change={d.min_rel_change:g})"
            )
    return failures


# ---------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------


def _cmd_generate(args: argparse.Namespace) -> int:
    from tests.golden.cases import CASES, run_case

    manifest = load_manifest()
    cases = manifest.setdefault("cases", {})
    names = args.only or sorted(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        print(f"unknown cases: {unknown}", file=sys.stderr)
        return 2
    if args.force and not args.reason:
        print("--force needs --reason", file=sys.stderr)
        return 2
    prov = _provenance()
    for name in names:
        if case_path(name).exists() and not args.force:
            print(f"skip {name} (stored; use --force --reason ... to overwrite)")
            continue
        quantities = run_case(name)
        digest = save_case(name, quantities)
        entry = {"sha256": digest, "n_quantities": len(quantities), **prov}
        if args.force:
            entry["regenerated_reason"] = args.reason
        cases[name] = entry
        print(f"stored {name}: {len(quantities)} quantities")
    _write_manifest(manifest)
    return 0


def _cmd_check_manifest(_: argparse.Namespace) -> int:
    from tests.golden.cases import CASES

    problems = manifest_problems(set(CASES))
    for p in problems:
        print(p)
    return 1 if problems else 0


def manifest_problems(case_names: set[str]) -> list[str]:
    manifest = load_manifest().get("cases", {})
    problems = []
    for name in sorted(case_names):
        path = case_path(name)
        if not path.exists():
            problems.append(f"{name}: no stored golden values (run generate)")
            continue
        entry = manifest.get(name)
        if entry is None:
            problems.append(f"{name}: not in MANIFEST.json")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry.get("sha256"):
            problems.append(f"{name}: sha256 differs from MANIFEST.json")
    for name in sorted(set(manifest) - case_names):
        problems.append(f"{name}: in MANIFEST.json but no longer a case")
    return problems


def _cmd_report(args: argparse.Namespace) -> int:
    from tests.golden.cases import CASES, run_case

    declarations = load_declarations()
    rows: list[Comparison] = []
    failures: list[str] = []
    for name in sorted(CASES):
        if not case_path(name).exists():
            continue
        comps = compare_case(name, run_case(name), load_case(name))
        failures += judge(comps, declarations)
        rows += comps
    moved = [r for r in rows if r.status not in ("identical", "within_tol")]
    payload = {
        "rtol": rtol(),
        "strict": strict(),
        "provenance": _provenance(),
        "n_quantities": len(rows),
        "n_identical": sum(r.status == "identical" for r in rows),
        "n_within_tol": sum(r.status == "within_tol" for r in rows),
        "moved": [asdict(r) for r in moved],
        "within_tol": [asdict(r) for r in rows if r.status == "within_tol"],
        "failures": failures,
    }
    text = json.dumps(payload, indent=2, default=str)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(_markdown(moved), encoding="utf-8")
    print(
        f"{payload['n_quantities']} quantities: {payload['n_identical']} identical, "
        f"{payload['n_within_tol']} within tol, {len(moved)} moved, "
        f"{len(failures)} failures"
    )
    for f in failures:
        print("FAIL", f)
    return 1 if failures else 0


def _markdown(moved: list[Comparison]) -> str:
    lines = [
        "| case | quantity | change (max rel.) | finding | declared by |",
        "|---|---|---|---|---|",
    ]
    for r in moved:
        change = "structure" if math.isinf(r.max_rel) else f"{r.max_rel:.2%}"
        lines.append(
            f"| {r.case} | {r.quantity} | {change} | {', '.join(r.findings)} "
            f"| {', '.join(r.declared_by) or '**undeclared**'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.golden.harness")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="store golden values")
    g.add_argument("--only", nargs="+", metavar="CASE")
    g.add_argument("--force", action="store_true")
    g.add_argument("--reason", default="")
    g.set_defaults(func=_cmd_generate)
    r = sub.add_parser("report", help="compare against stored values")
    r.add_argument("--json")
    r.add_argument("--markdown")
    r.set_defaults(func=_cmd_report)
    m = sub.add_parser("check-manifest", help="verify stored files")
    m.set_defaults(func=_cmd_check_manifest)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
