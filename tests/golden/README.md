# Golden values

The golden-value cases are the numerical safety net of a release
cycle. They answer one question for every change of the code base:
*which numbers moved since the last release, and was that intended?*

## How it works

1. `cases.py` defines small, deterministic scenarios (image family,
   layered soil, other engines, conductors and earth-return models,
   post-solve matrices of the AP1 production path, generators, fits,
   export, safety, references, defaults). Each case returns a flat
   mapping `quantity name -> numpy array`.
2. `data/*.npz` stores the values **of the last release** together
   with `data/MANIFEST.json` (version, git commit, numpy / scipy /
   Python versions, platform and the SHA-256 of every file).
3. `tests/test_golden.py` recomputes every case and compares it with
   the stored values.

The stored values are not regenerated during a development cycle, so
the comparison always reports every change relative to the last
release. The current baseline is **v0.15.0** (commit `c97b067`).

## Comparison

| stored value | rule |
|---|---|
| float / complex | same shape; norm-wise deviation `max|new-old| / max|old|` |
| string, integer, boolean | exact equality |

| status | meaning |
|---|---|
| `identical` | bit-for-bit equal |
| `within_tol` | deviation ≤ `rtol` (default `1e-9`, env `GF_GOLDEN_RTOL`) |
| `changed` | larger deviation, or a shape / dtype / string change |
| `missing` / `new` | quantity disappeared / appeared |

`GF_GOLDEN_STRICT=1` treats `within_tol` as a failure. Against the
committed baseline (last release) a pure refactor that comes after a
round-off change can no longer be bit-identical; snapshot the branch
you start from and compare against that instead:

```bash
GF_GOLDEN_DATA_DIR=/tmp/base python -m tests.golden.harness generate
# ... refactor ...
GF_GOLDEN_DATA_DIR=/tmp/base GF_GOLDEN_STRICT=1 pytest tests/test_golden.py
```

In this snapshot mode the expected-change declarations are ignored:
nothing may move.

`GF_GOLDEN_STRICT=1` treats `within_tol` as a failure. Performance
refactors must pass in strict mode on the platform the values were
generated on (they are bit-identical by contract); across platforms
(different BLAS / libm) the default tolerance applies.

## Declaring an intended change

A change of numbers is only accepted with a declaration in
`expected_changes/<stream>.toml`:

```toml
[[change]]
cases = ["img2_rod_crosslayer"]      # fnmatch patterns
quantities = ["*"]                   # fnmatch patterns
findings = ["10.15"]
reason = "Cross-layer rods: one owner per (electrode, layer)."
max_rel_change = 10.0                # optional upper bound
min_rel_change = 1e-3                # optional: the change must happen
allow_shape_change = false           # optional (discretisation changes)
```

Undeclared changes fail. A declared change that exceeds
`max_rel_change`, or that does not reach `min_rel_change`, fails too.
When several declarations (of different streams) cover the same
quantity, their bounds add up; one declaration without a bound makes
the change unbounded.

## Commands

Run from the repository root:

```bash
pytest tests/test_golden.py                          # compare
GF_GOLDEN_STRICT=1 pytest tests/test_golden.py       # bit-identity
pytest tests/test_golden.py --golden-report out.json # with report
python -m tests.golden.harness report --markdown moved.md
python -m tests.golden.harness generate --only NEW_CASE
python -m tests.golden.harness check-manifest
```

`report --markdown` lists every moved quantity with its finding; it
feeds the numbers-change block of the CHANGELOG.

## Rules for cases

* Deterministic, fast (well under two seconds), public API only.
* Append-only: never change an existing case, add a new one.
* Residual-type quantities that are pure round-off (e.g. the rms of an
  exact fit) get an absolute noise floor in `NOISE_FLOORS`; a deviation
  below the floor counts as `within_tol`.
* Engines are built with `_engine()`, which pins `kernel="point"` as
  soon as the engine offers a `kernel` option. The `defaults_*` cases
  use plain defaults and record every user-visible change of a
  default.
* A new baseline is stored only at release time:
  `generate --force --reason "baseline vYYYY.MM.MICRO"`, after which
  the declarations in `expected_changes/` are removed.
