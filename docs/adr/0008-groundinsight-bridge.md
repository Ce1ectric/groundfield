# ADR-0008: groundinsight bridge — `BusType` export from a `rho-f` fit

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-05-08 |
| **Deciders** | Christian Ehlert |
| **Scope** | `groundfield.io.groundinsight`; bridge to the sister project `groundinsight` |

## Context

`groundfield` is the field-grade reference. `groundinsight` is the
reduced equivalent network model that runs the actual planning and
fault-current studies. The dissertation pipeline

```
PDE / field model  →  reduced rho-f model  →  measurement-anchored
                                              network model
```

requires a precise, regression-tested handover between
`groundfield` and `groundinsight`. The `rho-f` fit is now in place
in two flavours:

- `groundfield.postprocess.vector_fitting` — a Gustavsen/Semlyen
  rational fit $Z(s)$ in the Laplace variable $s = j\,2\pi f$ at
  **fixed soil** (output of one parametric run).
- `groundfield.postprocess.rho_f_standard` — the dissertation's
  physically-motivated 5-coefficient form
  $Z(\rho, f) = k_1\rho + (k_2 + j k_3)f + (k_4 + j k_5)f\rho$
  fitted across a soil-resistivity sweep.

Both produce a SymPy expression. `groundinsight.BusType` consumes
exactly such an expression as its `impedance_formula`. What is
missing is the actual transport: a documented file format, a
Python helper that builds a `BusType` and (optionally) writes it
to the `groundinsight` SQLite store, and a regression test that
proves the formula round-trips through `groundinsight`'s own
parser and produces the same numbers.

## Decision

### Symbol convention

`groundinsight.BusType.impedance_formula` is parsed with two free
symbols that the `compute_impedance` evaluator binds at run time:

- `f` — frequency in Hz
- `rho` — `Bus.specific_earth_resistance`, in $\Omega\,\mathrm{m}$

and the imaginary unit `j` (mapped to `sympy.I` by the validator).

Therefore:

- A **`RhoFStandardFit` exports natively**: its expression is
  already in `(rho, f)`.
- A **`VectorFitResult` exports via the substitution**
  $s \to j\,2\pi f$ inside `fit_to_sympy(...)`. The exported
  formula then depends on `f` only and is independent of `rho`.
  This is correct: a vector fit is bound to one specific soil
  configuration, and its `BusType` is meant to be paired with a
  `Bus` whose `specific_earth_resistance` matches that soil.
  Discrepancies are flagged at export time but not enforced.

### Transport: JSON + Python API, both first-class

The user has two equally supported paths:

1. **`to_bustype_dict(fit, ...)`** returns a JSON-ready Python
   `dict`; **`save_bustype_json(fit, path, ...)`** writes the
   neutral schema below to disk. This is the canonical, persistent
   exchange. The schema is versioned (`schema_version`), and
   `groundfield` will honour older versions through dedicated
   loaders. The JSON file is independent of the `groundinsight`
   Python package — anyone can produce or consume it.

2. **`to_bustype(fit, ...)`** returns a live
   `groundinsight.BusType` Pydantic instance via a **lazy import**
   (no top-level dependency on `groundinsight`). For users running
   in a notebook with both packages installed this is the
   one-liner that closes the family pipeline. If `groundinsight`
   is not installed the function raises a clear `ImportError`
   pointing at the JSON path or at the optional install
   `pip install groundfield[groundinsight]`.

### JSON schema (v1)

Single document, parseable in seconds, no MIME-type tricks. Layout
is intentionally close to the `groundinsight.BusType` Pydantic
schema so a programme that only knows the `groundinsight` shape
can still read the file by ignoring the metadata block.

```jsonc
{
  "schema": "groundfield.bustype",
  "schema_version": 1,
  "name": "TN_house_connection_5EFH",
  "description": "...",
  "system_type": "LV",
  "voltage_level": 0.4,
  "impedance_formula": "0.123*rho + 0.000456*f + 0.0*j*f + 1.2e-7*f*rho + 3.4e-8*j*f*rho",
  "samples": {
    "frequency_Hz": [0.1, 1.0, 10.0, 50.0, 100.0, 1000.0],
    "rho_Ohm_m":   [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    "Z_real_Ohm":  [...],
    "Z_imag_Ohm":  [...]
  },
  "metadata": {
    "fit_method":      "rho_f_standard",
    "fit_quality":     {"rms_error_Ohm": 0.012, "rms_relative": 0.004},
    "coefficients":    {"k1": 0.123, "k2": 4.56e-4, "k3": 0.0,
                        "k4": 1.2e-7, "k5": 3.4e-8},
    "source":          "groundfield 0.1.0",
    "created_at_utc":  "2026-05-08T14:32:11Z",
    "electrode_name":  "trafo_ring",
    "soil_summary":    "TwoLayerSoil(rho_1 sweep [50, 100, 300, 1000], rho_2=20, h_1=2.0)"
  }
}
```

For `vector_fit` the `metadata` block carries `n_poles`, the pole
list and the residue list; `coefficients` is replaced by
`{"R_inf": ..., "L_inf": ...}`. For `rho_f_standard` the
`samples.rho_Ohm_m` array spans all sampled rhos; for
`vector_fit` it is constant at the soil under which the fit was
done.

### Coupling

`groundinsight` becomes an **optional dependency** declared as the
extras group `[groundinsight]` in `pyproject.toml`. The library
imports `groundinsight` lazily inside the few functions that
return `BusType` instances or talk to the database, so

- `pip install groundfield` works as before;
- `pip install groundfield[groundinsight]` enables the
  Python-API path;
- the JSON path needs neither extra.

## Validation programme

1. **Symbol round-trip** — every formula produced by
   `to_bustype_*` is fed back through
   `groundinsight.utils.validations.validate_impedance_formula_value`
   and through `groundinsight.utils.impedance_calculator.compute_impedance`,
   and the resulting per-frequency complex values must match
   `fit.evaluate(...)` to $10^{-9}$ relative.
2. **JSON round-trip** — `save_bustype_json` followed by
   `load_bustype_json` reproduces every numerical field bit-exact
   and the formula string verbatim.
3. **End-to-end** — a transformer-station notebook
   (`notebooks/19_groundinsight_export.ipynb`) sweeps
   $\rho_1 \in \{50, 100, 300, 1000\}\,\Omega\,\mathrm{m}$ on a
   2-layer soil, fits the standard form, exports a `BusType`,
   loads it back into a `groundinsight.Network`, and reproduces
   the per-frequency $|Z|$ and $\arg Z$ within the fit RMS.
4. **Optional-dependency hygiene** — a test that monkey-patches
   `groundinsight` to be unimportable and verifies the
   `to_bustype` / `save_bustype_to_db` functions raise a clear
   `ImportError`, while `to_bustype_dict` / `save_bustype_json`
   keep working.

## Consequences

- The `rho-f` family is now closed-loop: `groundfield` produces a
  fit, exports it, and `groundinsight` consumes it without any
  manual stringification.
- The neutral JSON makes both packages forward-compatible: tabular
  ingestion on the `groundinsight` side will read the `samples`
  block; the formula stays the canonical artifact in v1.
- `BranchType` is **out of scope** for this ADR. The mutual
  coupling between two grounding clusters is not yet a `rho-f`
  product of `groundfield`; it is a follow-up once
  `compute_mutual_impedance` produces a calibrated formula in the
  same `(f, rho, l)` symbol set that `BranchType` expects.

## References

- ADR-0001, ADR-0007 — engine family and cross-layer support that
  produced the underlying physics.
- Vector fitting: Gustavsen & Semlyen, *IEEE T-PWRD* 14(3),
  1052–1061 (1999).
- Standard `rho-f` form: dissertation concept document
  (`0_forschungsfragen/main.tex`, kapitel 06).
- `groundinsight.models.core_models.BusType` — consumer of the
  exported formula.
