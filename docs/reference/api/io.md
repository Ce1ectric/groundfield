# IO and groundinsight

The `io` package is the boundary between `groundfield` and the rest of
the software family: the export of reduced `rho-f` models as
`groundinsight` `BusType` definitions (JSON file or live object, ADR-0008),
CSV tables with a provenance header, and legacy VTK files for ParaView.
Result files of a solve are described on [Result files](results-io.md).

## Mathematical and physical context

The reduction pipeline is

$$
\text{field model} \;\longrightarrow\; Z(\rho, f) \;\longrightarrow\;
\texttt{BusType.impedance\_formula} \;\longrightarrow\; \text{network model}.
$$

`groundinsight` evaluates the formula with two free symbols, `f` (Hz)
and `rho` (the `Bus.specific_earth_resistance` in Ω·m), and the
imaginary unit `j`. Both fit families are projected onto that set:

- **standard form** (`RhoFStandardFit`), fitted over a sweep of $\rho$
  and $f$: $Z = k_0 + k_1\rho + (k_2 + jk_3)f + (k_4 + jk_5)f\rho$,
  already in `(rho, f)`;
- **vector fit** (`VectorFitResult`) at one soil: a rational function
  of $s$, exported with $s \to j2\pi f$, so the formula depends on `f`
  only. The soil it belongs to must be given as `rho_at_fit`; for
  layered soil the convention is the upper-layer resistivity $\rho_1$,
  the parameter of the AP1 $\rho_1$ sweeps, so that both kinds of
  `BusType` pair with the same `Bus`.

Before a `BusType` is written, the exported formula (with its rounded
coefficients) is checked for passivity, $\operatorname{Re}Z \ge 0$ from
0 Hz to twice the highest fitted frequency; a violation warns with
`NonPassiveFitWarning` (or raises `NonPassiveFitError` with
`require_passive=True`). A formula read from a file is untrusted text:
`load_bustype_json` and `evaluate_spec` parse it with an allow-list
parser that builds the expression from the syntax tree and never
evaluates the text (numbers, `f`, `rho`, `j`, `pi`, `E`, arithmetic,
literal powers, `sqrt` / `exp` / `log`, within size bounds). Every node
is built unevaluated; the parser computes the numbers itself in bounded
floating point, and `evaluate_spec` evaluates the tree with its own
numeric function instead of SymPy's `lambdify`, so a crafted formula
cannot make SymPy expand or simplify it. Printing a parsed formula or
handing it to `lambdify` lets SymPy evaluate parts of it again: show the
formula text of an untrusted file instead. The parser collects like
terms of a sum itself; a real or imaginary part that cancels below
$2^{-100}$ times the largest magnitude of that part among the terms is
zero, the other part is kept. Non-finite numbers of a `BusType` document
(an unknown rms error, a non-finite sample) are written as `null`, so the
file is strict JSON. `load_bustype_json` also checks the rest of the
document: `name` and `system_type` are not objects or lists,
`description` is a string or `null`, and the free-form `metadata` nest at
most `MAX_METADATA_DEPTH` (32) levels; anything else raises `ValueError`.
`fit_quality_summary` reads only finite numbers and strings from the
metadata, so a loaded document never makes it fail.

**Validity.** The exported model inherits the quasi-static band
($f < 1\,\mathrm{kHz}$) and the linearity of the field model; it is meant
for fault-current, EPR and reduction studies, not for impulse or
lightning transients. One `BusType` is the impedance of one terminal
(the input impedance by default); mutual impedances (`BranchType`) are
not exported.

## Transport

| Path | Functions | Needs `groundinsight` |
|---|---|---|
| JSON file | `save_bustype_json`, `load_bustype_json`, `to_bustype_dict` | no |
| Python object | `to_bustype` (live `groundinsight.BusType`), `save_bustype_to_db` | yes (`pip install "groundfield[groundinsight]"`) |
| Check | `evaluate_spec(spec, frequencies, rho)`, `fit_quality_summary(spec)` | no |

## Example

A station earth (ring and rod) in two-layer soil, swept over the
upper-layer resistivity $\rho_1$ (fixed $\rho_2 = 100\,\Omega\,\mathrm{m}$,
$h_1 = 5\,\mathrm{m}$), fitted with the standard form and exported:

```python
import groundfield as gf

def station(rho_1):
    world = gf.create_world(soil=gf.TwoLayerSoil(rho_1=rho_1, rho_2=100.0, h_1=5.0))
    gf.create_electrode(world, "ring", name="station", center=(0.0, 0.0, 0.8), radius=2.0)
    gf.create_electrode(world, "rod", name="rod", position=(2.0, 0.0, 0.8), length=3.0)
    gf.create_conductor(world, name="bond", start="station", end="rod")   # ideal bond
    gf.create_source(world, attached_to="station", magnitude=1.0)
    return world

rhos = [30.0, 100.0, 300.0, 1000.0]
engine = gf.create_engine(frequencies=[50.0, 1000.0])
results = [engine.solve(station(rho)) for rho in rhos]
fit = gf.rho_f_standard_from_results(results, rhos)    # input impedance by default

path = gf.save_bustype_json(
    fit, "station.bustype.json", name="Station", system_type="Substation", voltage_level=20.0
)
spec = gf.load_bustype_json(path)
print(gf.fit_quality_summary(spec))      # ... max_rel=3.91%, quantity=input

Z_formula = gf.evaluate_spec(spec, frequencies=[50.0], rho=100.0)[0]
Z_field = results[1].input_impedance()[0]
assert abs(Z_formula - Z_field) / abs(Z_field) <= fit.max_relative + 1e-9
assert fit.max_relative < 0.05
```

Without inductive conductors the impedance does not depend on
frequency, and the frequency coefficients $k_2 \ldots k_5$ come out at
round-off level. Tutorial T05 shows the export of a frequency-dependent
system, including the vector-fit path with `rho_at_fit`.

## CSV and VTK

The CSV writers (`save_potential_path_csv`, `save_electrode_table_csv`,
`save_cluster_impedances_csv`) write UTF-8, comma-separated files at full
precision with a `#` header that records the package version, the
backend, the frequencies and the solve provenance;
`read_csv_with_provenance(path)` returns the table and the header. A
complex quantity is written as `<symbol>_re`, `<symbol>_im`,
`abs_<symbol>`. `overwrite=False` (default) refuses to replace a file.

`export_geometry_vtk(world, path)` writes the electrodes and conductors
as legacy ASCII `POLYDATA` (cell scalar `role`: 0 electrode,
1 conductor), `export_field_vtk(result, path, extent=..., z=...)` the
surface potential as `STRUCTURED_POINTS` with real and imaginary part.
Coordinates are double precision; $z$ points down, so apply a
*Transform* with scale $(1, 1, -1)$ in ParaView for the usual view.

```python
world = station(100.0)
result = results[1]
csv_path = gf.save_potential_path_csv(
    result, "profile.csv", start=(2.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), distance=30.0, n=31
)
table, header = gf.read_csv_with_provenance(csv_path)
gf.export_geometry_vtk(world, "station.vtk")
gf.export_field_vtk(result, "surface_potential.vtk", extent=(-10.0, 10.0, -10.0, 10.0), n=(41, 41))

assert len(table) == 2 * 31      # 31 points at each of the two frequencies
assert {"frequency_Hz", "phi_re", "phi_im", "abs_phi"} <= set(table.columns)
assert header["backend"] == "image_2layer"
```

## API: groundinsight bridge

::: groundfield.io.groundinsight
    options:
      heading_level: 3

## API: CSV

::: groundfield.io.csv
    options:
      heading_level: 3

## API: VTK

::: groundfield.io.vtk
    options:
      heading_level: 3
