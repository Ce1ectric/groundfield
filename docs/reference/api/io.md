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
  only. The soil it belongs to must be given as `rho_at_fit`.

**The `rho` of layered soil.** A `Bus` carries one resistivity. For
homogeneous soil it is $\rho$; for layered soil it is the effective
resistivity of the AP1 catalogue,

$$
\rho_\text{eff} = \rho_1^{\,1-w}\,\rho_2^{\,w}, \qquad
w = w_0\,(h_1/10\ \mathrm{m})^{-\beta},
$$

with the weights $(w_0, \beta)$ of the electrode class: $w_0 = 0.13$,
$\beta = 0.8$ of the LV networks by default, or a preset of the AP2
hand-over such as `"station"` (0.06, 1.3) or `"lv_group_2_p10"`
(`gf.effective_resistivity(soil, preset=...)`, see
[Soil](soil.md#effective-resistivity)). It is the value the AP2 network
model gives a bus. A vector fit therefore passes
`rho_at_fit=gf.effective_resistivity(soil, preset=...)`, and a standard
form is fitted over the effective resistivities of the solved soils,
`gf.rho_f_standard_from_results(results, [gf.effective_resistivity(r.soil, preset=...) for r in results])`;
both kinds of `BusType` then pair with a `Bus` whose resistivity uses the
same preset. Every document records the convention as
`metadata["rho_definition"] = "effective_resistivity"` (documents of
0.15.0 have no such entry) and the preset you name with `rho_preset=...`
(a key of `gf.soil.EFFECTIVE_RESISTIVITY_PRESETS`) as
`metadata["rho_preset"]`, `None` when none is named (user decision Q39;
documents of earlier versions load with `None`, `spec.rho_preset`). The
exporter cannot check the number it is given against the preset; the
paths that serve a bus can: `evaluate_spec(spec, f, rho, rho_preset=...)`
and `bustype_from_spec(spec, rho_preset=...)` (the live groundinsight
`BusType` of a loaded document) take the preset of the bus and emit
`RhoPresetMismatchWarning` when the document names another one — a preset
holds for its electrode class only (the AP1 station paired with
`"lv_group_4_p40"` misses the field solution by 39.9 % rms). The live
`BusType` of `to_bustype`, `save_bustype_to_db` and `bustype_from_spec`
has no metadata; with a preset its description ends with the line
`groundfield rho_preset: <name>` (a preset line in the given description
is replaced), which the groundinsight database keeps, and
`gf.bustype_rho_preset(bus_type)` reads it back, also from a `BusType`
loaded with `gi.load_bustypes_from_db()` (audit W3B-06; `None` without a
preset line). A blank `metadata["rho_preset"]` is refused on load. Describe
the layers, and explicit weights $w_0$, $\beta$, in `soil_summary`.

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
$h_1 = 5\,\mathrm{m}$), fitted with the standard form over the effective
resistivities of the four soils (weights of the `"station"` preset) and
exported:

```python
import groundfield as gf

def station(rho_1):
    world = gf.create_world(soil=gf.TwoLayerSoil(rho_1=rho_1, rho_2=100.0, h_1=5.0))
    gf.create_electrode(world, "ring", name="station", center=(0.0, 0.0, 0.8), radius=2.0)
    gf.create_electrode(world, "rod", name="rod", position=(2.0, 0.0, 0.8), length=3.0)
    gf.create_conductor(world, name="bond", start="station", end="rod")   # ideal bond
    gf.create_source(world, attached_to="station", magnitude=1.0)
    return world

engine = gf.create_engine(frequencies=[50.0, 1000.0])
results = [engine.solve(station(rho_1)) for rho_1 in (30.0, 100.0, 300.0, 1000.0)]
rhos = [gf.effective_resistivity(r.soil, preset="station") for r in results]  # 35.8 ... 711.6 Ohm m
fit = gf.rho_f_standard_from_results(results, rhos, max_relative_tol=0.10)  # input impedance

path = gf.save_bustype_json(
    fit, "station.bustype.json", name="Station", system_type="Substation", voltage_level=20.0,
    soil_summary="TwoLayerSoil(rho_1 sweep, rho_2=100, h_1=5)", rho_preset="station",
)
spec = gf.load_bustype_json(path)
print(gf.fit_quality_summary(spec))      # ... max_rel=7.48%, quantity=input, rho_preset=station

Z_formula = gf.evaluate_spec(spec, frequencies=[50.0], rho=rhos[3], rho_preset="station")[0]
Z_field = results[3].input_impedance()[0]
assert abs(Z_formula - Z_field) / abs(Z_field) <= fit.max_relative + 1e-9
assert spec.metadata["rho_definition"] == "effective_resistivity"
assert spec.rho_preset == "station"
assert abs(rhos[3] - 711.64) < 0.01 and fit.max_relative < 0.10
```

A bus whose resistivity comes from another preset is reported:

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    gf.evaluate_spec(spec, [50.0], gf.effective_resistivity(results[3].soil), rho_preset="lv_network")
assert any(issubclass(w.category, gf.RhoPresetMismatchWarning) for w in caught)
```

<!-- doctest-check
assert abs(fit.max_relative - 0.0748) < 5e-4
assert rhos[1] == 100.0     # rho_1 = rho_2: homogeneous, exact
lv_network = gf.rho_f_standard_from_results(
    results, [gf.effective_resistivity(r.soil) for r in results], max_relative_tol=0.15
)
assert abs(lv_network.max_relative - 0.1341) < 5e-4
-->

The station is compact and lies in the upper layer, so it sees less of
the lower layer than an LV network: the standard form fits it to 7.5 %
over $\rho_\text{eff}$ with the station weights, against 13.4 % with the
default weights of the LV networks. Over $\rho_1$ it would fit to 3.9 %,
because at fixed $\rho_2$ the constant $k_0$ absorbs the lower layer, but
the bus the `BusType` is paired with carries $\rho_\text{eff}$, here with
the station preset. Check `max_relative`, and see
[Soil](soil.md#effective-resistivity) for the presets and the validity
of $\rho_\text{eff}$. Without inductive conductors the impedance does not
depend on frequency, and the frequency coefficients $k_2 \ldots k_5$ come
out at round-off level. Tutorial T05 shows the export of a
frequency-dependent system, including the vector-fit path with
`rho_at_fit`.

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
