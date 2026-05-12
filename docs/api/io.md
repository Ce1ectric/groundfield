# IO

The ``io`` subpackage is the boundary between ``groundfield`` and the
rest of the software family. It carries no numerical core; every
function here is a translator between an in-memory ``groundfield``
artefact (a fit, a result) and an external representation (a JSON
file, a sister-project Pydantic model).

## Mathematical and physical context

The dissertation pipeline reads

$$
\text{PDE / field model} \;\longrightarrow\; \text{reduced } \rho\text{-}f \text{ model}
\;\longrightarrow\; \text{measurement-anchored network model}.
$$

The third step is owned by ``groundinsight`` (network-grade fault
analysis on the reduced equivalent model). The interface that closes
the loop is **``BusType.impedance_formula``** — a SymPy-compatible
string in the two free symbols ``f`` (frequency in Hz) and ``rho``
(``Bus.specific_earth_resistance`` in $\Omega\,\mathrm{m}$).

``io.groundinsight`` produces exactly that string from either of the
two ``rho-f`` fits supported by ``groundfield``:

- **Standard form** (``RhoFStandardFit``) — the dissertation's
  five-coefficient ansatz

  $$
  Z(\rho, f) \;=\; k_1 \rho \;+\; (k_2 + j k_3)\,f \;+\; (k_4 + j k_5)\,f\,\rho,
  $$

  fitted across a $\rho$-sweep. Already in the canonical
  ``(rho, f)`` symbol set.

- **Vector fit** (``VectorFitResult``) — a Gustavsen/Semlyen rational
  $Z(s)$ in the Laplace variable $s = j\,2\pi f$ at fixed soil. The
  exporter performs the symbolic substitution

  $$
  s \;\longrightarrow\; j\,2\pi f
  $$

  and renders the formula in $f$ alone. The underlying soil
  resistivity is recorded in the metadata block (``rho_at_fit_Ohm_m``)
  but does not appear in the formula — a vector fit is by
  construction bound to a specific soil.

## Validity

- **Frequency range** $f \le 1\,\mathrm{kHz}$ — both fit families
  inherit the dissertation's quasi-static assumption. The exported
  ``BusType`` is meaningful only inside this band; ``groundinsight``
  evaluates the formula for whatever frequencies are stored on its
  ``Network``, so the user is responsible for staying within the band.
- **Linearity** — the underlying field model is solved on the
  $\nabla \cdot (\sigma\,\nabla\Phi) = 0$ regime; saturation effects in
  electrodes (e.g. ionisation under impulse currents) are *not*
  modelled. Exports are therefore appropriate for protection,
  reduction-factor and EPR studies, not for transient
  surge/lightning models.
- **Multi-port reduction** — the current API exports a *single*
  ``BusType``, i.e. the driving-point impedance of one cluster.
  ``BranchType`` (mutual impedance) is out of scope for ADR-0008 and
  reserved for a follow-up.

## Transport

Two equally supported paths, see
[ADR-0008](../adr/0008-groundinsight-bridge.md) for the design
rationale and the exact JSON schema (v1):

| Path        | Function                                                | ``groundinsight`` install required? |
|-------------|---------------------------------------------------------|-------------------------------------|
| JSON file   | ``save_bustype_json`` / ``load_bustype_json``           | no                                  |
| Python API  | ``to_bustype`` / ``save_bustype_to_db``                 | yes (extras group ``[groundinsight]``) |
| Hybrid      | ``to_bustype_dict`` (JSON-ready dict; build & inspect)  | no                                  |

## CSV exports — `groundfield.io.csv`

Three convenience writers that turn a :class:`FieldResult`
(and optionally its companion :class:`World`) into
machine-readable, tool-agnostic CSV files. They wrap the existing
``postprocess`` helpers — no new science here, just a clean
disk format for sharing AP1 results across notebooks,
spreadsheets, and downstream pipelines.

```python
import groundfield as gf

# 1. Sample the potential along a path and dump (s, x, y, z, freq, phi).
gf.save_potential_path_csv(
    result, "out/phi_radial.csv",
    start=(1.0, 0.0, 0.0), direction=(1, 0, 0),
    distance=30.0, n=200,
)
# 2. Dump the per-electrode summary.
gf.save_electrode_table_csv(result, "out/electrodes.csv", world=world)
# 3. Dump the per-cluster summary (members are flattened to a string).
gf.save_cluster_impedances_csv(result, "out/clusters.csv")
```

All writers use UTF-8, comma-separated, with a header row;
floating-point values are written at full precision (`%.17g`)
so files round-trip without loss of accuracy.

### API reference — CSV

::: groundfield.io.csv

## VTK exports — `groundfield.io.vtk`

Legacy ASCII VTK writers (no `pyvista` / `vtk` Python bindings
required). Two file flavours:

| Function                  | VTK dataset           | Use case                                          |
|---------------------------|-----------------------|---------------------------------------------------|
| `export_geometry_vtk`     | `POLYDATA`            | inspect electrodes + conductors in ParaView       |
| `export_field_vtk`        | `STRUCTURED_POINTS`   | render the surface potential as a heatmap / iso  |

```python
gf.export_geometry_vtk(world, "out/world.vtk")
gf.export_field_vtk(
    result, "out/phi_surface.vtk",
    extent=(-30, 30, -20, 20), z=0.0, n=(200, 150),
)
```

The geometry export carries an integer `role` cell-data scalar
(0 = electrode, 1 = conductor) so colour-by-role works in
ParaView without further configuration. The field export
includes `potential_re` and `potential_im` so above-DC AP1
studies remain visible.

### API reference — VTK

::: groundfield.io.vtk

## Example

### JSON path (no `groundinsight` install needed)

```python
import numpy as np
import groundfield as gf
from groundfield.postprocess.vector_fitting import rho_f_from_field_result
from groundfield.io.groundinsight import (
    save_bustype_json, load_bustype_json, to_bustype_dict,
)

# 1. Run a field study and produce a vector fit.
result = ...  # FieldResult from world.solve(engine)
fit = rho_f_from_field_result(result, electrode_name="g1", n_poles=3)

# 2. Export to a versioned JSON file.
save_bustype_json(
    fit,
    path="bus_type_substation.json",
    name="SubstationBus",
    description="Substation grounding grid (vector-fitted, AP1 ref).",
    system_type="Substation",
    voltage_level=20,
)

# 3. Load and inspect later (no groundinsight needed).
spec = load_bustype_json("bus_type_substation.json")
print(spec.name, spec.metadata["fit_method"])
```

### Python API path (live `groundinsight.BusType`)

```python
from groundfield.io.groundinsight import to_bustype, save_bustype_to_db

# Requires: pip install groundfield[groundinsight]
bus_type = to_bustype(
    fit,
    name="SubstationBus",
    description="Substation grounding grid (vector-fitted, AP1 ref).",
    system_type="Substation",
    voltage_level=20,
)
# bus_type is a live groundinsight.models.core_models.BusType instance
# and can be wired straight into a Network:
import groundinsight as gi
net = gi.create_network(name="AP1", frequencies=[50.0, 250.0])
gi.create_bus(name="bus_substation", type=bus_type, network=net,
              specific_earth_resistance=100.0)

# Or persist to the groundinsight SQLite database in one call:
save_bustype_to_db(fit, db_path="ap1.db",
                   name="SubstationBus", system_type="Substation",
                   voltage_level=20)
```

`evaluate_spec(spec, frequencies, rho)` re-evaluates an exported
formula at arbitrary $(f, \rho)$ points without round-tripping through
``groundinsight``.

## API reference

::: groundfield.io.groundinsight
