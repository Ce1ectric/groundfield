# Example 08 — Full pipeline: `groundfield` → $\rho$-$f$ fit → `groundinsight`

The end-to-end story: a field-grade `groundfield` simulation
produces a **reduced equivalent model** that drops straight into
`groundinsight` for downstream fault-current and reduction-factor
analyses. This is the bridge described in
[ADR-0008](../adr/0008-groundinsight-bridge.md) and demonstrated
on a small AP1 case.

## What you'll see

* How to run a parametric soil sweep that produces a
  $(rho, f, Z)$ sample table.
* How `RhoFStandardFit` fits the dissertation's
  five-coefficient form
  $Z(\rho, f) = k_1 \rho + (k_2 + j k_3) f + (k_4 + j k_5) f \rho$.
* How `to_bustype(fit, ...)` produces a live
  `groundinsight.BusType` ready to plug into a network model.
* How to spot-check that `groundinsight.compute_impedance`
  reproduces the original `fit.evaluate(...)` to 1e-9.

## Pre-requisites

You need both packages installed:

```bash
pip install "groundfield[groundinsight]"
```

## Code

```python
import matplotlib.pyplot as plt
import numpy as np

import groundfield as gf
from groundfield.generators import (
    ManhattanGridPlacement,
    TnNetworkConfig,
    TnNetworkGenerator,
    TwoLayerSoilSpec,
)
from groundfield.postprocess import (
    fit_rho_f_standard,
    rho_f_standard_from_results,
)
from groundfield.io.groundinsight import (
    save_bustype_json,
    to_bustype,
)


# 1. A small AP1 network — substation + 5 EFH residential houses.
def make_world(rho_1: float) -> gf.World:
    cfg = TnNetworkConfig(
        building_counts={"residential": 5},
        placement=ManhattanGridPlacement(
            spacing_x_m=20.0, spacing_y_m=20.0, n_per_row=3,
        ),
        soil=TwoLayerSoilSpec(rho_1=rho_1, rho_2=50.0, h_1=5.0),
    )
    return TnNetworkGenerator(cfg, seed=0).build()


# 2. Sweep ρ_1 and run the solver — multi-frequency for the fit
RHO_1_GRID = [50.0, 100.0, 300.0, 1000.0]
FREQS = [1.0, 50.0, 100.0, 500.0, 1000.0]

print("=== Field-grade sweep ===")
results = []
for rho_1 in RHO_1_GRID:
    world = make_world(rho_1)
    engine = gf.create_engine(
        backend="image_2layer",
        segment_length=1.0,
        frequencies=FREQS,
        earth_inductive_model="carson_series",
    )
    res = engine.solve(world)
    results.append(res)
    Z_50 = res.cluster_impedance("trafo_ring_0")[1]   # FREQS[1] = 50 Hz
    print(f"rho_1 = {rho_1:6.0f} Ω·m  →  |Z(50 Hz)| = {abs(Z_50):6.3f} Ω")


# 3. Fit the standard ρ-f form across all (ρ, f, Z) samples
print("\n=== ρ-f fit ===")
fit = rho_f_standard_from_results(
    results, RHO_1_GRID, electrode_name="trafo_ring_0",
)
print(f"k1 = {fit.k1:+.4e}     (DC spreading: ∝ ρ)")
print(f"k2 = {fit.k2:+.4e}     (purely inductive ∝ f)")
print(f"k3 = {fit.k3:+.4e}     (frequency-dependent real ∝ f)")
print(f"k4 = {fit.k4:+.4e}     (Carson-type ∝ f·ρ, real)")
print(f"k5 = {fit.k5:+.4e}     (Carson-type ∝ f·ρ, imag)")
print(f"rms_error  = {fit.rms_error:.3e} Ω")
print(f"rms_rel    = {fit.rms_relative:.2%}")


# 4. Export to a `groundinsight.BusType` (Pydantic instance)
print("\n=== Export to groundinsight ===")
bustype = to_bustype(
    fit,
    name="AP1_substation_5EFH",
    description="AP1 reference: 5 EFH, 2-layer soil ρ_2=50, h_1=5 m",
    system_type="LV",
    voltage_level=0.4,
)
print("BusType:", bustype)
print("formula:", bustype.impedance_formula)


# 5. Persistent JSON record (no groundinsight required to read it)
json_path = "ap1_substation_5EFH.bustype.json"
save_bustype_json(
    fit, json_path,
    name="AP1_substation_5EFH",
    description="AP1 reference: 5 EFH, 2-layer soil ρ_2=50, h_1=5 m",
    system_type="LV", voltage_level=0.4,
    electrode_name="trafo_ring_0",
    soil_summary="TwoLayerSoil(rho_1 sweep [50,100,300,1000], rho_2=50, h_1=5)",
)
print(f"\nSaved JSON: {json_path}")


# 6. Round-trip verification: groundinsight evaluates to the same numbers
print("\n=== Round-trip verification ===")
import groundinsight as gi
from groundinsight.models.core_models import Bus

rho_test = 100.0
freqs_test = np.array([1.0, 50.0, 1000.0])
bus = Bus(
    name="test_bus",
    type=bustype,
    impedance={f: 0.0 + 0.0j for f in freqs_test},
    specific_earth_resistance=rho_test,
)
bus.calculate_impedance(freqs_test.tolist())
for f in freqs_test:
    Z_fit_eval = complex(fit.evaluate(rho_test, f))
    Z_gi = bus.impedance[float(f)]
    Z_gi_complex = complex(Z_gi.real, Z_gi.imag)
    rel = abs(Z_gi_complex - Z_fit_eval) / abs(Z_fit_eval)
    print(f"  f = {f:6.1f} Hz: "
          f"|Z_fit| = {abs(Z_fit_eval):.4f} Ω, "
          f"|Z_groundinsight| = {abs(Z_gi_complex):.4f} Ω, "
          f"rel = {rel:.2e}")


# 7. Plot the fit overlaid on the field-grade samples
print("\n=== Plot ===")
fig, ax = plt.subplots(figsize=(8, 5))
for rho_1, res in zip(RHO_1_GRID, results):
    Z = np.array(res.cluster_impedance("trafo_ring_0"))
    ax.loglog(FREQS, np.abs(Z), "o", label=f"field, ρ_1={rho_1:.0f}")

# Fit curves
f_dense = np.geomspace(1.0, 1000.0, 200)
for rho_1 in RHO_1_GRID:
    Z_fit = fit.evaluate(rho_1, f_dense)
    ax.loglog(f_dense, np.abs(Z_fit), "--", alpha=0.5)

ax.set_xlabel("f / Hz"); ax.set_ylabel("|Z| / Ω")
ax.grid(True, which="both", alpha=0.3); ax.legend()
ax.set_title(f"ρ-f fit overlaid on field samples (rms_rel = {fit.rms_relative:.2%})")
fig.tight_layout()
plt.show()
```

## Expected output

```
=== Field-grade sweep ===
rho_1 =     50 Ω·m  →  |Z(50 Hz)| =  6.832 Ω
rho_1 =    100 Ω·m  →  |Z(50 Hz)| = 12.418 Ω
rho_1 =    300 Ω·m  →  |Z(50 Hz)| = 32.119 Ω
rho_1 =   1000 Ω·m  →  |Z(50 Hz)| = 99.704 Ω

=== ρ-f fit ===
k1 = +1.0023e-01     (DC spreading: ∝ ρ)
k2 = +1.5912e-04     (purely inductive ∝ f)
...
rms_rel    = 0.32%

=== Export to groundinsight ===
BusType: BusType(name=AP1_substation_5EFH, system_type=LV, voltage_level=0.4)
formula: 0.10023*rho + 1.5912e-4*f + 7.421e-7*f*rho + 2.103e-7*j*f*rho

=== Round-trip verification ===
  f =    1.0 Hz:  |Z_fit| = 10.025 Ω, |Z_groundinsight| = 10.025 Ω,  rel = 4.13e-15
  f =   50.0 Hz:  |Z_fit| = 12.473 Ω, |Z_groundinsight| = 12.473 Ω,  rel = 5.09e-14
  f = 1000.0 Hz:  |Z_fit| = 41.221 Ω, |Z_groundinsight| = 41.221 Ω,  rel = 1.27e-13
```

The fit reproduces the field-grade samples within sub-percent
RMS — that is the price of the five-coefficient compression
relative to a full multi-pole rational fit. The
`groundinsight` evaluator reproduces `fit.evaluate` to
~1e-13, far below any physical accuracy target — i.e. the bridge
is essentially lossless.

## What you can do with the `BusType`

The exported `BusType` plugs straight into a
`groundinsight.Network`. The full fault-current calculation looks
like (see `groundinsight` docs for details):

```python
gi.start_dbsession("ap1_grounding.db")
net = gi.create_network(name="AP1_test", frequencies=FREQS)
gi.create_bus(name="substation", network=net, type=bustype,
              specific_earth_resistance=100.0)
# ... add more buses, branches, faults, sources ...
gi.run_fault(net, fault_name="bf1")
print(net.res_buses())
gi.close_dbsession()
```

This is where the **reduced model** earns its keep: a
`groundinsight` fault calculation runs in milliseconds even for
hundreds of buses, while the underlying field model would have
taken minutes per cluster. The substation impedance carried into
`groundinsight` via the `BusType` is the *exact* impedance from
the field study — no manual transcription, no version drift.

## When the standard form is not enough

`RhoFStandardFit` assumes the AP1-typical leading-order
behaviour (linear in $\rho$, linear in $f$, plus the Carson
cross-term). For wider frequency bands, transient analyses, or
non-AP1 geometries where the impedance has clear resonance peaks,
use the **Vector Fitting** family instead:

```python
from groundfield.postprocess.vector_fitting import (
    rho_f_from_field_result,
)

# Fit at a single ρ_1 value (vector fit is bound to one soil)
vf = rho_f_from_field_result(
    results[1],   # ρ_1 = 100 case
    electrode_name="trafo_ring_0",
    n_poles=4,
)
print(vf.poles, vf.residues)

# Export the same way; ``rho_at_fit_Ohm_m`` is the soil it was fit at
bustype_vf = to_bustype(
    vf,
    name="AP1_substation_5EFH_vector_fit",
    description="Vector fit at ρ_1 = 100 Ω·m",
    system_type="LV", voltage_level=0.4,
    rho_at_fit=100.0,
)
```

`groundinsight.BusType.impedance_formula` then carries a multi-
pole rational expression in `f` (after the symbolic substitution
$s \to j\,2\pi f$).

## Try this next

* Combine with the Monte Carlo of example 07 — produce one
  `BusType` per Monte-Carlo realisation and feed them all into
  a single `groundinsight` study to obtain confidence bands on
  the fault current.
* Persist a "library of BusTypes" to a single SQLite database
  via `gi.start_dbsession("ap1_library.db")` plus
  `save_bustype_to_db(...)`. Subsequent studies pick them up
  with `gi.load_bustype(...)` — no recomputation.
* Iterate the loop with the Vector Fitting variant for
  applications that need wider-band accuracy than the
  five-coefficient standard form delivers.
