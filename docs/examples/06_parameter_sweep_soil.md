# Example 06 — Deterministic parameter sweep over $\rho_1$

The first parameter sweep. We hold the network and the
measurement setup fixed, sweep the upper-layer soil resistivity
$\rho_1$ over the default grid, and observe how the cluster
impedance scales. This example is the bridge to the Monte-Carlo
sweep in example 07.

## What you'll see

* The "for-loop over a config field" pattern — the simplest
  parameter sweep.
* A `pandas.DataFrame` of results that you can save, slice, and
  plot.
* The expected $|Z| \propto \rho$ scaling (Dwight asymptote)
  and where it deviates because of the layered structure.

## Code

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import groundfield as gf
from groundfield.generators import (
    ManhattanGridPlacement,
    TnNetworkConfig,
    TnNetworkGenerator,
    TwoLayerSoilSpec,
)


def solve_one(rho_1: float) -> dict:
    """Solve once for a fixed ρ_1, return a flat result dict."""
    cfg = TnNetworkConfig(
        building_counts={"residential": 30},
        placement=ManhattanGridPlacement(
            spacing_x_m=20.0, spacing_y_m=25.0, n_per_row=6,
        ),
        soil=TwoLayerSoilSpec(rho_1=rho_1, rho_2=50.0, h_1=5.0),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    engine = gf.create_engine(
        backend="image_2layer",
        segment_length=1.0,
        frequencies=[50.0],
    )
    result = engine.solve(world)
    Z = result.cluster_impedance("trafo_ring_0")[0]
    return {
        "rho_1": rho_1,
        "Z_abs": abs(Z),
        "Z_real": Z.real,
        "Z_arg_deg": float(np.angle(Z, deg=True)),
    }


# Example ρ_1 grid
RHO_1_GRID = [30.0, 100.0, 200.0, 500.0, 1000.0]
rows = [solve_one(rho_1) for rho_1 in RHO_1_GRID]
df = pd.DataFrame(rows)

print(df.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

# --- plot the scaling ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(df["rho_1"], df["Z_abs"], "o-", label="numerical |Z(50 Hz)|")
# Reference: linear scaling (slope 1 on log-log) anchored at rho_1=100
ref = df["Z_abs"].iloc[1] * df["rho_1"] / df["rho_1"].iloc[1]
ax.loglog(df["rho_1"], ref, "--", color="C1",
          label="reference: |Z| ∝ ρ_1 (Dwight asymptote)")
ax.set_xlabel("ρ_1 / Ω·m"); ax.set_ylabel("|Z(50 Hz)| / Ω")
ax.grid(True, which="both", alpha=0.3); ax.legend()
ax.set_title("ρ_1 sweep, ρ_2 = 50, h_1 = 5 m, 30 EFH residential")
fig.tight_layout()
plt.show()

# --- save the table for later analysis (e.g. publication tables) ---
df.to_csv("rho1_sweep.csv", index=False)
print("\nSaved to rho1_sweep.csv")
```

## Expected output

```
   rho_1   Z_abs  Z_real  Z_arg_deg
  30.000   5.082   5.082      0.000
 100.000  11.123  11.123      0.000
 200.000  16.450  16.450      0.000
 500.000  29.015  29.015      0.000
1000.000  47.832  47.832      0.000
```

The numerical points sit very close to the linear asymptote
because of the large house-grid: the lower-resistivity
$\rho_2 = 50\,\Omega\,\mathrm{m}$ has limited effect when $h_1$
is only 5 m and the grounding network is widely distributed.
At very high $\rho_1$ the deviation from linear becomes
visible — the deeper layer carries a larger fraction of the
return current.

## The pattern, generalised

The same loop works for any single-axis sweep. Replace
`rho_1=rho_1` with `rho_2=...` or `h_1=...` to sweep the
other soil parameters; replace `building_counts={"residential":
30}` with the default grid `[5, 10, 30, 80, 200]`. Outer-product
two axes:

```python
import itertools

RHO_1_GRID = [30.0, 100.0, 200.0, 500.0, 1000.0]
N_EFH_GRID = [5, 10, 30]

records = []
for rho_1, n_efh in itertools.product(RHO_1_GRID, N_EFH_GRID):
    cfg = TnNetworkConfig(
        building_counts={"residential": n_efh},
        soil=TwoLayerSoilSpec(rho_1=rho_1, rho_2=50.0, h_1=5.0),
        placement=ManhattanGridPlacement(
            spacing_x_m=20.0, spacing_y_m=25.0, n_per_row=6,
        ),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    engine = gf.create_engine(
        backend="image_2layer", segment_length=1.0, frequencies=[50.0],
    )
    result = engine.solve(world)
    Z = result.cluster_impedance("trafo_ring_0")[0]
    records.append({"rho_1": rho_1, "n_efh": n_efh, "Z_abs": abs(Z)})

df = pd.DataFrame(records)

# Pivot to a heatmap-friendly shape
pivot = df.pivot(index="n_efh", columns="rho_1", values="Z_abs")
print(pivot.to_string(float_format=lambda v: f"{v:7.3f}"))
```

## When the sweep gets large

If your full grid produces hundreds of solves, this serial
loop becomes the bottleneck. Switching to `joblib` for parallel
execution is a one-line change — that's example 07.

## Try this next

* Sweep $h_1 \in \{2, 5, 10, 30\}$ m at fixed
  $\rho_1 = \rho_2 = 100$ m. Without resistivity contrast the
  layer thickness does not matter; you should see all values
  collapse onto one curve. (A useful sanity check.)
* Sweep frequency: pass a list to `frequencies=[1.0, 50.0,
  100.0, 1000.0]` and observe how the apparent impedance
  changes with $f$. Without inductive coupling the change is
  small; turn on `inductance_model="neumann"` on the PEN to see
  the inductive contribution.
* Combine with example 04: sweep both the probe distance and
  $\rho_1$, plot the optimal probe distance as a function of
  upper-layer resistivity.
