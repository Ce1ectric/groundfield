# Example 07 — Monte-Carlo sweep with `joblib`

The full AP1 study has thousands of configurations. The
deterministic sweep of example 06 is fine for a few dozen
solves; for thousands you want **parallel execution** and
**stochastic distributions** on the parameters that you don't
know exactly. This example shows both, ending with statistical
bands (median plus inter-quartile range) on the apparent
substation impedance.

## What you'll see

* The full Monte-Carlo pattern: distributions in the config →
  per-realisation `cfg.sample(rng)` → solve → store result.
* `joblib.Parallel` for multi-core execution.
* Reproducible seeding so a stopped run can be resumed.
* Persistent storage in Parquet (small, fast, survives kernel
  restarts).
* Aggregation into median + 25 / 75 % quantiles per soil
  resistivity.

## Code

```python
"""ap1_monte_carlo.py — runnable as a script.

   Usage:
       python ap1_monte_carlo.py
   It writes ap1_mc.parquet next to the script.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import groundfield as gf
from groundfield.generators import (
    Categorical,
    Discrete,
    LogNormal,
    ManhattanGridPlacement,
    Normal,
    PenConfig,
    TnNetworkConfig,
    TnNetworkGenerator,
    TwoLayerSoilSpec,
)


# ---------------------------------------------------------------------
# 1. Stochastic config — every numeric field can be a Distribution
# ---------------------------------------------------------------------


def make_cfg(rho_1: float, n_efh: int) -> TnNetworkConfig:
    """Stochastic config for one (rho_1, n_efh) cell of the grid.

    Two of the three soil parameters are random, plus a small position
    jitter for the houses, plus a random house-electrode kind drawn
    per realisation. The remaining axes (rho_1, n_efh) are scanned
    deterministically by the outer grid.
    """
    return TnNetworkConfig(
        building_counts={"residential": n_efh},
        placement=ManhattanGridPlacement(
            spacing_x_m=20.0, spacing_y_m=25.0, n_per_row=6,
            jitter_m=Normal(mean=0.0, std=2.0,
                            truncate_low=-3.5, truncate_high=3.5),
        ),
        soil=TwoLayerSoilSpec(
            rho_1=rho_1,
            rho_2=LogNormal.from_moments(mean=50.0, std=15.0),
            h_1=Discrete(values=[5.0, 10.0, 30.0]),
        ),
        pen=PenConfig(inductance_model=None),  # AP1 production default
    )


# ---------------------------------------------------------------------
# 2. Solve one realisation
# ---------------------------------------------------------------------


def solve_one(rho_1: float, n_efh: int, seed: int) -> dict:
    """One Monte-Carlo realisation: sample distributions with ``seed``,
    build, solve, return a flat dict."""
    cfg = make_cfg(rho_1, n_efh)
    world, resolved = TnNetworkGenerator(cfg, seed=seed).sample_world()
    engine = gf.create_engine(
        backend="image_2layer",
        segment_length=1.0,
        frequencies=[50.0],
        earth_inductive_model="carson_series",
    )
    result = engine.solve(world)
    Z = result.cluster_impedance("trafo_ring_0")[0]
    return {
        # control variables
        "rho_1": rho_1,
        "n_efh": n_efh,
        "seed": seed,
        # drawn values (so we can post-hoc check the distributions)
        "rho_2_drawn": float(resolved.soil.rho_2),
        "h_1_drawn": float(resolved.soil.h_1),
        # observables
        "Z_abs": abs(Z),
        "Z_real": Z.real,
        "Z_imag": Z.imag,
    }


# ---------------------------------------------------------------------
# 3. Build the parameter × realisation grid
# ---------------------------------------------------------------------


RHO_1_GRID = [30.0, 100.0, 200.0, 500.0, 1000.0]
N_EFH_GRID = [10, 30]            # extend to [5, 10, 30, 80, 200] for full AP1
N_REALISATIONS = 30               # K — try 10 first to confirm timing

jobs = [
    (rho_1, n_efh, seed)
    for rho_1, n_efh in itertools.product(RHO_1_GRID, N_EFH_GRID)
    for seed in range(N_REALISATIONS)
]
print(f"Total jobs: {len(jobs)}")


# ---------------------------------------------------------------------
# 4. Run — single-core for one job to estimate the cost first
# ---------------------------------------------------------------------


t0 = time.perf_counter()
sample = solve_one(*jobs[0])
t_one = time.perf_counter() - t0
print(f"One realisation took {t_one:.1f} s")
print(f"Total expected single-core: {t_one * len(jobs) / 60:.1f} min")
print(f"Estimated 12-core wall: {t_one * len(jobs) / 12 / 60:.1f} min")


# ---------------------------------------------------------------------
# 5. Run — parallel
# ---------------------------------------------------------------------


t0 = time.perf_counter()
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(solve_one)(rho_1, n_efh, seed) for (rho_1, n_efh, seed) in jobs
)
t_total = time.perf_counter() - t0
print(f"Total wall-clock: {t_total/60:.1f} min")

df = pd.DataFrame(results)
df.to_parquet("ap1_mc.parquet")
print(f"Saved {len(df)} rows to ap1_mc.parquet")


# ---------------------------------------------------------------------
# 6. Aggregate and plot
# ---------------------------------------------------------------------


df = pd.read_parquet("ap1_mc.parquet")  # works as a re-entry point too

agg = (df.groupby(["rho_1", "n_efh"])["Z_abs"]
         .agg(["median", lambda s: s.quantile(0.25),
               lambda s: s.quantile(0.75), "count"])
         .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
         .reset_index())
print(agg.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

fig, ax = plt.subplots(figsize=(8, 5))
for n_efh, sub in agg.groupby("n_efh"):
    line, = ax.plot(sub["rho_1"], sub["median"], "o-", label=f"{n_efh} EFH")
    ax.fill_between(sub["rho_1"], sub["q25"], sub["q75"],
                    alpha=0.2, color=line.get_color())
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("ρ_1 / Ω·m"); ax.set_ylabel("|Z(50 Hz)| / Ω")
ax.grid(True, which="both", alpha=0.3); ax.legend()
ax.set_title(f"AP1 Monte Carlo — median and IQR ({N_REALISATIONS} draws each)")
fig.tight_layout()
plt.show()
```

## What you should observe

* **Linear in log-log** — the median curves are essentially
  straight lines with slope ~1 (Dwight asymptote
  $|Z| \propto \rho_1$).
* **The IQR shrinks with $n_\text{EFH}$** — more parallel
  electrodes means the cluster impedance is less sensitive to
  the soil-parameter randomness.
* **The lower curve sits below the upper curve** — adding houses
  drops the cluster impedance via parallel grounding.

## How to read this for the dissertation

The IQR widths quantify the **uncertainty band** on the
substation impedance for a given number of buildings and a given
upper-layer resistivity, taking realistic ranges of
$\rho_2$ and $h_1$ into account. That is exactly the band you
need for the AP1 reduction-factor and reliability statements.

## Tips and pitfalls

### Reproducibility

Each call to `solve_one(rho_1, n_efh, seed)` builds the world
deterministically from `seed` — the same `(rho_1, n_efh, seed)`
triple always gives the same result. To resume a stopped run,
load the existing Parquet, find which `(rho_1, n_efh, seed)`
triples are missing, and re-run only those:

```python
df_done = pd.read_parquet("ap1_mc.parquet")
done = set(df_done[["rho_1", "n_efh", "seed"]].itertuples(index=False, name=None))
remaining = [job for job in jobs if job not in done]
```

### Estimate before launching

The script prints the expected wall-clock for the full Monte
Carlo *before* the heavy run. For 200 EFH × 30 realisations the
single-core estimate may be days — don't accidentally launch
that without `n_jobs=-1` and a comfortable laptop.

### Watch out for shared state

`joblib.Parallel` serialises the function and its arguments via
`cloudpickle`. The `solve_one` function above is **stateless**:
every call rebuilds the world from scratch. That is the safe
pattern. Don't try to share a pre-built `World` across workers
— pickling a `World` is not supported.

### When to add `inductance_model="neumann"`

If your study question is the **galvanic** effect of soil
heterogeneity, leave PEN-Neumann off. It costs ~3× per solve
and changes $|Z|$ by ~1 % at 50 Hz. If you study **inductive
distortion of the measurement**, see example 05 — and budget
for the increased solve time when projecting Monte-Carlo
wall-clocks.

## Try this next

* Extend `N_EFH_GRID` to `[5, 10, 30, 80, 200]` and increase
  `N_REALISATIONS` to 50. This is the AP1 production sweep.
  Plan for an overnight run on 12 cores for the first three
  cells; the 200-EFH cell is the multi-day case (see the
  [performance guide](../performance.md) for projections).
* Combine the Monte Carlo with the measurement setup of
  example 05 — you'll then have a Monte-Carlo distribution of
  the *measurement error* under realistic measurement
  conditions, which is the AP1 Analysis-2 deliverable.
* Continue with example 08 to see how the same pipeline feeds
  back into `groundinsight` for fault-current studies.
