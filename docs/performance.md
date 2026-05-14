# Performance and scaling

This page collects the empirical performance characteristics of
the `image_2layer` solver pipeline (with optional Carson /
Sommerfeld earth-return correction and Neumann mutual inductance)
and gives concrete recommendations for parameter studies and
Monte-Carlo sweeps. The numbers come from
[`notebooks/21_performance_tuning.ipynb`](https://github.com/Ce1ectric/groundfield/blob/main/notebooks/21_performance_tuning.ipynb)
on a typical laptop and are reproducible end-to-end.

## TL;DR

For a typical study, the three configuration choices that
dominate wall-clock time are:

1. **`earth_inductive_model="carson_series"` instead of `"sommerfeld"`.**
   On the quasi-static frequency band ($f \le 1\,\mathrm{kHz}$) and with
   long parallel measurement / PEN structures both models give
   identical impedances; Carson is up to **~1200×** faster. Use
   `"sommerfeld"` only for spot-checks where the parallel-wires
   assumption is invalid (very short, strongly tilted leads).
2. **`segment_length ≤ 1.0 m`.** Coarser discretisation produces
   numerically wrong impedances (50 % error and more on
   $|Z|$ — this is *not* a small effect). The runtime difference
   between `2 m` and `1 m` is barely a factor of 2; do not trade
   accuracy here.
3. **`PenConfig(inductance_model=None)` unless you specifically
   study induced effects on the PEN.** PEN-Neumann adds ~1 % to
   $|Z|$ at 50 Hz for a 3× cost. Switch it on only when the
   inductive coupling between the PEN trunk and the measurement
   leads is the question under study.

Every other knob is secondary.

## Empirical characteristics

The numbers below come from a 5-EFH baseline with
`MeasurementSetupConfig`, two-layer soil
($\rho_1 = 100$, $\rho_2 = 50$, $h_1 = 5$ m), 50 Hz, on a single
core. Three runs are taken, the median reported.

| Configuration | t / s | Δ vs. baseline |
|---|---|---|
| `earth_model="perfect_mirror"`, no neumann anywhere | 0.06 | ×1 |
| `earth_model="carson_series"`, only leads neumann | 0.13 | ×2.2 |
| `earth_model="carson_series"`, all conductors neumann | 0.37 | ×6.2 |
| `earth_model="sommerfeld"`, all conductors neumann | 69.7 | ×1160 |

Discretisation cost (carson, no PEN-neumann):

| `segment_length` | n_segments | t / s | $\|Z(50\,\text{Hz})\|$ |
|---|---|---|---|
| 4.0 m | 69 | 0.05 | 6.04 Ω (50 % off — **wrong**) |
| 2.0 m | 114 | 0.06 | 6.37 Ω (50 % off — **wrong**) |
| 1.0 m | 223 | 0.13 | 14.29 Ω (converged) |
| 0.5 m | 438 | 0.40 | 14.32 Ω (converged) |

Frequency scaling is approximately linear in `len(frequencies)`:
each frequency adds roughly the cost of one matrix rebuild plus
the LU solve; the per-frequency overhead scales with the active
inductance model.

## Wall-clock estimates for typical study sizes

Assuming converged settings (`segment_length=1.0 m`, Carson,
PEN-Neumann off, leads-Neumann on):

| Study size | n_segments | t/solve (1 freq) | t/solve (4 freq) |
|---|---|---|---|
| 5 EFH | ~440 | 0.4 s | 1.5 s |
| 30 EFH | ~2 500 | ~5 s | ~20 s |
| 80 EFH | ~6 000 | ~30 s | ~2 min |
| 200 EFH | ~15 000 | ~3–5 min | ~10–15 min |

These are for a single deterministic solve. Empirically the
intrinsic scaling (after subtracting the ~0.3 s per-call
overhead) is close to $O(N^{2.3})$, consistent with the dense
$O(N^2)$ matrix build plus an $O(N^3)$ LU step that grows from
sub-dominant at small $N$ to dominant near $N \sim 5\,000$.

## Monte-Carlo strategy

For a typical large study the parameter axes alone produce
$5 \times 4 \times 4 \times 5 \times 3 \times 3 = 3\,600$
deterministic configurations. With $K$ stochastic realisations
per configuration (typical $K = 20$–$100$) you reach $10^4$ to
$10^5$ solves. That is too much for a single core but very
tractable with parallelisation.

### Recommended pattern

```python
# studies/parameter_sweep.py
from joblib import Parallel, delayed

import groundfield as gf
from groundfield.generators import (
    LogNormal, TnNetworkConfig, TnNetworkGenerator, TwoLayerSoilSpec,
    Discrete, MeasurementSetupConfig, MeasurementInjectionConfig,
    MeasurementProbeConfig, overhead_lead, PenConfig,
)


def make_cfg(rho_1, n_efh):
    """One stochastic config — every numeric field can be a Distribution
    so cfg.sample(rng) inside the worker draws independently per seed."""
    return TnNetworkConfig(
        building_counts={"residential": n_efh},
        soil=TwoLayerSoilSpec(
            rho_1=rho_1,
            rho_2=LogNormal.from_moments(mean=50.0, std=15.0),
            h_1=Discrete(values=[5.0, 10.0, 30.0]),
        ),
        pen=PenConfig(inductance_model=None),
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(
                position_xy=(120.0, 0.0),
                feed_lead=overhead_lead(),
            ),
            probe=MeasurementProbeConfig(
                position_xy=(50.0, 0.0),
                lead=overhead_lead(),
            ),
        ),
    )


def solve_one(rho_1: float, n_efh: int, seed: int) -> dict:
    """One Monte-Carlo realisation. Returns a flat dict; aggregate
    later into a DataFrame."""
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
        "rho_1": rho_1, "n_efh": n_efh, "seed": seed,
        "rho_2_drawn": float(resolved.soil.rho_2),
        "h_1_drawn": float(resolved.soil.h_1),
        "Z_abs": abs(Z),
        "Z_arg_deg": float(__import__("numpy").angle(Z, deg=True)),
    }


# Parameter grid + Monte-Carlo realisations per cell
RHO_1_GRID = [30.0, 100.0, 200.0, 500.0, 1000.0]
N_EFH_GRID = [5, 10, 30]   # add 80, 200 for the full default set
N_REALISATIONS = 20

jobs = [
    (rho_1, n_efh, seed)
    for rho_1 in RHO_1_GRID
    for n_efh in N_EFH_GRID
    for seed in range(N_REALISATIONS)
]

# joblib distributes across local cores; n_jobs=-1 uses all of them
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(solve_one)(rho_1, n_efh, seed) for (rho_1, n_efh, seed) in jobs
)

import pandas as pd
df = pd.DataFrame(results)
df.to_parquet("parameter_sweep.parquet")  # cheap binary; survives kernel restarts
```

### Throughput estimate

On a 12-core laptop with the recommended settings:

| Per-solve cost | Per-cell M-C cost (K=20) | Full default grid M-C |
|---|---|---|
| 30 EFH @ 5 s | ~10 s wall (parallel) | ~1.5 h |
| 80 EFH @ 30 s | ~50 s wall (parallel) | ~7 h |
| 200 EFH @ 4 min | ~7 min wall (parallel) | ~50 h |

Overnight or weekend runs cover the full parameter axis comfortably for
sizes ≤ 80 EFH. The 200-EFH × full Monte-Carlo case is the one
that motivates the ACA roadmap below.

### Tips for keeping Monte Carlo robust

1. **Always derive the per-realisation seed from a deterministic
   integer.** `seed = realisation_index` (or
   `seed = 1000 * config_index + realisation_index`) makes
   results bit-exactly reproducible after the fact.
2. **Persist intermediate results to disk** (`parquet` /
   `feather` are fast and small) so a kernel restart does not
   throw away half a day of compute.
3. **Time one realisation first.** Multiply by the total
   workload to see if the run is tractable before launching the
   full sweep. Notebook 21 shows you where the dominant cost
   lives.
4. **Run a 100-element pilot Monte Carlo first** at a single
   `(n_efh, ρ_1)` cell. Quantile bands stabilise around K ≈ 20–30
   for the typical Lognormal soil; full $K = 100$ is rarely
   needed.

### What does *not* work today

- **Don't expect speed-ups from `n_jobs > n_physical_cores`** —
  the dense LU solve is BLAS-bound and uses every core in a
  single solve already.
- **Don't try to share a `TnNetworkConfig` across processes by
  pickling it as part of the `Parallel(...)` call** — the
  Pydantic discriminated unions (Distributions) round-trip
  cleanly through joblib's `cloudpickle`, but the *world*
  produced from `gen.build()` does not. Always rebuild inside
  the worker (as the example above does).
- **Don't use `inductance_model="neumann"` on every house's
  bonding wires** by accident — only PEN, the measurement leads
  and (rarely) the substation bonds need it. Default settings
  enforce this.

## Roadmap: when ACA + iterative solver becomes worth it

The dense matrix is $O(N^2)$ memory and $O(N^3)$ LU. For
$N \gtrsim 5\,000$ both costs become sharp, and a Monte-Carlo
sweep over 200 EFH × 1 000 realisations is multi-week today on
a single workstation. Two algorithmic upgrades are on the
[CHANGELOG roadmap](CHANGELOG-Roadmap-Punkt) and would change
this regime:

1. **ACA (Adaptive Cross Approximation)** for the dense reaction
   matrix — brings memory from $O(N^2)$ to $O(N \log N)$ and
   preserves accuracy to user-set tolerance.
2. **GMRES with block-Jacobi preconditioner** for the linear
   system — replaces the dense LU with an iterative solve whose
   per-iteration cost benefits from the ACA-compressed matrix.

Together, these would push 200-EFH × full Monte Carlo from
~weeks to ~hours on the same hardware. They are deferred
because the current typical study scope (≤ 80 EFH single shot,
≤ 30 EFH × Monte Carlo) is well within the dense-solver regime.

## Reproducing these numbers yourself

`notebooks/21_performance_tuning.ipynb` runs the exact
benchmarks. The `PRESET = 'fast'` mode finishes in well under
10 minutes on any modern laptop. Run after every solver-side
change to spot regressions early.
