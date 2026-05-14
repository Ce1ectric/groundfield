# Example 05 — Inductive coupling on the measurement leads

The galvanic measurement of example 04 is the textbook ideal,
but it ignores a real-world effect: the **metallic measurement
leads** that physically connect the source / measurement device
to the auxiliary electrode and the voltage probe. Those leads
carry current and generate magnetic fields, and a long parallel
voltage-measurement lead picks up an additional **induced EMF**
from the current-feed lead. This example quantifies how much
this distorts the apparent impedance.

## What you'll see

* How `feed_lead` and `lead` add metallic conductors to the
  measurement setup.
* What `inductance_model="neumann"` does (mutual inductance
  via the Neumann double-line integral).
* The choice of earth-return correction — `carson_series` is
  the default path.
* A direct comparison of the apparent impedance with and
  without inductive coupling.

## Code

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import groundfield as gf
from groundfield.generators import (
    ManhattanGridPlacement,
    MeasurementInjectionConfig,
    MeasurementProbeConfig,
    MeasurementSetupConfig,
    PenConfig,
    TnNetworkConfig,
    TnNetworkGenerator,
    TwoLayerSoilSpec,
    overhead_lead,
)


def make_cfg(*, with_leads: bool) -> TnNetworkConfig:
    """Same network, two measurement variants:

    - ``with_leads=False``: galvanic only.
    - ``with_leads=True``:  metallic feed and probe leads.
    """
    if with_leads:
        injection = MeasurementInjectionConfig(
            position_xy=(200.0, 0.0),
            feed_lead=overhead_lead(),       # bare overhead at z = 0
        )
        probe = MeasurementProbeConfig(
            position_xy=(124.0, 0.0),         # 62 % point
            lead=overhead_lead(),
        )
    else:
        injection = MeasurementInjectionConfig(
            position_xy=(200.0, 0.0),
        )
        probe = MeasurementProbeConfig(
            position_xy=(124.0, 0.0),
        )
    return TnNetworkConfig(
        building_counts={"residential": 10},
        placement=ManhattanGridPlacement(
            spacing_x_m=20.0, spacing_y_m=20.0, n_per_row=5,
        ),
        soil=TwoLayerSoilSpec(rho_1=100.0, rho_2=50.0, h_1=5.0),
        # PEN inductance off — here we are studying the inductive
        # coupling between the *measurement* leads, not the PEN
        # trunk. Switching this on adds ~1 % at 50 Hz at a 3×
        # cost; see the performance guide.
        pen=PenConfig(inductance_model=None),
        measurement=MeasurementSetupConfig(injection=injection, probe=probe),
    )


def solve(cfg: TnNetworkConfig, frequency_Hz: float) -> tuple[complex, complex]:
    """Returns (Z_true_cluster, Z_apparent_measured)."""
    world = TnNetworkGenerator(cfg, seed=0).build()
    engine = gf.create_engine(
        backend="image_2layer",
        segment_length=1.0,
        frequencies=[frequency_Hz],
        # Carson is the default path; Sommerfeld would be 1000×
        # slower and gives identical results in this regime.
        earth_inductive_model="carson_series",
    )
    result = engine.solve(world)
    Z_true = result.cluster_impedance("trafo_ring_0")[0]
    phi_subst = result.electrode_potentials["trafo_ring_0"][0]
    phi_probe = result.electrode_potentials["probe_rod_0"][0]
    Z_meas = (phi_subst - phi_probe) / cfg.source_magnitude_A
    return Z_true, Z_meas


# Sweep frequencies — the inductive distortion grows linearly in f.
FREQS = [1.0, 50.0, 250.0, 500.0, 1000.0]
records = []
for f in FREQS:
    Z_galv_true, Z_galv = solve(make_cfg(with_leads=False), f)
    Z_ind_true, Z_ind = solve(make_cfg(with_leads=True), f)
    records.append({
        "f_Hz": f,
        "|Z_true|": abs(Z_galv_true),
        "|Z_apparent_galvanic|": abs(Z_galv),
        "|Z_apparent_inductive|": abs(Z_ind),
        "arg_Z_apparent_galvanic_deg": np.angle(Z_galv, deg=True),
        "arg_Z_apparent_inductive_deg": np.angle(Z_ind, deg=True),
        "abs_diff_pct": 100.0 * (abs(Z_ind) - abs(Z_galv)) / abs(Z_galv),
    })

df = pd.DataFrame(records)
print(df.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

# --- plot magnitude and phase of the apparent impedance ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(df["f_Hz"], df["|Z_apparent_galvanic|"], "o-", label="galvanic only")
axes[0].plot(df["f_Hz"], df["|Z_apparent_inductive|"], "s--", label="with inductive leads")
axes[0].axhline(df["|Z_true|"].iloc[0], color="C2", ls=":", label="true |Z|")
axes[0].set_xscale("log")
axes[0].set_xlabel("f / Hz"); axes[0].set_ylabel("|Z_apparent| / Ω")
axes[0].grid(True, which="both", alpha=0.3); axes[0].legend()
axes[0].set_title("Magnitude")

axes[1].plot(df["f_Hz"], df["arg_Z_apparent_galvanic_deg"], "o-", label="galvanic only")
axes[1].plot(df["f_Hz"], df["arg_Z_apparent_inductive_deg"], "s--", label="with inductive leads")
axes[1].set_xscale("log")
axes[1].set_xlabel("f / Hz"); axes[1].set_ylabel("arg Z / °")
axes[1].grid(True, which="both", alpha=0.3); axes[1].legend()
axes[1].set_title("Phase")

fig.suptitle("Measurement error from inductive coupling")
fig.tight_layout()
plt.show()
```

## What you should observe

* **Magnitude error grows with frequency**: at 50 Hz the
  inductive distortion of the apparent impedance is small
  (sub-percent for typical geometries), but at 1 kHz it is
  several percent — the precise number depends on lead length,
  spacing, and soil resistivity.
* **Phase shift**: the galvanic-only measurement reads a real
  impedance (phase ≈ 0°). With the inductive leads, the phase
  becomes positive (inductive) and grows with $f$ — a clear
  marker that the measurement is no longer reading the pure
  cluster impedance.
* **The true cluster impedance** (the green dotted line) is
  what you would *want* to recover. The gap between the dashed
  curve and the dotted line is the measurement error.

## Configuration knobs that matter

`overhead_lead()` returns a `MeasurementLeadConfig` with these
defaults — all overridable:

```python
overhead_lead(
    wire_radius_m=0.005,
    inductance_model="neumann",   # set to None to disable inductance
    segment_length_m=5.0,         # finer for higher frequencies
)
```

For a buried-cable measurement lead use `buried_lead()` instead;
it changes the conductor type and depth.

For the **worst-case** scenario the user has often referenced —
"the overhead current-feed lead runs directly above the buried
PEN cable" — choose the auxiliary electrode along the same x-axis
as the substation's nearest cable cabinet. The feed lead then
runs straight over the PEN trunk for most of its length, and
the mutual-inductance term is maximal.

## Try this next

* Set `pen=PenConfig(inductance_model="neumann")` to additionally
  capture the inductive coupling **into** the PEN trunk — that
  changes the substation cluster impedance directly. Cost: 3×
  slower.
* Move the probe to the 62 % point of the *current* aux distance
  (~ 124 m for a 200 m aux) and observe how much the inductive
  distortion of the apparent impedance differs from the galvanic
  ideal.
* Run a parameter sweep over the lead spacing — see example 06
  for the deterministic-sweep pattern and example 07 for the
  Monte-Carlo version.
