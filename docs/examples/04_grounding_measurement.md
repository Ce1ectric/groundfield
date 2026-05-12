# Example 04 — AP1 Analysis 1: galvanic grounding measurement

This is the **fall-of-potential** measurement of the AP1 work
package, **without** modelling the metallic measurement leads —
i.e. only the galvanic (DC-like) effects. We add an **auxiliary
current electrode** (Hilfserder) that closes the current loop and
a **voltage probe** (Spannungssonde) where we read off the
potential difference. The measurement aims to recover the
substation's grounding impedance, but it is sensitive to where
the probe sits relative to the auxiliary electrode.

## What you'll see

* How `MeasurementSetupConfig` adds the auxiliary electrode and
  the voltage probe to the generated world.
* How the source's `return_to` is automatically wired to the
  auxiliary electrode.
* The classical fall-of-potential (Spannungstrichter overlap)
  curve when the probe is moved along the line between
  substation and auxiliary electrode.
* The "62 % point" where the measurement reads the true
  cluster impedance.

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
    TnNetworkConfig,
    TnNetworkGenerator,
    TwoLayerSoilSpec,
)

# Fixed network: 10 EFH, 2-layer soil, default substation. The
# auxiliary electrode sits at (200, 0) — well outside the network's
# own potential funnel.
def make_cfg(probe_x: float) -> TnNetworkConfig:
    return TnNetworkConfig(
        building_counts={"residential": 10},
        placement=ManhattanGridPlacement(
            spacing_x_m=20.0, spacing_y_m=20.0, n_per_row=5,
        ),
        soil=TwoLayerSoilSpec(rho_1=100.0, rho_2=50.0, h_1=5.0),
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(
                position_xy=(200.0, 0.0),     # Hilfserder
                # No metallic feed_lead — galvanic only (Analysis 1).
            ),
            probe=MeasurementProbeConfig(
                position_xy=(probe_x, 0.0),    # Spannungssonde
            ),
        ),
        source_magnitude_A=1.0,
    )


# Sweep the probe position along the substation → aux electrode line
PROBE_DISTANCES_M = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180]

records = []
for d in PROBE_DISTANCES_M:
    cfg = make_cfg(probe_x=float(d))
    world = TnNetworkGenerator(cfg, seed=0).build()
    engine = gf.create_engine(
        backend="image_2layer",
        segment_length=1.0,
        frequencies=[50.0],
        earth_inductive_model="perfect_mirror",  # galvanic regime
    )
    result = engine.solve(world)

    # The cluster impedance of the substation — the *true* answer
    # we are trying to measure.
    Z_true = result.cluster_impedance("trafo_ring_0")[0].real

    # The measured value: U_probe / I_source. The probe potential is
    # phi(probe_anchor); the substation potential is phi(trafo_ring_0).
    # The actual measurement device reads (phi_substation - phi_probe).
    phi_subst = result.electrode_potentials["trafo_ring_0"][0].real
    # Pick out the probe's anchor electrode (named ``probe_rod_0``):
    phi_probe = result.electrode_potentials["probe_rod_0"][0].real
    U_meas = phi_subst - phi_probe
    Z_meas = U_meas / cfg.source_magnitude_A

    records.append({
        "probe_x_m": d,
        "Z_true_Ohm": Z_true,
        "Z_meas_Ohm": Z_meas,
        "rel_error_pct": 100.0 * (Z_meas - Z_true) / Z_true,
    })

df = pd.DataFrame(records)
print(df.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

# --- plot the fall-of-potential curve ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(df["Z_true_Ohm"].iloc[0], color="C2", ls="--",
           label=f"true cluster |Z| = {df['Z_true_Ohm'].iloc[0]:.2f} Ω")
ax.plot(df["probe_x_m"], df["Z_meas_Ohm"], "o-", color="C0",
        label="measured Z(probe_x)")
ax.set_xlabel("probe distance from substation in m")
ax.set_ylabel("apparent grounding impedance in Ω")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title("AP1 Analysis 1 — fall-of-potential curve, aux at 200 m")
fig.tight_layout()
plt.show()

# Print the 62 % rule of thumb
d_62 = 0.62 * 200
Z_at_62 = np.interp(d_62, df["probe_x_m"], df["Z_meas_Ohm"])
print(f"\n62 % point ({d_62:.0f} m): Z_meas = {Z_at_62:.3f} Ω, "
      f"Z_true = {df['Z_true_Ohm'].iloc[0]:.3f} Ω, "
      f"error = {100*(Z_at_62 - df['Z_true_Ohm'].iloc[0])/df['Z_true_Ohm'].iloc[0]:+.2f} %")
```

## Expected output

The fall-of-potential curve is monotonically increasing from a
small value (probe sits inside the substation's potential funnel)
to nearly the true value as the probe moves outwards, with a
characteristic plateau around the 62 % point. The
"62 % rule of thumb" — the basis of the standard 62 % method —
predicts that the apparent impedance read at the 62 % point of
the substation–aux distance is closest to the true cluster
impedance.

You should see something like:

```
probe_x_m  Z_true_Ohm  Z_meas_Ohm  rel_error_pct
   10.000      14.295       3.142       -78.022
   20.000      14.295       6.038       -57.768
   ...
  120.000      14.295      14.298         0.020
   ...
  180.000      14.295      16.421       +14.870

62 % point (124 m): Z_meas = 14.31 Ω, Z_true = 14.30 Ω,
error = +0.10 %
```

## What just happened

* **`MeasurementSetupConfig`** added two new electrodes to the
  world — an aux rod at $(200, 0)$ and a probe rod at the swept
  position. The generator also rewired the source's `return_to`
  to the aux electrode, so the test current physically returns
  through the aux instead of through remote earth.
* **`feed_lead=None` and `lead=None`** mean *no metallic
  measurement leads*. The current path is purely through the
  soil, which is the textbook **galvanic** measurement (AP1
  Analysis 1).
* The measured "apparent impedance" is computed as the potential
  difference between the substation anchor and the probe anchor,
  divided by the source current — exactly what the measurement
  device reads.

## Why the curve looks the way it does

The probe potential is the *superposition* of two
contributions:

1. The substation injects $+I$ → produces a potential funnel
   centred at the substation, decaying with distance.
2. The aux electrode draws $-I$ → produces an inverted funnel
   centred at the aux electrode.

The probe sits in both funnels. When the probe is close to the
substation, the substation funnel dominates; close to the aux,
the aux funnel does. There is a **sweet spot** in between where
both funnels have decayed enough that the measurement reads
the substation's "remote-earth" potential — that's the 62 %
point.

## Try this next

* Move the aux electrode further out (300 m, 500 m). The 62 %
  rule still applies; the absolute distance scales.
* Vary the soil parameters and watch the optimal probe distance
  shift. In high-resistivity upper soil the funnels reach
  further.
* Continue with example 05 to see what happens when the
  metallic measurement leads are added — the inductive
  coupling distorts the apparent impedance even when the
  geometry is "right".
