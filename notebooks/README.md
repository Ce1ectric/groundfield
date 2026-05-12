# Notebooks

This folder collects the parameter studies and reference cases for
`groundfield`. Each notebook is self-contained: the soil model,
geometry, and frequency grid are defined at the top, and the end of
the notebook produces the reduced `rho-f` model that can be imported
into `groundinsight`.

Existing notebooks:

- `01_smoke_test.ipynb` — quick end-to-end check of the top-level API
  (`create_world`, `create_electrode`, `create_conductor`,
  `create_source`, `create_engine`, `run_simulation`) plus the plot
  family (`plot_potential_contour`, `plot_potential_profile`,
  `plot_potential_radial`). The "without / with connection conductor"
  comparison demonstrates the cluster logic.
- `02_two_layer.ipynb` — 2-layer engine (`image_2layer`, Tagg/Sunde):
  plausibility check in the limit $\rho_1 = \rho_2$, parameter sweep
  over $K$ and $h_1$, trumpet comparison homogeneous vs. 2-layer.
- `03_cross_engine.ipynb` — cross-validation between the `image` /
  `image_2layer` backends and the new `mom` backend (Galerkin
  Method-of-Moments). Shows the cluster-impedance agreement across
  the full K range, exact K=0 collapse, and side-by-side surface
  potential profiles.
- `04_image_nlayer.ipynb` — `image_nlayer` engine (n-layer
  dispatcher): single rod sanity, two interconnected rods (potential
  profile, cluster impedance, current split), 2-layer sweep, and the
  controlled error for `n ≥ 3`.
- `05_cim.ipynb` — Complex Image Method (`cim`): inspection of the
  matrix-pencil fit of $\Gamma_1(\lambda)$, single rod, three-layer
  reference case, and two interconnected rods across a soil sweep
  (homogeneous / 2-layer / 3-layer).
- `06_mom_sommerfeld.ipynb` — direct Sommerfeld-quadrature MoM
  (`mom_sommerfeld`): kernel sanity, single rod, two interconnected
  rods, and a hard-contrast validation table used as the reference
  in the cross-engine comparison.
- `07_bem.ipynb` — boundary-element collocation (`bem`): single rod,
  two interconnected rods (incl. surface potential profile), and a
  mesh-refinement sanity check.
- `08_fem.ipynb` — axisymmetric volume PDE (`fem`):
  equivalent-hemisphere reduction, single rod cross-check against
  `image`, 2-layer trend, and two interconnected rods (the
  hemisphere-pair limit `R_pair ≈ R_single / 2`).
- `09_cross_engine_extended.ipynb` — every engine side by side:
  homogeneous and 2-layer single rod, 2-layer sweep, two
  interconnected rods (surface potential overlay), plus a
  programmatic `compare_engines` summary.

Planned notebooks for work package 1:

- `ap1_ring_homogeneous.ipynb` — ring electrode in homogeneous soil,
  analytical reference (Sunde / Dwight).
- `ap1_ring_two_layer.ipynb` — parameter sweep over $\rho_1$,
  $\rho_2$, $h_1$.
- `ap1_tn_distribution_network.ipynb` — substation + house
  connections + cable cabinets, PEN and MV cable shield.
- `ap1_measurement_vs_fault_case.ipynb` — comparison of measurement
  configuration and fault case.

These notebooks materialise as the corresponding modules become
production-ready; until then this README acts as a placeholder.
