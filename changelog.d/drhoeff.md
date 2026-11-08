### Added

- **`effective_resistivity` (10.84, user decision Q6):** the effective
  resistivity of a homogeneous or two-layer soil in Ω·m,
  ρ_eff = ρ₁^(1−w) · ρ₂^w with w = w0 · (h₁ / h_ref)^(−β), h_ref = 10 m and
  the AP1 calibration w0 = 0.13, β = 0.8 (group-specific values as `w0=`,
  `beta=`, `h_ref=`). `gf.effective_resistivity(soil)` (also
  `groundfield.soil.effective_resistivity`, or with the keywords `rho_1`,
  `rho_2`, `h_1`) returns ρ itself for homogeneous soil and for stacks of
  equal layers, without round-off (the plain formula gives
  200.00000000000003 Ω·m for 200 / 200 Ω·m, h₁ = 10 m); reduces a
  `MultiLayerSoil` to its canonical stack (equal adjacent layers merged, as
  the layered backends do, 10.48) and raises `ValueError` for three or more
  distinct layers; limits w to [0, 1], so ρ_eff stays between ρ₁ and ρ₂ (the
  unlimited formula leaves that interval below h₁ = 0.78 m: 17.9 Ω·m for
  100 / 30 Ω·m at h₁ = 0.5 m; with the limit such a thin upper layer gives
  ρ₂). Inside the calibration range of the AP1 catalogue (h₁ = 5 … 30 m,
  ρ₁ = 30 … 1000 Ω·m, ρ₂ = 30 … 200 Ω·m) the value is bit-identical to the
  AP1 / AP2 scripts' `r1 ** (1 - w) * r2 ** w` and agrees with a 50-digit
  evaluation to 1.6e-16. Test: `tests/test_effective_resistivity.py`.

### Changed

- ⚠ **The `rho` of a BusType in layered soil is the effective resistivity
  (10.84, user decision Q6):** `rho_at_fit` of a vector fit and the `rho`
  samples of a standard form fitted in layered soil are ρ_eff of the soil
  they were computed in, `effective_resistivity(soil)`, the
  `specific_earth_resistance` the AP2 network model gives a groundinsight
  bus. This replaces the upper-layer resistivity ρ₁ that the 10.84 entry of
  this release cycle had made the convention; on the AP1 grid ρ_eff differs
  from ρ₁ by −54.8 % (1000 over 30 Ω·m, h₁ = 5 m) to +53.6 % (30 over
  200 Ω·m, h₁ = 5 m). The error message for a missing `rho_at_fit`, the
  docstrings of the four exporters, the IO and Soil references, ADR-0008
  and tutorial T05 state it. Every BusType document records the convention
  as `metadata["rho_definition"] = "effective_resistivity"` (documents of
  0.15.0 have no such entry). No function computes a different number: the
  exporter writes the `rho` it is given. Affected: BusTypes exported in
  layered soil with ρ₁ as `rho`. A compact electrode in the upper layer
  follows ρ₁ more closely than ρ_eff of the AP1 networks: the station of the
  IO example (2 m ring and 3 m rod) swept over ρ₁ fits the standard form to
  13.4 / 8.3 / 3.6 % over ρ_eff and to 3.9 / 1.9 / 0.6 % over ρ₁ for
  h₁ = 5 / 10 / 30 m; check `max_relative`. Golden cases: none. Tests:
  `tests/test_io_groundinsight.py::test_rho_at_fit_convention_for_layered_soil_is_the_effective_resistivity`,
  `::test_bustype_metadata_records_the_rho_definition`,
  `::test_layered_soil_exports_pair_with_an_effective_resistivity_bus`.
