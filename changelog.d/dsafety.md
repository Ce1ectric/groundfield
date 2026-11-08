### Added
- **Touch points: `touch_voltage_at_part` (10.21; user decision Q3):** the
  EN 50522 touch voltage of an accessible metal part above ground — a
  station housing or its door, a mast, a fence post, a cable cabinet, a
  service connection box. The part is given by its horizontal position and
  the electrode or junction it is bonded to (`bonded_to=`), whose potential
  it carries; a part bonded to a PEN junction expresses a transferred
  potential. The feet stand on the circle of radius `distance` (1 m) around
  the part (`n_angles`, default 24), optionally only in given `directions`,
  and the positions inside the outlines in `exclude` (the housing or the
  building, where nobody can stand) are dropped. Returns the touch-voltage
  phasors and azimuths per foot position like `touch_voltage_envelope`
  (and the positions with `return_points=True`). Available as
  `gf.touch_voltage_at_part` and `gf.safety.touch_voltage_at_part`. A
  service connection box 40 m from a ring earth (100 A, 100 Ω·m), bonded to
  the PEN, has 1081 V of the 1121 V earth potential rise as touch voltage.
  Tests: `tests/test_touch_voltage_at_part.py::test_part_on_a_wire_matches_touch_voltage_on_the_offset_curve`,
  `tests/test_touch_voltage_at_part.py::test_part_at_a_rod_head_against_the_analytic_surface_potential`,
  `tests/test_touch_voltage_at_part.py::test_part_bonded_to_a_pen_junction_carries_the_transferred_potential`.
- **How-to "Assess touch voltages"** (`docs/how-to/assess-touch-voltages.md`):
  touch points of a compact station, the transferred potential over the
  PEN, the meshes of an open-air grid, and the comparison with the
  permissible values; tutorial T08 adds a fence bonded to the substation
  grid (199.5 V at the corner post against 140.7 V 1 m outside the grid,
  `mom`, 150 A) and an earth fault that is not cleared.

### Changed
- ⚠ **`touch_voltage_envelope` samples no interior by default (10.21; user
  decision Q3):** inside an electrode's footprint stands the station, the
  house, the steel structure or the mast, so the default is now
  `interior=False` for every electrode kind: the curve `distance` outside
  the wires only. The mesh sampling of the development snapshots (the
  centres of the meshes of grids and of generated mesh foundations, the
  centre of a ring) is the explicit `interior="meshes"`, for open-air grids
  such as a substation yard; `interior=True` also samples the inside of a
  closed polyline. `interior=None` is rejected with a hint. Against 0.15.0
  the default envelope is the footprint's offset curve of 10.21: a ring of
  10 m radius at 0.8 m (100 A, 100 Ω·m, `image`, point kernel) keeps
  139.4 V (the development snapshots gave 182.6 V, its centre; with the
  default line kernel 142.2 / 185.5 V), a 20 m perimeter loop (1 × 1 grid)
  141.4 V instead of 158.3 V in its centre, and the 20 m × 20 m grid with
  4 × 4 meshes 111.4 V (unchanged: the maximum is 1 m outside a corner, the
  mesh centres reach 63.0 V). Affected: envelopes of rings, grids and
  generated foundations computed with the development snapshots; pass
  `interior="meshes"` for open-air grids. Golden cases: none (no case
  evaluates the envelope). Tests:
  `tests/test_touch_voltage_envelope.py::test_envelope_samples_no_interior_by_default`,
  `tests/test_touch_voltage_envelope.py::test_release_note_envelopes_with_the_new_default`.
- ⚠ **EN 50522 permissible touch voltage beyond 10 s (10.42; user decision
  Q4):** `permissible_touch_voltage_en50522(t_f)` follows Figure 8 /
  Table B.4 of EN 50522:2022 up to 10 s and returns 80 V for every
  t_f > 10 s, including `math.inf` for an earth fault that is not cleared
  (resonant-earthed networks). The curve of Figure 8 ends at 10 s; the
  note to Figure 8 (5.4.1) and the text after Table B.4 (B.3) allow 80 V
  for current-flow durations considerably longer than 10 s, below the
  85 V end value of the table. 0.15.0 returned 85 V beyond 10 s (now
  −5.9 %) and rejected `math.inf`. `permissible_prospective_touch_voltage_en50522`
  uses the same U_Tp: beyond 10 s 80 V + 50 mA · R_A / HF (1000 Ω shoes
  on 100 Ω·m soil: 137.5 V instead of 142.5 V). Affected: assessments
  with fault durations above 10 s, e.g. resonant-earthed networks. Golden
  cases: none (`safety_en50522_curve` covers 0.05 … 10 s). Tests:
  `tests/test_safety.py::test_en50522_beyond_ten_seconds_is_80_volt`,
  `tests/test_safety.py::test_prospective_touch_voltage_beyond_ten_seconds_uses_80_volt`.

### Removed
- **`long_duration=`** of `permissible_touch_voltage_en50522` and
  `permissible_prospective_touch_voltage_en50522` (added during this
  release cycle, never released): 80 V beyond 10 s is the default. Test:
  `tests/test_safety.py::test_en50522_has_no_long_duration_keyword`.

### Deprecated
- Nothing.
