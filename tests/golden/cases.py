"""Golden-value cases: the numerical safety net of the release.

Every case builds a small, deterministic scenario, runs it through the
public API and returns a flat ``{quantity_name: numpy array}`` mapping.
:mod:`tests.golden.harness` stores these arrays once (from the code of
the last release) and compares every later run against them.

Rules for cases
---------------
* **Deterministic.** Fixed seeds, fixed geometry, no wall-clock values,
  no file system, no network.
* **Fast.** A case must run in well under two seconds on a laptop; the
  whole set runs in the default test suite.
* **Public API only.** Cases call what a user would call, so internal
  refactors stay covered without touching this file.
* **Forward compatible.** Engines are built with :func:`_engine`, which
  pins ``kernel="point"`` as soon as the engine offers a ``kernel``
  option. The pinned cases therefore keep guarding bit-identical
  refactors after the default kernel changes; the cases in the
  ``defaults_*`` group use the plain defaults on purpose and record
  every user-visible change of a default.
* **Append-only.** Do not change an existing case. Add a new case
  instead (``python -m tests.golden.harness generate --only NAME``),
  so that the stored values keep their meaning.

Quantity naming
---------------
``phi/<electrode>``
    Electrode potential per frequency (complex).
``I/<electrode>``
    Leakage current of the electrode per frequency (complex).
``Zc/<electrode>``
    :meth:`FieldResult.cluster_impedance` per frequency.
``potential/f<k>``
    :meth:`FieldResult.potential` at the probe points of the case.
``ps/pos``, ``ps/len``, ``ps/I``, ``ps/owner``
    Discretisation: point-source midpoints, represented lengths,
    currents and owning electrode names.
``world/json``
    Canonical JSON of a generated :class:`World` (generator cases).
"""

from __future__ import annotations

import inspect
import json
import warnings
from collections.abc import Callable

import numpy as np

import groundfield as gf

__all__ = ["CASES", "run_case"]

Quantities = dict[str, np.ndarray]

CASES: dict[str, Callable[[], Quantities]] = {}

#: Absolute noise floors of residual-type quantities (see
#: :func:`tests.golden.harness.noise_floor`): ``case -> {pattern: floor}``.
#: The rms error of an exact synthetic vector fit is pure round-off
#: (1.3e-13 ohm with numpy 2.5, 2.4e-10 ohm with numpy 2.4).
NOISE_FLOORS: dict[str, dict[str, float]] = {
    "fit_vector_fit_synthetic": {"rms": 1e-8},
}


def _case(func: Callable[[], Quantities]) -> Callable[[], Quantities]:
    """Register ``func`` under its name (without the ``case_`` prefix)."""
    name = func.__name__.removeprefix("case_")
    if name in CASES:
        raise RuntimeError(f"duplicate golden case {name!r}")
    CASES[name] = func
    return func


def run_case(name: str) -> Quantities:
    """Run one case with all warnings silenced (deprecations included)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = CASES[name]()
    return {k: np.asarray(v) for k, v in out.items()}


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _engine(**kw: object) -> gf.Engine:
    """``gf.create_engine`` with the point kernel pinned when available."""
    params = inspect.signature(gf.create_engine).parameters
    if "kernel" in params and "kernel" not in kw:
        kw["kernel"] = "point"
    return gf.create_engine(**kw)  # type: ignore[arg-type]


def _surface_line(x0: float, x1: float, n: int, y: float = 0.0) -> np.ndarray:
    xs = np.linspace(x0, x1, n)
    return np.column_stack([xs, np.full(n, y), np.zeros(n)])


def _with_buried(points: np.ndarray, extra: list[tuple[float, float, float]]):
    return np.vstack([points, np.asarray(extra, dtype=float)])


def _result_quantities(
    res: gf.FieldResult,
    *,
    probes: np.ndarray | None = None,
    cluster_of: tuple[str, ...] = (),
    point_sources: bool = True,
    prefix: str = "",
) -> Quantities:
    out: Quantities = {}
    for name in sorted(res.electrode_potentials):
        out[f"{prefix}phi/{name}"] = np.asarray(
            res.electrode_potentials[name], dtype=complex
        )
    for name in sorted(res.electrode_currents):
        out[f"{prefix}I/{name}"] = np.asarray(
            res.electrode_currents[name], dtype=complex
        )
    for name in cluster_of:
        out[f"{prefix}Zc/{name}"] = np.asarray(
            res.cluster_impedance(name), dtype=complex
        )
    if probes is not None:
        for k in range(len(res.frequencies)):
            out[f"{prefix}potential/f{k}"] = np.asarray(
                res.potential(probes, frequency_index=k), dtype=complex
            )
    if point_sources and res.point_sources:
        ps = res.point_sources
        out[f"{prefix}ps/pos"] = np.array([p.position for p in ps], dtype=float)
        out[f"{prefix}ps/len"] = np.array([p.length for p in ps], dtype=float)
        out[f"{prefix}ps/I"] = np.array([p.current for p in ps], dtype=complex)
        out[f"{prefix}ps/owner"] = np.array([p.electrode_name for p in ps], dtype=str)
    return out


def _world_json(world: gf.World) -> np.ndarray:
    payload = json.dumps(world.model_dump(mode="json"), sort_keys=True)
    return np.array(payload, dtype=str)


def _ideal_link(world: gf.World, name: str, a: str, b: str) -> None:
    """Galvanically ideal bond (collapses ``a`` and ``b`` into one cluster)."""
    gf.create_conductor(
        world,
        name=name,
        start=a,
        end=b,
        conductor_type="bare_copper",
        cross_section=1.0,
        coupling_to_soil="isolated",
    )


_H100 = dict(resistivity=100.0)


# ---------------------------------------------------------------------
# image family, homogeneous soil
# ---------------------------------------------------------------------


@_case
def case_img_rod_hom() -> Quantities:
    w = gf.create_world(soil=gf.HomogeneousSoil(**_H100))
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    res = w.solve(_engine(backend="image", frequencies=[50.0], segment_length=0.5))
    probes = _with_buried(_surface_line(0.0, 20.0, 11), [(1.0, 0.0, 1.0)])
    return _result_quantities(res, probes=probes, cluster_of=("rod",))


def _mixed_cluster_world() -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(**_H100))
    gf.create_electrode(w, "ring", name="ring", center=(0.0, 0.0, 0.8), radius=4.0)
    for i, (x, y) in enumerate([(4.0, 0.0), (0.0, 4.0), (-4.0, 0.0), (0.0, -4.0)]):
        gf.create_electrode(w, "rod", name=f"rod{i}", position=(x, y, 0.8), length=2.0)
    gf.create_electrode(
        w, "strip", name="strip", start=(4.5, 0.0, 0.8), end=(14.0, 0.0, 0.8)
    )
    gf.create_electrode(
        w,
        "polyline",
        name="poly",
        vertices=[(-4.5, 0.0, 0.8), (-10.0, 0.0, 0.8), (-10.0, 6.0, 0.8)],
    )
    gf.create_electrode(
        w,
        "star",
        name="star",
        center=(0.0, -12.0, 0.6),
        n_arms=3,
        arm_length=3.0,
    )
    for i in range(4):
        _ideal_link(w, f"b_rod{i}", "ring", f"rod{i}")
    _ideal_link(w, "b_strip", "ring", "strip")
    _ideal_link(w, "b_poly", "ring", "poly")
    _ideal_link(w, "b_star", "ring", "star")
    gf.create_source(w, attached_to="ring", magnitude=1.0)
    return w


@_case
def case_img_mixed_cluster_hom() -> Quantities:
    w = _mixed_cluster_world()
    res = w.solve(_engine(backend="image", frequencies=[0.0, 50.0], segment_length=0.5))
    probes = _with_buried(
        _surface_line(-20.0, 20.0, 17), [(0.0, 0.0, 0.0), (2.0, 2.0, 0.0)]
    )
    out = _result_quantities(res, probes=probes, cluster_of=("ring", "star"))
    out["safety/touch_1m"] = np.array(
        [
            gf.touch_voltage(res, w, electrode=name, distance=1.0)
            for name in ("ring", "strip", "star")
        ],
        dtype=complex,
    )
    out["safety/step"] = np.array(
        [
            gf.step_voltage(res, position=(x, 0.0, 0.0), step=1.0)
            for x in (3.0, 5.0, 15.0)
        ],
        dtype=complex,
    )
    return out


@_case
def case_img_two_clusters_lumped_hom() -> Quantities:
    w = gf.create_world(soil=gf.HomogeneousSoil(**_H100))
    gf.create_electrode(w, "rod", name="A", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="B", position=(30.0, 0.0, 0.5), length=3.0)
    gf.create_conductor(
        w,
        name="pen",
        start="A",
        end="B",
        conductor_type="pen",
        resistivity=2.82e-8,
        cross_section=50e-6,
        coupling_to_soil="isolated",
    )
    gf.create_source(w, attached_to="A", magnitude=1.0)
    res = w.solve(_engine(backend="image", frequencies=[50.0], segment_length=0.5))
    probes = _surface_line(-10.0, 40.0, 11)
    out = _result_quantities(res, probes=probes, cluster_of=("A", "B"))
    out["split"] = np.asarray(gf.split_factor(res, w), dtype=complex)
    return out


# ---------------------------------------------------------------------
# layered soil
# ---------------------------------------------------------------------


@_case
def case_img2_rod_crosslayer() -> Quantities:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=1.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    res = w.solve(
        _engine(backend="image_2layer", frequencies=[50.0], segment_length=0.5)
    )
    probes = _with_buried(_surface_line(0.0, 20.0, 9), [(1.0, 0.0, 2.0)])
    return _result_quantities(res, probes=probes, cluster_of=("rod",))


@_case
def case_img2_ring_rods_within_layer() -> Quantities:
    soil = gf.TwoLayerSoil(rho_1=300.0, rho_2=50.0, h_1=2.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "ring", name="ring", center=(0.0, 0.0, 0.6), radius=3.0)
    for i, (x, y) in enumerate([(3.0, 0.0), (-3.0, 0.0)]):
        gf.create_electrode(w, "rod", name=f"rod{i}", position=(x, y, 0.6), length=1.0)
        _ideal_link(w, f"b{i}", "ring", f"rod{i}")
    gf.create_source(w, attached_to="ring", magnitude=1.0)
    res = w.solve(
        _engine(backend="image_2layer", frequencies=[50.0], segment_length=0.5)
    )
    probes = _surface_line(0.0, 30.0, 9)
    return _result_quantities(res, probes=probes, cluster_of=("ring",))


@_case
def case_img2_grid_upper_resistive() -> Quantities:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=5.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w,
        "grid_mesh",
        name="grid",
        corner=(-5.0, -5.0, 0.6),
        size=(10.0, 10.0),
        n_x=2,
        n_y=2,
    )
    gf.create_source(w, attached_to="grid", magnitude=1.0)
    res = w.solve(
        _engine(backend="image_2layer", frequencies=[50.0], segment_length=1.0)
    )
    probes = _surface_line(0.0, 25.0, 6)
    return _result_quantities(res, probes=probes, cluster_of=("grid",))


@_case
def case_imgN_two_layer_multilayer_rod() -> Quantities:
    soil = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(resistivity=200.0, thickness=1.5),
            gf.SoilLayer(resistivity=50.0, thickness=None),
        ]
    )
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.3), length=1.0)
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    res = w.solve(
        _engine(backend="image_nlayer", frequencies=[50.0], segment_length=0.25)
    )
    return _result_quantities(res, cluster_of=("rod",))


@_case
def case_mom_sommerfeld_three_layer_rod() -> Quantities:
    soil = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(resistivity=200.0, thickness=1.5),
            gf.SoilLayer(resistivity=50.0, thickness=3.0),
            gf.SoilLayer(resistivity=500.0, thickness=None),
        ]
    )
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.3), length=1.0)
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    res = w.solve(
        _engine(backend="mom_sommerfeld", frequencies=[50.0], segment_length=0.25)
    )
    return _result_quantities(res, cluster_of=("rod",))


@_case
def case_cim_rod_two_layer() -> Quantities:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=5.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    res = w.solve(_engine(backend="cim", frequencies=[50.0], segment_length=0.5))
    return _result_quantities(res, cluster_of=("rod",))


# ---------------------------------------------------------------------
# other engines
# ---------------------------------------------------------------------


def _ring_rod_world(soil: object) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "ring", name="ring", center=(0.0, 0.0, 0.6), radius=2.0)
    gf.create_electrode(w, "rod", name="rod", position=(2.0, 0.0, 0.6), length=1.5)
    _ideal_link(w, "b", "ring", "rod")
    gf.create_source(w, attached_to="ring", magnitude=1.0)
    return w


@_case
def case_mom_ring_rod_hom() -> Quantities:
    w = _ring_rod_world(gf.HomogeneousSoil(**_H100))
    res = w.solve(_engine(backend="mom", frequencies=[50.0], segment_length=0.5))
    return _result_quantities(res, cluster_of=("ring",))


@_case
def case_bem_ring_rod_hom() -> Quantities:
    w = _ring_rod_world(gf.HomogeneousSoil(**_H100))
    res = w.solve(_engine(backend="bem", frequencies=[50.0], segment_length=0.5))
    return _result_quantities(res, cluster_of=("ring",))


@_case
def case_mom_sommerfeld_rod_two_layer() -> Quantities:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=3.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.5), length=1.5)
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    res = w.solve(
        _engine(backend="mom_sommerfeld", frequencies=[50.0], segment_length=0.5)
    )
    return _result_quantities(res, cluster_of=("rod",))


@_case
def case_fem_rod_hom() -> Quantities:
    w = gf.create_world(soil=gf.HomogeneousSoil(**_H100))
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    res = w.solve(_engine(backend="fem", frequencies=[50.0], segment_length=0.5))
    return _result_quantities(res, cluster_of=("rod",), point_sources=False)


# ---------------------------------------------------------------------
# conductors and earth-return models
# ---------------------------------------------------------------------


def _galvanic_pair(model: str, soil: object | None = None):
    w = gf.create_world(soil=soil or gf.HomogeneousSoil(**_H100))
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_electrode(w, "rod", name="g2", position=(20.0, 0.0, 0.5), length=2.0)
    gf.create_conductor(
        w,
        name="lead",
        start="g1",
        end="g2",
        conductor_type="bare_copper",
        wire_radius=0.004,
        resistivity=1.72e-8,
        cross_section=50e-6,
        discretize_segment_length=5.0,
        coupling_to_soil="galvanic",
        inductance_model="neumann",
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = _engine(
        backend="image",
        frequencies=[0.0, 50.0, 1000.0],
        segment_length=0.5,
        earth_inductive_model=model,
    )
    return w, eng


def _make_galvanic_case(model: str) -> Callable[[], Quantities]:
    def case() -> Quantities:
        w, eng = _galvanic_pair(model)
        res = w.solve(eng)
        return _result_quantities(
            res, probes=_surface_line(-5.0, 25.0, 7), cluster_of=("g1", "g2")
        )

    case.__name__ = f"case_cond_galvanic_neumann_{model}"
    return case


for _model in ("perfect_mirror", "carson_series", "sommerfeld"):
    _case(_make_galvanic_case(_model))


@_case
def case_cond_isolated_pen_sommerfeld_two_layer() -> Quantities:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=5.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="s", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="h", position=(60.0, 0.0, 0.5), length=1.5)
    gf.create_conductor(
        w,
        name="pen",
        start="s",
        end="h",
        conductor_type="pen",
        wire_radius=0.006,
        resistivity=2.82e-8,
        cross_section=95e-6,
        discretize_segment_length=20.0,
        coupling_to_soil="isolated",
        inductance_model="neumann",
    )
    gf.create_source(w, attached_to="s", magnitude=1.0)
    eng = _engine(
        backend="image_2layer",
        frequencies=[50.0, 1000.0],
        segment_length=0.5,
        earth_inductive_model="sommerfeld",
    )
    res = w.solve(eng)
    return _result_quantities(res, cluster_of=("s", "h"))


@_case
def case_cond_coupled_leads_open_circuit() -> Quantities:
    w = gf.create_world(soil=gf.HomogeneousSoil(**_H100))
    for name, x, y in (("g1", 0, 0), ("g2", 20, 0), ("m1", 0, 1), ("m2", 20, 1)):
        gf.create_electrode(
            w, "rod", name=name, position=(float(x), float(y), 0.5), length=1.5
        )
    for name, a, b in (("src_lead", "g1", "g2"), ("meas_lead", "m1", "m2")):
        gf.create_conductor(
            w,
            name=name,
            start=a,
            end=b,
            conductor_type="bare_copper",
            wire_radius=0.004,
            resistivity=1.72e-8,
            cross_section=50e-6,
            discretize_segment_length=5.0,
            coupling_to_soil="galvanic",
            inductance_model="neumann",
        )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    res = w.solve(
        _engine(
            backend="image",
            frequencies=[50.0, 500.0],
            segment_length=0.5,
            earth_inductive_model="perfect_mirror",
        )
    )
    return _result_quantities(res, point_sources=False)


# ---------------------------------------------------------------------
# post-solve matrices (AP1 production path)
# ---------------------------------------------------------------------


def _foundation_pair_world(concrete_model: str) -> tuple[gf.World, list[str]]:
    spec = gf.GroundingSystemSpec(
        electrodes=[
            gf.FoundationElectrodeSpec(
                style="ring",
                size_m=10.0,
                depth_m=0.7,
                wire_radius_m=0.0075,
                concrete_rho_ohm_m=100.0,
                concrete_thickness_m=0.05,
                concrete_model=concrete_model,
            )
        ],
        bond_conductor_type="bare_copper",
    )
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=30.0, rho_2=100.0, h_1=5.0))
    rng = np.random.default_rng(1)
    a = spec.build_at(w, site_xy=(0.0, 0.0), name_prefix="h0", rng=rng)
    b = spec.build_at(w, site_xy=(30.0, 0.0), name_prefix="h1", rng=rng)
    return w, [a, b]


def _make_mutual_case(concrete_model: str) -> Callable[[], Quantities]:
    def case() -> Quantities:
        w, anchors = _foundation_pair_world(concrete_model)
        eng = _engine(backend="image", segment_length=2.0, image_max_terms=500)
        probes = np.array([(0.0, 0.0, 0.0), (30.0, 0.0, 0.0)])
        Z = gf.solve_mutual_matrix(w, eng, anchors, probes)
        field = _with_buried(_surface_line(-10.0, 40.0, 11), [(15.0, 5.0, 0.0)])
        R = gf.solve_mutual_field(w, eng, anchors, field)
        return {
            "Z_G": np.asarray(Z, dtype=complex),
            "R": np.asarray(R, dtype=complex),
            "world/json": _world_json(w),
        }

    case.__name__ = f"case_mutual_foundations_{concrete_model}_concrete"
    return case


for _cm in ("lumped", "distributed"):
    _case(_make_mutual_case(_cm))


@_case
def case_mutual_three_rods_crosslayer() -> Quantities:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=1.0)
    w = gf.create_world(soil=soil)
    anchors = [("g0", 0.0, 0.0), ("g1", 12.0, 0.0), ("g2", 6.0, 9.0)]
    for name, x, y in anchors:
        gf.create_electrode(w, "rod", name=name, position=(x, y, 0.5), length=3.0)
    eng = _engine(backend="image", segment_length=0.5, frequencies=[50.0])
    probes = np.array([[x, y, 0.0] for _, x, y in anchors])
    Z = gf.solve_mutual_matrix(w, eng, [a[0] for a in anchors], probes)
    return {"Z_G": np.asarray(Z, dtype=complex)}


# ---------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------


@_case
def case_gen_tn_default() -> Quantities:
    from groundfield.generators import TnNetworkConfig, TnNetworkGenerator

    cfg = TnNetworkConfig(building_counts={"residential": 5})
    w = TnNetworkGenerator(cfg, seed=0).build()
    res = w.solve(_engine(backend="image_2layer", frequencies=[50.0]))
    out = _result_quantities(res, probes=_surface_line(-30.0, 60.0, 10))
    out["world/json"] = _world_json(w)
    return out


@_case
def case_gen_tn_measurement_overhead_leads() -> Quantities:
    from groundfield.generators import (
        MeasurementInjectionConfig,
        MeasurementProbeConfig,
        MeasurementSetupConfig,
        TnNetworkConfig,
        TnNetworkGenerator,
        overhead_lead,
    )

    cfg = TnNetworkConfig(
        building_counts={"residential": 3},
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(
                position_xy=(150.0, 0.0), feed_lead=overhead_lead()
            ),
            probe=MeasurementProbeConfig(position_xy=(50.0, 0.0), lead=overhead_lead()),
        ),
    )
    w = TnNetworkGenerator(cfg, seed=0).build()
    res = w.solve(
        _engine(
            backend="image_2layer",
            frequencies=[50.0],
            earth_inductive_model="perfect_mirror",
        )
    )
    out = _result_quantities(res, point_sources=False)
    out["world/json"] = _world_json(w)
    return out


@_case
def case_gen_layout_builder() -> Quantities:
    from groundfield.generators import OrtsnetzLayout

    fps = []
    for i in range(4):
        cx, cy = 20.0 + 25.0 * i, 18.0 if i % 2 == 0 else -18.0
        fps.append(
            gf.BuildingFootprint(
                polygon_xy_m=[
                    (cx - 5.0, cy - 5.0),
                    (cx + 5.0, cy - 5.0),
                    (cx + 5.0, cy + 5.0),
                    (cx - 5.0, cy + 5.0),
                ],
                building_use="residential",
                osm_id=1000 + i,
            )
        )
    layout = OrtsnetzLayout.from_footprints(fps, substation_xy=(0.0, 0.0))
    layout.add_pen_cable(start="substation", end_xy=(110.0, 0.0))
    layout.connect_buildings()
    w = layout.to_world(seed=0)
    res = w.solve(_engine(backend="image", frequencies=[50.0]))
    out = _result_quantities(res, point_sources=False)
    out["world/json"] = _world_json(w)
    return out


# ---------------------------------------------------------------------
# fits, export, safety, references
# ---------------------------------------------------------------------


def _synthetic_response(f: np.ndarray) -> np.ndarray:
    s = 2j * np.pi * f
    p = -2.0 * np.pi * 300.0
    return 2.0 + 0.004 * s + 900.0 / (s - p)


@_case
def case_fit_vector_fit_synthetic() -> Quantities:
    f = np.array([0.0, 10.0, 30.0, 50.0, 100.0, 200.0, 400.0, 700.0, 1000.0])
    fit = gf.vector_fit(f, _synthetic_response(f), n_poles=2, include_L_inf=True)
    grid = np.linspace(0.0, 1000.0, 21)
    return {
        "eval": np.asarray(fit.evaluate(grid), dtype=complex),
        "rms": np.asarray(float(fit.rms_error)),
    }


def _rho_sweep_results() -> tuple[list[gf.FieldResult], list[float]]:
    rhos = [30.0, 100.0, 300.0]
    results = []
    for rho in rhos:
        w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=rho))
        gf.create_electrode(w, "rod", name="s", position=(0.0, 0.0, 0.5), length=3.0)
        gf.create_electrode(w, "rod", name="h", position=(40.0, 0.0, 0.5), length=3.0)
        gf.create_conductor(
            w,
            name="pen",
            start="s",
            end="h",
            conductor_type="pen",
            wire_radius=0.006,
            resistivity=2.82e-8,
            cross_section=95e-6,
            discretize_segment_length=10.0,
            coupling_to_soil="isolated",
            inductance_model="neumann",
        )
        gf.create_source(w, attached_to="s", magnitude=1.0)
        eng = _engine(
            backend="image",
            frequencies=[0.0, 50.0, 250.0, 1000.0],
            segment_length=0.5,
            earth_inductive_model="perfect_mirror",
        )
        results.append(w.solve(eng))
    return results, rhos


@_case
def case_fit_rho_f_standard_sweep() -> Quantities:
    results, rhos = _rho_sweep_results()
    fit = gf.rho_f_standard_from_results(results, rhos, "s")
    rho_grid = np.array([30.0, 100.0, 300.0, 30.0, 300.0])
    f_grid = np.array([0.0, 50.0, 250.0, 1000.0, 1000.0])
    out: Quantities = {
        f"coef/{k}": np.asarray(float(getattr(fit, k)))
        for k in ("k1", "k2", "k3", "k4", "k5")
    }
    out |= {
        "eval": np.asarray(fit.evaluate(rho_grid, f_grid), dtype=complex),
        "samples/Z": np.array(
            [res.cluster_impedance("s") for res in results], dtype=complex
        ),
    }
    spec = gf.BusTypeSpec.from_dict(
        gf.to_bustype_dict(
            fit, name="golden_rho_f", system_type="TN", voltage_level=0.4
        )
    )
    f_eval = [0.0, 50.0, 250.0, 1000.0]
    out["bustype/eval"] = np.array(
        [gf.evaluate_spec(spec, f_eval, rho) for rho in (30.0, 100.0, 300.0)],
        dtype=complex,
    )
    return out


@_case
def case_safety_en50522_curve() -> Quantities:
    t = np.concatenate(
        [np.geomspace(0.05, 10.0, 25), [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]]
    )
    return {
        "U_Tp": np.array([gf.permissible_touch_voltage_en50522(float(x)) for x in t])
    }


@_case
def case_references_dwight_ieee80() -> Quantities:
    from groundfield.references import dwight1936 as d
    from groundfield.references.ieee80 import grid_resistance_sverak

    vals = [
        d.rod(100.0, 3.0, 0.01),
        d.two_rods_far(100.0, 3.0, 0.01, 10.0),
        d.two_rods_close(100.0, 3.0, 0.01, 1.0),
        d.horizontal_wire(100.0, 10.0, 0.005, 0.5),
        d.right_angle_wire(100.0, 10.0, 0.005, 0.5),
        d.n_point_star(100.0, 5.0, 0.005, 0.5, 4),
        d.buried_ring(100.0, 6.0, 0.01, 0.8),
        d.horizontal_strip(100.0, 10.0, 0.04, 0.004, 0.5),
        d.horizontal_round_plate(100.0, 0.5, 1.0),
        d.vertical_round_plate(100.0, 0.5, 1.0),
        d.hemisphere(100.0, 1.0),
        grid_resistance_sverak(
            rho=100.0, total_conductor_length=200.0, area=400.0, depth=0.5
        ),
    ]
    return {"values": np.array(vals, dtype=float)}


# ---------------------------------------------------------------------
# defaults: record every user-visible change of a default
# ---------------------------------------------------------------------


@_case
def case_defaults_rod_hom() -> Quantities:
    w = gf.create_world(soil=gf.HomogeneousSoil(**_H100))
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_source(w, attached_to="rod")
    res = w.solve(gf.create_engine())
    return _result_quantities(
        res, probes=_surface_line(0.0, 10.0, 6), cluster_of=("rod",)
    )


@_case
def case_defaults_two_layer_ring_auto() -> Quantities:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=200.0, rho_2=50.0, h_1=3.0))
    gf.create_electrode(w, "ring", name="ring", center=(0.0, 0.0, 0.7), radius=5.0)
    gf.create_source(w, attached_to="ring")
    res = w.solve(gf.create_engine())
    return _result_quantities(
        res, probes=_surface_line(0.0, 20.0, 6), cluster_of=("ring",)
    )


@_case
def case_defaults_mixed_cluster_inductive() -> Quantities:
    w, _ = _galvanic_pair("sommerfeld")
    res = w.solve(gf.create_engine(frequencies=[50.0, 1000.0]))
    return _result_quantities(res, cluster_of=("g1",), point_sources=False)


# ---------------------------------------------------------------------
# engine family, review pass 10 (stream eng1; stored by the coordinator)
# ---------------------------------------------------------------------


@_case
def case_mom_two_port_passive_ring_two_layer() -> Quantities:
    """10.29: a source-free ring floats (zero net current) next to a rod."""
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="A", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_electrode(w, "ring", name="B", center=(4.0, 0.0, 0.8), radius=1.5)
    gf.create_source(w, attached_to="A", magnitude=1.0)
    res = w.solve(_engine(backend="mom", frequencies=[50.0], segment_length=0.5))
    return _result_quantities(res, cluster_of=("A",))


@_case
def case_cim_grid_unequal_segments() -> Quantities:
    """10.30: length-weighted read-back on 0.6 m and 0.667 m segments."""
    soil = gf.TwoLayerSoil(rho_1=300.0, rho_2=50.0, h_1=6.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w,
        "grid_mesh",
        name="grid",
        corner=(-3.0, -2.0, 0.6),
        size=(6.0, 4.0),
        n_x=2,
        n_y=2,
    )
    gf.create_source(w, attached_to="grid", magnitude=1.0)
    res = w.solve(_engine(backend="cim", frequencies=[50.0], segment_length=0.7))
    return _result_quantities(res, cluster_of=("grid",))


@_case
def case_fem_rod_two_layer_deep() -> Quantities:
    """10.49: radial resolution kept for a 30 m top layer."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=30.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    res = w.solve(_engine(backend="fem", frequencies=[50.0], segment_length=0.5))
    return _result_quantities(res, cluster_of=("rod",), point_sources=False)


@_case
def case_mom_sommerfeld_duplicated_stack() -> Quantities:
    """10.48: [100, 400, 400] over [2, 3] is solved as two layers."""
    soil = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(resistivity=100.0, thickness=2.0),
            gf.SoilLayer(resistivity=400.0, thickness=3.0),
            gf.SoilLayer(resistivity=400.0, thickness=None),
        ]
    )
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    res = w.solve(
        _engine(backend="mom_sommerfeld", frequencies=[50.0], segment_length=0.5)
    )
    return _result_quantities(res, cluster_of=("rod",))
