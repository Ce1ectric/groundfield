"""Tests for the Complex-Image-Method backend (``cim``).

Checks that the matrix-pencil fit of Γ_1(λ) collapses to zero on
homogeneous soil, that on two-layer soils ``cim`` agrees with the
closed-form Tagg/Sunde series within 2 %, and that ``cim`` is
self-consistent across simple multi-layer reductions (the
``ρ_2 = ρ_1 = ρ_3`` collapse should give the homogeneous result).
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf
from groundfield.references import dwight1936 as dw
from groundfield.solver._layered import LayerStack
from groundfield.solver.cim import fit_complex_images

SEG = 0.05


def _world(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_cim_homogeneous_collapses_to_image() -> None:
    """For HomogeneousSoil the CIM fit produces zero images."""
    stack = LayerStack(rhos=np.array([100.0]), h=np.zeros(0))
    fit = fit_complex_images(stack)
    assert fit.a.size == 0
    assert fit.beta.size == 0


def test_cim_homogeneous_matches_image() -> None:
    """``cim`` on homogeneous soil agrees with ``image`` to within 2 %."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng_image = gf.create_engine(backend="image", segment_length=SEG)
    eng_cim = gf.create_engine(backend="cim", segment_length=SEG)
    Z_image = eng_image.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_cim = eng_cim.solve(_world(soil)).cluster_impedance("g1")[0].real
    assert abs(Z_cim - Z_image) / Z_image < 0.02


@pytest.mark.parametrize("rho_2", [50.0, 200.0, 500.0])
def test_cim_two_layer_matches_image_2layer(rho_2: float) -> None:
    """For 2-layer soil ``cim`` and ``image_2layer`` must agree within 5 %."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=rho_2, h_1=2.0)
    eng_il2 = gf.create_engine(backend="image_2layer", segment_length=SEG)
    eng_cim = gf.create_engine(backend="cim", segment_length=SEG)
    Z_il2 = eng_il2.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_cim = eng_cim.solve(_world(soil)).cluster_impedance("g1")[0].real
    assert abs(Z_cim - Z_il2) / Z_il2 < 0.05, (
        f"rho_2={rho_2}: cim={Z_cim:.3f} vs il2={Z_il2:.3f}"
    )


def test_cim_three_layer_rejected() -> None:
    """n_layers >= 3 is rejected loudly (audit 2026-07-08, WP-E).

    The historic n >= 3 complex-image kernel implemented an
    incomplete Green's function (single image family, no multiple
    reflections); until a complete kernel exists the backend raises
    and points at mom_sommerfeld / fem.

    The stack has three *distinct* resistivities: an equal-resistivity
    stack is homogeneous after canonicalisation (10.48) and is solved,
    see ``tests/test_layer_stack.py``.
    """
    multilayer = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(resistivity=100.0, thickness=2.0),
            gf.SoilLayer(resistivity=300.0, thickness=2.0),
            gf.SoilLayer(resistivity=50.0, thickness=None),
        ]
    )
    eng = gf.create_engine(backend="cim", segment_length=SEG)
    with pytest.raises(NotImplementedError, match="mom_sommerfeld"):
        eng.solve(_world(multilayer))


def test_cim_rod_matches_dwight() -> None:
    """A rod in homogeneous soil matches Dwight to within 5 %."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng = gf.create_engine(backend="cim", segment_length=SEG)
    Z = eng.solve(_world(soil)).cluster_impedance("g1")[0].real
    R_dw = dw.rod(rho=100.0, length=1.5, radius=0.005)
    assert abs(Z - R_dw) / R_dw < 0.05


@pytest.mark.finding("10.76")
def test_cim_metadata() -> None:
    """``cim`` is a deprecated alias: no fit diagnostics, the image result.

    Up to 0.15.0 the result carried ``cim_n_images`` / ``cim_rms`` /
    ``reduces_to`` describing a fit no path used. Since 10.76 the name
    runs the image auto-dispatch; ``n >= 3`` still raises and points at
    ``mom_sommerfeld``.
    """
    eng = gf.create_engine(backend="cim", segment_length=SEG)

    soil_2 = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
    with pytest.warns(DeprecationWarning, match="2027.5"):
        res_2 = eng.solve(_world(soil_2))
    assert res_2.metadata["backend"] == "cim"
    assert "cim_n_images" not in res_2.metadata
    assert "cim_rms" not in res_2.metadata
    il2 = gf.create_engine(backend="image_2layer", segment_length=SEG)
    assert res_2.electrode_potentials == il2.solve(_world(soil_2)).electrode_potentials

    soil_3 = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=100.0, h=2.0),
            gf.SoilLayer(rho=400.0, h=2.0),
            gf.SoilLayer(rho=50.0, h=None),
        ]
    )
    # n >= 3 is rejected since the WP-E audit fix.
    with pytest.raises(NotImplementedError, match="mom_sommerfeld"):
        eng.solve(_world(soil_3))


# ---------------------------------------------------------------------
# 10.30 — cim reports the length-weighted node potential like the image
# family (the F10 fix of 0.15.0 had missed this backend)
# ---------------------------------------------------------------------


def _rod_world(soil, *, z0: float, length: float, name: str = "g1") -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name=name, position=(0.0, 0.0, z0), length=length)
    gf.create_source(w, attached_to=name, magnitude=1.0)
    return w


@pytest.mark.finding("10.30")
@pytest.mark.parametrize(
    ("soil", "z0", "length", "ds"),
    [
        # 3 m rod split at h_1 = 2 m: 5 x 0.4 m + 3 x 0.333 m segments
        pytest.param(
            gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0),
            0.0,
            3.0,
            0.4,
        ),
        # the pass-9 F10 shape with a 5 cm remainder below h_1 = 5 m (a
        # remainder shorter than the wire diameter is merged since 10.32)
        pytest.param(
            gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=5.0),
            0.5,
            4.55,
            0.5,
        ),
        # upper layer only: a grid mesh with 3 m x 2 m cells gives
        # 0.6 m and 0.667 m segments inside one electrode
        (gf.TwoLayerSoil(rho_1=300.0, rho_2=50.0, h_1=6.0), None, None, 0.7),
    ],
)
def test_cim_is_bit_identical_to_image_2layer_for_unequal_segments(
    soil, z0, length, ds
) -> None:
    """v0.15.0: +4.10 % on the first case (71.060210 vs 68.261634 Ω)."""

    def world() -> gf.World:
        if z0 is not None:
            return _rod_world(soil, z0=z0, length=length)
        w = gf.create_world(soil=soil)
        gf.create_electrode(
            w,
            "grid_mesh",
            name="g1",
            corner=(-3.0, -2.0, 0.6),
            size=(6.0, 4.0),
            n_x=2,
            n_y=2,
        )
        gf.create_source(w, attached_to="g1", magnitude=1.0)
        return w

    res_cim = gf.create_engine(backend="cim", segment_length=ds, kernel="point").solve(
        world()
    )
    res_il2 = gf.create_engine(
        kernel="point", backend="image_2layer", segment_length=ds
    ).solve(world())
    lengths = {ps.length for ps in res_cim.point_sources}
    assert len({round(x, 9) for x in lengths}) > 1, "case must have unequal segments"
    assert res_cim.electrode_potentials == res_il2.electrode_potentials
    assert res_cim.electrode_currents == res_il2.electrode_currents
    assert res_cim.cluster_impedance("g1") == res_il2.cluster_impedance("g1")
    # 10.76: the alias runs image_2layer itself.
    ran = res_cim.metadata.get("dispatched_to", res_cim.metadata["effective_backend"])
    assert ran == "image_2layer"


@pytest.mark.finding("10.30")
@pytest.mark.parametrize(
    ("h_1", "port_a", "tol"),
    [
        # grid mesh (0.6 m and 0.667 m segments) inside the upper layer:
        # closed-form kernel, round-off level
        (5.0, "grid", 1e-10),
        # rod through h_1 = 2 m (split into unequal segments): cross-layer
        # quadrature kernel
        (2.0, "rod", 1e-7),
    ],
)
def test_cim_two_port_transfer_impedance_is_reciprocal(h_1, port_a, tol) -> None:
    """Transfer impedances from two solves agree.

    Port A carries segments of unequal length, for which the unweighted
    read-back of v0.15.0 broke reciprocity (4.0e-2 on the cross-layer
    case, R2-03).
    """
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=h_1)

    def world(src: str) -> gf.World:
        w = gf.create_world(soil=soil)
        if port_a == "grid":
            gf.create_electrode(
                w,
                "grid_mesh",
                name="A",
                corner=(-3.0, -2.0, 0.6),
                size=(6.0, 4.0),
                n_x=2,
                n_y=2,
            )
        else:
            gf.create_electrode(
                w, "rod", name="A", position=(0.0, 0.0, 0.0), length=3.0
            )
        gf.create_electrode(w, "ring", name="B", center=(6.0, 0.0, 0.8), radius=1.5)
        gf.create_source(w, attached_to=src, magnitude=1.0)
        return w

    eng = gf.create_engine(
        backend="cim", segment_length=0.7 if port_a == "grid" else 0.4
    )
    r_a, r_b = eng.solve(world("A")), eng.solve(world("B"))
    z_ba = r_a.electrode_potentials["B"][0] / r_a.electrode_currents["A"][0]
    z_ab = r_b.electrode_potentials["A"][0] / r_b.electrode_currents["B"][0]
    assert abs(z_ab - z_ba) / abs(z_ba) < tol


# ---------------------------------------------------------------------
# 10.49 — fit_complex_images: "converged" bounds the maximum error
# ---------------------------------------------------------------------


@pytest.mark.finding("10.49")
@pytest.mark.parametrize(
    ("rhos", "h"),
    [
        ([956.0, 43.9, 1081.9], [7.173, 0.695]),
        ([2078.1, 20.2, 404.8], [0.077, 8.813]),
        ([13.5, 60.4, 7.0, 273.6, 5.8], [0.109, 6.296, 1.002, 3.059]),
    ],
)
def test_fit_converged_flag_bounds_the_maximum_error(rhos, h) -> None:
    """v0.15.0 reported ``converged=True`` for these stacks although the
    model missed Γ_1 by 2.4e-3 to 2.6e-3 between the fitting samples
    (RMS 1e-11 on the samples, 9.4e-4 on the verification grid, R2-16).
    """
    import warnings

    from groundfield.solver._layered import reflection_gamma

    stack = LayerStack(rhos=np.array(rhos), h=np.array(h))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_complex_images(stack, rms_tol=1e-3)
    h_min = float(np.min(stack.h[1:]))
    lam = np.concatenate([[0.0], np.logspace(-5, np.log10(1e3 / h_min), 4000)])
    model = (np.exp(-2.0 * lam[:, None] * fit.beta[None, :]) @ fit.a).real
    max_err = float(np.max(np.abs(model - reflection_gamma(stack, lam))))
    assert max_err > 1e-3  # the stacks are genuinely hard for the pencil
    assert fit.converged is False
    assert fit.max_error > 1e-3


@pytest.mark.finding("10.49")
def test_fit_failure_is_reported_with_the_maximum_error() -> None:
    from groundfield.solver.cim import ComplexImageFitWarning

    stack = LayerStack(rhos=np.array([956.0, 43.9, 1081.9]), h=np.array([7.173, 0.695]))
    with pytest.warns(ComplexImageFitWarning, match="maximum"):
        fit_complex_images(stack)


@pytest.mark.finding("10.49")
def test_fit_of_an_easy_stack_still_converges() -> None:
    stack = LayerStack(rhos=np.array([100.0, 400.0, 50.0]), h=np.array([2.0, 3.0]))
    fit = fit_complex_images(stack)
    assert fit.converged is True
    assert fit.max_error <= 1e-3
