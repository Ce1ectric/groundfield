"""Potential plots: component selection and axis labels (10.73).

Up to v0.15.0 every potential plot drew Re φ and labelled it
"Potential φ in V". With phase-shifted sources (a 180° return source) or
inductive coupling at 1 kHz Re φ differs from |φ|; the plots now take
``component="real" | "imag" | "abs" | "phase"`` and label what they draw.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import groundfield as gf  # noqa: E402

_LABELS = {
    "real": "Re φ in V",
    "imag": "Im φ in V",
    "abs": "|φ| in V",
    "phase": "arg φ in °",
}


@pytest.fixture(scope="module")
def rotated() -> tuple[gf.World, gf.FieldResult]:
    """Rod fed with 10 A at 90°: the potential is purely imaginary."""
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_source(w, attached_to="g1", magnitude=10.0, phase_deg=90.0)
    eng = gf.create_engine(backend="image", segment_length=0.5, frequencies=[50.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return w, eng.solve(w)


def _phi_line(result, x: np.ndarray) -> np.ndarray:
    pts = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    return result.potential(pts)


@pytest.mark.finding("10.73")
@pytest.mark.parametrize("component", ["real", "imag", "abs", "phase"])
def test_profile_plots_and_labels_the_requested_component(rotated, component) -> None:
    _, res = rotated
    fig = gf.plot_potential_profile(
        res, start=(1.0, 0.0, 0.0), distance=10.0, n=11, component=component
    )
    ax = fig.axes[0]
    phi = _phi_line(res, np.linspace(1.0, 11.0, 11))
    expected = {
        "real": phi.real,
        "imag": phi.imag,
        "abs": np.abs(phi),
        "phase": np.degrees(np.angle(phi)),
    }[component]
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), expected, rtol=1e-12, atol=1e-12
    )
    assert ax.get_ylabel() == _LABELS[component]
    plt.close(fig)


@pytest.mark.finding("10.73")
def test_default_component_is_the_real_part_with_an_honest_label(rotated) -> None:
    w, res = rotated
    fig = gf.plot_potential_radial(
        res, around="g1", world=w, r_max=5.0, n=5, depths=(0.0,)
    )
    assert fig.axes[0].get_ylabel() == "Re φ in V"
    plt.close(fig)
    fig = gf.plot_potential_contour(res, extent=(-5, 5, -5, 5), n=12)
    assert fig.axes[-1].get_ylabel() == "Re φ in V"  # colour bar
    plt.close(fig)


@pytest.mark.finding("10.73")
def test_surface_plot_component_and_log_label(rotated) -> None:
    w, res = rotated
    fig = gf.plot_surface_potential(res, w, n=12, padding_m=5.0, component="abs")
    assert fig.axes[-1].get_ylabel() == "|φ| in V"
    plt.close(fig)
    fig = gf.plot_surface_potential(res, w, n=12, padding_m=5.0, log=True)
    assert fig.axes[-1].get_ylabel() == "|Re φ| in V (log scale)"
    plt.close(fig)
    with pytest.raises(ValueError, match="phase"):
        gf.plot_surface_potential(res, w, n=12, component="phase", log=True)
    with pytest.raises(ValueError, match="component"):
        gf.plot_potential_contour(
            res, extent=(-5, 5, -5, 5), n=12, component="magnitude"
        )
