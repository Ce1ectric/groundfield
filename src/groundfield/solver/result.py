"""Result object of a simulation run.

A :class:`FieldResult` is the unified return value of
:meth:`Engine.solve` and :meth:`World.solve`. It contains:

- ``electrode_potentials`` / ``electrode_currents`` — node values per
  frequency, used to derive input and transfer impedances.
- ``point_sources`` — the discretised current distribution. The
  post-processing layer can use these to evaluate the potential at
  arbitrary field points (contour plots, profiles, touch and step
  voltages).
- ``soil_resistivity`` and ``soil`` — provide the parameters needed to
  evaluate the appropriate Green's function (homogeneous or 2-layer)
  inside :meth:`potential`.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    TwoLayerSoil,
)

__all__ = ["FieldResult", "PointSource"]


# Local discriminated union of all soil types — kept here to avoid a
# circular import.
_SoilUnion = Union[HomogeneousSoil, TwoLayerSoil, MultiLayerSoil]


# 3-D coordinate
_Point3D = tuple[float, float, float]


class PointSource(BaseModel):
    """A discretised point current source (segment midpoint).

    Filled by the backend; not instantiated directly by the user.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    position: _Point3D = Field(..., description="(x, y, z) midpoint in m.")
    current: list[complex] = Field(
        ..., description="Current per frequency in A (complex phasor)."
    )
    electrode_name: str = Field(..., description="Owning electrode.")
    length: float = Field(..., gt=0.0, description="Represented length in m.")


class FieldResult(BaseModel):
    """Result of a field computation."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    backend: str
    frequencies: list[float]
    electrode_potentials: dict[str, list[complex]] = Field(default_factory=dict)
    electrode_currents: dict[str, list[complex]] = Field(default_factory=dict)
    point_sources: list[PointSource] = Field(default_factory=list)
    soil_resistivity: float | None = Field(
        default=None,
        description=(
            "Effective resistivity used to evaluate the image-charge sum. "
            "For layered soils this stores $\\rho_1$; ``soil`` then "
            "carries the full layered model."
        ),
    )
    soil: _SoilUnion | None = Field(
        default=None,
        description=(
            "Full soil model used in the solution. :meth:`potential` reads "
            "this to pick the correct Green's-function kernel "
            "(homogeneous vs. 2-layer)."
        ),
    )
    clusters: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Cluster mapping: ``electrode_name -> sorted list of all "
            "electrodes that share the same galvanic cluster (including "
            "the electrode itself). Stand-alone electrodes map to "
            "``[name]``."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def grounding_impedance(self, electrode_name: str) -> list[complex]:
        """Input impedance $Z(f) = U/I$ of an electrode.

        Parameters
        ----------
        electrode_name : str
            Name of the electrode whose input impedance is requested.
            Must be present in both :attr:`electrode_potentials` and
            :attr:`electrode_currents`.

        Returns
        -------
        list[complex]
            Complex impedance per frequency in ohms, one entry per
            entry in :attr:`frequencies`. Frequencies for which the
            electrode current is exactly zero return ``nan``.

        Raises
        ------
        KeyError
            If ``electrode_name`` is unknown to the result.

        Notes
        -----
        For galvanically connected electrodes (cluster with more than
        one member) this quantity is the cluster potential divided by
        the electrode's share of the total current. The physically
        meaningful quantity is the **cluster impedance** (see
        :meth:`cluster_impedance`).
        """
        if electrode_name not in self.electrode_potentials:
            raise KeyError(f"No potential data for '{electrode_name}'.")
        if electrode_name not in self.electrode_currents:
            raise KeyError(f"No current data for '{electrode_name}'.")
        u = self.electrode_potentials[electrode_name]
        i = self.electrode_currents[electrode_name]
        return [
            (uk / ik) if ik != 0 else complex("nan") for uk, ik in zip(u, i)
        ]

    def cluster_impedance(self, electrode_name: str) -> list[complex]:
        """Grounding impedance of the galvanic cluster containing an electrode.

        Definition: $Z_{\\text{cluster}}(f) =
        \\varphi_{\\text{cluster}}/\\sum_{e \\in c} I_e$. For a
        stand-alone electrode this is identical to
        :meth:`grounding_impedance`. For connected electrodes it
        corresponds to the parallel combination of the individual
        grounding admittances.

        Parameters
        ----------
        electrode_name : str
            Name of any electrode in the target cluster. The cluster
            members are looked up in :attr:`clusters`; if the
            electrode does not appear there it is treated as a
            stand-alone cluster ``[electrode_name]``.

        Returns
        -------
        list[complex]
            Complex cluster impedance per frequency in ohms, one entry
            per entry in :attr:`frequencies`. Frequencies for which
            the summed cluster current is exactly zero return ``nan``.

        Raises
        ------
        KeyError
            If the resolved cluster is empty.
        """
        members = self.clusters.get(electrode_name, [electrode_name])
        if not members:
            raise KeyError(f"No cluster for '{electrode_name}'.")
        u = self.electrode_potentials[members[0]]
        i_sum = [
            sum(self.electrode_currents[m][k] for m in members)
            for k in range(len(u))
        ]
        return [
            (uk / ik) if ik != 0 else complex("nan")
            for uk, ik in zip(u, i_sum)
        ]

    # ------------------------------------------------------------------
    # Potential evaluation at arbitrary field points
    # ------------------------------------------------------------------

    def potential(
        self,
        points: np.ndarray,
        frequency_index: int = 0,
        min_distance: float = 1e-3,
    ) -> np.ndarray:
        """Evaluate the potential at field points (image-charge sum).

        Picks the appropriate Green's-function kernel automatically:

        - **homogeneous** (:class:`HomogeneousSoil` or no ``self.soil``
          set but ``soil_resistivity`` available): classic image-charge
          sum $1/r + 1/r_{\\text{img}}$.
        - **2-layer** (:class:`TwoLayerSoil`, or :class:`MultiLayerSoil`
          with exactly two layers): Tagg/Sunde series with adaptive
          truncation (tolerance $10^{-6}$, at most 100 terms).
        - **1-layer** (:class:`MultiLayerSoil` with a single layer):
          degenerate case, dispatched to the homogeneous kernel.

        Parameters
        ----------
        points
            Field points, array of shape ``(M, 3)`` in metres.
        frequency_index
            Index into :attr:`frequencies`. Default 0.
        min_distance
            Numerical cutoff for 1/r singularities, in metres.

        Returns
        -------
        phi : np.ndarray, shape (M,)
            Complex potential in V.

        Raises
        ------
        NotImplementedError
            If ``self.soil`` is a :class:`MultiLayerSoil` with three or
            more layers. The post-solve potential path does not yet
            carry an n-layer Green's-function kernel. The solve itself
            is still correct via the ``cim`` / ``mom_sommerfeld`` /
            ``bem`` backends — only the explicit
            :meth:`potential` evaluation is unavailable for n ≥ 3.
            Cluster impedances, electrode currents and potentials
            stored in :attr:`electrode_potentials` /
            :attr:`cluster_impedance` remain accessible.

        Notes
        -----
        Prior to 0.2.0 a :class:`MultiLayerSoil` with n ≥ 3 silently
        fell through to the homogeneous kernel, returning incorrect
        potentials, profiles, touch / step voltages and VTK exports.
        The error is now raised explicitly so the regime is visible
        rather than silent.
        """
        if not self.point_sources:
            raise RuntimeError(
                "FieldResult has no point_sources — the backend did not "
                "populate the post-processing data (stub?)."
            )

        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.shape[1] != 3:
            raise ValueError(
                f"points must have shape (M, 3), got {pts.shape}."
            )

        sources = np.array([ps.position for ps in self.point_sources])
        currents = np.array(
            [ps.current[frequency_index] for ps in self.point_sources],
            dtype=complex,
        )

        # Dispatch to the kernel that matches the effective number of
        # soil layers. A ``MultiLayerSoil`` with one or two layers is
        # mathematically equivalent to the homogeneous / two-layer
        # case and is cast on the fly. For n ≥ 3 there is no
        # closed-form image-charge series (Γ_1(λ) is no longer
        # constant in λ — see ``solver/image_nlayer.py``), so the
        # call is rejected explicitly rather than silently falling
        # back to the homogeneous kernel.
        if isinstance(self.soil, TwoLayerSoil):
            return self._potential_two_layer(
                pts, sources, currents, self.soil, min_distance
            )
        if isinstance(self.soil, MultiLayerSoil):
            n = len(self.soil.layers)
            if n == 1:
                # Degenerate 1-layer MultiLayerSoil: use the single
                # layer's resistivity as the homogeneous case.
                return self._potential_homogeneous(
                    pts,
                    sources,
                    currents,
                    min_distance,
                    rho_override=float(self.soil.layers[0].resistivity),
                )
            if n == 2:
                # Two-layer MultiLayerSoil: cast to TwoLayerSoil and
                # reuse the Tagg/Sunde kernel.
                two_layer = TwoLayerSoil(
                    rho_1=float(self.soil.layers[0].resistivity),
                    rho_2=float(self.soil.layers[1].resistivity),
                    h_1=float(self.soil.layers[0].thickness),
                )
                return self._potential_two_layer(
                    pts, sources, currents, two_layer, min_distance
                )
            raise NotImplementedError(
                f"FieldResult.potential: MultiLayerSoil with n={n} ≥ 3 "
                "layers is not supported on the post-solve potential "
                "path. The solve itself is correct (use the 'cim', "
                "'mom_sommerfeld', or 'bem' backend), but a closed-form "
                "n-layer Green's-function kernel is not yet wired into "
                "FieldResult.potential. Access cluster impedances and "
                "electrode potentials directly via "
                "result.electrode_potentials / result.cluster_impedance "
                "instead, or reduce the soil model to "
                "TwoLayerSoil / HomogeneousSoil for explicit field "
                "evaluation."
            )
        return self._potential_homogeneous(pts, sources, currents, min_distance)

    # -- private helpers per Green's-function kind --------------------

    def _potential_homogeneous(
        self,
        pts: np.ndarray,
        sources: np.ndarray,
        currents: np.ndarray,
        min_distance: float,
        rho_override: float | None = None,
    ) -> np.ndarray:
        rho = (
            float(rho_override)
            if rho_override is not None
            else self.soil_resistivity
        )
        if rho is None:
            raise RuntimeError(
                "FieldResult.soil_resistivity is not set — homogeneous "
                "image-charge evaluation requires it."
            )
        image_sources = sources.copy()
        image_sources[:, 2] = -image_sources[:, 2]
        diff_real = pts[:, None, :] - sources[None, :, :]
        diff_image = pts[:, None, :] - image_sources[None, :, :]
        r_real = np.linalg.norm(diff_real, axis=2)
        r_image = np.linalg.norm(diff_image, axis=2)
        np.maximum(r_real, min_distance, out=r_real)
        np.maximum(r_image, min_distance, out=r_image)
        kernel = (1.0 / r_real) + (1.0 / r_image)
        return (rho / (4.0 * np.pi)) * (
            kernel @ currents.real + 1j * (kernel @ currents.imag)
        )

    def _potential_two_layer(
        self,
        pts: np.ndarray,
        sources: np.ndarray,
        currents: np.ndarray,
        soil: TwoLayerSoil,
        min_distance: float,
        max_terms: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Tagg/Sunde series for 2-layer soil."""
        K = soil.reflection_coefficient
        h_1 = soil.h_1
        rho_1 = soil.rho_1

        diff_xy = pts[:, None, 0:2] - sources[None, :, 0:2]
        delta_sq = np.einsum("mnk,mnk->mn", diff_xy, diff_xy)
        z_field = pts[:, 2:3]
        z_src = sources[None, :, 2]

        def _series_for(real_currents: np.ndarray) -> np.ndarray:
            """Core series for a real-valued current distribution."""
            phi = np.zeros(pts.shape[0], dtype=float)
            # n = 0 (two images, weight 1)
            for sign_zs in (+1, -1):
                z_img = sign_zs * z_src
                r = np.sqrt(delta_sq + (z_field - z_img) ** 2)
                np.maximum(r, min_distance, out=r)
                phi += (1.0 / r) @ real_currents
            # n = 1, 2, ...
            abs_K = abs(K)
            for n in range(1, max_terms + 1):
                K_n = K ** n
                acc = np.zeros(pts.shape[0], dtype=float)
                for sign_n in (+1, -1):
                    for sign_zs in (+1, -1):
                        z_img = sign_n * 2.0 * n * h_1 + sign_zs * z_src
                        r = np.sqrt(delta_sq + (z_field - z_img) ** 2)
                        np.maximum(r, min_distance, out=r)
                        acc += (1.0 / r) @ real_currents
                phi += K_n * acc
                if abs_K ** n < tol:
                    break
            return phi

        phi_re = _series_for(currents.real)
        phi_im = _series_for(currents.imag)
        return (rho_1 / (4.0 * np.pi)) * (phi_re + 1j * phi_im)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Compact one-line description of the result.

        Returns
        -------
        str
            Human-readable summary listing the backend name, the
            number of frequencies, the number of electrodes for which
            potentials were stored and the number of discretised
            point sources. Intended for logging and notebook output;
            the format is informational and not stable across
            versions.
        """
        return (
            f"FieldResult(backend='{self.backend}', "
            f"n_freq={len(self.frequencies)}, "
            f"n_electrodes={len(self.electrode_potentials)}, "
            f"n_segments={len(self.point_sources)})"
        )
