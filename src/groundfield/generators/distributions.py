"""Probability distributions for stochastic generator parameters.

This module defines the distribution catalogue used by
:mod:`groundfield.generators`. Every numerical field of a
:class:`~groundfield.generators.base.GeneratorConfig` may be either a
fixed value or an instance of one of the classes here. The user
exposes the parameter as

>>> from groundfield.generators.distributions import (
...     Constant, Discrete, Normal, LogNormal, Uniform, Weibull,
...     Categorical,
... )
>>> rho_1 = LogNormal(mu=5.0, sigma=0.7)   # log-normal soil resistivity
>>> n_efh = Discrete(values=[5, 10, 30, 80, 200])

A subsequent ``cfg.sample(rng)`` call resolves these into concrete
values that the generator can use to build a :class:`World`.

Mathematical conventions
------------------------
Each subclass documents the parameterisation used. We follow the
NumPy / SciPy convention rather than e.g. the MATLAB one:

- ``Normal(mean, std)`` — :math:`\\mathcal{N}(\\mu = \\text{mean}, \\sigma^2 = \\text{std}^2)`.
  Optional truncation via rejection sampling.
- ``LogNormal(mu, sigma)`` — :math:`X = e^{Y}` with
  :math:`Y \\sim \\mathcal{N}(\\mu, \\sigma^2)`. Class method
  :py:meth:`LogNormal.from_moments` accepts the *physical* mean and
  std and back-solves for $\\mu, \\sigma$.
- ``Weibull(shape, scale)`` — pdf
  :math:`f(x) = (k/\\lambda)\\,(x/\\lambda)^{k-1}\\,e^{-(x/\\lambda)^k}`,
  shape :math:`k > 0`, scale :math:`\\lambda > 0`.
- ``Uniform(low, high)`` — uniform on :math:`[\\text{low}, \\text{high})`.
- ``Discrete(values, weights?)`` — finite numeric set.
- ``Categorical(values, weights?)`` — finite string set (electrode
  kinds, soil types, ...).

Reproducibility
---------------
Every ``.sample(rng)`` call takes a :class:`numpy.random.Generator` and
makes exactly the canonical number of draws documented in the class.
Combined with a fixed seed this gives bit-exact reproducibility of
Monte Carlo studies (see ADR-0009).

References
----------
- NumPy random API: https://numpy.org/doc/stable/reference/random/generator.html
- SciPy stats: https://docs.scipy.org/doc/scipy/reference/stats.html
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "Distribution",
    "Constant",
    "Uniform",
    "Normal",
    "LogNormal",
    "Weibull",
    "Discrete",
    "Categorical",
    "AnyDistribution",
]


class Distribution(BaseModel):
    """Abstract base for probability distributions.

    Subclasses must:

    1. set a literal ``kind`` field (used as the JSON discriminator),
    2. implement :meth:`sample` returning a single draw.

    The base class is intentionally a regular Pydantic model rather
    than an :class:`abc.ABC` so it composes cleanly with discriminated
    unions in :class:`pydantic.BaseModel` field annotations.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str

    def sample(self, rng: np.random.Generator) -> Any:  # pragma: no cover
        raise NotImplementedError(
            f"{type(self).__name__}.sample is abstract; use a concrete subclass."
        )


# ---------------------------------------------------------------------
# Continuous numerical distributions
# ---------------------------------------------------------------------


class Constant(Distribution):
    """Degenerate distribution returning ``value`` deterministically.

    Useful as a placeholder or in code paths that always expect a
    :class:`Distribution` instance.
    """

    kind: Literal["constant"] = "constant"
    value: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(self.value)


class Uniform(Distribution):
    """Continuous uniform on $[\\text{low}, \\text{high})$.

    The bounds are stored as Pydantic fields ``low`` and ``high``;
    the model validator enforces ``high > low``.
    """

    kind: Literal["uniform"] = "uniform"
    low: float
    high: float

    @model_validator(mode="after")
    def _check_bounds(self) -> "Uniform":
        if not (self.high > self.low):
            raise ValueError(
                f"Uniform: high must exceed low (got low={self.low}, "
                f"high={self.high})"
            )
        return self

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


class Normal(Distribution):
    """Normal distribution $\\mathcal{N}(\\mu, \\sigma^2)$ with optional truncation.

    Truncation is implemented via rejection sampling. A
    :class:`RuntimeError` is raised if more than ``max_attempts``
    rejections occur in a row, which protects against pathological
    bound configurations (e.g. ``truncate_low`` more than ten standard
    deviations away from the mean).
    """

    kind: Literal["normal"] = "normal"
    mean: float
    std: float = Field(gt=0.0)
    truncate_low: Optional[float] = None
    truncate_high: Optional[float] = None
    max_attempts: int = Field(default=1000, gt=0)

    @model_validator(mode="after")
    def _check_truncation(self) -> "Normal":
        lo, hi = self.truncate_low, self.truncate_high
        if lo is not None and hi is not None and not (hi > lo):
            raise ValueError(
                f"Normal: truncate_high must exceed truncate_low "
                f"(got low={lo}, high={hi})"
            )
        return self

    def sample(self, rng: np.random.Generator) -> float:
        for _ in range(self.max_attempts):
            x = float(rng.normal(self.mean, self.std))
            if self.truncate_low is not None and x < self.truncate_low:
                continue
            if self.truncate_high is not None and x > self.truncate_high:
                continue
            return x
        raise RuntimeError(
            f"Normal.sample: rejection sampling exhausted {self.max_attempts} "
            f"attempts; check truncation bounds versus (mean, std)."
        )


class LogNormal(Distribution):
    """Log-normal distribution.

    The default parameterisation matches NumPy's: ``mu`` and ``sigma``
    are the mean and standard deviation **of the underlying normal**
    $Y \\sim \\mathcal{N}(\\mu, \\sigma^2)$, and the log-normal sample
    is $X = e^{Y}$.

    Use :meth:`from_moments` to construct an instance from the
    *physical* mean and standard deviation of $X$ instead.
    """

    kind: Literal["lognormal"] = "lognormal"
    mu: float
    sigma: float = Field(gt=0.0)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.lognormal(self.mu, self.sigma))

    @classmethod
    def from_moments(cls, mean: float, std: float) -> "LogNormal":
        """Build a :class:`LogNormal` from the physical mean and std.

        Solves $\\mu = \\ln(\\text{mean}^2/\\sqrt{\\text{std}^2 + \\text{mean}^2})$
        and $\\sigma^2 = \\ln(1 + \\text{std}^2/\\text{mean}^2)$.

        Parameters
        ----------
        mean
            Target physical mean $\\mathbb{E}[X] > 0$.
        std
            Target physical standard deviation $\\sqrt{\\mathrm{Var}(X)} > 0$.
        """
        if mean <= 0.0:
            raise ValueError(f"LogNormal.from_moments: mean must be > 0 (got {mean}).")
        if std <= 0.0:
            raise ValueError(f"LogNormal.from_moments: std must be > 0 (got {std}).")
        var = std * std
        sigma_sq = math.log(1.0 + var / (mean * mean))
        mu = math.log(mean) - 0.5 * sigma_sq
        return cls(mu=mu, sigma=math.sqrt(sigma_sq))


class Weibull(Distribution):
    """Weibull distribution with shape $k$ and scale $\\lambda$.

    pdf $f(x) = (k/\\lambda)\\,(x/\\lambda)^{k-1}\\,e^{-(x/\\lambda)^k}$
    on $x \\ge 0$. NumPy's ``rng.weibull(k)`` returns a sample from the
    Weibull with scale 1; we multiply by ``scale`` to obtain the
    desired scale parameter.
    """

    kind: Literal["weibull"] = "weibull"
    shape: float = Field(gt=0.0, description="Shape parameter k > 0.")
    scale: float = Field(default=1.0, gt=0.0, description="Scale parameter lambda > 0.")

    def sample(self, rng: np.random.Generator) -> float:
        return float(self.scale * rng.weibull(self.shape))


# ---------------------------------------------------------------------
# Discrete / categorical distributions
# ---------------------------------------------------------------------


class Discrete(Distribution):
    """Discrete numerical distribution over a finite value set.

    Mirrors typical parameter axes such as
    $n_\\text{EFH} \\in \\{5, 10, 30, 80, 200\\}$. ``weights`` may be
    supplied to bias the choice; if omitted the values are sampled
    uniformly.
    """

    kind: Literal["discrete"] = "discrete"
    values: list[float] = Field(min_length=1)
    weights: Optional[list[float]] = None

    @model_validator(mode="after")
    def _check_weights(self) -> "Discrete":
        if self.weights is not None:
            if len(self.weights) != len(self.values):
                raise ValueError(
                    f"Discrete: weights length {len(self.weights)} does not "
                    f"match values length {len(self.values)}."
                )
            if any(w < 0 for w in self.weights):
                raise ValueError("Discrete: weights must be non-negative.")
            if sum(self.weights) <= 0:
                raise ValueError("Discrete: weights sum must be positive.")
        return self

    def sample(self, rng: np.random.Generator) -> float:
        if self.weights is None:
            idx = int(rng.integers(0, len(self.values)))
        else:
            p = np.asarray(self.weights, dtype=float)
            p = p / p.sum()
            idx = int(rng.choice(len(self.values), p=p))
        return float(self.values[idx])


class Categorical(Distribution):
    """Categorical distribution over a finite string set.

    Used for non-numeric parameters such as the electrode kind per
    house ($\\in \\{\\text{foundation}, \\text{rod}, \\text{mesh}\\}$). ``weights``
    biases the choice; uniform if omitted.
    """

    kind: Literal["categorical"] = "categorical"
    values: list[str] = Field(min_length=1)
    weights: Optional[list[float]] = None

    @model_validator(mode="after")
    def _check_weights(self) -> "Categorical":
        if self.weights is not None:
            if len(self.weights) != len(self.values):
                raise ValueError(
                    f"Categorical: weights length {len(self.weights)} does not "
                    f"match values length {len(self.values)}."
                )
            if any(w < 0 for w in self.weights):
                raise ValueError("Categorical: weights must be non-negative.")
            if sum(self.weights) <= 0:
                raise ValueError("Categorical: weights sum must be positive.")
        return self

    @field_validator("values")
    @classmethod
    def _check_unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("Categorical: values must be unique strings.")
        return v

    def sample(self, rng: np.random.Generator) -> str:
        if self.weights is None:
            idx = int(rng.integers(0, len(self.values)))
        else:
            p = np.asarray(self.weights, dtype=float)
            p = p / p.sum()
            idx = int(rng.choice(len(self.values), p=p))
        return str(self.values[idx])


# ---------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------


# AnyDistribution is the discriminated-union form used in
# GeneratorConfig field annotations to enable Pydantic to deserialise
# the right subclass from JSON given the ``kind`` field.
AnyDistribution = Annotated[
    Union[Constant, Uniform, Normal, LogNormal, Weibull, Discrete, Categorical],
    Field(discriminator="kind"),
]
