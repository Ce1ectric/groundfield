r"""Foundation-electrode penetration masks for the layout builder.

Part of :mod:`groundfield.generators.layout`. A penetration study equips
a fraction :math:`p` of the houses with a foundation electrode. The mask
is deterministic and nested in :math:`p`: house :math:`i` is equipped iff
:math:`r_i < p`, where :math:`r_i \in [0, 1)` is a stable pseudo-random
number derived from a key of the footprint and a salt.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from groundfield.geo.footprint import BuildingFootprint

__all__: list[str] = []


def _stable_uniform(salt: int, key: object) -> float:
    r"""Deterministic uniform :math:`[0, 1)` draw from a salt + key.

    Built on MD5 (chosen for stable cross-Python-version hashing,
    not for cryptographic strength) so the value depends only on
    the textual representation of the inputs and is reproducible
    across runs, processes and Python versions.
    """
    blob = f"{salt}|{key}".encode("utf-8")
    digest = hashlib.md5(blob).digest()[:8]
    n = int.from_bytes(digest, "big")
    return n / float(1 << 64)


class _PenetrationMixin:
    """Penetration-mask method of :class:`~groundfield.generators.layout.LvNetworkLayout`."""

    if TYPE_CHECKING:  # pragma: no cover - attributes of the layout model
        footprints: list[BuildingFootprint]

    def foundation_mask(
        self,
        penetration: float,
        *,
        salt: int = 0,
    ) -> list[bool]:
        r"""Deterministic per-house Bernoulli mask for foundation
        electrodes.

        For each footprint a stable pseudo-random number
        :math:`r_i \in [0, 1)` is derived from the footprint's
        ``osm_id`` (or its index as a fallback) and the user-
        supplied ``salt``. The mask entry is ``True`` iff
        :math:`r_i < p` where :math:`p =` ``penetration``. Two
        properties hold:

        * **Reproducible.** Calling
          ``layout.foundation_mask(0.15)`` always returns the same
          mask for the same set of footprints, regardless of when
          or in which process you call it.
        * **Nested in p.** For a fixed ``salt``, every house with
          a foundation at :math:`p_1` also has one at
          :math:`p_2 > p_1` — increasing the penetration only
          *grows* the foundation-equipped subset. This is the
          natural property for penetration-rate sweep studies.

        Parameters
        ----------
        penetration
            Penetration rate :math:`p \in [0, 1]`. ``0`` returns
            an all-``False`` mask, ``1`` an all-``True`` mask.
        salt
            Optional integer that shifts the mask deterministically
            into a different realisation. Useful for Monte-Carlo
            studies where the user wants several independent runs
            at the *same* penetration rate.

        Returns
        -------
        list of bool
            One entry per footprint, in the layout's declared
            order. Length equals ``len(self.footprints)``.

        Examples
        --------
        >>> mask_15 = layout.foundation_mask(0.15)
        >>> mask_30 = layout.foundation_mask(0.30)
        >>> all(b1 <= b2 for b1, b2 in zip(mask_15, mask_30))   # nested
        True
        >>> layout.foundation_mask(0.15) == layout.foundation_mask(0.15)
        True
        """
        if not 0.0 <= penetration <= 1.0:
            raise ValueError(
                "LvNetworkLayout.foundation_mask: penetration must be "
                f"in [0, 1], got {penetration}."
            )
        out: list[bool] = []
        for i, fp in enumerate(self.footprints):
            key = fp.osm_id if fp.osm_id is not None else i
            r = _stable_uniform(salt, key)
            out.append(r < penetration)
        return out
