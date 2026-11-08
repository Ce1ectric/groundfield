"""Touch and step voltages: the curated namespace ``gf.safety``.

Prospective touch and step voltages evaluated from the surface
potential of a solved :class:`~groundfield.FieldResult` — around an
electrode or at user-defined touch points (accessible metal parts) —
and the permissible touch voltage of EN 50522. The functions are the same
objects as the flat top-level names
(``gf.safety.touch_voltage is gf.touch_voltage``); see
:mod:`groundfield.postprocess.safety` for the definitions, units and
the validity of each quantity.
"""

from groundfield.postprocess.safety import (
    NEAR_FIELD_DISTANCE_M,
    UniformCurrentFieldWarning,
    permissible_prospective_touch_voltage_en50522,
    permissible_touch_voltage_en50522,
    step_voltage,
    touch_voltage,
    touch_voltage_at_part,
    touch_voltage_envelope,
    touch_voltage_spectrum,
)

__all__ = [
    "touch_voltage",
    "touch_voltage_envelope",
    "touch_voltage_at_part",
    "touch_voltage_spectrum",
    "step_voltage",
    "permissible_touch_voltage_en50522",
    "permissible_prospective_touch_voltage_en50522",
    "UniformCurrentFieldWarning",
    "NEAR_FIELD_DISTANCE_M",
]
