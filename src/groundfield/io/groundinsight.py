"""Export reduzierter Modelle an ``groundinsight``.

Die zentrale Funktion dieses Moduls ist das Fit- und Export-Paar für
``rho-f``-Modelle:

1. ``fit_rho_f(result, ...)`` passt eine parametrische Darstellung
   ``Z(rho, f)`` an die per ``FieldStudy`` berechnete Eingangsimpedanz
   an. Erlaubt sind Formel-Familien, die ``groundinsight`` als
   ``BusType.impedance_formula`` verarbeitet.
2. ``export_bustype(fit, name, description, ...)`` erzeugt einen
   ``BusType`` im ``groundinsight``-Sinn, der direkt in eine Netzmodell-
   Datenbank geschrieben werden kann.

Todo
----
* Entscheidung über das bevorzugte Fit-Format: Polynom in ``(f, rho)``,
  rationale Approximation (Vector Fitting) oder physikalisch motivierte
  Ausdrücke (z.B. Sunde-Ausdruck für homogene Böden).
* Rückkopplung mit ``groundinsight.core_models.BusType`` auf Schema-
  Ebene sicherstellen (derselbe SymPy-Ausdruck muss dort parsbar sein).
"""

from __future__ import annotations

__all__: list[str] = []
