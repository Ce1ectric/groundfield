"""Bodenmodelle für ``groundfield``.

Dieses Subpackage beschreibt das Ausbreitungsmedium Erdreich. Unterstützt
werden homogene, geschichtete (zwei- und mehrschichtige) sowie
frequenzabhängige Bodenmodelle im Sinne von Visacro/Alipio. Die Modelle
liefern die für den Feldlöser benötigten effektiven elektromagnetischen
Parameter ``rho(f)``, ``eps_r(f)`` und die zugehörigen Green'schen
Funktionen bzw. Spiegelladungskoeffizienten.

Contents
--------
HomogeneousSoil
    Homogenes Erdreich mit konstantem spezifischem Widerstand.
TwoLayerSoil
    Zwei-Schicht-Modell (Oberschicht/Unterschicht) mit Schichtdicke.
MultiLayerSoil
    Mehrschicht-Modell mit beliebig vielen horizontalen Schichten.
FrequencyDependentSoil
    Frequenzabhängiges Modell nach Visacro/Alipio.

Notes
-----
Für Arbeitspaket 1 der Dissertation ist zunächst das Zwei-Schicht-Modell
maßgeblich (Parameterraum: variierende Widerstände und Schichtdicken).
Die übrigen Klassen sind als Stubs vorgesehen und werden sukzessive
implementiert.
"""

from __future__ import annotations

__all__: list[str] = []
