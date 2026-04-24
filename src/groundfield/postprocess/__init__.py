"""Auswertung der Feldlösung.

Aus der gelösten Potentialverteilung werden die für Ingenieur und
Dissertation interessanten Größen extrahiert:

* Erdungsspannung ``U_E`` am Einspeisungspunkt,
* Berührungs- und Schrittspannungen an beliebigen Messpunkten,
* Potentialverlauf entlang definierter Pfade (Messharke, Traversen),
* Stromdichte im Erdreich und Stromaufteilung auf Rückleiterpfade,
* Reduktionsfaktor pro Frequenz,
* 2D/3D-Feldverläufe für Plotting.

Contents
--------
potential
    Punkt- und Pfadauswertung des skalaren Potentials.
voltages
    Aggregierte Spannungsgrößen (EPR, UT, US).
currents
    Stromdichtefelder und Leiterstromaufteilung.
plots
    Matplotlib-Helfer (analog zu ``groundinsight.plotting``).
"""

from __future__ import annotations

__all__: list[str] = []
