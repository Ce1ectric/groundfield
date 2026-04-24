"""Datenklassen für Bodenmodelle.

Die hier definierten Klassen sind reine Datencontainer (Pydantic v2). Sie
beschreiben die Geometrie und die elektromagnetischen Parameter des
Erdreichs. Die eigentliche Auswertung (z.B. Green'sche Funktion, Bild-
ladungssumme, Reflexionsfaktor) erfolgt in ``groundfield.solver`` über
die hier bereitgestellten Daten.

Todo
----
* Implementierung ``HomogeneousSoil``, ``TwoLayerSoil``, ``MultiLayerSoil``.
* Ergänzung der frequenzabhängigen Modelle (Visacro, Alipio, Messier).
"""

from __future__ import annotations

__all__: list[str] = []
