"""Geometrien von Erdern und Erdungsanlagen.

Dieses Subpackage stellt die 3D-Geometrie der Erdungsanlage bereit. Ein
Erder besteht aus einer Menge von Drahtsegmenten (``Segment``) mit Start-
und Endpunkt, Draht-Radius, Material und einer Zuordnung zu einem oder
mehreren elektrischen Knoten. Die Subpackage-Funktionen erzeugen daraus
ein Mesh für den Feldlöser und stellen typische Grundformen zur Verfügung.

Contents
--------
Segment
    Einzelnes Drahtsegment (Zylinderabschnitt) im Erdreich.
Electrode
    Zusammensetzung aus mehreren ``Segment`` zu einer elektrischen Einheit.
HorizontalRingElectrode, HorizontalStripElectrode,
VerticalRodElectrode, MeshGridElectrode, FoundationElectrode
    Parametrisierte Standard-Erdergeometrien nach DIN VDE 0101 und
    IEC 62305.
Mesher
    Segmentierung eines Erders in finite Abschnitte für den Löser.
Notes
-----
Die Klassen sind zunächst Stubs; die konkrete Implementierung entsteht
begleitend zu Arbeitspaket 1 (TN-Ortsnetz). Die Geometrie ist bewusst vom
Löser getrennt, damit Geometrien sowohl für MoM- als auch für FEM-Löser
genutzt werden können.
"""

from __future__ import annotations

__all__: list[str] = []
