"""Kopplung zwischen Leitern, Schirmen und Erdreich.

Dieses Subpackage stellt die Kopplungs-Beziehungen bereit, die aus den
Leiter-Geometrien und den Materialkenngrößen folgen:

* galvanische Kopplung über gemeinsame Erdungsknoten,
* induktive Kopplung zwischen parallelen Leitern (Neumann-Integrale),
* Carson-Korrektur für den Erdrückleiter bei Frequenzen unterhalb
  weniger kHz,
* kapazitive Kopplung zwischen Leiter und Erdreich (bei Bedarf).

Leitfrage aus Arbeitspaket 1
----------------------------
Wie stark wirken Koppelung und Rückleiterpfade im Niedrigfrequenzbereich,
und wann ist Diffusionsfeld/Carson überhaupt relevant? Die hier
angesiedelten Routinen liefern die Zahlenbasis für diese Bewertung.
"""

from __future__ import annotations

__all__: list[str] = []
