# Konzepte

## Positionierung in der Softwarefamilie

```
  groundmeas   ──▶   groundinsight   ◀──   groundfield
  (Messung)         (reduziertes                (Feldmodell,
                     Netzmodell)                 PDE-Referenz)
```

`groundfield` ist das feldtheoretische Referenzwerkzeug. Es liefert die
Wahrheit gegenüber der sich `groundinsight` messen lassen muss, und die
Datenbasis gegen die `groundmeas` ausgewertet werden kann.

## Modellannahmen

- **Quasi-statisch** im Erdreich für $f \lesssim 1\,\mathrm{kHz}$.
  Das elektrische Skalarpotential erfüllt dann $\nabla \cdot
  (\sigma \nabla \varphi) = 0$.
- **Carson-Korrektur** für den Erdrückleiter oberirdischer/unter-
  irdischer Leiter; kein Vollwellen-Modell.
- **Geschichtetes Erdreich** als horizontale Halbräume mit stück-
  weise konstanten Leitfähigkeiten.
- **Dünndraht-Approximation** für Erder und Leiter (Momentenmethode).

## Das `rho-f`-Modell

Ziel einer Feldrechnung in `groundfield` ist nicht nur ein numerisches
Ergebnis, sondern eine Kompression der Lösung auf wenige, parametrisch
lesbare Größen. Für den Zweipol-Fall besteht diese Reduktion im
`rho-f`-Modell:

$$
Z(\rho, f) = R_{0}(\rho) + X(\rho, f)
$$

wobei $R_{0}$ die niederfrequente Ausbreitungsresistanz ist und $X$ die
frequenzabhängige Reaktanzanteile zusammenfasst. Die Koeffizienten
werden an die Feldlösung angepasst und landen anschließend als
`BusType.impedance_formula` in `groundinsight`.

## Arbeitspaket 1

`groundfield` ist das primäre Werkzeug für **Arbeitspaket 1**. Dort
werden TN-Ortsnetze mit Ortsnetzstation, Hausanschlüssen und
Kabelverteilern im geschichteten Erdreich untersucht. Die Kernfragen
sind:

1. Welchen Einfluss hat die entfernte Stromeinspeisung auf das
   Erdungsmessergebnis?
2. Wie stark wirken Koppelung und Rückleiterpfade im niedrigen
   Frequenzbereich?
3. Lassen sich robuste Aussagen für typische Ortsnetz-
   konfigurationen ableiten?

Die Antworten auf diese Fragen sind der erste Baustein der
Dissertation; `groundfield` liefert die dafür nötige numerische Basis.
