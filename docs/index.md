# groundfield

**Numerische Feldberechnung von Erdungsanlagen.**

`groundfield` ist ein Open-Source-Python-Paket für die physikalische
Referenzmodellierung vernetzter Erdungssysteme. Innerhalb der
`groundmeas` / `groundinsight` / `groundfield` Softwarefamilie bildet
`groundfield` die feldtheoretische Seite ab: Bodenmodelle,
Erdergeometrien, Leiter und deren Kopplungen werden als 3D-Problem im
Erdreich formuliert und numerisch gelöst.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Was `groundfield` tut

Die realen Erdungsanlagen in Mittel- und Niederspannungsnetzen sind
gekoppelte, dreidimensionale und frequenzabhängige Systeme. Für die
Planung, Bewertung und Beobachtung braucht es ein Modell, das die
tatsächliche Stromaufteilung im Fehlerfall physikalisch plausibel
beschreibt. `groundfield` erzeugt genau dieses Referenzmodell — bewusst
als *Referenz, nicht als Endprodukt*: aus der Feldlösung wird ein
reduziertes Ersatzmodell (`rho-f`) abgeleitet, das in `groundinsight`
als `BusType`- oder `BranchType`-Formel weiter verwendet wird.

## Workflow

```mermaid
---
title: Rechenweg in groundfield
---
flowchart TD
    start((Start))
    soil[Bodenmodell definieren]
    geom[Geometrie + Leiter festlegen]
    study[FieldStudy zusammenbauen]
    solve[solve]
    post[Postprocess: Potential, UT, Ströme]
    reduce[Reduktion auf rho-f]
    export[Export an groundinsight]
    finish((End))

    start --> soil
    soil --> geom
    geom --> study
    study --> solve
    solve --> post
    solve --> reduce
    reduce --> export
    post --> finish
    export --> finish
```

## Nächste Schritte

- [Installation](installation.md) — Poetry-Setup, Docs-Gruppe, VS Code.
- [Schnelleinstieg](quickstart.md) — erste Feldrechnung.
- [Konzepte](concepts.md) — was `groundfield` von einem reinen FEM-Tool
  unterscheidet.
- [API-Referenz](api/index.md) — generiert per `mkdocstrings` aus den
  Docstrings.
