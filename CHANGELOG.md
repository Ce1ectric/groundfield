# Changelog

All notable changes to `groundfield` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Change categories follow the Keep-a-Changelog vocabulary:

- **Added** — new features and public API.
- **Changed** — behaviour changes to existing public API.
- **Deprecated** — features that still work but will be removed.
- **Removed** — features taken out of the public API.
- **Fixed** — bug fixes.
- **Security** — vulnerability fixes.
- **Docs** — documentation-only changes.
- **Internal** — refactors, tests, packaging, CI; no observable behaviour change.

The backlog of ideas that are not yet scheduled is kept at the end of this
file under **Roadmap**. During regular work, add your entry under the
matching category in `[Unreleased]`; the release script
(`scripts/release.py`) moves the whole `[Unreleased]` block into a new
version section when a release is cut.

---

## [Unreleased]

_No changes yet._

---

## [0.1.0] — 2026-04-24

Initial project skeleton for `groundfield`.

### Added

- Paketstruktur analog zu `groundinsight` mit `src/groundfield/` und
  Subpackages `soil`, `geometry`, `conductors`, `solver`, `coupling`,
  `postprocess`, `io`, `utils`.
- Poetry-Konfiguration (`pyproject.toml`) mit Dev- und Docs-Gruppen.
- MkDocs-Material-Stub (`mkdocs.yml`, `docs/`) für die spätere
  Dokumentations-Site auf GitHub Pages.
- Release-Automatisierung (`scripts/release.py`) und Third-Party-
  Lizenz-Report (`scripts/generate_third_party_licenses.py`) analog zu
  `groundinsight`.
- GitHub-Actions-Workflows `ci.yml`, `docs.yml`, `release.yml`.
- `CITATION.cff`, `LICENSE` (MIT), `.gitignore`, `CLAUDE.md` (Kontext
  für die KI-Assistenten).
- Erster Smoke-Test (`tests/test_import.py`).

### Internal

- Projekt-Hülle als Grundlage für **Arbeitspaket 1** der Dissertation
  zu vernetzten Erdungssystemen: TN-Ortsnetz mit geschichtetem
  Erdreich, Hausanschlüssen und Kabelverteilern.

---

## Roadmap

Diese Liste bündelt Feature-Ideen, die noch keiner konkreten Version
zugeordnet sind. Sie wird während der Arbeit an Arbeitspaket 1
laufend ergänzt.

### Kernfunktionalität

- MoM-Löser mit Bildladungen für homogenes Erdreich.
- Erweiterung auf geschichtetes Erdreich (geschlossene Lösung und
  numerische Quadratur der Sommerfeld-Integrale).
- FEM-Backend (optional, über `scikit-fem`).
- Carson-Korrektur für den Erdrückleiter unterhalb 1 kHz.
- Vector-Fitting für die Anpassung der `rho-f`-Kurve.

### Integration

- Direkter Export von `BusType` und `BranchType` in
  `groundinsight`-Datenbanken.
- Import der Messgeometrien aus `groundmeas` (Positionen, Längen,
  Hilfselektroden-Setup).

### Doku und Typisierung

- Referenzfall-Bibliothek für AP1: Einfamilienhaus-Cluster
  (5/10/30/80/200), Klein- und Mittel-Gewerbe, Kabelverteiler.
- Notebook-Suite, die den Parameterraum aus der Dissertations-
  Präsentation abdeckt.

[Unreleased]: https://github.com/Ce1ectric/groundfield/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.1.0
