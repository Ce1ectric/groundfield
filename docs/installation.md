# Installation

`groundfield` benötigt **Python 3.12 oder neuer** und wird mit
[Poetry](https://python-poetry.org/) verwaltet.

## Aus dem Git-Checkout

```bash
git clone https://github.com/Ce1ectric/groundfield.git
cd groundfield
poetry install
```

Dev-Abhängigkeiten (pytest, black, ipykernel) kommen aus der Gruppe
`dev` und werden per Default mit installiert.

## Dokumentations-Gruppe

Für das lokale Bauen der Dokumentation:

```bash
poetry install --with docs
poetry run mkdocs serve
```

## PyPI

Sobald die erste Version freigegeben ist:

```bash
pip install groundfield
```

## Integration in die Softwarefamilie

`groundfield` arbeitet eng mit `groundinsight` und `groundmeas`
zusammen. Wenn diese in derselben Poetry-Umgebung liegen sollen, kann
sie im Monorepo-Stil als Pfad-Abhängigkeit eingetragen werden:

```toml
[tool.poetry.dependencies]
groundinsight = { path = "../groundinsight", develop = true }
groundmeas = { path = "../groundmeas", develop = true }
```

Dieser Schritt ist optional und wird üblicherweise nur für
integrations-nahe Entwicklung genutzt.
