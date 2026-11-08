# Contribute to the documentation

The documentation follows the Diátaxis scheme: four kinds of pages, each
with one job.

| Section | Job | Form |
|---|---|---|
| `docs/tutorials/` | teach by doing, with checks | executed Jupyter notebooks |
| `docs/how-to/` | solve one task | Markdown with executed code blocks |
| `docs/reference/` | describe every public name | mkdocstrings pages, tables |
| `docs/explanation/` | explain the physics and the methods | Markdown with equations, generated validation tables |

Three rules hold everywhere: English; every number comes from code that
runs (a notebook cell, an executed block, a generated table) or cites the
test or evidence file it comes from; and every page about a module or an
engine states the mathematical-physical context, not only the API.

## Write or change a tutorial notebook

- One notebook per tutorial, `docs/tutorials/TNN_short_name.ipynb`,
  listed in `docs/tutorials/index.md` and in the navigation (`mkdocs.yml`).
- Narrative first: what we compute and why, the physics in two or three
  sentences with the formula where it helps, then the code.
- Cross-check the key numbers with a second method (a closed form, a
  limit, a second engine, a convergence study) and put the checks into
  `assert` cells, so the notebook fails when a number drifts.
- No network: OpenStreetMap data comes from a cache file under
  `docs/tutorials/data/`; files the notebook writes go to a temporary
  directory.
- The notebook runs top to bottom in less than 120 s on a laptop and is
  committed **with its outputs**: the site renders the stored outputs
  (mkdocs-jupyter with `execute: false`). Execute it in place from the
  notebook's directory, then format the code cells:

```bash
cd docs/tutorials
jupyter nbconvert --to notebook --execute --inplace T01_first_grounding_impedance.ipynb
ruff format T01_first_grounding_impedance.ipynb
```

- `tests/test_tutorials.py` (marker `slow`) executes every tutorial with
  nbclient and fails on an error, a failed `assert` or a deprecation
  warning in the outputs:

```bash
python -m pytest -m slow tests/test_tutorials.py
```

## Write or change a how-to page

Code blocks in Markdown pages (and in `README.md`) are executed by
`tests/test_doc_snippets.py`, page by page, blocks in document order in
one namespace, in a temporary working directory. A block either runs or
is marked; the markers are HTML comments and invisible on the site:

````markdown
<!-- skip-doctest: needs the Overpass API (network) -->
```python
footprints, projector = gf.query_and_project(51.9, 10.8, 150.0)
```

<!-- doctest-module: my_factories.py -->
```python
def network(seed): ...
```

<!-- doctest-allow-warnings: UniformCurrentFieldWarning -->
```python
gf.touch_voltage_envelope(result, world, electrode="grid")
```

<!-- doctest-check
assert abs(z_in.real - 33.50) < 0.01
-->
````

- `skip-doctest: reason` — an illustrative fragment; the reason is
  mandatory.
- `doctest-module: name.py` — the block is written to that file in the
  working directory (which is on `sys.path`), for code that must be
  importable, e.g. by the worker processes of `sweep(n_jobs=...)`.
- `doctest-allow-warnings: A, B` — the block may emit these warnings.
  Any other warning that comes from groundfield or from the block itself
  fails the page — in particular a deprecated name.
- `doctest-check` — a hidden check, executed in the page's namespace.
  Every how-to page needs at least one `assert`.

Run the harness for one page while you write it:

```bash
python -m pytest tests/test_doc_snippets.py -k how-to/plot-results
```

## Build the site locally

```bash
poetry install --with docs
poetry run mkdocs serve          # live preview at http://127.0.0.1:8000
poetry run mkdocs build --strict # what CI runs: every warning is an error
```

`--strict` fails on broken links, unknown cross-references and malformed
docstrings. Link to other pages with relative paths to the source file
(`../reference/glossary.md`, `../tutorials/T03_interconnected_systems_and_impedances.ipynb`)
and to API objects with mkdocstrings references
(`[FieldResult][groundfield.FieldResult]`); do not use reST roles such as
`:class:` in Markdown pages. Math uses `$...$` and `$$...$$`.

The published site is versioned with `mike`: every release gets its own
version of the documentation, `latest` points to the newest release and
`dev` to the main branch. The deployment runs in CI; you do not need
`mike` locally to write pages.
