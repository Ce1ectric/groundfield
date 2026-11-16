# Contribute to the documentation

This guide shows how to write or change a page of this documentation —
a tutorial notebook, a how-to page, an explanation or reference page — so
that its code runs, its numbers are checked and the strict site build
passes. Read it before you open a pull request that touches `docs/`.

**Prerequisites:** a checkout of the repository with
`poetry install --with dev,docs` (pytest, ruff and nbclient from the `dev`
group; MkDocs, mkdocs-jupyter and with it nbconvert from the `docs`
group).

## 1. Pick the kind of page

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

## 2. Write or change a tutorial notebook

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

- `tests/test_tutorials.py` checks without executing anything that every
  tutorial is committed with its outputs and without errors, stays below
  1.5 MB, needs no network and has at least three `assert` statements.
  With the marker `slow` it executes every tutorial with nbclient (300 s
  per notebook) and fails on an error, a failed `assert` or a deprecation
  warning in the outputs:

```bash
python -m pytest -m slow tests/test_tutorials.py
```

## 3. Write or change a how-to page

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

A how-to page opens with the task it solves and who needs it, lists its
prerequisites, then gives the steps in order, says what the reader gets
and how to check it, and ends with links to the tutorial, explanation and
reference pages of the topic.

## 4. Write or change an explanation or reference page

- **Generated blocks.** The validation page and the blocks between
  `<!-- generated:begin <name> -->` and `<!-- generated:end <name> -->` on
  the engine pages are written by `scripts/generate_validation_page.py`
  from the stored measurements (`docs/explanation/validation/measurements.json`)
  and the bounds in `tests/benchmarks/tolerances.py`. Do not edit them by
  hand: change the bound or the benchmark and render again.

```bash
python scripts/generate_validation_page.py            # render from the stored measurements
python scripts/generate_validation_page.py --measure  # run the benchmarks, then render
python scripts/generate_validation_page.py --check    # exit 1 if a generated file is stale
```

- **Purpose first.** Each page opens with one or two sentences: what the
  page is for and who should read it. Introduce each symbol where it first
  appears and give each equation one sentence that says what it means
  physically.
- **Provenance out of the reading flow.** Audit ids, finding numbers, long
  measurement lists and the history of a number are the provenance of a
  statement: keep them, but in a collapsible block
  (`??? note "Measurements and audit trail"`, from pymdownx.details), a
  footnote or a closing provenance section. Never delete a measured limit;
  the limit itself stays in the text.
- **Math.** `$…$` inline and `$$…$$` on their own lines; inside a Markdown
  table no `|` in math (write `\lvert … \rvert`), no escaped underscores,
  no math in headings.
- **Tables** with short cells and at most about six columns; put long
  explanations in the text below.
- **Pinned phrases.** `tests/test_explanation_pages.py`, `tests/test_docs_*.py`
  and the tests of several modules assert phrases of these pages — for
  example the headings every engine page carries
  (`## Mathematical and physical context`, `## Validity range`,
  `## Validation`, `## Literature`) and the measured limits of the
  [known limitations](../reference/known-limitations.md). Run them after a
  rewrite, and keep the fact a failing test protects.

## 5. Build the site locally

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

**How to check it:** before you push, run the docs tests and the strict
build:

```bash
python -m pytest -q tests/test_doc_snippets.py tests/test_docs_*.py tests/test_explanation_pages.py tests/test_validation_page.py
python scripts/generate_validation_page.py --check
poetry run mkdocs build --strict
```

## See also

- [How-to guides](index.md): the existing recipes, as models.
- [Tutorials](../tutorials/index.md): the notebooks and how they are run.
- [Validation and benchmarks](../explanation/validation.md): the generated
  page.
