# Installation

`groundfield` requires **Python 3.12 or newer** and uses
[Poetry](https://python-poetry.org/) for dependency management.

## From the Git checkout

```bash
git clone https://github.com/Ce1ectric/groundfield.git
cd groundfield
poetry install
```

Dev dependencies (pytest, black, ipykernel) live in the ``dev`` group
and are installed by default.

## Documentation group

To build the documentation locally:

```bash
poetry install --with docs
poetry run mkdocs serve
```

## Executable documentation

Every fenced ``python`` block under `docs/` is executed by the test
suite (`tests/test_pass9_doc_snippets.py`), so a snippet that has
drifted away from the API fails CI instead of failing the reader:

```bash
poetry run pytest tests/test_pass9_doc_snippets.py
```

All blocks of one page run **in document order into one shared
namespace**, exactly as a reader would work through the page, so a
later block may reuse the `world` or `result` built earlier on the
same page. Snippets run on the `matplotlib` `Agg` backend, with
`plt.show()` neutralised and the working directory redirected to a
temporary folder — a snippet that writes CSV / VTK / JSON therefore
cannot litter the checkout.

Blocks that are *not* runnable programs opt out with an HTML comment
on the line directly above the fence (invisible in the rendered
page). Always state a reason:

```markdown
<!-- skip-doctest: schema sketch, not runnable code -->
```

Use it for schema / signature sketches (typical in the ADRs),
snippets that need the Overpass network or another optional
dependency, and study templates that would run for hours. A whole
page opts out with `<!-- skip-doctest-page: reason -->` anywhere in
the file. Pages that are runnable but take minutes belong in the
`SLOW_PAGES` map of the harness and then only run with
`GF_DOC_SNIPPETS_SLOW=1`; trimming the demo (coarser
`segment_length`, fewer realisations) is usually the better fix,
because a snippet too slow for CI is also too slow for the reader
who copy-pastes it.

## PyPI

Once the first release is published:

```bash
pip install groundfield
```

## Integration with the sister projects

`groundfield` is designed to work alongside `groundinsight` and
`groundmeas`. When the three projects share a Poetry environment they
can be wired up as path dependencies:

```toml
[tool.poetry.dependencies]
groundinsight = { path = "../groundinsight", develop = true }
groundmeas = { path = "../groundmeas", develop = true }
```

This step is optional and is usually only relevant for
integration-heavy development.
