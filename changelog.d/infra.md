### Changed

- **Calendar versioning:** releases are numbered `YYYY.MM.MICRO` in PEP 440
  form (`2026.11.0`; month without leading zero, no hyphen; `MICRO` counts
  the earlier releases of the month; tags `v2026.11.0`). The deprecation
  policy is stated at the top of this changelog. Test:
  `tests/test_release_script.py::test_calver_version_from_date_and_tags`.
- **Release tool (10.09):** `python scripts/release.py calver [--date
  YYYY-MM-DD] [--dry-run]` derives the version from the date and the
  existing tags; `python scripts/release.py set X.Y.Z` takes an explicit
  canonical PEP 440 calendar version (pre-releases such as `2026.11.0rc1`
  included). The tool updates `pyproject.toml`, `__version__`,
  `CITATION.cff` (`version`, `date-released`) and turns `[Unreleased]` into
  `## [X] - YYYY-MM-DD`, then commits and tags; it never pushes. The
  `major` / `minor` / `patch` bumps and the options `--no-push`,
  `--allow-dirty` and `--allow-empty-changelog` are gone. Test:
  `tests/test_release_script.py`.
- **Package metadata (10.11):** metadata moved from the legacy
  `[tool.poetry]` tables to `[project]` (PEP 621; license as SPDX expression
  `MIT`, PEP 639); explicit classifiers for Python 3.12, 3.13 and 3.14
  (the auto-generated 3.15 classifier is gone), OS independent,
  Scientific/Engineering, `Typing :: Typed`; build backend bounded to
  `poetry-core>=2.2.0,<3.0.0`. Test:
  `tests/test_packaging.py::test_metadata_uses_the_project_table`.
- **Typed package (10.02):** `groundfield` ships a `py.typed` marker, so type
  checkers use its annotations. Test:
  `tests/test_packaging.py::test_wheel_contents`.
- **Development standards (10.94, 10.80):** ruff lint configuration (F, E,
  W, I, B, UP, D with NumPy convention, N with a physics naming
  allow-list, C901 with complexity 15), mypy configuration with the
  pydantic plugin, pre-commit hooks (ruff format, ruff check, nbstripout,
  file hygiene), ruff 0.17.4 and mypy 2.4.1 pinned in the dev group,
  black removed. CI runs lint, types (non-blocking), tests on Python 3.12,
  3.13 and 3.14 with `pytest -n auto` and coverage, a strict docs build on
  pull requests, and a non-blocking weekly dependency and workflow audit.

### Removed

- **`polars` dependency (10.11):** it was never imported. It is still
  installed with the `groundinsight` extra, because groundinsight 0.3.0
  requires it. Test:
  `tests/test_packaging.py::test_polars_is_neither_a_dependency_nor_imported`.
- **Console scripts and the top-level `scripts` package (10.02):** the wheel
  no longer installs `release`, `generate-third-party-licenses` or a
  `scripts` package. Run the tools from a checkout:
  `python scripts/release.py ...`,
  `python scripts/generate_third_party_licenses.py`. Test:
  `tests/test_packaging.py::test_no_scripts_package_and_no_console_scripts`.

### Security

- **Package collision with groundinsight (10.02):** the v0.15.0 wheel
  installed the same `scripts/release.py` and `release` command as
  groundinsight 0.3.0; installing both overwrote one with the other and
  uninstalling one broke the other. The wheel now contains only
  `groundfield/` and its metadata. Test:
  `tests/test_packaging.py::test_wheel_contents`.
- **CI supply chain (10.03):** every workflow starts with `permissions: {}`
  and grants each job only what it needs; every action is pinned to a full
  commit SHA with the tag in a comment; checkouts do not persist
  credentials (except the docs deployment, which pushes `gh-pages`);
  `snok/install-poetry` and `softprops/action-gh-release` are replaced by
  a pinned `pip install poetry` and the GitHub CLI; tag names reach shell
  code only through environment variables. The release workflow checks
  that the tag equals the version in all three version files, runs the
  full CI as a gate before building, and keeps PyPI trusted publishing.
  zizmor 1.32.0 reports 26 findings on the v0.15.0 workflows and none on
  the new ones.
- **Locked dependencies (10.04):** `pip-audit` on the exported lock (all
  groups and extras) found 100 known advisories in 8 packages; after
  updating pillow (13.0.0), tornado (6.6), urllib3 (2.8.0),
  pymdown-extensions (12.3), certifi (2026.10.23), idna (3.20), pytest
  (9.2.0) and mkdocs-material (9.8.1) it finds none. numpy, scipy, sympy,
  pydantic, matplotlib and pandas keep their locked versions.
- **MathJax (10.08):** the documentation loaded MathJax from a floating
  `cdn.jsdelivr.net/npm/mathjax@3` URL without integrity check. MathJax
  3.2.2 is now vendored with the site (60 files byte-identical to the
  verified npm release, pinned by `SHA256SUMS`); the site loads no
  third-party JavaScript. Test: `tests/test_docs_assets.py`.
- **Release tool (10.09):** the v0.15.0 tool ran git in the caller's
  working directory, rewrote the version files before it validated the
  changelog (an aborted run left bumped files behind), committed on a
  detached HEAD or onto an existing tag, and could not finalise a
  pre-release. It now runs git in the repository root, validates
  everything before it writes (clean tree, branch, tag, agreeing version
  files, non-empty `[Unreleased]`) and restores the files if the commit
  fails. Tests: `tests/test_release_script.py` (marked `finding("10.09")`).

### Docs

- **Formula rendering:** the MathJax configuration started a second
  typesetting pass concurrently with MathJax's own; every formula was
  typeset twice (the copy lands in the hidden assistive-MathML region),
  and pages that autoload a TeX extension (`\boldsymbol` in ADR-0003 and
  ADR-0004) failed with "Cannot read properties of null (reading
  'replaceChild')". Re-typesetting after instant navigation is now
  serialised through `MathJax.startup.promise`.
