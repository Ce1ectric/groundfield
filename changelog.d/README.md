# Changelog fragments

Work streams of the calendar-versioned release write their changelog
entries here, one file per stream (`core1.md`, `core2.md`, `post.md`,
…), instead of editing `CHANGELOG.md` directly. The release
coordinator merges the fragments into the `[Unreleased]` block and
deletes this directory before the release is cut.

Format of a fragment:

```markdown
### Added
- **<topic> (10.NN):** what changed for the user. Test: `tests/test_x.py::test_y`.

### Changed
- ⚠ **<topic> (10.NN):** what changed; numbers move by <magnitude> for
  <who is affected>. Golden cases: <case names>.

### Fixed
- **<topic> (10.NN):** symptom, cause, fix. Test: `tests/test_x.py::test_y`.

### Deprecated / Removed / Security
- …
```

Rules: English; every `Fixed` entry names its test; every entry that
changes numbers is marked ⚠ and names the affected golden cases; no
local paths; no `$$` display math inside list items.
