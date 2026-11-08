# ADR-0013: Calendar versioning

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-11-07 |
| **Deciders** | Project maintainers |
| **Scope** | version numbers of `groundfield` (`pyproject.toml`, `groundfield.__version__`, `CITATION.cff`), the release tool `scripts/release.py`, the changelog, the deprecation policy and the versioned documentation |

## Context

Up to 0.15.0 `groundfield` used semantic-versioning numbers below 1.0:
sixteen releases in about six months, most of them minor releases that
changed numerical results. The numbers said little:

- **Compatibility** was never the question a user of a reference model
  asks; the question is *which results changed*. A 0.x minor release
  could move an impedance by tens of per cent, a patch release could fix
  a security problem without touching a number. The version number carried
  neither fact.
- **Dating** matters for a dissertation and its catalogue of computed
  networks: a result has to be traceable to the code that produced it,
  and "computed with 2026.11.0" places it in time; "computed with
  0.15.0" needs a lookup.
- Review pass 10 asked for one release with everything (physics fixes,
  performance, architecture, documentation) and for an explicit list of
  numbers-changing releases. A 1.0.0 would have promised a stability that
  a research code under active validation does not offer.

## Decision

1. **Version scheme `YYYY.M.MICRO`** in PEP 440 canonical form: year,
   month without leading zero, and `MICRO` = the number of earlier final
   releases in that month (the first release of November 2026 is
   `2026.11.0`; a second one would be `2026.11.1`; January 2027 starts at
   `2027.1.0`). Pre-releases append `aN`, `bN` or `rcN`; post- and
   development releases `.postN` / `.devN`. A deleted tag never reuses a
   number.
2. **One release tool.** `python scripts/release.py calver [--date
   YYYY-MM-DD] [--dry-run]` computes the next version;
   `python scripts/release.py set X` sets an explicit one (also to
   finalise a pre-release). The tool runs every check before it writes
   anything: repository root, clean tree, a branch checked out, the three
   version sources agree, the new version is valid and above the current
   one, its tag does not exist, `[Unreleased]` of the changelog is not
   empty, no unmerged `changelog.d/` fragment remains. It writes the three
   version sources and the changelog heading, commits and tags — and never
   pushes. The tool is not installed with the package (finding 10.02); it
   runs from a checkout.
3. **Numbers instead of semantics.** Every release that moves computed
   numbers carries a ⚠ "Numbers that change" block in its release notes
   (quantity, magnitude, who is affected), and the changelog keeps a table
   of all numbers-changing releases. Every result records the producing
   version in `metadata["provenance"]`.
4. **Deprecation policy.** A deprecated name keeps working, emits a
   `DeprecationWarning` naming the replacement and the earliest removal
   version, and is removed no earlier than **two releases and six months**
   after the release that announced it: names deprecated in `2026.11.0`
   go in `2027.5` at the earliest.
5. **Versioned documentation.** The documentation is published per
   release with `mike` (one documentation version per calendar release,
   `latest` for the newest release, `dev` for the main branch).

## Consequences

**Positive.** A version number dates a result; the changelog answers the
question users ask (which numbers moved); the release tool cannot leave
half-bumped version files behind; removal dates of deprecated names are
computable from the announcing version.

**Negative.** The version no longer signals compatibility; users must read
the ⚠ block of the release notes. Tools that compare versions numerically
must not assume `MAJOR < 1` semantics; `2026.11.0` sorts above `0.15.0`
under PEP 440, so upgrades work.

**Neutral.** The first calendar release follows 0.15.0 directly; there is
no 1.0.0. Pass-5's guard against a hard-coded version in the project
instructions is kept by the release tool.

## References

- Python Packaging Authority. PEP 440 — Version Identification and
  Dependency Specification.
- Keep a Changelog, version 1.1.0 (format of `CHANGELOG.md`).
- Calendar versioning, <https://calver.org> (scheme catalogue).
- Review pass 10 (2026-11-07), decision questions 1 and 11, findings
  10.02, 10.09 and 10.93.
