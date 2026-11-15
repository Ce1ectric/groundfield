---
title: Audit history
description: The review and audit passes of groundfield, what they found and in which release the fixes shipped.
---

# Audit history

groundfield is a reference model: its numbers feed a catalogue of real
low-voltage networks, rho-f models for groundinsight and touch-voltage
assessments. A number that changes therefore needs a recorded reason. Since
May 2026 the code has been reviewed in ten numbered review passes and one
physics and numerics audit. Each pass produced findings, each fix a regression
test, and the releases below shipped them.

Up to 0.15.0 the full narrative of every pass stood in the changelog. Since
2026.11.0 the [changelog](../changelog.md) holds concise release notes, and the
narratives are kept here, verbatim, one page per pass.

| Pass | Date | Reviewed | Fixes released in | Finding identifiers |
|---|---|---|---|---|
| [Pass 1](pass-01.md) | 2026-05-10 | development state between 0.1.0 and 0.2.0 | not labelled (backlog) | none |
| [Pass 2](pass-02.md) | 2026-05-10 | not stated | not labelled (backlog) | none |
| [Pass 3](pass-03.md) | 2026-05-12 | not stated | not labelled (backlog) | none |
| [Pass 4](pass-04.md) | 2026-05-12 | not stated | 0.5.0 | none |
| [Pass 5](pass-05.md) | 2026-05-13 | not stated | 0.5.0 | none |
| [Pass 6](pass-06.md) | 2026-05-14 | 0.5.0 | 0.6.0 (one fix) | none |
| [Pass 7](pass-07.md) | 2026-05-18 | three days after 0.6.0 | 0.7.0 | none |
| [Pass 8](pass-08.md) | 2026-05-25 | ten days after 0.6.0 | recorded in the 0.7.0 notes | none |
| [Audit of 2026-07-08](audit-2026-07-08.md) | 2026-07-08 | 0.10.0 | 0.11.0 | work packages WP-A to WP-F |
| [Pass 9](pass-09.md) | 2026-07-28 | 0.14.0 | 0.14.1 and 0.15.0 | F01–F57 |
| [Pass 10](pass-10.md) | 2026-11-07 | 0.15.0 | 2026.11.0 | 10.01–10.95 |

## Reading the pages

- A page reproduces what was written at the time. File paths, test modules,
  notebook numbers and API names refer to the code of that time; later
  releases renamed or removed some of them (see the changelog).
- Passes 1 to 8 recorded their findings as a *backlog* of work to do. Not every
  backlog entry was implemented, and later passes re-listed what was still
  open, so the same finding can appear on several pages.
- Where a later pass found a published statement to be wrong, the page carries
  a correction box above the original text; the text itself is not changed.
  Pass 9 has such a box (the corrections of finding 10.82), and so has the
  pass-10 page for statements of the work-stream fragments that later audits
  found wrong.
- *Roadmap candidates* of a pass are ideas, not commitments; the current plan
  is on the [roadmap](../roadmap.md).

## Finding identifiers

- Passes 1 to 8 name findings by their subject only.
- The audit of 2026-07-08 groups its fixes into work packages WP-A to WP-F and
  quotes the audit's own finding labels (such as P4 and P5) in passing.
- Pass 9 numbers its findings F01 to F57.
- Pass 10 numbers its findings 10.01 to 10.95 and keeps the pass-9 number where
  a finding was already known (for example F01/F30 with 10.16).
- The adversarial audits of the release preparation of 2026.11.0 number their
  findings per audit and round (for example A2G-04, D3A-02, R6A-01, W1A-02,
  W3R2B-08); the pass-10 page records them with the streams that fixed them.
- From 2026.11.0 on, regression tests carry the marker
  `@pytest.mark.finding("10.NN")`; `pytest --finding 10.28` runs the tests of
  one finding.

## How these pages were made

The pages were cut from `CHANGELOG.md` by a script when the release notes were
condensed in 2026.11.0, so the text is the published text: reST roles are
rendered as code or math, one relative link points to the current page, and
every block stands under a heading that names where it was published. The
feature descriptions of the release notes were condensed in the changelog and
not moved; the complete text as published is in the history of
`CHANGELOG.md` in the repository (for example at the tag `v0.15.0`).
