"""Golden-value regression tests (see ``tests/golden/README.md``).

Each case of :mod:`tests.golden.cases` is recomputed and compared with
the values stored from the last release. A number that moves fails the
test unless a stream declared the change, with its finding, in
``tests/golden/expected_changes/``.
"""

from __future__ import annotations

import pytest

from tests.golden import harness
from tests.golden.cases import CASES, run_case

pytestmark = pytest.mark.golden

_DECLARATIONS = harness.load_declarations()


def test_manifest_matches_stored_files() -> None:
    """Stored files and ``MANIFEST.json`` agree (no silent regeneration)."""
    problems = harness.manifest_problems(set(CASES))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_case(name: str, golden_report: list) -> None:
    if not harness.case_path(name).exists():
        pytest.fail(
            f"no stored golden values for new case {name!r}; run "
            f"'python -m tests.golden.harness generate --only {name}'"
        )
    comparisons = harness.compare_case(name, run_case(name), harness.load_case(name))
    failures = harness.judge(comparisons, _DECLARATIONS)
    golden_report.extend(comparisons)
    assert not failures, "\n".join(failures)
