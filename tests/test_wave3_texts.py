"""Texts of wave 3 that the audits W3A / W3B checked against measurements.

The numbers themselves are tested where they are computed
(`tests/test_carson_series_networks.py`, `tests/test_sommerfeld_long_segments.py`);
these tests keep the coordinator's texts from falling back to the figures the
audits refuted.
"""

from __future__ import annotations

import pathlib

import pytest

import groundfield as gf

ROOT = pathlib.Path(gf.__file__).resolve().parents[2]


def _text(relpath: str) -> str:
    return " ".join((ROOT / relpath).read_text(encoding="utf-8").split())


# audit W3B-02
@pytest.mark.finding("10.20")
def test_the_limitations_state_the_carson_series_error_of_networks_at_both_frequencies() -> (
    None
):
    """carson_series reads X 21 ... 44 % high at 50 Hz on generated networks
    (the 10-house radial +41 %); the limitations gave 9 ... 15 % at 1 kHz only."""
    kl = _text("docs/reference/known-limitations.md")
    assert "X 9 … 15 %" not in kl
    assert "X 4 … 21 % and R 2 … 11 % high at 1 kHz and X 21 … 44 % high at 50 Hz" in kl


# audit W3B-09
def test_the_pass10_rows_of_q34_and_q39_say_where_they_went() -> None:
    """Rows 34 and 39 said "next release" in their status column and linked a
    roadmap section that no longer lists them."""
    text = (ROOT / "docs/audit/pass-10.md").read_text(encoding="utf-8")
    rows = {
        line.split("|")[1].strip(): line
        for line in text.splitlines()
        if line.startswith(("| 34 |", "| 39 |"))
    }
    for q in ("34", "39"):
        assert "done in 2026.11.0" in rows[q].split("|")[4], q
        assert "scheduled-for-the-next-release" not in rows[q], q
