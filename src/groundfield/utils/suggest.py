"""Name suggestions for error messages ("did you mean ...?").

Misspelled names are the most common input error in notebook work: an
electrode kind (``"rood"``), a keyword (``positon=``), an electrode
name in a conductor end. The helpers here turn such a miss into an
actionable message by naming the closest valid candidates. Matching is
case-insensitive and ranks candidates by the similarity ratio of
:class:`difflib.SequenceMatcher` (Ratcliff/Obershelp: twice the number
of matching characters over the total length), as
:func:`difflib.get_close_matches` does.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

__all__ = ["did_you_mean", "suggest"]


def suggest(
    name: str,
    candidates: Iterable[str],
    *,
    n: int = 3,
    cutoff: float = 0.6,
) -> list[str]:
    """Return the candidates closest to ``name``, best match first.

    Parameters
    ----------
    name
        The name that was not found.
    candidates
        Valid names. Duplicates are ignored; the original spelling of a
        candidate is returned.
    n
        Maximum number of suggestions.
    cutoff
        Similarity threshold in ``[0, 1]`` of
        :func:`difflib.get_close_matches`; candidates below it are not
        suggested.

    Returns
    -------
    list of str
        Up to ``n`` candidates, ordered by decreasing similarity and
        alphabetically among equally similar ones (so messages are
        reproducible). A case-insensitive exact match comes first; an
        empty list means nothing is close enough.
    """
    by_lower: dict[str, str] = {}
    for cand in candidates:
        by_lower.setdefault(str(cand).lower(), str(cand))
    key = str(name).lower()
    matcher = difflib.SequenceMatcher(b=key)
    scored: list[tuple[float, str]] = []
    for low, original in by_lower.items():
        matcher.set_seq1(low)
        ratio = 2.0 if low == key else matcher.ratio()
        if ratio >= cutoff:
            scored.append((-ratio, original))
    scored.sort()
    return [original for _, original in scored[:n]]


def did_you_mean(
    name: str,
    candidates: Iterable[str],
    *,
    n: int = 1,
    cutoff: float = 0.6,
) -> str:
    """Return a ``" Did you mean 'x'?"`` sentence, or ``""``.

    Parameters
    ----------
    name, candidates, n, cutoff
        See :func:`suggest`.

    Returns
    -------
    str
        A sentence with a leading space, ready to be appended to an
        error message, or the empty string when nothing is close.

    Examples
    --------
    >>> did_you_mean("positon", ["position", "length"])
    " Did you mean 'position'?"
    """
    hits = suggest(name, candidates, n=n, cutoff=cutoff)
    if not hits:
        return ""
    quoted = [f"'{h}'" for h in hits]
    if len(quoted) == 1:
        return f" Did you mean {quoted[0]}?"
    return f" Did you mean {', '.join(quoted[:-1])} or {quoted[-1]}?"
