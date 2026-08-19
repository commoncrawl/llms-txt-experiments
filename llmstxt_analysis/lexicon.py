"""Anchored lexicon matching.

The policy and abuse tracks each evaluate dozens of regexes against every
document. Running them all is wasteful: almost every document fails almost
every pattern. Each entry therefore carries a set of cheap lowercase *anchors*
- literal substrings that any match must contain - which are tested with
``in`` (C ``memmem``, ~GB/s) before the regex is run at all.

Semantics are identical to running every regex; only the cost changes. The
invariant "every string the regex can match contains at least one anchor" is
checked by ``tests/test_lexicon.py`` against the regexes' own examples.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    slug: str
    anchors: tuple[str, ...]
    pattern: re.Pattern
    level: int = 0


class Lexicon:
    """An ordered collection of anchored patterns."""

    def __init__(self, entries: list[Entry]):
        self.entries = entries

    def matches(self, text: str, text_lower: str | None = None) -> list[str]:
        low = text_lower if text_lower is not None else text.lower()
        out = []
        for e in self.entries:
            if e.anchors and not any(a in low for a in e.anchors):
                continue
            if e.pattern.search(text):
                out.append(e.slug)
        return out

    def matches_min(self, text: str, text_lower: str | None = None,
                    min_hits: int = 2) -> list[str]:
        """Slugs matching at least ``min_hits`` times.

        For topical lexicons a single occurrence is usually incidental — a car
        service that runs "casino trips", a charity's "casino night" — while
        genuine spam repeats the vocabulary throughout. Counting is capped by
        the caller's scan window, so it stays linear.
        """
        low = text_lower if text_lower is not None else text.lower()
        out = []
        for e in self.entries:
            if e.anchors and not any(a in low for a in e.anchors):
                continue
            hits = 0
            for _ in e.pattern.finditer(text):
                hits += 1
                if hits >= min_hits:
                    out.append(e.slug)
                    break
        return out

    def matches_with_level(self, text: str, text_lower: str | None = None) -> tuple[list[str], int]:
        low = text_lower if text_lower is not None else text.lower()
        out, level = [], 0
        for e in self.entries:
            if e.anchors and not any(a in low for a in e.anchors):
                continue
            if e.pattern.search(text):
                out.append(e.slug)
                level = max(level, e.level)
        return out, level


def build(spec: list[tuple], flags: int = re.I) -> Lexicon:
    """``spec`` entries are ``(slug, anchors, pattern)`` or ``(level, slug, anchors, pattern)``."""
    entries = []
    for item in spec:
        if len(item) == 4:
            level, slug, anchors, pat = item
        else:
            level, (slug, anchors, pat) = 0, item
        entries.append(
            Entry(slug, tuple(a.lower() for a in anchors), re.compile(pat, flags), level)
        )
    return Lexicon(entries)
