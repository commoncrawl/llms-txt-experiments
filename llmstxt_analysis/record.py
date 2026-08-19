"""The unit of analysis: one WARC response record, plus derived views of it.

Derived views (lines, head, stripped prose, links, ...) are computed once per
record and shared by every extractor, so that the corpus text is scanned as few
times as possible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from urllib.parse import urlsplit

# Head window used by cheap prefix tests, generator detection and language ID.
HEAD_CHARS = 4000
# Window scanned by the policy/abuse lexicons. Bounded so that multi-megabyte
# llms-full.txt dumps do not dominate runtime; policy and injection text is
# overwhelmingly placed in the preamble (or, occasionally, the trailer).
SCAN_HEAD = 20000
SCAN_TAIL = 4000

_LINK_RE = re.compile(
    r"\[([^\]\n]{0,200})\]\("
    r"\s*(?:<([^>\n]{1,2000})>|([^)\s]{1,2000}))\s*"
    r"(?:\"[^\"]*\")?\)"
)
_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+")
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")
_NONASCII_RUN_RE = re.compile(r"[^\x00-\x7f]+")
# Space-separated scripts (Cyrillic, Greek, ...) yield one run per word while
# scripts written without spaces (Japanese, Chinese) yield one run per phrase.
# Collapsing adjacent placeholders makes the skeleton script-independent.
# Punctuation touching a run is absorbed too: a sentence ending in an ASCII "."
# and one ending in a full-width "。" must produce the same signature.
# \x01 is itself a non-word character, so one flat class spans a whole run of
# placeholders and the punctuation between them — no nested quantifiers, and
# therefore no backtracking blowup on long punctuation runs.
_PLACEHOLDER_RUN_RE = re.compile(r"[^\w\x02\x03]*\x01[^\w\x02\x03]*")
_MD_SYNTAX_RE = re.compile(r"[#*_`>\[\]()|~-]+")

BOM = "﻿"


def safe_host(url: str) -> str:
    """Hostname of ``url``, or "" if it cannot be parsed.

    Crawl data contains URLs that ``urlsplit`` rejects outright — most often a
    bracketed authority that is not a valid IPv6 literal, which raises
    ``ValueError: Invalid IPv6 URL``. One such link target in one document is
    enough to kill an extraction worker, so every parse goes through here.
    """
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def registered_domain(host: str) -> str:
    """Approximate registrable domain.

    Deliberately not a Public Suffix List lookup: this is only used to decide
    whether a link is on-site, where the two-or-three-label heuristic is
    accurate enough and orders of magnitude cheaper.
    """
    if not host:
        return ""
    labels = host.lower().rstrip(".").split(".")
    if len(labels) <= 2:
        return ".".join(labels)
    # co.uk, com.au, com.br, co.jp, ...
    if len(labels[-2]) <= 3 and len(labels[-1]) <= 3 and labels[-2] in _SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


_SECOND_LEVEL = {
    "co", "com", "net", "org", "gov", "edu", "ac", "or", "ne", "go", "in", "pp",
}


@dataclass  # not slots=True: cached_property needs an instance __dict__
class Record:
    """One llms.txt response, with its location in the corpus.

    ``dataset_index`` is the record's position in the Hugging Face dataset
    split — i.e. ``load_dataset(repo, config)["train"][dataset_index]`` is this
    record. It is the shard's base offset plus the row's position within the
    shard, which is exactly how the parquet files are concatenated. Storing it
    means a finding can be re-fetched from the dataset years later even if the
    live URL has changed or gone.
    """

    shard: int
    rg: int
    rg_row: int
    dataset_index: int
    url: str
    content: str
    http_headers: str
    warc_date: str
    warc_ip: str
    payload_type: str

    # ---- URL views -------------------------------------------------------
    @cached_property
    def _split(self):
        try:
            return urlsplit(self.url)
        except ValueError:
            return urlsplit("")

    @cached_property
    def host(self) -> str:
        try:
            return (self._split.hostname or "").lower()
        except ValueError:
            return ""

    @cached_property
    def regdomain(self) -> str:
        return registered_domain(self.host)

    @cached_property
    def tld(self) -> str:
        h = self.host
        return h.rsplit(".", 1)[-1] if "." in h else ""

    @property
    def is_full(self) -> bool:
        return self.url.rstrip("/").endswith("llms-full.txt")

    # ---- Text views ------------------------------------------------------
    @cached_property
    def body(self) -> str:
        """Content with any BOM removed."""
        return self.content[1:] if self.content[:1] == BOM else self.content

    @cached_property
    def head(self) -> str:
        return self.body[:HEAD_CHARS]

    @cached_property
    def head_lower(self) -> str:
        return self.head.lower()

    @cached_property
    def scan(self) -> str:
        """Bounded window used by the policy and abuse lexicons."""
        b = self.body
        if len(b) <= SCAN_HEAD + SCAN_TAIL:
            return b
        return b[:SCAN_HEAD] + "\n\n" + b[-SCAN_TAIL:]

    @cached_property
    def scan_lower(self) -> str:
        return self.scan.lower()

    @cached_property
    def lines(self) -> list[str]:
        return self.body.split("\n")

    @cached_property
    def stripped_lines(self) -> list[str]:
        return [ln.strip() for ln in self.lines]

    @cached_property
    def nonempty(self) -> list[str]:
        return [ln for ln in self.stripped_lines if ln]

    @cached_property
    def first_line(self) -> str:
        return self.nonempty[0] if self.nonempty else ""

    @cached_property
    def links(self) -> list[tuple[str, str]]:
        """Markdown links as (text, target), target de-angle-bracketed."""
        return [
            (m.group(1), (m.group(2) if m.group(2) is not None else m.group(3)).strip())
            for m in _LINK_RE.finditer(self.body)
        ]

    @cached_property
    def bare_urls(self) -> list[str]:
        return _URL_RE.findall(self.head)

    @cached_property
    def prose(self) -> str:
        """Head text with link bullets, URLs and markdown syntax removed.

        Used for language ID and topic modelling so that link dumps and
        template boilerplate do not dominate the signal.
        """
        keep = []
        n = 0
        for ln in self.stripped_lines:
            if not ln:
                continue
            if ln.startswith(("- [", "* [", "+ [")) or ln.startswith("|"):
                continue
            ln = _URL_RE.sub(" ", ln)
            ln = _LINK_RE.sub(r"\1", ln)
            ln = _MD_SYNTAX_RE.sub(" ", ln)
            ln = _WS_RE.sub(" ", ln).strip()
            if len(ln) < 3:
                continue
            keep.append(ln)
            n += len(ln)
            if n >= HEAD_CHARS:
                break
        return "\n".join(keep)[:HEAD_CHARS]

    @cached_property
    def skeleton(self) -> str:
        """Normalised template skeleton for exact-match near-duplicate grouping.

        The signature must be identical for two sites running the same
        generator, so the site-specific parts are dropped first: the title
        heading, the summary blockquote and every link bullet (which is where
        one site's page list differs from another's). What remains is the
        vendor's boilerplate and section structure. URLs, e-mail addresses,
        numbers and non-ASCII runs are then erased.

        Erasing non-ASCII runs makes the signature *script*-independent, not
        *language*-independent: a template whose boilerplate is translated
        still yields one cluster per language, because the surrounding Latin
        words differ. Detecting a generator by fingerprint is what unifies
        those; skeleton clustering is what catches template families whose
        generator is unknown.

        If nothing but site-specific lines remain, the full normalised text is
        used instead, so link-only documents are compared against each other
        rather than all collapsing onto the empty string.
        """
        keep = []
        n = 0
        for ln in self.body[:20000].split("\n"):
            s = ln.strip()
            if not s:
                continue
            if s.startswith("# ") or s.startswith(">") or s.startswith(("- ", "* ", "+ ", "|")):
                continue
            keep.append(s)
            n += len(s)
            if n >= 6000:
                break
        text = " ".join(keep)
        if len(text) < 80:
            text = self.body[:8000]
        s = text.lower()
        s = _URL_RE.sub(" \x02 ", s)
        s = _EMAIL_RE.sub(" \x03 ", s)
        s = _NONASCII_RUN_RE.sub("\x01", s)
        s = _PLACEHOLDER_RUN_RE.sub(" \x01 ", s)
        s = _NUM_RE.sub("0", s)
        s = _WS_RE.sub(" ", s)
        return s.strip()[:4000]

    @cached_property
    def headers(self) -> dict[str, str]:
        import json

        try:
            d = json.loads(self.http_headers) if self.http_headers else {}
        except (ValueError, TypeError):
            return {}
        return {str(k).lower(): str(v) for k, v in d.items()} if isinstance(d, dict) else {}
