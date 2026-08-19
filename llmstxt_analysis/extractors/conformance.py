"""Track A — conformance with the llms.txt specification (https://llmstxt.org/).

The spec:

    # Title                      <- required, the only required element
    > Summary blockquote         <- recommended
    Free prose (no headings)     <- optional
    ## Section
    - [name](url): notes         <- file list
    ## Optional                  <- may be skipped for shorter context

``doc_kind`` first decides whether the response is markdown at all; the
conformance ladder only applies to markdown documents.
"""
from __future__ import annotations

import re

from ..record import Record, registered_domain, safe_host
from ..registry import BOOL, F32, I32, LIST_STR, STR, RowExtractor, register

_H1_RE = re.compile(r"^# +\S", re.M)
_H2_RE = re.compile(r"^## +\S", re.M)
_H3_RE = re.compile(r"^#{3,6} +\S", re.M)
_LINK_BULLET_RE = re.compile(r"^ *[-*+] +\[[^\]\n]*\]\([^)\s]+\)", re.M)
_BULLET_NOTES_RE = re.compile(r"^ *[-*+] +\[[^\]\n]*\]\([^)\s]+\) *[:\-–] +\S", re.M)
_OPTIONAL_RE = re.compile(r"^## +optional\s*$", re.M | re.I)
_HTML_RE = re.compile(r"<\s*(?:html|body|head|div|script|meta|!doctype)\b", re.I)
_ROBOTS_RE = re.compile(r"^(?:user-agent|disallow|allow|sitemap|crawl-delay)\s*:", re.M | re.I)
_YAML_KEY_RE = re.compile(r"^[a-z_][a-z0-9_-]{2,30}:\s*(?:$|[\[\"'a-z0-9])", re.M | re.I)
_MD_MARKER_RE = re.compile(r"^(?:#{1,6} |> |[-*+] |\d+\. |```)", re.M)

CONFORMANCE_LABELS = {
    0: "not markdown",
    1: "markdown, no H1",
    2: "H1 only",
    3: "H1 + summary",
    4: "H1 + summary + link sections",
}


def _doc_kind(rec: Record) -> str:
    body = rec.body
    if not body.strip():
        return "empty"
    head = rec.head.lstrip()
    if head[:1] in "{[":
        return "json"
    if head.startswith("---") and _YAML_KEY_RE.search(head[:1500]):
        return "yaml_frontmatter"
    if _HTML_RE.search(rec.head[:1000]):
        return "html"
    robots_hits = len(_ROBOTS_RE.findall(rec.head))
    if robots_hits >= 2 and not _H1_RE.search(rec.head):
        return "robots_txt"
    # A bare policy document written as top-level YAML (no --- fence).
    if len(_YAML_KEY_RE.findall(rec.head[:1500])) >= 4 and not _H1_RE.search(rec.head[:1500]):
        return "yaml"
    if _MD_MARKER_RE.search(rec.head):
        return "markdown"
    return "plain"


@register
class ConformanceExtractor(RowExtractor):
    NAME = "conformance"
    TRACK = "A"
    FIELDS = {
        "doc_kind": STR,
        "has_bom": BOOL,
        "starts_with_h1": BOOL,
        "n_h1": I32,
        "h1_text": STR,
        "n_h2": I32,
        "n_h3": I32,
        "has_blockquote": BOOL,
        "blockquote_follows_h1": BOOL,
        "has_prose_body": BOOL,
        "has_optional_section": BOOL,
        "n_link_bullets": I32,
        "n_links": I32,
        "n_links_with_notes": I32,
        "n_relative_links": I32,
        "n_offsite_links": I32,
        "n_offsite_domains": I32,
        "offsite_ratio": F32,
        "conformance_level": I32,
        "conf_flags": LIST_STR,
    }

    def extract(self, rec: Record) -> dict:
        kind = _doc_kind(rec)
        body = rec.body
        head = rec.head

        h1s = _H1_RE.findall(body)
        n_h1 = len(h1s)
        n_h2 = len(_H2_RE.findall(body))
        n_h3 = len(_H3_RE.findall(body))

        # First non-empty line is an H1 (the spec's opening element).
        first = rec.first_line
        starts_with_h1 = first.startswith("# ")
        h1_text = ""
        for ln in rec.nonempty[:80]:
            if ln.startswith("# "):
                h1_text = ln[2:].strip()[:200]
                break

        # Summary blockquote: '>' line within the first few non-empty lines.
        lead = rec.nonempty[:8]
        has_bq = any(ln.startswith(">") for ln in lead)
        bq_follows_h1 = False
        for i, ln in enumerate(lead[:-1]):
            if ln.startswith("# ") and lead[i + 1].startswith(">"):
                bq_follows_h1 = True
                break

        # Prose body: a non-heading, non-bullet, non-blockquote line of real
        # length appearing before the first H2.
        has_prose = False
        for ln in rec.nonempty[:120]:
            if ln.startswith("## "):
                break
            if ln.startswith(("#", ">", "-", "*", "+", "|", "```")):
                continue
            if len(ln) >= 40:
                has_prose = True
                break

        links = rec.links
        n_links = len(links)
        n_rel = 0
        offsite = 0
        offdoms: set[str] = set()
        home = rec.regdomain
        for _txt, tgt in links:
            if tgt.startswith(("http://", "https://")):
                rd = registered_domain(safe_host(tgt))
                if home and rd and rd != home:
                    offsite += 1
                    offdoms.add(rd)
            elif tgt.startswith(("mailto:", "tel:", "#", "data:", "javascript:")):
                continue
            else:
                n_rel += 1

        n_bullets = len(_LINK_BULLET_RE.findall(body))
        n_notes = len(_BULLET_NOTES_RE.findall(body))
        has_optional = bool(_OPTIONAL_RE.search(body))

        if kind not in ("markdown", "yaml_frontmatter"):
            level = 0
        elif n_h1 == 0:
            level = 1
        elif not (has_bq and bq_follows_h1):
            level = 2
        elif n_h2 == 0 or n_bullets == 0:
            level = 3
        else:
            level = 4

        flags = []
        if n_h1 > 1:
            flags.append("multiple_h1")
        if n_h1 >= 1 and not starts_with_h1:
            flags.append("h1_not_first")
        if n_rel:
            flags.append("relative_links")
        if n_links == 0:
            flags.append("no_links")
        if has_optional:
            flags.append("has_optional_section")
        if n_links and n_notes == 0:
            flags.append("links_without_notes")
        if n_h3:
            flags.append("deep_headings")
        if rec.content[:1] == "﻿":
            flags.append("bom")
        if n_bullets and n_bullets < n_links / 2:
            flags.append("links_outside_bullets")

        return {
            "doc_kind": kind,
            "has_bom": rec.content[:1] == "﻿",
            "starts_with_h1": starts_with_h1,
            "n_h1": n_h1,
            "h1_text": h1_text,
            "n_h2": n_h2,
            "n_h3": n_h3,
            "has_blockquote": has_bq,
            "blockquote_follows_h1": bq_follows_h1,
            "has_prose_body": has_prose,
            "has_optional_section": has_optional,
            "n_link_bullets": min(n_bullets, 2**31 - 1),
            "n_links": min(n_links, 2**31 - 1),
            "n_links_with_notes": min(n_notes, 2**31 - 1),
            "n_relative_links": min(n_rel, 2**31 - 1),
            "n_offsite_links": min(offsite, 2**31 - 1),
            "n_offsite_domains": len(offdoms),
            "offsite_ratio": (offsite / n_links) if n_links else 0.0,
            "conformance_level": level,
            "conf_flags": flags,
        }
