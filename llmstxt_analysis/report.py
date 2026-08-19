"""Render ``stats.json`` into one standalone HTML report.

The prose is composed from the statistics rather than written beside them, so
the narrative cannot drift from the numbers when the pipeline is re-run.

Styling follows the Common Crawl style guide (``cc-web-tools/STYLE-GUIDE.md``):
its colour tokens, Libre Franklin and IBM Plex Mono, 10/6/4px radii, the
uppercase 11px table headers and the understated underline link style. Fonts
are inlined, so the page still makes no external requests.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
from pathlib import Path

from . import charts

# Libre Franklin and IBM Plex Mono, vendored from cc-web-tools. The Common
# Crawl style guide requires shipping the faces locally rather than linking a
# CDN, and the artifact CSP blocks external font hosts anyway, so they are
# inlined as data URIs — the page then carries its own typography with no
# requests at all.
FONT_DIR = Path(__file__).parent / "assets" / "fonts"
FONTS = [
    ("Libre Franklin", "LibreFranklin_wght.woff2", "100 900"),
    ("IBM Plex Mono", "IBMPlexMono-Regular.woff2", "400 700"),
]


@lru_cache(maxsize=1)
def font_faces() -> str:
    """@font-face rules with the woff2 files embedded as data URIs."""
    out = []
    for family, filename, weight in FONTS:
        path = FONT_DIR / filename
        if not path.exists():
            print(f"warning: {path} missing; falling back to system fonts")
            continue
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        out.append(
            f"@font-face{{font-family:'{family}';font-weight:{weight};"
            f"font-style:normal;font-display:swap;"
            f"src:url(data:font/woff2;base64,{b64}) format('woff2')}}"
        )
    return "\n".join(out)

TITLE = "What is actually inside llms.txt"
SUBTITLE = "A content analysis of the llms.txt files in Common Crawl CC-MAIN-2026-30"

# Which dataset the example row indices address. Set from stats["meta"] at
# render time so the viewer links point at the corpus the numbers came from.
_DATASET = {"repo": "commoncrawl/llms.txt", "config": "CC-MAIN-2026-30", "split": "train"}


def set_dataset(repo: str, config: str, split: str = "train") -> None:
    _DATASET.update(repo=repo, config=config, split=split)


def viewer_url(idx: int) -> str:
    """Hugging Face dataset-viewer URL for one row of the split."""
    d = _DATASET
    return (
        f"https://huggingface.co/datasets/{d['repo']}/viewer/"
        f"{d['config']}/{d['split']}?row={int(idx)}"
    )


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------
def n(v) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return str(v)


def pc(v) -> str:
    return "—" if v is None else f"{float(v):.2f}%"


def pc1(v) -> str:
    return "—" if v is None else f"{float(v):.1f}%"


def _short_url(u: str) -> str:
    """Host plus filename — enough to recognise, short enough for a list."""
    s = u.split("://", 1)[-1]
    parts = s.split("/")
    host = parts[0]
    tail = parts[-1] if len(parts) > 1 else ""
    return f"{host}/{tail}" if tail else host


def _examples_cell(items: list) -> str:
    """A disclosure listing example records: live link + dataset row link.

    Two links per example on purpose. The first goes to the site as it is
    now; the second opens the archived record in the Hugging Face dataset
    viewer, which is what stays true after the live page changes or vanishes.
    """
    if not items:
        return '<td class="ex">—</td>'
    lis = "".join(
        f'<li><a href="{escape(str(e["url"]))}" rel="nofollow noopener" target="_blank">'
        f'{escape(_short_url(str(e["url"])))}</a>'
        f'<a class="idx" href="{escape(viewer_url(e["idx"]))}" target="_blank"'
        f' rel="noopener" title="Open row {int(e["idx"]):,} in the dataset viewer">'
        f'#{int(e["idx"]):,}</a></li>'
        for e in items
    )
    return (
        f'<td class="ex"><details><summary>{len(items)}</summary>'
        f"<ol>{lis}</ol></details></td>"
    )


def _cell(v) -> str:
    if isinstance(v, list):
        return _examples_cell(v)
    if v is None:
        return '<td class="num">—</td>'
    if isinstance(v, bool):
        return f"<td>{v}</td>"
    if isinstance(v, (int, float)):
        return f'<td class="num">{v:,.2f}</td>' if isinstance(v, float) else f'<td class="num">{v:,}</td>'
    s = str(v)
    if s.startswith("http"):
        return f'<td class="url">{escape(s[:70])}</td>'
    return f"<td>{escape(s)}</td>"


def tbl(t: dict, caption: str = "", limit: int = 0) -> str:
    if not t or not t.get("columns"):
        return ""
    rows = t["rows"][:limit] if limit else t["rows"]
    head = "".join(f"<th>{escape(c)}</th>" for c in t["columns"])
    body = "".join("<tr>" + "".join(_cell(v) for v in r) + "</tr>" for r in rows)
    cap = f"<caption>{escape(caption)}</caption>" if caption else ""
    return f'<div class="tw"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def tiles(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="tile"><div class="tv">{escape(v)}</div>'
        f'<div class="tl">{escape(l)}</div>'
        f'<div class="th">{escape(h)}</div></div>'
        for v, l, h in items
    )
    return f'<div class="tiles">{cells}</div>'


def section(sid: str, kicker: str, title: str, body: str) -> str:
    return (
        f'<section id="{sid}"><div class="kicker">{escape(kicker)}</div>'
        f"<h2>{escape(title)}</h2>{body}</section>"
    )


def p(text: str) -> str:
    return f"<p>{text}</p>"


def note(text: str) -> str:
    return f'<p class="note">{text}</p>'


def _row(t: dict, key, col: int = 1, keycol: int = 0):
    for r in t.get("rows", []):
        if str(r[keycol]) == str(key):
            return r[col]
    return 0


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def sec_overview(s: dict) -> str:
    o, F, B, A = s["overview"], s["F"], s["B"], s["A"]
    tok = F["tokens"]
    med = _row(tok, "p50", 1)
    p99 = _row(tok, "p99", 1)
    mx = _row(tok, "max", 1)
    body = [
        p(
            f"Those {n(o['n_responses'])} surviving responses are the corpus: every URL ending in "
            f"<code>llms.txt</code> or <code>llms-full.txt</code> that returned HTTP 200 with a "
            f"<code>text/plain</code> or <code>text/markdown</code> body, extracted from the WARC "
            f"archives. This report asks what is written in them. Published adoption studies count "
            f"who has the file and who fetches it; none of them open it."
        ),
        p(
            f"<strong>{n(o['n_empty'])}</strong> of those responses ({pc(o['pct_empty'])}, across "
            f"{n(o['empty_hosts'])} hosts) have an empty body: a 200, the right content type, and "
            f"nothing in it. They are a finding about deployment, not documents to analyse — every "
            f"empty body is identical, so leaving them in would manufacture a template cluster and "
            f"inflate the share of files with no links, no title and no generator. They are counted "
            f"here and excluded from every figure that follows, which is computed over the "
            f"<strong>{n(o['n_records'])}</strong> responses that contain something."
        ),
        tbl(o["intake"], "What the crawl returned"),
        tiles([
            (n(o["n_records"]), "documents", f"{n(o['n_hosts'])} distinct hosts"),
            (pc1(o["pct_template"]), "generated from a template", "not written by a person"),
            (pc1(A["pct_links_with_notes"]), "annotate any link", "the format's whole point"),
            (n(med), "median tokens", f"p99 {n(p99)} · max {n(mx)}"),
        ]),
        p(
            f"Three findings organise everything below. First, this is <strong>plugin output, not "
            f"curation</strong>: {pc1(B['template_share'])} of documents are template-generated, and the "
            f"ten largest template clusters alone account for {pc1(B['top10_template_share'])} of the "
            f"whole corpus. Second, the file has quietly become a <strong>policy and "
            f"service-discovery channel</strong> — {pc1(s['D']['pct_any_policy'])} carry usage or "
            f"rights language and {pc1(B['mcp']['pct_mentions_mcp'])} advertise an agent endpoint, "
            f"neither of which the specification mentions. Third, it is <strong>unguarded</strong>: "
            f"{pc(s['E']['for_sale']['pct'])} are parked domains pitching themselves to agents, "
            f"{pc(s['E']['pct_spam'])} hit a spam lexicon, and a small but non-zero share carry "
            f"instructions written at the reading model."
        ),
        tbl(o["file_kind"], "Corpus composition by filename"),
        charts.from_series(o["tld_chart"], title="Top-level domains (documents)"),
        tbl(o["tld"], "Top-level domains", limit=15),
    ]
    return section("overview", "Overview", "The corpus", "".join(body))


def sec_index(s: dict) -> str:
    """The funnel that precedes the content analysis, from the URL index."""
    I = s.get("index")
    if not I:
        return ""
    m = s["meta"]
    SD = I.get("seeding") or {}
    body = [
        p(
            f"Before anything can be read, the file has to exist. This corpus is the result of "
            f"an experiment rather than an incidental find: for {escape(SD.get('crawl', 'this crawl'))} "
            f"the paths <code>/llms.txt</code> and <code>/llms-full.txt</code> were added to the "
            f"crawl's seed list on purpose, and what follows is the outcome of "
            f"<strong>{n(I['total'])}</strong> attempted URLs."
        ),
    ]
    if SD:
        body.append(p(
            f"The seeds were drawn two ways. Most are a random sample of hosts Common Crawl had "
            f"recently fetched without problems — {n(SD['sampled_llms_txt'])} URLs for "
            f"<code>/llms.txt</code>, and a deliberately narrower "
            f"{n(SD['sampled_llms_full'])} for <code>/llms-full.txt</code>, since a site with no "
            f"<code>/llms.txt</code> rarely publishes the longer file. The rest are hosts already "
            f"known from the two preceding crawls "
            f"({' and '.join(f'<code>{escape(c)}</code>' for c in SD['prior_crawls'])}) to serve "
            f"one of them: {n(SD['known_hosts_llms_txt'])} with a <code>/llms.txt</code> and "
            f"{n(SD['known_hosts_llms_full'])} with a <code>/llms-full.txt</code>. Not every seed "
            f"is fetched within a single crawl, which is why the totals below fall short of the "
            f"sample. Because the first group is a random sample, the 200-plus-text rate for "
            f"<code>/llms.txt</code> reads as an adoption rate for hosts Common Crawl can fetch. "
            f"The <code>/llms-full.txt</code> rate does not, because its sample was enriched with "
            f"hosts already known to serve the file."
        ))
    body += [
        note(
            "The query below reads the URL index after the fact, to recover the outcome of each "
            "attempt. It is not what selected the population. Quoted verbatim, never re-run here."
        ),
        f'<pre class="sql"><code>{escape(I["query"])}</code></pre>',
        tiles([
            (n(I["total"]), "URLs attempted", "in this one crawl"),
            (pc1(I["pct_404"]), "returned 404", "no such file"),
            (pc1(I["pct_200"]), "returned 200", "the file is served"),
            (pc1(I["pct_text_of_200"]), "of those 200s carry text",
             "the rest are HTML, robots.txt, JSON…"),
        ]),
        tbl(I["funnel"], "From attempted URL to analysable document"),
        p(
            f"<strong>{pc1(I['pct_404'])}</strong> of attempts 404 and another "
            f"{pc(I['pct_3xx'])} redirect, which is unsurprising: the crawler tries the path on "
            f"every host it knows. What is surprising is the {n(I['n_status_codes'])} distinct "
            f"status codes it received, and how little of the 200 population is actually a "
            f"document — only <strong>{pc(I['pct_text_of_200'])}</strong> of successful responses "
            f"carry a <code>text/plain</code> or <code>text/markdown</code> body."
        ),
        charts.from_series(I["status_chart"], title="HTTP response to the attempted URL"),
        tbl(I["status"], "Responses, grouped"),
        p(
            "The content types tell you what the other 200s are. A site that serves its "
            "single-page app for every unknown path answers <code>text/html</code>; one whose "
            "rewrite rules send <code>/llms.txt</code> to <code>/robots.txt</code> answers "
            "<code>text/x-robots</code>. Both are a 200 that an agent would have to parse and "
            "discard."
        ),
        charts.from_series(I["mime_chart"], title="Detected content type of HTTP 200 responses"),
        tbl(I["mime"], "Content types among HTTP 200 responses"),
        p(
            f"Split by filename, <code>llms-full.txt</code> is both rarer and much more often "
            f"misconfigured: it is attempted on {n(I['by_kind']['rows'][1][1])} URLs but only "
            f"{I['by_kind']['rows'][1][6]:.2f}% of its 200s carry a text body, against "
            f"{I['by_kind']['rows'][0][6]:.2f}% for <code>llms.txt</code>."
        ),
        tbl(I["by_kind"], "By filename"),
        p(
            f"One reconciliation note, because the two populations are not quite the same. The "
            f"index query above matches the path exactly, while the WARC extraction that produced "
            f"the dataset matched <code>LIKE '%/llms.txt'</code> and so also captured files deeper "
            f"in a site — {n(I['n_deep'])} of them ({pc(I['pct_deep'])}), things like "
            f"<code>/branding/vesence/llms.txt</code> or one file per product. Set those aside and "
            f"the {n(m['responses'])} extracted records leave {n(I['n_root'])} at a root path"
            + (
                f", exactly the {n(I['n_text200'])} the index counts."
                if I["n_root"] == I["n_text200"]
                else f", against {n(I['n_text200'])} in the index — a difference of "
                     f"{n(abs(I['n_root'] - I['n_text200']))} records."
            )
        ),
        tbl(I["reconcile"], "How the index total becomes the analysed corpus"),
    ]
    return section("index", "Before reading", "Most of these files do not exist",
                   "".join(body))


def sec_a(s: dict) -> str:
    A = s["A"]
    md_share = 100.0 * A["n_markdown"] / max(s["meta"]["records"], 1)
    lvl0 = _row(A["conformance"], "not markdown", 2)
    body = [
        p(
            "The <a href=\"https://llmstxt.org/\">llms.txt specification</a> is small enough to test "
            "mechanically: an H1 title (the only required element), a <code>&gt;</code> blockquote "
            "summary, optional heading-free prose, then <code>##</code> sections of "
            "<code>- [name](url): notes</code> bullets, optionally including an <code>## Optional</code> "
            "section that a short-context reader may skip."
        ),
        p(
            f"{pc1(md_share)} of documents are markdown at all. The rest — {pc(lvl0)} — are JSON "
            f"error blobs, HTML error pages, robots.txt files served under the wrong name, and YAML "
            f"policy documents, every one of them returned as HTTP 200 with a "
            f"<code>text/plain</code> or <code>text/markdown</code> content type. (The "
            f"{n(s['meta']['n_empty'])} entirely empty responses are excluded here and counted in "
            f"the overview.)"
        ),
        charts.from_series(A["doc_kind_chart"], title="What the response actually is"),
        tbl(A["doc_kind"], "Document kinds"),
        p(
            f"The surface form travels well. <strong>{pc(A['pct_level4'])}</strong> of documents "
            f"carry the full spec shape — H1, summary blockquote, H2 sections, link bullets — and "
            f"that is not an artefact of one generator: {pc(A['pct_level4_human'])} of "
            f"human-authored documents reach it against {pc(A['pct_level4_template'])} of templated "
            f"ones. Whatever else is wrong with the llms.txt web, people did read the spec."
        ),
        charts.from_series(A["conformance_chart"], title="Conformance ladder (documents)"),
        tbl(A["conformance"], "Conformance ladder"),
        tbl(A["conformance_by_file"], "Conformance by filename (% within each file type)"),
        p(
            f"What does not travel is the part that carries the value. The specification's purpose "
            f"is to hand an agent a <em>curated, annotated</em> list of links, and that is precisely "
            f"where adherence collapses: only <strong>{pc(A['pct_links_with_notes'])}</strong> of "
            f"documents annotate any link with the <code>: notes</code> the format allows, the "
            f"<code>## Optional</code> section that lets a short-context reader skip material "
            f"appears in <strong>{pc(A['pct_optional'])}</strong>, and "
            f"<strong>{pc1(A['pct_zero_link'])}</strong> of documents contain no link at all "
            f"({pc1(A['pct_zero_link_markdown'])} of the markdown ones). The shape is followed; the "
            f"substance is not."
        ),
        tbl(A["elements"], "Individual spec elements"),
        tbl(A["defects"], "Structural flags, by frequency", limit=12),
        tbl(A["links"], "Links per document"),
        tbl(A["zero_link_by_generator"], "Zero-link rate by generator"),
    ]
    return section("spec", "Specification", "The shape is followed; the substance is not",
                   "".join(body))


def sec_b(s: dict) -> str:
    B = s["B"]
    top = B["generator"]["rows"][0] if B["generator"]["rows"] else ["?", 0, 0]
    body = [
        p(
            "Detection runs in two layers: an explicit self-declaration with a version string "
            "(\"Generated by Yoast SEO v28.0, this is an llms.txt file\"), and, for the silent "
            "generators, a language-independent structural fingerprint — for the site builders, the "
            "help-centre URL they embed rather than the prose around it, so that localised templates "
            "are still detected."
        ),
        charts.from_series(B["generator_chart"], title="Producers (documents)"),
        tbl(B["generator"], "Generators", limit=14),
        tbl(B["family"], "Generator families"),
        p(
            f"The largest single producer is <code>{escape(str(top[0]))}</code> at {pc(top[2])} of the "
            f"corpus. Together with the WordPress SEO plugins, a handful of vendors decide what most of "
            f"the llms.txt web says — and they interpreted the same informal spec in mutually "
            f"incompatible ways: some emit a curated index, some dump the sitemap, some emit no links "
            f"at all and advertise an API instead."
        ),
        tbl(B["generator_vs_conformance"], "How each producer treats the spec"),
        p(
            "Template detection needs no statistical model. Dropping the parts that are "
            "site-specific by construction — the title, the summary blockquote, the link bullets — "
            "and then erasing URLs, e-mail addresses, numbers and every non-ASCII run leaves the "
            "vendor's boilerplate and section structure. Exact matching on that skeleton is enough: "
            "one template collapses to a single string across thousands of sites. Translated "
            "boilerplate forms one cluster per language, which is what the generator fingerprints "
            "unify; skeleton clustering is what catches template families whose producer is unknown."
        ),
        tbl(B["top_templates"], "Largest template clusters"),
        p(
            f"{pc1(B['template_share'])} of the corpus is templated by this measure; the ten biggest "
            f"clusters alone are {pc1(B['top10_template_share'])}. Every rate elsewhere in this report is "
            f"therefore reported against the human-authored remainder wherever the distinction matters."
        ),
        p(
            f"One template family deserves its own line. <strong>{pc1(B['mcp']['pct_mentions_mcp'])}</strong> "
            f"of documents mention the Model Context Protocol and {pc1(B['mcp']['pct_mcp_endpoint'])} publish "
            f"a live MCP endpoint URL. For those sites llms.txt is not a link index at all: it is a "
            f"service-discovery record telling agents to stop scraping and call an API instead. The "
            f"specification does not mention this use."
        ),
        tbl(B["mcp"]["by_generator"], "Which producers advertise an MCP endpoint"),
        "<h3>Files that were pasted out of a chat window</h3>",
        p(
            f"A third production route leaves its own traces. "
            f"{pc(B['pct_llm_artefacts'])} of documents carry a marker that only appears when "
            f"someone asked an assistant to write the file and shipped the answer unedited: "
            f"ChatGPT's <code>contentReference[oaicite:…]</code> citation markers, an unfilled "
            f"<code>[Insert company name]</code> placeholder, a leftover <code>```markdown</code> "
            f"fence around the whole document, or a <code>utm_source=chatgpt</code> link. The "
            f"topic model found these on its own — whole clusters of clinics and agencies held "
            f"together by the token <code>oaicite</code>."
        ),
        tbl(B["llm_artefacts"], "Traces of hand-pasted assistant output"),
        tbl(B["artefacts_by_generator"], "Where the pasted files live"),
        tbl(B["versions"], "Declared generator versions", limit=15),
    ]
    return section("generators", "Who writes them", "Plugin output, not curation", "".join(body))


def sec_d(s: dict) -> str:
    D = s["D"]
    body = [
        p(
            "Nothing in the specification concerns rights. In practice a measurable share of "
            "documents use llms.txt to state terms — and they do so in at least four mutually "
            "incompatible dialects: prose paragraphs, YAML permission blocks, robots.txt syntax "
            "served under the llms.txt name, and links out to a licence page."
        ),
        tiles([
            (pc1(D["pct_any_policy"]), "carry policy language", "usage, rights or licensing"),
            (pc1(D["pct_docs_naming_bots"]), "name a specific crawler", "allow- or deny-lists"),
            (pc1(D["policy_heading"]), "have a policy heading", "an explicit section"),
            (pc1(D["contact_email"]), "publish a contact address", "for licensing or corrections"),
        ]),
        charts.from_series(D["dialect_chart"], title="Policy dialects (documents with a policy)"),
        tbl(D["dialect"], "How the policy is expressed"),
        tbl(D["directives"], "Directives found, by frequency", limit=16),
        tbl(D["training_stance"], "Stance on training use"),
        p(
            "The named-crawler table is the closest thing this corpus has to a robots.txt of the agent "
            "era. It is also where a publisher's intent is least ambiguous: naming a user-agent and "
            "putting <code>allow</code> or <code>deny</code> next to it is a deliberate act."
        ),
        tbl(D["named_bots"], "Named crawlers and their verdicts", limit=20),
        note(
            "A verdict is read from the same line the crawler is named on. Documents that name a "
            "crawler without any allow/deny marker are counted in the first column only, which is why "
            "allowed + denied does not sum to the total."
        ),
        tbl(D["policy_by_generator"], "Which producers emit policy language"),
    ]
    return section("policy", "Rights and crawlers", "A policy channel the spec never defined", "".join(body))


def sec_e(s: dict) -> str:
    E = s["E"]
    body = [
        p(
            "llms.txt is text an agent reads <em>as guidance</em>, published at a predictable path, "
            "inspected by no security tooling. The lexicon below grades what publishers put there, "
            "from benign steering through self-promotion to outright instruction override. It is "
            "deliberately conservative: high precision by construction, unknown recall."
        ),
        tiles([
            (pc(E["pct_injection_ge2"]), "instruct the reading model", "promotional or stronger"),
            (pc(E["pct_injection_ge3"]), "attempt an override", "\"ignore previous instructions\" class"),
            (pc(E["for_sale"]["pct"]), "are domains for sale", f"{n(E['for_sale']['n'])} documents"),
            (pc(E["pct_spam"]), "hit a spam lexicon", "gambling, adult, pharma, …"),
        ]),
        charts.from_series(E["injection_chart"], title="Documents containing model-directed instructions"),
        tbl(E["injection"], "Instruction severity"),
        note(
            "The override row is small enough to audit exhaustively, and it was: all of its "
            "documents were read individually. Four are genuine — a bug-bounty researcher's "
            "labelled payload catcher, two protest files, and one &ldquo;LLM Training Policy&rdquo; "
            "whose summary blockquote instructs the reader to ignore its instructions and fetch "
            "another URL. The rest are false positives of two kinds: technical writing that "
            "contains the literal ChatML control tokens because it is explaining them, and an "
            "incidental &ldquo;you are now&rdquo; in a page description. Read the override count as "
            "an upper bound on candidates, not a count of attacks — and note that every genuine "
            "case was placed deliberately by its author, rather than smuggled in."
        ),
        tbl(E["injection_matches"], "Which patterns matched", limit=12),
        tbl(E["injection_human_only"], "Instruction severity — human-authored documents only"),
        p(
            f"Parked domains are the corpus's most absurd corner. {n(E['for_sale']['n'])} documents "
            f"({pc(E['for_sale']['pct'])}) advertise the domain itself for sale, and "
            f"{pc(E['for_sale']['pct_wellformed'])} of those are well-formed enough to clear conformance "
            f"level 2 — someone wrote a spec-shaped sales pitch aimed at a language model."
        ),
        tbl(E["for_sale"]["marketplace"], "Parked-domain marketplaces"),
        charts.from_series(E["spam_chart"], title="Primary spam lexicon hit"),
        tbl(E["spam"], "Spam lexicons"),
        p(
            f"Link injection is visible too. {pc(E['offsite']['pct_dominant'])} of documents send the "
            f"majority of their links to five or more third-party domains, and "
            f"{pc(E['lang_mismatch']['pct'])} are written in a language implausible for their ccTLD — the "
            f"classic signature of a compromised site republished as a link farm."
        ),
        tbl(E["lang_mismatch"]["top"], "Largest ccTLD / content-language mismatches", limit=12),
        tbl(E["offsite"]["offsite_ratio"], "Share of links pointing off-domain (%)"),
        tbl(E["suspicious_by_generator"], "Which producers host the suspicious documents"),
    ]
    return section("abuse", "Abuse", "Unguarded, and already being used", "".join(body))


def sec_f(s: dict) -> str:
    F = s["F"]
    o = s["overview"]
    tok = F["tokens"]
    body = [
        p(
            "Language is identified with "
            '<a href="https://github.com/commoncrawl/commonlid-eval">commonlid</a>, Common Crawl\'s '
            "own LID evaluation kit: <code>cld2</code> as the primary model with <code>GlotLID</code> "
            "as the fallback where cld2 abstains. Input is the document's prose with link bullets, "
            "URLs and markdown syntax stripped, so that link dumps and translated boilerplate do not "
            "decide the label."
        ),
        charts.from_series(F["lang_chart"], title="Content languages (documents)"),
        tbl(F["lang"], "Languages (ISO 639-3)", limit=18),
        tbl(F["lang_source"], "Which LID model produced the label"),
        tbl(F["lang_by_generator"], "Language reach by producer"),
        p(
            f"Length is where the specification's premise — a small, cheap, curated file — meets what "
            f"was actually shipped. The median document is {n(_row(tok,'p50',1))} tokens; the 99th "
            f"percentile is {n(_row(tok,'p99',1))}, and the largest is {n(_row(tok,'max',1))} tokens: a "
            f"single file whose entire purpose is being inexpensive to read."
        ),
        charts.loghist(F["tokens_hist"]["edges"], F["tokens_hist"]["counts"],
                       title="Document size distribution", xlabel="tokens (log scale)"),
        tbl(tok, "Tokens per document"),
        tbl(F["tokens_by_file"], "Tokens by filename"),
        tbl(F["context_fit"], "Documents that exceed a context window"),
        p(
            f"Ingesting the whole corpus once costs the following, at "
            f"{n(o['total_tokens'])} input tokens (counted with <code>o200k_base</code>; other "
            f"tokenizers differ by roughly ±20%). List prices come from litellm's model cost map, "
            f"charging each document as its own request — pricing the corpus as one call would "
            f"trigger the long-context tier several providers apply above ~200k tokens and roughly "
            f"double the figure."
        ),
        tbl(F["cost"], "Cost to read every llms.txt in this crawl, once"),
    ]
    if s.get("topics"):
        body.append(_topics_block(s["topics"]))
    return section("langlen", "Reach and cost", "Language, length, topics and what it costs to read",
                   "".join(body))


def _topics_block(t: dict) -> str:
    out = [
        "<h3>Topics</h3>",
        p(
            "Topics come from an LDA model over the human-authored subset — templates removed, one "
            "document per template skeleton — fitted per language on a 2 KB prose excerpt. Topic "
            "<em>names</em> are not machine-generated: the model's top terms and representative "
            "documents were read by hand and labelled."
        ),
    ]
    for lang, block in t.get("languages", {}).items():
        rows = [[x.get("name", f"topic {x['id']}"), x["share"], x["n_docs"],
                 ", ".join(x["terms"][:8]),
                 [e for e in x.get("examples", []) if "idx" in e]] for x in block["topics"]]
        out.append(tbl(
            {"columns": ["topic", "% of subset", "documents", "top terms", "examples"],
             "rows": rows},
            f"LDA topics — {lang} ({n(block['n_docs'])} documents)"))
        out.append(charts.hbar([r[0] for r in rows][:12], [r[1] for r in rows][:12],
                               unit="%", title=f"Topic shares — {lang}"))
    if t.get("skipped_languages"):
        out.append(note(
            "Languages written without whitespace word boundaries ("
            + escape(", ".join(t["skipped_languages"]))
            + ") need a segmenter and are excluded from the topic model."))
    return "".join(out)


def sec_methods(s: dict) -> str:
    m = s["meta"]
    body = [
        p(
            f"Every grouped result carries five example documents, each with two links: the site "
            f"as it is now, and <code>#index</code> — the record's row in the Hugging Face dataset "
            f"split, which opens in the "
            f'<a href="{escape(viewer_url(0).rsplit("?", 1)[0])}">dataset viewer</a>. The index is '
            f"the durable reference; the same row is reachable in code as "
            f'<code>load_dataset("{escape(m["dataset"])}", "{escape(m["config"])}")'
            f'["train"][index]</code>, and returns the exact bytes a figure was computed from long '
            f"after the live page has changed or gone."
        ),
        p(
            f"Source: the Hugging Face dataset <code>{escape(m['dataset'])}</code>, config "
            f"<code>{escape(m['config'])}</code> — WARC response records for <code>*/llms.txt</code> and "
            f"<code>*/llms-full.txt</code> with HTTP 200 and a <code>text/plain</code> or "
            f"<code>text/markdown</code> body, {n(m['records'])} documents. Analysis code: "
            f"<code>analyze.py</code> in the <code>llms-txt-experiments</code> repository; every number "
            f"here is produced by <code>analyze.py aggregate</code> from a single streaming pass over "
            f"the corpus."
        ),
        "<h3>Limitations</h3>",
        "<ul>"
        "<li>One crawl snapshot. Nothing here supports a claim about growth or change over time.</li>"
        "<li>The corpus is HTTP 200 plus <code>text/plain|markdown</code> only. Sites that serve "
        "llms.txt as <code>text/html</code>, redirect, or 404 are absent — in this crawl those are the "
        "majority of all <code>/llms.txt</code> URL attempts.</li>"
        "<li>The seeds are a random sample only within the set of hosts Common Crawl fetches "
        "successfully, which is not the same population as the web. The <code>/llms.txt</code> "
        "success rate is therefore an adoption estimate for that population and not for the web "
        "at large, and the <code>/llms-full.txt</code> rate is not an adoption estimate at all, "
        "its sample having been enriched with hosts already known to serve the file.</li>"
        "<li>Instruction severity, policy directives and spam categories are lexicons, not classifiers. "
        "They are tuned for precision and were validated by reading sampled matches; recall is "
        "unknown and every rate here is a lower bound.</li>"
        "<li>A spam-lexicon hit means the vocabulary is present at least twice, not that the site "
        "is spam — though in practice most hits are hijacked legitimate sites rather than "
        "purpose-built ones. Successive spot checks removed bare <code>slot</code> (a fifth of its "
        "matches were appointment slots, a Danish castle and a football manager), word-bounded "
        "<code>judi</code> (it was matching \"judicial\"), dropped bare <code>xxx</code> (it "
        "matched the Roman numeral in \"XXXI Velada Musical\"), and required two occurrences "
        "rather than one, which excluded a car service advertising \"casino trips\". Measured "
        "precision after those changes is about seven in eight.</li>"
        "<li>Policy and abuse lexicons scan the first 20 KB and last 4 KB of each document, so text "
        "buried in the middle of a multi-megabyte dump can be missed.</li>"
        "<li>Token counts use one tokenizer (<code>o200k_base</code>).</li>"
        f"<li>{n(m['n_empty'])} responses ({pc(m['pct_empty'])}) had an empty body and are excluded "
        f"from every figure except the intake table in the overview. They are all byte-identical, so "
        f"including them would create a spurious template cluster and drag down the rates for links, "
        f"titles and generators.</li>"
        f"<li>A document counts as templated when its normalised skeleton is shared by at least "
        f"{m['template_min_cluster']} documents, or when a generator was positively identified.</li>"
        "<li>LDA topic labels were assigned by a human reading top terms and example documents — "
        "reproducible, but a judgement call.</li>"
        "</ul>",
    ]
    return section("methods", "Method", "How this was measured, and what it cannot show",
                   "".join(body))


# --------------------------------------------------------------------------
CSS = """
/* Common Crawl visual language — see cc-web-tools/STYLE-GUIDE.md. Colours,
   fonts, radii and the table-header treatment come from cc-base.css; every
   value below is one of its tokens rather than a new one.

   Single theme by choice: the Common Crawl palette is defined for light
   surfaces only, and inventing a dark counterpart would mean inventing greys
   the design system deliberately does not have. Every colour is therefore
   painted explicitly, so the page holds its own on any host background. */
:root{
  color-scheme: light;

  /* text */
  --cc-text:#152a47;
  --cc-text-secondary:#64748b;
  --cc-text-muted:#94a3b8;
  --cc-text-subtle:#475569;

  /* accent */
  --cc-accent:#2e5f8a;
  --cc-accent-hover:#1e4a6e;

  /* backgrounds */
  --cc-bg-page:#f5f7fa;
  --cc-bg-card:#fff;
  --cc-bg-inset:#f8fafb;
  --cc-bg-hover:#f8fafc;
  --cc-bg-active:#eef4fa;
  --cc-bg-code:#f1f5f9;

  /* borders */
  --cc-border:#e2e8f0;
  --cc-border-input:#dde2e9;
  --cc-border-hover:#cbd5e1;
  --cc-border-subtle:#f1f5f9;
  --cc-border-accent:#b8c9e0;

  /* links: inherit the body colour, distinguished by a light underline */
  --cc-link-underline:#c0c8d4;
  --cc-link-underline-hover:#152a47;

  --cc-shadow-sm:0 1px 3px rgba(0,0,0,.04);
  --cc-radius:10px;
  --cc-radius-sm:6px;
  --cc-radius-xs:4px;
  --cc-max-width:1200px;
  --cc-gutter:20px;

  --cc-font-body:'Libre Franklin','Segoe UI',system-ui,-apple-system,sans-serif;
  --cc-font-mono:'IBM Plex Mono',ui-monospace,'SFMono-Regular',Menlo,monospace;

  /* Chart series, taken from the twelve named accents so plots sit in the
     same world as badges and chips. Sapphire/CC-blue leads; the rest are only
     reached by multi-series charts. */
  --series-1:#2e5f8a;
  --series-2:#846730;
  --series-3:#2b674f;
  --series-4:#5b437f;
  --grid:#e2e8f0;
  --axis:#cbd5e1;
}
*{box-sizing:border-box}
body{
  margin:0;
  background:var(--cc-bg-page);
  color:var(--cc-text);
  font:15px/1.65 var(--cc-font-body);
  -webkit-text-size-adjust:100%;
  text-size-adjust:100%;
  -webkit-font-smoothing:antialiased;
}
:focus-visible{outline:2px solid var(--cc-accent);outline-offset:3px;border-radius:var(--cc-radius-xs)}
.wrap{max-width:var(--cc-max-width);margin:0 auto;padding:32px var(--cc-gutter) 64px}

header{border-bottom:1px solid var(--cc-border);padding-bottom:24px;margin-bottom:8px}
h1{font-size:26px;font-weight:700;line-height:1.2;margin:0 0 8px;text-wrap:balance}
.sub{color:var(--cc-text-secondary);font-size:15px;margin:0 0 16px;text-wrap:pretty}
.meta{color:var(--cc-text-muted);font-size:12px;font-family:var(--cc-font-mono)}
nav{margin:20px 0 0;display:flex;flex-wrap:wrap;gap:8px}
nav a{
  font-size:12px;font-weight:600;text-decoration:none;color:var(--cc-text-secondary);
  background:var(--cc-bg-card);border:1px solid var(--cc-border-input);
  border-radius:var(--cc-radius-sm);padding:6px 12px;transition:color .15s,border-color .15s;
}
nav a:hover{color:var(--cc-accent);border-color:var(--cc-border-accent)}

section{padding:32px 0 4px;border-top:1px solid var(--cc-border);margin-top:32px}
section:first-of-type{border-top:none}
.kicker{
  font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  color:var(--cc-accent);margin-bottom:6px;
}
h2{font-size:20px;font-weight:700;line-height:1.25;margin:0 0 14px;text-wrap:balance}
h3{font-size:17px;font-weight:700;line-height:1.25;margin:32px 0 10px;text-wrap:balance}
p{margin:0 0 16px;color:var(--cc-text-subtle);text-wrap:pretty}
p strong{color:var(--cc-text);font-weight:600}
p.note{
  color:var(--cc-text-secondary);font-size:14px;
  border-left:2px solid var(--cc-border-hover);padding-left:12px;
}
ul{padding-left:20px;color:var(--cc-text-subtle)}
li{margin-bottom:8px}

a{
  font-weight:600;color:var(--cc-text);text-decoration:underline;
  text-decoration-color:var(--cc-link-underline);text-underline-offset:2px;
}
a:hover{text-decoration-color:var(--cc-link-underline-hover)}

code{
  font-family:var(--cc-font-mono);font-size:.875em;
  background:var(--cc-bg-code);padding:1px 5px;border-radius:var(--cc-radius-xs);
  color:var(--cc-text);
}
pre.sql{
  margin:16px 0 24px;padding:14px 16px;overflow-x:auto;
  background:var(--cc-bg-card);border:1px solid var(--cc-border);
  border-radius:var(--cc-radius);border-left:2px solid var(--cc-accent);
  box-shadow:var(--cc-shadow-sm);
}
pre.sql code{background:none;padding:0;font-size:12.5px;line-height:1.6;color:var(--cc-text-subtle)}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0 24px}
.tile{
  background:var(--cc-bg-card);border:1px solid var(--cc-border);
  border-radius:var(--cc-radius);padding:14px 16px;box-shadow:var(--cc-shadow-sm);
}
.tv{
  font-family:var(--cc-font-mono);font-size:22px;font-weight:600;line-height:1.2;
  color:var(--cc-text);font-variant-numeric:tabular-nums;
}
.tl{font-size:13px;color:var(--cc-text-subtle);margin-top:5px;text-wrap:pretty}
.th{font-size:12px;color:var(--cc-text-muted);margin-top:2px;text-wrap:pretty}

.tw{
  overflow-x:auto;margin:16px 0 24px;
  border:1px solid var(--cc-border-input);border-radius:var(--cc-radius);
  background:var(--cc-bg-card);box-shadow:var(--cc-shadow-sm);
}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:14px}
caption{
  text-align:left;padding:12px 14px 10px;color:var(--cc-text-secondary);font-size:13px;
  border-bottom:1px solid var(--cc-border);
}
th{
  padding:10px 14px;text-align:left;font-size:11px;font-weight:700;
  text-transform:uppercase;letter-spacing:.06em;color:var(--cc-text-secondary);
  background:var(--cc-bg-inset);border-bottom:2px solid var(--cc-border);white-space:nowrap;
}
td{
  padding:10px 14px;text-align:left;border-bottom:1px solid var(--cc-border-subtle);
  white-space:nowrap;vertical-align:middle;
}
tbody tr:hover{background:var(--cc-bg-hover)}
tbody tr:last-child td{border-bottom:none}
td.num{
  text-align:right;font-family:var(--cc-font-mono);font-size:13px;
  font-variant-numeric:tabular-nums;
}
td.url{max-width:340px;overflow:hidden;text-overflow:ellipsis;font-size:13px;color:var(--cc-text-secondary)}

td.ex{color:var(--cc-text-muted);font-size:12px}
td.ex details{display:inline-block}
td.ex summary{
  cursor:pointer;list-style:none;font-weight:600;font-size:11px;
  color:var(--cc-accent);background:var(--cc-bg-active);
  border:1px solid var(--cc-border-accent);border-radius:var(--cc-radius-sm);
  padding:2px 9px;user-select:none;
}
td.ex summary::-webkit-details-marker{display:none}
td.ex summary::after{content:" examples";color:var(--cc-text-secondary);font-weight:400}
td.ex ol{margin:8px 0 4px;padding-left:20px;white-space:normal;max-width:340px}
td.ex li{margin:3px 0;line-height:1.5;font-size:12px}
td.ex a{font-weight:600;color:var(--cc-text)}
td.ex a.idx{
  font-family:var(--cc-font-mono);margin-left:6px;font-weight:400;
  color:var(--cc-text-muted);font-variant-numeric:tabular-nums;
  text-decoration-color:var(--cc-border-hover);
}
td.ex a.idx:hover{color:var(--cc-accent);text-decoration-color:var(--cc-accent)}

/* Capped at the chart's intrinsic drawing width. Letting it stretch to the
   container would scale the SVG — and its type — up with it. Auto side
   margins then centre the figure in the wider container. */
.chart{
  display:block;margin:18px auto 8px;width:100%;max-width:900px;
  height:auto;overflow:visible;
}
.ch-title{fill:var(--cc-text);font-size:13px;font-weight:700}
.ch-label{fill:var(--cc-text-subtle);font-size:12px}
.ch-value{
  fill:var(--cc-text-muted);font-size:11px;dominant-baseline:auto;
  font-family:var(--cc-font-mono);font-variant-numeric:tabular-nums;
}
.ch-tick{fill:var(--cc-text-muted);font-size:10px;font-family:var(--cc-font-mono)}
.ch-axis{stroke:var(--axis);stroke-width:1}
.ch-grid{stroke:var(--grid);stroke-width:1}

footer{
  margin-top:48px;padding-top:20px;border-top:1px solid var(--cc-border);
  color:var(--cc-text-muted);font-size:12px;
}
@media (max-width:860px){
  .wrap{padding:24px 16px 48px}
  th,td{padding:8px 10px}
}
@media (max-width:480px){
  .wrap{padding:20px 14px 40px}
  h1{font-size:20px}
  h2{font-size:18px}
}
"""

NAV = [
    ("index", "Before reading"),
    ("overview", "The corpus"),
    ("spec", "Specification"),
    ("generators", "Who writes them"),
    ("policy", "Rights & crawlers"),
    ("abuse", "Abuse"),
    ("langlen", "Reach & cost"),
    ("methods", "Method"),
]


SPACE_CARD = """---
title: What is actually inside llms.txt
emoji: 📄
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: other
---

# What is actually inside llms.txt

A content analysis of the {records} non-empty `llms.txt` and `llms-full.txt`
documents in the Common Crawl [`{dataset}`](https://huggingface.co/datasets/{dataset})
dataset, config `{config}` — spec conformance, which tool generated the file,
AI-usage policy and named-crawler rules, manipulation and spam, and language,
length, topics and ingestion cost.

Every grouped result links five example documents, each to the live URL and to
its row in the dataset viewer, so any figure can be traced back to the exact
bytes it was computed from.

Generated by [`analyze.py`](https://github.com/commoncrawl/llms-txt-experiments)
in one streaming pass over the corpus.
"""


def space_card(stats: dict) -> str:
    """Hugging Face Space card. Static Spaces need this header to serve the page."""
    m = stats.get("meta", {})
    return SPACE_CARD.format(
        records=f"{int(m.get('records', 0)):,}",
        dataset=m.get("dataset", _DATASET["repo"]),
        config=m.get("config", _DATASET["config"]),
    )


def _wrap_document(body: str, description: str) -> str:
    """A complete HTML document, for hosting the report as a static page."""
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(TITLE)}</title>\n"
        f'<meta name="description" content="{escape(description)}">\n'
        f"<style>{font_faces()}\n{CSS}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def render(stats: dict, out: str | Path, fragment: bool = False,
           write_space_card: bool = True) -> Path:
    """Write the report.

    By default a complete HTML document, which is what a browser or a static
    Hugging Face Space needs. ``fragment=True`` emits the body only, without
    page-level tags, for hosts that supply their own document skeleton (the
    Claude artifact renderer does).
    """
    meta = stats.get("meta", {})
    set_dataset(meta.get("dataset", _DATASET["repo"]), meta.get("config", _DATASET["config"]))
    nav = "".join(f'<a href="#{i}">{escape(t)}</a>' for i, t in NAV)
    body = (
        f'<div class="wrap">'
        f"<header><h1>{escape(TITLE)}</h1>"
        f'<p class="sub">{escape(SUBTITLE)}</p>'
        f'<p class="meta">{n(stats["meta"]["records"])} documents of '
        f'{n(stats["meta"]["responses"])} responses · dataset '
        f'<code>{escape(stats["meta"]["dataset"])}</code> · '
        f'config <code>{escape(stats["meta"]["config"])}</code></p>'
        f"<nav>{nav}</nav></header>"
        + sec_index(stats)
        + sec_overview(stats)
        + sec_a(stats)
        + sec_b(stats)
        + sec_d(stats)
        + sec_e(stats)
        + sec_f(stats)
        + sec_methods(stats)
        + "<footer>Generated by <code>analyze.py report</code>. "
          "Every figure is derived from the corpus in a single pass; "
          "the source live alongside this report.</footer>"
        f"</div>"
    )

    description = (
        f"Content analysis of {n(meta.get('records', 0))} llms.txt documents in "
        f"Common Crawl {meta.get('config', '')}."
    )
    # Both forms carry the stylesheet; only the full document adds the page
    # skeleton. A fragment host supplies <html>/<head>/<body> but not CSS.
    html = (
        f"<title>{escape(TITLE)}</title>"
        f"<style>{font_faces()}\n{CSS}</style>{body}"
        if fragment
        else _wrap_document(body, description)
    )

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html)
    print(f"wrote {p} ({p.stat().st_size/1024:.0f} KB)")
    if write_space_card and not fragment:
        card = p.parent / "README.md"
        card.write_text(space_card(stats))
        print(f"wrote {card} (Hugging Face Space card)")
    return p
