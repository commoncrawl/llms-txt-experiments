"""End-to-end tests over the real-data fixture shard.

These run the production code paths — ``extract`` (including the parquet
writer), ``derive``, ``topics``, ``aggregate``, ``report`` and ``spotcheck`` —
rather than re-implementing them, so a change that breaks the pipeline breaks
these tests.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow.dataset as ds
import pytest

from llmstxt_analysis.registry import feature_schema


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------
def test_extract_writes_both_outputs(derived_dir, records):
    feats = ds.dataset(str(Path(derived_dir) / "features"), format="parquet").to_table()
    assert feats.num_rows == len(records)
    assert set(feats.schema.names) == set(feature_schema().names)
    assert feats.schema.equals(feature_schema()), "written schema drifted from the registry"

    topic = ds.dataset(str(Path(derived_dir) / "topic_corpus"), format="parquet").to_table()
    assert topic.num_rows <= feats.num_rows
    assert set(topic.schema.names) == {"shard", "rg", "rg_row", "lang", "text"}


def test_topic_corpus_holds_only_unattributed_prose(derived_dir):
    feats = ds.dataset(str(Path(derived_dir) / "features"), format="parquet").to_table().to_pylist()
    topic = ds.dataset(str(Path(derived_dir) / "topic_corpus"), format="parquet").to_table().to_pylist()
    by_key = {(f["shard"], f["rg"], f["rg_row"]): f for f in feats}
    assert topic, "fixture should yield some topic documents"
    for t in topic:
        f = by_key[(t["shard"], t["rg"], t["rg_row"])]
        assert f["generator_id"] == "unknown", "templated documents must not reach the topic model"
        assert len(t["text"]) >= 200
        assert len(t["text"]) <= 2000
        assert "http://" not in t["text"] and "https://" not in t["text"]


def test_extract_is_deterministic(shards_root, tmp_path):
    from llmstxt_analysis.extract import run

    a, b = tmp_path / "a", tmp_path / "b"
    run(shards_root, a, workers=1)
    run(shards_root, b, workers=1)
    ta = ds.dataset(str(a / "features"), format="parquet").to_table().to_pylist()
    tb = ds.dataset(str(b / "features"), format="parquet").to_table().to_pylist()
    assert ta == tb


def test_locators_round_trip_to_the_original_records(derived_dir, shards_root, records):
    """(shard, rg, rg_row) must address exactly the record it was written for."""
    from llmstxt_analysis.corpus import fetch_records

    feats = ds.dataset(str(Path(derived_dir) / "features"), format="parquet").to_table().to_pylist()
    picks = feats[:: max(1, len(feats) // 25)]
    locs = [(f["shard"], f["rg"], f["rg_row"]) for f in picks]
    got = fetch_records(shards_root, locs)
    assert len(got) == len(set(locs))
    for f in picks:
        rec = got[(f["shard"], f["rg"], f["rg_row"])]
        assert rec.url == f["url"]
        assert len(rec.body) == f["n_chars"]


# --------------------------------------------------------------------------
# derive
# --------------------------------------------------------------------------
def test_derived_columns_are_consistent(frame):
    df = frame
    assert len(df) > 0
    assert (df["dup_count"] >= 1).all()
    assert df.loc[df["is_exact_dup"], "dup_count"].min() >= 2
    assert (df["template_cluster_size"] >= 1).all()
    # Every positively identified generator counts as templated.
    assert df.loc[df["generator_source"] != "none", "is_template"].all()
    assert not (df["is_template"] & df["is_human_authored"]).any()
    dom = df.loc[df["offsite_dominant"], "n_offsite_domains"]
    assert dom.empty or dom.min() >= 5
    assert not df.loc[df["is_template"], "lang_mismatch"].any()


def test_template_clustering_groups_a_real_vendor_template(frame):
    """The fixture holds several Wix documents from unrelated sites."""
    wix = frame[frame["generator_id"] == "wix"]
    assert len(wix) >= 3
    assert wix["host"].nunique() == len(wix), "fixture Wix docs are from distinct sites"
    biggest = wix["skeleton_sha1"].value_counts().iloc[0]
    assert biggest >= 3, "same-language Wix boilerplate must collapse to one skeleton"


def test_dataset_index_is_a_dense_ordered_offset(frame, shards_root):
    """The index must address the record in the concatenated HF split."""
    import pyarrow.parquet as pq

    from llmstxt_analysis.corpus import find_shards

    idx = frame["dataset_index"]
    assert idx.is_unique
    assert idx.min() == 0
    total = sum(pq.ParquetFile(s.path).metadata.num_rows for s in find_shards(shards_root))
    assert idx.max() == total - 1
    ordered = frame.sort_values(["shard", "rg", "rg_row"])["dataset_index"]
    assert ordered.is_monotonic_increasing, "index must follow shard/row-group order"


def test_derived_frame_is_cached_and_invalidated(derived_dir):
    """Second load comes from parquet; touching the inputs rebuilds it."""
    import time

    from llmstxt_analysis import derive

    derive.clear_cache(derived_dir)
    assert not derive.cache_info(derived_dir, 3)["fresh"]

    a = derive.load(derived_dir, 3)
    info = derive.cache_info(derived_dir, 3)
    assert info["fresh"] and info["size_mb"] >= 0

    t0 = time.perf_counter()
    b = derive.load(derived_dir, 3)
    cached_seconds = time.perf_counter() - t0
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)
    assert cached_seconds < 5

    # A different threshold is a different frame, so it must not be served
    # from the same entry.
    assert derive.cache_path(derived_dir, 3) != derive.cache_path(derived_dir, 9)

    # Touching a feature part changes the fingerprint.
    before = derive.fingerprint(derived_dir, 3)
    part = next(derive.features_dir(derived_dir).glob("*.parquet"))
    part.touch()
    assert derive.fingerprint(derived_dir, 3) != before


def test_list_columns_get_string_mirrors(frame):
    for col in ("conf_flags", "policy_directives", "named_bots", "spam_categories"):
        assert col + "_str" in frame.columns
    row = frame[frame["n_named_bots"] > 0].iloc[0]
    assert row["named_bots_str"]
    assert set(row["named_bots_str"].split(",")) == set(row["named_bots"])


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def urlindex_json(derived_dir, records):
    """A URL-index result whose funnel actually lands on the fixture corpus."""
    from llmstxt_analysis.urlindex import scan, write

    csv = derived_dir / "index.csv"
    lines = ["crawl,url,url_host_name,fetch_status,content_mime_detected"]
    # Every fixture record as a served text document, plus misses around it.
    for r in records:
        host = r.url.split("/")[2]
        mime = "text/markdown" if r.is_full else "text/plain"
        lines.append(f"CC-MAIN-2026-30,{r.url},{host},200,{mime}")
    for i in range(50):
        lines.append(f"CC-MAIN-2026-30,https://miss{i}.example/llms.txt,miss{i}.example,404,text/html")
    for i in range(10):
        lines.append(f"CC-MAIN-2026-30,https://html{i}.example/llms.txt,html{i}.example,200,text/html")
    csv.write_text("\n".join(lines) + "\n")
    return write(scan(csv), derived_dir / "urlindex.json")


@pytest.fixture(scope="module")
def stats(derived_dir, urlindex_json):
    from llmstxt_analysis.aggregate import build

    return build(derived_dir, None, template_min_cluster=3, urlindex_path=urlindex_json)


def test_url_index_funnel_is_reported(stats, records):
    """The pre-analysis funnel has to be present and internally consistent."""
    idx = stats["index"]
    assert idx["total"] == len(records) + 60
    assert idx["pct_404"] == pytest.approx(100 * 50 / idx["total"], abs=0.01)
    assert idx["n_text200"] == len(records)
    # Only some 200s carry text; that ratio is the point of the section.
    assert idx["pct_text_of_200"] < 100

    funnel = [r[1] for r in idx["funnel"]["rows"]]
    assert funnel == sorted(funnel, reverse=True), "a funnel only narrows"
    assert funnel[0] == idx["total"]

    recon = dict(idx["reconcile"]["rows"])
    assert recon["dataset: records extracted from WARC"] == stats["meta"]["responses"]
    assert recon["  of which at a root path"] + recon["  of which deeper in the site"] == \
        stats["meta"]["responses"]
    assert recon["analysed here (non-empty)"] == stats["meta"]["records"]


def test_url_index_section_renders_with_the_query(html, stats):
    assert 'id="index"' in html
    assert "<pre class=\"sql\">" in html
    assert "url_path = &#x27;/llms.txt&#x27;" in html or "url_path = '/llms.txt'" in html
    assert f"{stats['index']['total']:,}" in html


def test_aggregate_works_without_a_url_index(derived_dir):
    """The funnel is optional; its absence must not break the report."""
    from llmstxt_analysis.aggregate import build
    from llmstxt_analysis.report import render

    s = build(derived_dir, None, template_min_cluster=3, urlindex_path=derived_dir / "nope.json")
    assert "index" not in s
    out = derived_dir / "no-index.html"
    render(s, out, write_space_card=False)
    assert 'id="overview"' in out.read_text()


def test_stats_have_every_track(stats):
    assert set(stats) >= {"meta", "overview", "A", "B", "D", "E", "F"}
    assert stats["meta"]["records"] > 0


def _walk_tables(node, path="root"):
    if isinstance(node, dict):
        if "columns" in node and "rows" in node:
            yield path, node
        for k, v in node.items():
            yield from _walk_tables(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_tables(v, f"{path}[{i}]")


def test_every_table_is_well_formed(stats):
    tables = list(_walk_tables(stats))
    assert len(tables) > 20
    for path, t in tables:
        width = len(t["columns"])
        for row in t["rows"]:
            assert len(row) == width, f"ragged row in {path}"


def test_percentages_are_in_range(stats):
    def check(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, (int, float)) and (k.startswith("pct") or k == "share"):
                    assert 0 <= v <= 100, f"{path}.{k} = {v}"
                check(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                check(v, f"{path}[{i}]")

    check(stats, "root")


def test_counts_reconcile_with_the_frame(stats, frame):
    n = len(frame)
    assert stats["meta"]["records"] == n
    assert stats["overview"]["n_records"] == n
    assert stats["overview"]["n_llms_txt"] + stats["overview"]["n_llms_full"] == n
    assert stats["overview"]["total_tokens"] == int(frame["n_tokens"].sum())
    doc_kinds = dict((r[0], r[1]) for r in stats["A"]["doc_kind"]["rows"])
    assert sum(doc_kinds.values()) == n
    assert doc_kinds == frame["doc_kind"].value_counts().to_dict()


def test_empty_responses_are_counted_but_excluded(stats, frame, derived_dir, shards_root):
    """Empties skew everything: identical bodies, no links, no title, no generator."""
    from llmstxt_analysis.derive import load

    everything = load(derived_dir, 3, use_cache=False, drop_empty=False)
    n_empty = int(everything["is_empty"].sum())
    assert n_empty > 0, "fixture must contain empty responses for this to mean anything"

    # Reported in the intake table, absent from the analysis frame.
    assert stats["meta"]["n_empty"] == n_empty
    assert stats["meta"]["responses"] == len(everything)
    assert stats["meta"]["records"] == len(everything) - n_empty
    assert not frame["is_empty"].any()
    assert "empty" not in dict(r[:2] for r in stats["A"]["doc_kind"]["rows"])
    assert sum(r[1] for r in stats["overview"]["intake"]["rows"]) == len(everything)

    # Nothing downstream may reference an empty document.
    assert (frame["n_chars"] > 0).all()
    for row in stats["B"]["top_templates"]["rows"]:
        assert row[0] != n_empty, "the empty-body cluster must not appear as a template"

    # But they stay reachable for inspection.
    from llmstxt_analysis.spotcheck import run

    text = run(derived_dir, shards_root, "doc_kind == 'empty'", n=2, chars=50)
    assert f"matches: {n_empty} /" in text
    assert "URL: http" in text


def test_cost_table_uses_litellm_prices(stats):
    rows = stats["F"]["cost"]["rows"]
    assert rows, "no models resolved in litellm's cost map"
    total_tokens = stats["overview"]["total_tokens"]
    from litellm import model_cost

    for model, per_m, ctx, cost in rows:
        assert per_m > 0 and ctx > 0
        assert per_m == pytest.approx(model_cost[model]["input_cost_per_token"] * 1e6, rel=1e-6)
        # The displayed cost is rounded to cents.
        assert cost == pytest.approx(per_m * total_tokens / 1e6, abs=0.01)


def test_stats_are_json_serialisable(stats, tmp_path):
    from llmstxt_analysis.aggregate import write

    out = tmp_path / "stats.json"
    write(stats, out)
    assert json.loads(out.read_text())["meta"]["records"] == stats["meta"]["records"]


# --------------------------------------------------------------------------
# topics
# --------------------------------------------------------------------------
def test_topics_fit_and_assign(derived_dir, tmp_path):
    from llmstxt_analysis import topics as T

    out = tmp_path / "topics"
    raw = T.fit(derived_dir, ["eng", "jpn"], out=out, n_topics=3, min_docs=5,
                template_min_cluster=3)
    assert "jpn" in raw["skipped_languages"], "unsegmented scripts must be skipped"
    assert "eng" in raw["languages"]
    block = raw["languages"]["eng"]
    assert len(block["topics"]) == 3
    assert sum(t["n_docs"] for t in block["topics"]) == block["n_docs"]
    assert all(t["terms"] for t in block["topics"])

    names = tmp_path / "names.yaml"
    ids = [t["id"] for t in block["topics"]]
    names.write_text(f"eng:{ids[0]}: Local business\n")
    merged = T.assign(out / "topics_raw.json", names, tmp_path / "topics.json")
    labelled = {t["id"]: t["name"] for t in merged["languages"]["eng"]["topics"]}
    assert labelled[ids[0]] == "Local business"
    assert labelled[ids[1]].startswith("unlabeled-")
    assert len(merged["unlabeled"]) == 2


def test_topic_name_parser_handles_comments_and_quotes(tmp_path):
    from llmstxt_analysis.topics import load_names

    p = tmp_path / "n.yaml"
    p.write_text('# a comment\n\neng:0: "Docs & APIs"\neng:1: Local services\n')
    names = load_names(p)
    assert names["eng:0"] == "Docs & APIs"
    assert names["eng:1"] == "Local services"


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def report_dir(stats, tmp_path_factory):
    """The publishable directory: a complete index.html plus the Space card."""
    from llmstxt_analysis.report import render

    out = tmp_path_factory.mktemp("report")
    render(stats, out / "index.html")
    return out


@pytest.fixture(scope="module")
def html(stats, tmp_path_factory):
    """The fragment form, as embedded by a host that supplies the skeleton."""
    from llmstxt_analysis.report import render

    out = tmp_path_factory.mktemp("fragment") / "r.html"
    render(stats, out, fragment=True)
    return out.read_text()


def test_default_output_is_a_complete_document(report_dir):
    """A static Space serves index.html directly, so it must be a real page."""
    doc = (report_dir / "index.html").read_text()
    assert doc.startswith("<!doctype html>")
    assert '<html lang="en">' in doc
    assert '<meta charset="utf-8">' in doc
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in doc
    assert "<title>" in doc and "</head>" in doc
    assert doc.rstrip().endswith("</html>")
    assert doc.count("<body>") == 1 and doc.count("</html>") == 1
    assert '<div class="wrap">' in doc


def test_space_card_is_written_next_to_the_report(report_dir, stats):
    """Static Spaces need `sdk: static` frontmatter or they will not serve."""
    card = report_dir / "README.md"
    assert card.exists(), "the Space needs a README.md alongside index.html"
    text = card.read_text()
    assert text.startswith("---\n")
    front = text.split("---", 2)[1]
    assert "sdk: static" in front
    assert "app_file: index.html" in front
    assert "title:" in front
    assert f"{stats['meta']['records']:,}" in text
    assert stats["meta"]["dataset"] in text


def test_fragment_form_omits_page_level_tags(html):
    assert "<!doctype" not in html.lower()
    for tag in ("<html>", "<html ", "<head>", "<head ", "<body>", "<body "):
        assert tag not in html.lower()
    assert "<title>" in html.lower()


def test_report_is_self_contained(html):
    """A strict CSP blocks every external *request*; the page must need none.

    Anchor hrefs are navigations, not requests, so example links to the crawled
    documents are fine. What must not appear is anything the browser fetches on
    load: scripts, stylesheets, images, fonts, iframes.
    """
    lowered = html.lower()
    assert "<script" not in lowered
    assert "<iframe" not in lowered
    assert "<img" not in lowered
    assert "<link" not in lowered
    assert re.search(r'\ssrc\s*=', lowered) is None, "no fetched sub-resources"
    assert "@import" not in html
    # Fonts are inlined as data URIs; nothing else may reference a URL in CSS.
    for m in re.finditer(r"url\(([^)]*)\)", html):
        assert m.group(1).startswith("data:font/woff2;base64,"), m.group(1)[:60]
    for m in re.finditer(r'href\s*=\s*"([^"]+)"', html):
        url = m.group(1)
        assert url.startswith(("#", "http://", "https://")), url


def test_example_links_carry_a_dataset_index(html, stats):
    """Every example is a live link *and* a durable dataset offset."""
    assert "load_dataset" in html, "the report must explain how to resolve an index"
    assert re.search(r'<a class="idx" href="[^"]+row=\d+"[^>]*>#[\d,]+</a>', html)

    ex = stats["B"]["generator"]["rows"][0][-1]
    assert ex and all({"url", "idx"} <= set(e) for e in ex)
    for e in ex:
        assert f'href="{e["url"]}"' in html
        assert f'#{e["idx"]:,}</a>' in html


def test_example_index_links_to_the_dataset_viewer(html, stats):
    """The row id must open that row in the Hugging Face dataset viewer."""
    from llmstxt_analysis.report import viewer_url

    repo, config = stats["meta"]["dataset"], stats["meta"]["config"]
    assert viewer_url(100) == (
        f"https://huggingface.co/datasets/{repo}/viewer/{config}/train?row=100"
    )
    for e in stats["B"]["generator"]["rows"][0][-1]:
        assert f'href="{viewer_url(e["idx"])}"' in html
    assert f"https://huggingface.co/datasets/{repo}/viewer/{config}/train?row=" in html


def test_report_has_no_page_level_tags(html):
    lowered = html.lower()
    for tag in ("<!doctype", "<html>", "<html ", "<head>", "<head ", "<body>", "<body "):
        assert tag not in lowered, f"{tag} must be supplied by the artifact wrapper"
    assert "<title>" in lowered


def test_report_paints_itself_explicitly(html):
    """Single-theme by choice, so nothing may be inherited from the host.

    The Common Crawl palette exists for light surfaces only, so the report
    commits to it rather than inventing dark counterparts. That is only safe if
    the page paints its own ground and every colour — a transparent body would
    borrow a dark host background and render navy text on it.
    """
    assert "color-scheme: light" in html
    assert "background:var(--cc-bg-page)" in html
    assert "color:var(--cc-text)" in html
    # No colour may be defined only inside a media query.
    for block in re.findall(r"@media[^{]*\{(.*?)\n\}", html, re.S):
        assert "--cc-" not in block, "tokens must not be redefined per media query"


def test_report_follows_the_common_crawl_style_guide(html):
    """Colours, fonts and the table-header treatment come from cc-base.css."""
    for token, value in [
        ("--cc-text", "#152a47"),
        ("--cc-text-secondary", "#64748b"),
        ("--cc-text-muted", "#94a3b8"),
        ("--cc-accent", "#2e5f8a"),
        ("--cc-bg-page", "#f5f7fa"),
        ("--cc-bg-card", "#fff"),
        ("--cc-border", "#e2e8f0"),
    ]:
        assert f"{token}:{value}" in html, token

    assert "'Libre Franklin'" in html and "'IBM Plex Mono'" in html
    assert html.count("@font-face") == 2, "both faces inlined, no CDN"
    assert "fonts.googleapis.com" not in html and "fonts.gstatic.com" not in html

    # The uppercase 11px header is the strongest CC identifier; keep it exact.
    header = re.search(r"\nth\{(.*?)\}", html, re.S).group(1)
    assert "font-size:11px" in header
    assert "font-weight:700" in header
    assert "text-transform:uppercase" in header
    assert "letter-spacing:.06em" in header

    # Understated links: body colour plus a light underline.
    link = re.search(r"\na\{(.*?)\}", html, re.S).group(1)
    assert "text-decoration:underline" in link
    assert "var(--cc-link-underline)" in link

    assert "--cc-radius:10px" in html and "--cc-radius-sm:6px" in html
    assert "#000" not in html, "the guide forbids pure black"


def test_report_contains_every_section(html):
    for sid in ("overview", "spec", "generators", "policy", "abuse", "langlen", "methods"):
        assert f'id="{sid}"' in html
    assert html.count("<table>") > 20
    assert html.count("<svg") >= 6


def test_report_does_not_reference_the_project_plan(html):
    """The report has to stand on its own, not index into the analysis plan."""
    assert not re.search(r"\bTrack [A-H]\b", html)
    assert not re.search(r"\b[A-H] · ", html)


def test_report_numbers_come_from_the_stats(html, stats):
    assert f'{stats["overview"]["n_records"]:,}' in html
    assert f'{stats["overview"]["n_hosts"]:,}' in html


def test_charts_are_not_scaled_up(html):
    """An upscaled viewBox scales its text too, so the CSS cap must match.

    Drawing at 900 and displaying at 1160 would render every 12px axis label
    at 15px, and the labels looked oversized on desktop before this was fixed.
    """
    from llmstxt_analysis.charts import WIDTH

    assert f"max-width:{WIDTH}px" in html, "CSS cap must equal the drawing width"
    for vb in re.findall(r'viewBox="0 0 (\d+) \d+"', html):
        assert int(vb) == WIDTH, f"chart drawn at {vb}, CSS caps at {WIDTH}"


def test_charts_are_centred_in_the_container(html):
    """Capped narrower than the container, so it needs auto side margins."""
    rule = re.search(r"\n\.chart\{(.*?)\}", html, re.S).group(1)
    assert re.search(r"margin:[^;]*\bauto\b", rule), rule.strip()


def test_prose_uses_the_full_container(html):
    para = re.search(r"\np\{(.*?)\}", html, re.S).group(1)
    assert "max-width" not in para
    assert re.search(r"\nul\{(.*?)\}", html, re.S).group(1).count("max-width") == 0


def test_charts_use_theme_tokens_not_baked_colours(html):
    svgs = re.findall(r"<svg.*?</svg>", html, re.S)
    assert svgs
    for svg in svgs:
        assert "#" not in re.sub(r"<title>.*?</title>", "", svg, flags=re.S), (
            "chart colours must be CSS custom properties so both themes work"
        )
        assert "var(--series-1)" in svg or "var(--" in svg


# --------------------------------------------------------------------------
# standalone figures (blog post)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def figure_svgs(stats, tmp_path_factory):
    from llmstxt_analysis.figures import write

    out = tmp_path_factory.mktemp("figures")
    write(stats, out, png=False)
    paths = sorted(out.glob("*.svg"))
    assert paths, "no figures emitted from the fixture stats"
    return {p.name: p.read_text() for p in paths}


def test_standalone_figures_are_parseable_xml(figure_svgs):
    """A .svg file is parsed as XML, and without xmlns it is rejected outright.

    The report's fragments have no namespace because they are inlined into HTML,
    where it is implied. Copying one to disk yields a broken image.
    """
    import xml.etree.ElementTree as ET

    for name, svg in figure_svgs.items():
        root = ET.fromstring(svg)
        assert root.tag == "{http://www.w3.org/2000/svg}svg", f"{name}: {root.tag}"
        assert "%" not in root.get("width", ""), f"{name}: needs an intrinsic width"


def test_standalone_figures_resolve_every_token(figure_svgs):
    """librsvg does not implement CSS custom properties: var() rasterises black."""
    for name, svg in figure_svgs.items():
        assert "var(--" not in svg, f"{name} still carries an unresolved token"


def test_standalone_figures_carry_their_own_styling(figure_svgs):
    """No host stylesheet, so unstyled axes get stroke:none and disappear."""
    from llmstxt_analysis.figures import TOKENS

    for name, svg in figure_svgs.items():
        assert "<style>" in svg, f"{name} has no styling of its own"
        assert f"stroke:{TOKENS['--axis']}" in svg, f"{name}: axis would be invisible"
        assert TOKENS["--series-1"] in svg, f"{name}: bars would paint black"
        assert 'fill="#fff"' in svg, f"{name}: needs a backdrop for the muted labels"


def test_standalone_figures_have_no_baked_in_title(figure_svgs):
    """The style guide puts the chart title in a heading above the image."""
    for name, svg in figure_svgs.items():
        assert 'class="ch-title"' not in svg, f"{name} duplicates its markdown heading"


# --------------------------------------------------------------------------
# spotcheck
# --------------------------------------------------------------------------
def test_spotcheck_returns_real_content(derived_dir, shards_root):
    from llmstxt_analysis.spotcheck import run

    text = run(derived_dir, shards_root, "generator_id == 'wix'", n=2, chars=400)
    assert "filter: generator_id == 'wix'" in text
    assert "URL: http" in text
    assert "_api/mcp" in text or "Model Context Protocol" in text


def test_spotcheck_handles_no_matches(derived_dir, shards_root):
    from llmstxt_analysis.spotcheck import run

    assert "no records match" in run(derived_dir, shards_root, "n_chars < 0")


def test_spotcheck_rejects_a_bad_filter(derived_dir, shards_root):
    from llmstxt_analysis.spotcheck import run

    with pytest.raises(SystemExit):
        run(derived_dir, shards_root, "this is not a query(")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_wires_up_every_subcommand(shards_root, derived_dir, tmp_path, capsys):
    from llmstxt_analysis.cli import main

    assert main(["info", "--dataset", str(shards_root)]) == 0
    assert "registered extractors" in capsys.readouterr().out

    assert main(["summary", "--derived", str(derived_dir), "--template-min-cluster", "3"]) == 0
    assert "records:" in capsys.readouterr().out

    stats = tmp_path / "s.json"
    assert main(["aggregate", "--derived", str(derived_dir), "--out", str(stats),
                 "--template-min-cluster", "3"]) == 0
    rep = tmp_path / "r.html"
    assert main(["report", "--stats", str(stats), "--out", str(rep)]) == 0
    assert rep.exists() and rep.stat().st_size > 20_000
