"""Turn the features table into the statistics the report renders.

Output is a single JSON document. Every entry is either a scalar, a
``{"columns": [...], "rows": [...]}`` table, or a series for a chart, so the
report layer stays free of pandas.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .derive import drop_empty_rows, load
from .extractors.abuse import SEVERITY_LABELS
from .extractors.conformance import CONFORMANCE_LABELS
from .urlindex import STATUS_GROUPS

# Model panel for the ingestion-cost table. Names must exist in litellm's
# cost map; missing ones are skipped with a warning rather than failing.
COST_MODELS = [
    "gpt-5",
    "gpt-4o-mini",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "gemini-2.5-flash",
]
CONTEXT_WINDOWS = [32_000, 128_000, 200_000, 1_000_000]
PERCENTILES = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]

# Every grouped result carries this many example records, each as a live URL
# plus the record's index in the Hugging Face dataset split.
N_EXAMPLES = 5
EXAMPLES_COL = "examples"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> dict:
    return {"columns": list(columns), "rows": [list(r) for r in rows]}


def _pct(n: float, d: float) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


def _sample_rows(frame: pd.DataFrame, k: int = N_EXAMPLES) -> list[dict]:
    """A deterministic spread of example records from ``frame``.

    Each example carries the live URL *and* the record's index in the Hugging
    Face dataset split, so it stays retrievable when the URL's content changes
    or the site disappears.
    """
    if frame is None or frame.empty:
        return []
    take = frame.sample(min(k, len(frame)), random_state=0) if len(frame) > k else frame
    take = take.sort_values("dataset_index")
    return [
        {"url": str(r.url), "idx": int(r.dataset_index)}
        for r in take[["url", "dataset_index"]].itertuples(index=False)
    ]


def _examples_by_group(frame: pd.DataFrame | None, key: pd.Series | None,
                       values: Sequence[Any], k: int = N_EXAMPLES) -> dict:
    if frame is None or key is None:
        return {}
    mask = key.isin(set(values))
    if not mask.any():
        return {}
    sub = frame.loc[mask, ["url", "dataset_index"]].copy()
    sub["_g"] = key.loc[mask]
    return {g: _sample_rows(part, k) for g, part in sub.groupby("_g", sort=False)}


def counts(series: pd.Series, total: int | None = None, top: int = 0,
           label: str = "value", frame: pd.DataFrame | None = None) -> dict:
    """Value counts, with example records per group when ``frame`` is given."""
    vc = series.value_counts(dropna=False)
    if top:
        vc = vc.head(top)
    total = total if total is not None else int(series.shape[0])
    ex = _examples_by_group(frame, series, [k for k in vc.index if not pd.isna(k)])
    cols = [label, "documents", "% of corpus"] + ([EXAMPLES_COL] if frame is not None else [])
    rows = []
    for k, v in vc.items():
        row = [("(none)" if pd.isna(k) else k), int(v), _pct(int(v), total)]
        if frame is not None:
            row.append(ex.get(k, []))
        rows.append(row)
    return table(cols, rows)


def list_counts(series: pd.Series, total: int, top: int = 0, label: str = "value",
                frame: pd.DataFrame | None = None) -> dict:
    """Counts over a list-valued column, with example records per value."""
    c: Counter = Counter()
    for v in series:
        if v is None or len(v) == 0:
            continue
        c.update(set(v))
    items = c.most_common(top or None)

    ex: dict = {}
    if frame is not None and items:
        wanted = {k for k, _ in items}
        # v is a numpy array from parquet; `v or []` raises on arrays.
        flat = series.map(
            lambda v: [x for x in set(v if v is not None else []) if x in wanted])
        mask = flat.map(bool)
        if mask.any():
            sub = frame.loc[mask, ["url", "dataset_index"]].copy()
            sub["_g"] = flat.loc[mask]
            sub = sub.explode("_g")
            ex = {g: _sample_rows(part) for g, part in sub.groupby("_g", sort=False)}

    cols = [label, "documents", "% of corpus"] + ([EXAMPLES_COL] if frame is not None else [])
    rows = []
    for k, v in items:
        row = [k, v, _pct(v, total)]
        if frame is not None:
            row.append(ex.get(k, []))
        rows.append(row)
    return table(cols, rows)


def quantiles(series: pd.Series, qs=PERCENTILES) -> dict:
    s = series.dropna()
    rows = [["min", int(s.min()) if len(s) else 0]]
    rows += [[f"p{int(q*1000)/10:g}", int(s.quantile(q))] for q in qs]
    rows += [["max", int(s.max()) if len(s) else 0], ["mean", int(s.mean()) if len(s) else 0]]
    return table(["statistic", "value"], rows)


def series_for_chart(series: pd.Series, top: int = 12) -> dict:
    vc = series.value_counts().head(top)
    return {"labels": [str(k) for k in vc.index], "values": [int(v) for v in vc.values]}


# --------------------------------------------------------------------------
# tracks
# --------------------------------------------------------------------------
def overview(df: pd.DataFrame, everything: pd.DataFrame | None = None) -> dict:
    n = len(df)
    hosts = df["host"].nunique()
    raw = everything if everything is not None else df
    n_empty = int(raw["is_empty"].sum()) if "is_empty" in raw.columns else 0
    return {
        "n_responses": len(raw),
        "n_empty": n_empty,
        "pct_empty": _pct(n_empty, len(raw)),
        "empty_hosts": int(raw.loc[raw["is_empty"], "host"].nunique()) if n_empty else 0,
        "intake": table(
            ["response", "documents", "% of responses"],
            [
                ["contains a document", n, _pct(n, len(raw))],
                ["empty body (excluded below)", n_empty, _pct(n_empty, len(raw))],
            ],
        ),
        "n_records": n,
        "n_hosts": int(hosts),
        "n_regdomains": int(df["regdomain"].nunique()),
        "n_llms_txt": int((~df["is_full"]).sum()),
        "n_llms_full": int(df["is_full"].sum()),
        "total_chars": int(df["n_chars"].sum()),
        "total_tokens": int(df["n_tokens"].sum()),
        "pct_exact_dup": _pct(int(df["is_exact_dup"].sum()), n),
        "pct_template": _pct(int(df["is_template"].sum()), n),
        "pct_human_authored": _pct(int(df["is_human_authored"].sum()), n),
        "file_kind": counts(df["file_kind"], n, label="file", frame=df),
        "tld": counts(df["tld"], n, top=20, label="TLD", frame=df),
        "tld_chart": series_for_chart(df["tld"], 15),
        "payload_type": counts(df["payload_type"], n, top=6, label="MIME (WARC-identified)", frame=df),
        "http_server": counts(df["http_server"].str.split("/").str[0].str.lower(), n, top=12,
                              label="Server header"),
    }


def track_a(df: pd.DataFrame) -> dict:
    n = len(df)
    md = df[df["doc_kind"] == "markdown"]
    lvl = df["conformance_level"].map(CONFORMANCE_LABELS)
    by_kind = (
        df.groupby("file_kind")["conformance_level"]
        .apply(lambda s: s.map(CONFORMANCE_LABELS).value_counts(normalize=True) * 100)
        .unstack(fill_value=0)
    )
    rows = []
    for lab in CONFORMANCE_LABELS.values():
        rows.append([
            lab,
            round(float(by_kind.get(lab, pd.Series(dtype=float)).get("llms.txt", 0.0)), 2),
            round(float(by_kind.get(lab, pd.Series(dtype=float)).get("llms-full.txt", 0.0)), 2),
        ])
    human = df[df["is_human_authored"]]
    tmpl = df[df["is_template"]]
    return {
        "pct_level4": _pct(int((df["conformance_level"] == 4).sum()), n),
        "pct_level4_human": _pct(int((human["conformance_level"] == 4).sum()), max(len(human), 1)),
        "pct_level4_template": _pct(int((tmpl["conformance_level"] == 4).sum()), max(len(tmpl), 1)),
        "pct_links_with_notes": _pct(int((df["n_links_with_notes"] > 0).sum()), n),
        "pct_optional": _pct(int(df["has_optional_section"].sum()), n),
        "doc_kind": counts(df["doc_kind"], n, label="document kind", frame=df),
        "doc_kind_chart": series_for_chart(df["doc_kind"], 10),
        "conformance": counts(lvl, n, label="conformance level", frame=df),
        "conformance_chart": series_for_chart(lvl, 6),
        "conformance_by_file": table(["conformance level", "% of llms.txt", "% of llms-full.txt"], rows),
        "elements": table(
            ["spec element", "documents", "% of corpus", "% of markdown docs"],
            [
                ["starts with H1", int(df["starts_with_h1"].sum()),
                 _pct(int(df["starts_with_h1"].sum()), n), _pct(int(md["starts_with_h1"].sum()), len(md))],
                ["has any H1", int((df["n_h1"] > 0).sum()),
                 _pct(int((df["n_h1"] > 0).sum()), n), _pct(int((md["n_h1"] > 0).sum()), len(md))],
                ["summary blockquote after H1", int(df["blockquote_follows_h1"].sum()),
                 _pct(int(df["blockquote_follows_h1"].sum()), n),
                 _pct(int(md["blockquote_follows_h1"].sum()), len(md))],
                ["prose body before first H2", int(df["has_prose_body"].sum()),
                 _pct(int(df["has_prose_body"].sum()), n), _pct(int(md["has_prose_body"].sum()), len(md))],
                ["at least one H2 section", int((df["n_h2"] > 0).sum()),
                 _pct(int((df["n_h2"] > 0).sum()), n), _pct(int((md["n_h2"] > 0).sum()), len(md))],
                ["link bullets", int((df["n_link_bullets"] > 0).sum()),
                 _pct(int((df["n_link_bullets"] > 0).sum()), n),
                 _pct(int((md["n_link_bullets"] > 0).sum()), len(md))],
                ["link notes (': description')", int((df["n_links_with_notes"] > 0).sum()),
                 _pct(int((df["n_links_with_notes"] > 0).sum()), n),
                 _pct(int((md["n_links_with_notes"] > 0).sum()), len(md))],
                ["'## Optional' section", int(df["has_optional_section"].sum()),
                 _pct(int(df["has_optional_section"].sum()), n),
                 _pct(int(md["has_optional_section"].sum()), len(md))],
                ["byte-order mark", int(df["has_bom"].sum()),
                 _pct(int(df["has_bom"].sum()), n), _pct(int(md["has_bom"].sum()), len(md))],
            ],
        ),
        "defects": list_counts(df["conf_flags"], n, label="flag", frame=df),
        "links": quantiles(df["n_links"]),
        "pct_zero_link": _pct(int((df["n_links"] == 0).sum()), n),
        "pct_zero_link_markdown": _pct(int((md["n_links"] == 0).sum()), len(md)),
        "zero_link_by_generator": table(
            ["generator", "documents", "% with zero links"],
            [
                [g, int(sub.shape[0]), _pct(int((sub["n_links"] == 0).sum()), sub.shape[0])]
                for g, sub in df.groupby("generator_id")
                if sub.shape[0] >= max(200, len(df) // 2000)
            ],
        ),
        "n_markdown": len(md),
    }


def track_b(df: pd.DataFrame) -> dict:
    n = len(df)
    gen = counts(df["generator_id"], n, top=20, label="generator", frame=df)
    fam = counts(df["generator_family"], n, label="family", frame=df)
    versions = (
        df[(df["generator_version"] != "") & df["generator_version"].notna()]
        .groupby(["generator_id", "generator_version"])
        .size()
        .sort_values(ascending=False)
        .head(20)
    )
    clusters = (
        df.groupby("skeleton_sha1")
        .agg(size=("skeleton_sha1", "size"), gen=("generator_id", "first"),
             lang=("lang", lambda s: s.mode().iat[0] if len(s) else ""))
        .sort_values("size", ascending=False)
        .head(15)
    )
    cluster_ex = _examples_by_group(df, df["skeleton_sha1"], list(clusters.index))
    return {
        "generator": gen,
        "generator_chart": series_for_chart(df["generator_id"], 12),
        "family": fam,
        "source": counts(df["generator_source"], n, label="detection", frame=df),
        "versions": table(["generator", "version", "documents"],
                          [[a, b, int(v)] for (a, b), v in versions.items()]),
        "top_templates": table(
            ["cluster size", "% of corpus", "generator", "top language", EXAMPLES_COL],
            [[int(r.size), _pct(int(r.size), n), r.gen, r.lang, cluster_ex.get(r.Index, [])]
             for r in clusters.itertuples()],
        ),
        "template_share": _pct(int(df["is_template"].sum()), n),
        "top10_template_share": _pct(int(clusters["size"].head(10).sum()), n),
        "mcp": {
            "pct_mentions_mcp": _pct(int(df["mentions_mcp"].sum()), n),
            "n_mcp_endpoints": int((df["mcp_endpoint"] != "").sum()),
            "pct_mcp_endpoint": _pct(int((df["mcp_endpoint"] != "").sum()), n),
            "by_generator": counts(df.loc[df["mentions_mcp"], "generator_id"],
                                   int(df["mentions_mcp"].sum()), top=10, label="generator"),
        },
        "llm_artefacts": list_counts(df["llm_artefacts"], n, label="artefact", frame=df),
        "pct_llm_artefacts": _pct(int(df["has_llm_artefacts"].sum()), n),
        "artefacts_by_generator": counts(
            df.loc[df["has_llm_artefacts"], "generator_id"],
            max(int(df["has_llm_artefacts"].sum()), 1), top=8, label="generator"),
        "generator_vs_conformance": table(
            ["generator", "documents", "% conformance >= 3", "median links", "median tokens"],
            [
                [g, int(sub.shape[0]),
                 _pct(int((sub["conformance_level"] >= 3).sum()), sub.shape[0]),
                 int(sub["n_links"].median()), int(sub["n_tokens"].median())]
                for g, sub in df.groupby("generator_id")
                if sub.shape[0] >= max(200, len(df) // 2000)
            ],
        ),
    }


def track_d(df: pd.DataFrame) -> dict:
    n = len(df)
    pol = df[df["has_any_policy"]]
    bots = Counter()
    allowed = Counter()
    denied = Counter()
    for nb, al, dn in zip(df["named_bots"], df["bots_allowed"], df["bots_denied"]):
        if nb is not None and len(nb):
            bots.update(set(nb))
        if al is not None and len(al):
            allowed.update(set(al))
        if dn is not None and len(dn):
            denied.update(set(dn))
    top_bots = [name for name, _ in bots.most_common(25)]
    bot_set = set(top_bots)
    named_flat = df["named_bots"].map(
        lambda v: [x for x in (v if v is not None else []) if x in bot_set])
    bot_mask = named_flat.map(bool)
    bot_ex: dict = {}
    if bot_mask.any():
        sub = df.loc[bot_mask, ["url", "dataset_index"]].copy()
        sub["_g"] = named_flat.loc[bot_mask]
        sub = sub.explode("_g")
        bot_ex = {g: _sample_rows(part) for g, part in sub.groupby("_g", sort=False)}

    bot_rows = []
    for name, total in bots.most_common(25):
        a, d = allowed.get(name, 0), denied.get(name, 0)
        bot_rows.append([name, total, a, d, _pct(d, a + d) if (a + d) else None,
                         bot_ex.get(name, [])])
    return {
        "pct_any_policy": _pct(len(pol), n),
        "dialect": counts(df["policy_dialect"], n, label="dialect", frame=df),
        "dialect_chart": series_for_chart(df.loc[df["policy_dialect"] != "none", "policy_dialect"], 6),
        "directives": list_counts(df["policy_directives"], n, label="directive", frame=df),
        "training_stance": counts(df["training_stance"], n, label="stance", frame=df),
        "training_stance_nonnone": counts(
            df.loc[df["training_stance"] != "none", "training_stance"],
            int((df["training_stance"] != "none").sum()), label="stance"),
        "named_bots": table(
            ["crawler", "documents naming it", "allowed", "denied", "% denied (of decided)",
             EXAMPLES_COL],
            bot_rows),
        "n_docs_naming_bots": int((df["n_named_bots"] > 0).sum()),
        "pct_docs_naming_bots": _pct(int((df["n_named_bots"] > 0).sum()), n),
        "policy_by_generator": counts(pol["generator_id"], len(pol), top=10, label="generator"),
        "policy_heading": _pct(int(df["has_policy_heading"].sum()), n),
        "contact_email": _pct(int(df["has_contact_email"].sum()), n),
    }


def track_index(idx: dict, everything: pd.DataFrame, n_analysed: int) -> dict:
    """The pre-analysis funnel, from the URL index.

    Also reconciles the index against the analysed corpus. The two Athena
    queries differ in one clause: the index one matches ``url_path =
    '/llms.txt'`` exactly, while the WARC repackage that built the dataset used
    ``url_path LIKE '%/llms.txt'`` and so also picked up files in
    subdirectories. Reporting the difference is what makes the funnel add up.
    """
    total = idx["total"]
    status = {int(k): v for k, v in idx["status"].items()}
    by_kind = idx["by_kind"]

    grouped = []
    for label, pred in STATUS_GROUPS:
        v = sum(c for s, c in status.items() if pred(s))
        grouped.append([label, v, _pct(v, total)])

    text200 = sum(k["text200"] for k in by_kind.values())
    s200 = status.get(200, 0)
    funnel = [
        ["URLs attempted", total, 100.0, "every /llms.txt and /llms-full.txt in the crawl"],
        ["responded 200", s200, _pct(s200, total), "the file exists and was served"],
        ["…with a text body", text200, _pct(text200, total),
         "text/plain or text/markdown — the analysable population"],
    ]

    paths = everything["url"].str.replace(r"^https?://[^/]+", "", regex=True)
    n_root = int(paths.isin(["/llms.txt", "/llms-full.txt"]).sum())
    n_deep = len(everything) - n_root

    mimes = list(idx["mime_200"].items())
    mime_rows = [[m, c, _pct(c, s200)] for m, c in mimes[:12]]
    other = sum(c for _, c in mimes[12:])
    if other:
        mime_rows.append([f"other ({len(mimes) - 12} types)", other, _pct(other, s200)])

    kind_rows = []
    for kind in ("llms.txt", "llms-full.txt"):
        k = by_kind[kind]
        t = k["total"]
        kind_rows.append([
            kind, t, _pct(k["s404"], t), _pct(k["s3012"], t),
            _pct(k["s200"], t), _pct(k["text200"], t), _pct(k["text200"], max(k["s200"], 1)),
        ])

    return {
        "query": idx["query"],
        "seeding": idx.get("seeding"),
        "total": total,
        "n_status_codes": idx["n_status_codes"],
        "pct_404": _pct(status.get(404, 0), total),
        "pct_3xx": _pct(sum(c for s, c in status.items() if 300 <= s < 400), total),
        "pct_200": _pct(s200, total),
        "pct_text200": _pct(text200, total),
        "pct_text_of_200": _pct(text200, s200),
        "n_text200": text200,
        "funnel": table(["stage", "URLs", "% of attempted", "what it means"], funnel),
        "status": table(["response", "URLs", "% of attempted"], grouped),
        "status_chart": {"labels": [r[0] for r in grouped], "values": [r[1] for r in grouped]},
        "mime": table(["content type detected (HTTP 200)", "URLs", "% of 200s"], mime_rows),
        "mime_chart": {"labels": [m for m, _ in mimes[:10]],
                       "values": [c for _, c in mimes[:10]]},
        "by_kind": table(
            ["file", "URLs attempted", "% 404", "% 301/302", "% 200",
             "% 200 + text body", "% of its 200s"],
            kind_rows),
        "reconcile": table(
            ["population", "documents"],
            [
                ["URL index: HTTP 200 with a text body", text200],
                ["dataset: records extracted from WARC", len(everything)],
                ["  of which at a root path", n_root],
                ["  of which deeper in the site", n_deep],
                ["analysed here (non-empty)", n_analysed],
            ],
        ),
        "n_root": n_root,
        "n_deep": n_deep,
        "pct_deep": _pct(n_deep, len(everything)),
    }


def _mismatch_table(df: pd.DataFrame) -> dict:
    """ccTLD / content-language pairs that should not co-occur, with examples."""
    mm = df.loc[df["lang_mismatch"]]
    if mm.empty:
        return table(["TLD", "content language", "documents", EXAMPLES_COL], [])
    pairs = mm.groupby(["tld", "lang"]).size().sort_values(ascending=False).head(15)
    key = mm["tld"].astype(str) + "|" + mm["lang"].astype(str)
    ex = _examples_by_group(mm, key, [f"{t}|{l}" for t, l in pairs.index])
    return table(
        ["TLD", "content language", "documents", EXAMPLES_COL],
        [[t, l, int(v), ex.get(f"{t}|{l}", [])] for (t, l), v in pairs.items()],
    )


def track_e(df: pd.DataFrame) -> dict:
    n = len(df)
    sev = df["injection_severity"].map(SEVERITY_LABELS)
    human = df[df["is_human_authored"]]
    return {
        "injection": counts(sev, n, label="severity", frame=df),
        "injection_chart": series_for_chart(sev[sev != "none"], 4),
        "injection_matches": list_counts(df["injection_matches"], n, label="pattern", frame=df),
        "injection_human_only": counts(
            human["injection_severity"].map(SEVERITY_LABELS), len(human), label="severity"),
        "pct_injection_ge2": _pct(int((df["injection_severity"] >= 2).sum()), n),
        "pct_injection_ge3": _pct(int((df["injection_severity"] >= 3).sum()), n),
        "for_sale": {
            "n": int(df["for_sale"].sum()),
            "pct": _pct(int(df["for_sale"].sum()), n),
            "marketplace": counts(df.loc[df["for_sale"], "for_sale_marketplace"],
                                  int(df["for_sale"].sum()), label="marketplace"),
            "pct_wellformed": _pct(
                int(((df["for_sale"]) & (df["conformance_level"] >= 2)).sum()),
                max(int(df["for_sale"].sum()), 1)),
        },
        "spam": list_counts(df["spam_categories"], n, label="lexicon", frame=df),
        "pct_spam": _pct(int(df["has_spam"].sum()), n),
        "spam_chart": series_for_chart(
            df.loc[df["n_spam_categories"] > 0, "spam_categories"].map(lambda v: v[0]), 8),
        "offsite": {
            "pct_dominant": _pct(int(df["offsite_dominant"].sum()), n),
            "offsite_ratio": quantiles((df.loc[df["n_links"] > 0, "offsite_ratio"] * 100).round()),
            "n_offsite_domains": quantiles(df.loc[df["n_links"] > 0, "n_offsite_domains"]),
        },
        "lang_mismatch": {
            "pct": _pct(int(df["lang_mismatch"].sum()), n),
            "top": _mismatch_table(df),
        },
        "pct_suspicious": _pct(int(df["is_suspicious"].sum()), n),
        "suspicious_by_generator": counts(
            df.loc[df["is_suspicious"], "generator_id"],
            max(int(df["is_suspicious"].sum()), 1), top=10, label="generator"),
    }


def _cost_rows(total_tokens: int) -> list[list]:
    """Cost of reading every document once, at litellm's list prices.

    Each document is priced as its own request, which is how an agent would
    actually fetch them. Note that passing the corpus total to
    ``litellm.cost_per_token`` in one call would instead trigger the
    long-context tier some providers charge above ~200k tokens and roughly
    double the figure; that is not the scenario being costed.
    """
    from litellm import model_cost

    rows = []
    for m in COST_MODELS:
        info = model_cost.get(m)
        rate = (info or {}).get("input_cost_per_token")
        if not rate:
            continue
        rows.append([
            m,
            round(rate * 1e6, 3),
            int(info.get("max_input_tokens") or 0),
            round(rate * total_tokens, 2),
        ])
    return rows


def track_f(df: pd.DataFrame) -> dict:
    n = len(df)
    human = df[df["is_human_authored"]]
    total_tokens = int(df["n_tokens"].sum())
    fit_rows = []
    for w in CONTEXT_WINDOWS:
        over = int((df["n_tokens"] > w).sum())
        fit_rows.append([f"{w:,}", over, _pct(over, n)])
    lang_top = df["lang"].value_counts().head(20)
    return {
        "lang": counts(df["lang"], n, top=25, label="language (ISO 639-3)", frame=df),
        "lang_chart": series_for_chart(df.loc[df["lang"] != "und", "lang"], 15),
        "lang_source": counts(df["lang_source"], n, label="LID model", frame=df),
        "lang_human": counts(human["lang"], len(human), top=15, label="language", frame=human),
        "n_languages": int((df["lang"].value_counts() >= 10).sum()),
        "lang_by_generator": table(
            ["generator", "documents", "distinct languages (>=10 docs)", "% English"],
            [
                [g, int(sub.shape[0]),
                 int((sub["lang"].value_counts() >= 10).sum()),
                 _pct(int((sub["lang"] == "eng").sum()), sub.shape[0])]
                for g, sub in df.groupby("generator_id")
                if sub.shape[0] >= max(500, len(df) // 1000)
            ],
        ),
        "chars": quantiles(df["n_chars"]),
        "tokens": quantiles(df["n_tokens"]),
        "words": quantiles(df["n_words"]),
        "tokens_by_file": table(
            ["file", "documents", "median tokens", "p90", "p99", "max"],
            [[k, int(sub.shape[0]), int(sub["n_tokens"].median()),
              int(sub["n_tokens"].quantile(0.9)), int(sub["n_tokens"].quantile(0.99)),
              int(sub["n_tokens"].max())]
             for k, sub in df.groupby("file_kind")],
        ),
        "tokens_hist": _log_hist(df["n_tokens"]),
        "context_fit": table(["context window (tokens)", "documents exceeding", "% of corpus"], fit_rows),
        "total_tokens": total_tokens,
        "cost": table(
            ["model", "input $/1M tokens", "context window", "cost to ingest whole corpus ($)"],
            _cost_rows(total_tokens)),
        "lang_top_for_topics": [str(k) for k in lang_top.index],
    }


def _log_hist(series: pd.Series, nbins: int = 24) -> dict:
    import numpy as np

    s = series[series > 0]
    if s.empty:
        return {"edges": [], "counts": []}
    lo, hi = np.log10(max(s.min(), 1)), np.log10(s.max())
    edges = np.logspace(lo, hi, nbins + 1)
    cnt, _ = np.histogram(s, bins=edges)
    return {"edges": [float(x) for x in edges], "counts": [int(x) for x in cnt]}


# --------------------------------------------------------------------------
def build(derived: str | Path, topics_path: str | Path | None = None,
          template_min_cluster: int = 50,
          urlindex_path: str | Path | None = None) -> dict:
    # The full frame is loaded once so the excluded rows can be counted and
    # reported; every figure below is computed on the analysis frame.
    everything = load(derived, template_min_cluster, drop_empty=False)
    df = drop_empty_rows(everything)
    n_empty = len(everything) - len(df)
    stats: dict[str, Any] = {
        "meta": {
            "dataset": "commoncrawl/llms.txt",
            "config": "CC-MAIN-2026-30",
            "responses": len(everything),
            "records": len(df),
            "n_empty": n_empty,
            "pct_empty": _pct(n_empty, len(everything)),
            "template_min_cluster": template_min_cluster,
        },
        "overview": overview(df, everything),
        "A": track_a(df),
        "B": track_b(df),
        "D": track_d(df),
        "E": track_e(df),
        "F": track_f(df),
    }
    if urlindex_path is None:
        base = Path(derived)
        urlindex_path = (base.parent if base.name == "features" else base) / "urlindex.json"
    if urlindex_path and Path(urlindex_path).exists():
        stats["index"] = track_index(
            json.loads(Path(urlindex_path).read_text()), everything, len(df))
    if topics_path and Path(topics_path).exists():
        stats["topics"] = json.loads(Path(topics_path).read_text())
    return stats


def write(stats: dict, out: str | Path) -> None:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats, indent=1, default=str))
    print(f"wrote {p} ({p.stat().st_size/1e6:.2f} MB)")
