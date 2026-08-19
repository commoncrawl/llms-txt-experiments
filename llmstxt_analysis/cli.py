"""Command line interface: ``uv run analyze.py <subcommand>``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_DERIVED = "data/derived"


def _p(parser: argparse.ArgumentParser, *names: str) -> None:
    """Attach the common path options a subcommand needs."""
    if "shards" in names:
        from .corpus import DEFAULT_CONFIG, DEFAULT_REPO

        parser.add_argument("--dataset", default=DEFAULT_REPO,
                            help="Hugging Face dataset repo id, or a local directory of "
                                 "parquet shards (used by the tests)")
        parser.add_argument("--config", default=DEFAULT_CONFIG,
                            help="dataset config (crawl) to read")
    if "derived" in names:
        parser.add_argument("--derived", default=DEFAULT_DERIVED,
                            help="directory holding features/ and topic_corpus/")
    if "template" in names:
        parser.add_argument("--template-min-cluster", type=int, default=50,
                            help="skeleton cluster size at which a document counts as templated")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="analyze.py",
        description="Content analysis of llms.txt files from Common Crawl.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="list shards and registered extractors")
    _p(p, "shards")

    p = sub.add_parser("extract", help="stream shards -> features table")
    _p(p, "shards")
    p.add_argument("--out", default=DEFAULT_DERIVED)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-shards", type=int, default=0)
    p.add_argument("--limit-rows", type=int, default=0, help="rows per shard (0 = all)")

    p = sub.add_parser("urlindex",
                       help="reduce the URL-index CSV to the pre-analysis funnel")
    p.add_argument("--csv", default=None, help="default: data/llms-txt-cc-main-2026-30.csv.zst")
    _p(p, "derived")
    p.add_argument("--out", default=None, help="default: <derived>/urlindex.json")

    p = sub.add_parser("summary", help="quick console summary of the features table")
    _p(p, "derived", "template")

    p = sub.add_parser("cache", help="derived-frame cache (build / info / clear)")
    p.add_argument("action", choices=["build", "info", "clear"], nargs="?", default="info")
    _p(p, "derived", "template")

    p = sub.add_parser("spotcheck", help="print real records matching a filter")
    _p(p, "derived", "shards")
    p.add_argument("--filter", default="", help="pandas query expression over the features table")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--chars", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)

    p = sub.add_parser("topics", help="LDA topic model (fit / assign)")
    tsub = p.add_subparsers(dest="topics_cmd", required=True)
    pf = tsub.add_parser("fit", help="fit LDA and dump top terms for manual labelling")
    _p(pf, "derived", "template")
    pf.add_argument("--languages", default="eng")
    pf.add_argument("--n-topics", type=int, default=20)
    pf.add_argument("--min-docs", type=int, default=3000)
    pf.add_argument("--max-docs", type=int, default=200_000)
    pf.add_argument("--out", default=None, help="default: <derived>/topics")
    pa = tsub.add_parser("assign", help="attach manual names to fitted topics")
    _p(pa, "derived")
    pa.add_argument("--raw", default=None, help="default: <derived>/topics/topics_raw.json")
    pa.add_argument("--names", default="topic_names.yaml")
    pa.add_argument("--out", default=None, help="default: <derived>/topics.json")

    p = sub.add_parser("aggregate", help="features -> stats.json")
    _p(p, "derived", "template")
    p.add_argument("--topics", default=None, help="default: <derived>/topics.json if present")
    p.add_argument("--urlindex", default=None,
                   help="default: <derived>/urlindex.json if present")
    p.add_argument("--out", default=None, help="default: <derived>/stats.json")

    p = sub.add_parser("report", help="stats.json -> standalone HTML report")
    p.add_argument("--stats", default=None, help="default: <derived>/stats.json")
    _p(p, "derived")
    p.add_argument("--out", default="report/index.html",
                   help="output path; the directory is what gets uploaded to the Space")
    p.add_argument("--fragment", action="store_true",
                   help="emit the body only, for hosts that supply their own "
                        "<html>/<head> skeleton (e.g. a Claude artifact)")
    p.add_argument("--no-space-card", action="store_true",
                   help="skip writing the Space README.md next to the report")

    p = sub.add_parser("figures",
                       help="stats.json -> standalone SVG/PNG charts for the blog post")
    p.add_argument("--stats", default=None, help="default: <derived>/stats.json")
    _p(p, "derived")
    p.add_argument("--out", default="docs/blog-post/img",
                   help="directory to write the chart files into")
    p.add_argument("--no-png", action="store_true",
                   help="skip rasterising (PNG needs rsvg-convert on PATH)")

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "info":
        from . import extractors  # noqa: F401
        from .corpus import cache_root, resolve_shards, shard_info
        from .registry import all_extractors, feature_schema

        shards = resolve_shards(args.dataset, args.config)
        print(f"dataset {args.dataset} config {args.config}: {len(shards)} shard(s)")
        print(f"HF cache: {cache_root()}")
        total = 0
        for s in shards:
            i = shard_info(s)
            total += i["rows"]
            print(f"  [{i['shard']:>2}] {i['file']}  rows={i['rows']:>8}  "
                  f"row_groups={i['row_groups']:>3}  {i['size_mb']} MB")
        print(f"  total rows: {total}")
        print("\nregistered extractors:")
        for cls in all_extractors():
            print(f"  {cls.NAME:<14} track={cls.TRACK:<5} fields={len(cls.FIELDS)}")
        print(f"\nfeature schema: {len(feature_schema())} columns")
        return 0

    if args.cmd == "extract":
        from .extract import run

        run(args.dataset, args.out, workers=args.workers,
            limit_shards=args.limit_shards, limit_rows=args.limit_rows,
            config=args.config)
        return 0

    if args.cmd == "summary":
        from .derive import load

        df = load(args.derived, args.template_min_cluster)
        n = len(df)
        print(f"records: {n}   hosts: {df['host'].nunique()}")
        for col in ["doc_kind", "conformance_level", "generator_id", "lang",
                    "policy_dialect", "training_stance", "injection_severity"]:
            print(f"\n--- {col} ---")
            vc = df[col].value_counts().head(12)
            for k, v in vc.items():
                print(f"  {str(k):<22} {v:>8}  {100*v/n:6.2f}%")
        print(f"\ntemplates: {100*df['is_template'].mean():.2f}%   "
              f"human-authored: {100*df['is_human_authored'].mean():.2f}%   "
              f"exact dups: {100*df['is_exact_dup'].mean():.2f}%")
        print(f"tokens: median={df['n_tokens'].median():.0f} "
              f"p99={df['n_tokens'].quantile(0.99):.0f} max={df['n_tokens'].max():.0f}")
        return 0

    if args.cmd == "urlindex":
        from .urlindex import DEFAULT_CSV, scan, write

        result = scan(args.csv or DEFAULT_CSV)
        write(result, args.out or Path(args.derived) / "urlindex.json")
        return 0

    if args.cmd == "cache":
        from .derive import cache_info, clear_cache, load

        if args.action == "clear":
            print(f"removed {clear_cache(args.derived)} cache file(s)")
            return 0
        if args.action == "build":
            load(args.derived, args.template_min_cluster, verbose=True)
        info = cache_info(args.derived, args.template_min_cluster)
        print(f"cache: {info['path']}")
        print(f"  fresh: {info['fresh']}   size: {info['size_mb']} MB")
        if info["stale_entries"]:
            print(f"  superseded entries present: {', '.join(info['stale_entries'])}")
        return 0

    if args.cmd == "spotcheck":
        from .spotcheck import run

        run(args.derived, args.dataset, args.filter, n=args.n, chars=args.chars,
            seed=args.seed, out=args.out, config=args.config)
        return 0

    if args.cmd == "topics":
        from . import topics as T

        derived = Path(args.derived)
        if args.topics_cmd == "fit":
            T.fit(derived, [l.strip() for l in args.languages.split(",") if l.strip()],
                  out=args.out or derived / "topics", n_topics=args.n_topics,
                  min_docs=args.min_docs, template_min_cluster=args.template_min_cluster,
                  max_docs=args.max_docs)
        else:
            T.assign(args.raw or derived / "topics" / "topics_raw.json",
                     args.names, args.out or derived / "topics.json")
        return 0

    if args.cmd == "aggregate":
        from .aggregate import build, write

        derived = Path(args.derived)
        topics = args.topics or (derived / "topics.json")
        stats = build(derived, topics, args.template_min_cluster, args.urlindex)
        write(stats, args.out or derived / "stats.json")
        return 0

    if args.cmd == "report":
        from .report import render

        stats_path = Path(args.stats or Path(args.derived) / "stats.json")
        stats = json.loads(stats_path.read_text())
        render(stats, args.out, fragment=args.fragment,
               write_space_card=not args.no_space_card)
        return 0

    if args.cmd == "figures":
        from .figures import write

        stats_path = Path(args.stats or Path(args.derived) / "stats.json")
        stats = json.loads(stats_path.read_text())
        write(stats, args.out, png=not args.no_png)
        return 0

    return 1
