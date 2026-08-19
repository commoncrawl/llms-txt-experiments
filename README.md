# llms-txt-experiments

Experiments on the usage of [llms.txt](https://llmstxt.org/) files in the Common Crawl corpus

Datasets:
- S3: https://data.commoncrawl.org/projects/llms-txt-analysis/index.html
- HF: https://huggingface.co/datasets/commoncrawl/llms.txt

The analysis of this dataset is available as a Hugging Face space: 🤗 [commoncrawl/llms.txt-report](https://huggingface.co/spaces/commoncrawl/llms.txt-report)

## Install

You can install the project dependencies via uv:

```bash
uv sync
```

## How the data was collected

Procedure and results from an experiment performed during July crawl (CC-MAIN-2026-30):

- Take a random sample from the successfully fetched hosts of the first seed crawl cycle. These hosts are approved to be crawlable now.
  - 5,167,831 `/llms.txt`
  - 1,761,228 `/llms-full.txt`. Draw a smaller sample, because if there's no `/llms.txt`, it's very unlikely that a `/llms-full.txt` is provided. See next point how to proceed.
- Look into URL index of the preceding two crawl (CC-MAIN-2026-21 and CC-MAIN-2026-25), to get a list of sites providing a `/llms.txt` or `/llms-full.txt`.
  - 27,394 hosts with `/llms.txt` found
  - 15,763 hosts with `/llms-full.txt` found
  - Overlap
    - 13,583 hosts with both `/llms.txt` and `/llms-full.txt`.
    - 2,180 hosts with only `/llms-full.txt`. Of course, the `/llms.txt` might have been captured in an earlier crawl.
    - 13,811 hosts with only `/llms.txt`.
      - For these hosts the `/llms-full.txt` was added as seeds.
- Results for `/llms.txt`
  ```
   4895688  total
   3348471  HTTP 404
        68.40%    of total
    365545  HTTP 301/302
         7.47%    of total
   1092709  HTTP 200
        22.32%    of total
    573659  HTTP 200 and (text/markdown or text/plain)
        11.72%    of total
        52.50%    of HTTP 200
  ```
- Results for `/llms-full.txt`
  ```
   1667437  total
   1232826  HTTP 404
        73.94%    of total
    122580  HTTP 301/302
         7.35%    of total
    194498  HTTP 200
        11.66%    of total
     13643  HTTP 200 and (text/markdown or text/plain)
         0.82%    of total
         7.01%    of HTTP 200
  ```


## WARC repackage

Extract WARC response records from the main crawl archive with `llms.txt` and `llms-full.txt` fetches:

```bash
mkdir -p data/cc-repackage/

# build fetch job CSV (no fetch, query costs ~ 0.30 USD)
cdxt -v repackage \
    --target-source sql --engine athena \
    --query "SELECT warc_filename, warc_record_offset, warc_record_length, url
             FROM ccindex
             WHERE subset = 'warc'
                AND crawl = 'CC-MAIN-2026-30'
                AND url_query IS NULL
                AND (url_path LIKE '%/llms.txt' or url_path LIKE '%/llms-full.txt')
                AND fetch_status = 200
                AND content_mime_detected IN ('text/plain', 'text/markdown')" \
    --athena-database ccindex \
    --confirm-cost \
    --range-jobs-output ./data/cc-repackage/llms-txt-CC-MAIN-2026-30_ranges.csv --no-fetch

# repackage from fetch job CSV (~ 8 GB)
CDXT_UVLOOP=1 cdxt -v repackage \
    --target-source csv --csv-path ./data/cc-repackage/llms-txt-CC-MAIN-2026-30_ranges.csv \
    --warc-download-prefix s3://commoncrawl \
    --prefix ./data/cc-repackage/llms-txt-CC-MAIN-2026-30 \
    --processes 4 --parallel_readers 32
```

## WARC2HF

Convert the WARC files into a [Hugging Face datset](https://huggingface.co/datasets/commoncrawl/llms.txt):

```bash
uv run warc2hf.py --hf-repo-id=commoncrawl/llms.txt --config-name CC-MAIN-2026-30 \
    --warc ./data/cc-repackage/llms-txt-CC-MAIN-2026-30-*.warc.gz
```


## Content analysis

A content analysis of what is actually written inside the llms.txt files in
`CC-MAIN-2026-30` — spec conformance, which tool generated the file, AI-usage
policy, abuse, and language/length/topic.

```bash
uv run analyze.py --help   # info | urlindex | extract | cache | summary | spotcheck
                           # topics | aggregate | report | figures
uv run pytest tests/ -q    # unit + integration tests over a real-data fixture
```

The report opens with the URL-index funnel — all 6.6M attempted `/llms.txt`
fetches, their status codes and content types.

Those attempts are the seeding experiment described in [How the data was
collected](#how-the-data-was-collected) above, not an incidental by-product of
the crawl, and the funnel section says so. The sample sizes it quotes live in
`llmstxt_analysis/urlindex.py::SEEDING`, next to the index query they belong
with; they flow into `stats.json` and are rendered from there. Update `SEEDING`
when running this against another crawl — the numbers are quoted from the
experiment, not derived from the data.

The pipeline is a registry of small feature extractors (`llmstxt_analysis/extractors/`)
run in one streaming pass over the parquet shards; adding an analysis means
adding one class. Output is a single standalone HTML report.

## Blog post figures

The figures of the blog post can be reproduced with this command:

```bash
uv run analyze.py figures --stats data/derived/stats.json --out docs/blog-post/img
```

## Publish the report as a Hugging Face Space

`analyze.py report` writes `report/index.html` (a complete, self-contained HTML
document — no external requests) together with `report/README.md`, the Space
card whose `sdk: static` header is what makes the Hub serve the page. The
`report/` directory is therefore exactly the contents of the Space.

```bash
# regenerate the report and its Space card
uv run analyze.py report --stats data/derived/stats.json --out report/index.html

# one-off: create the static Space
hf repos create commoncrawl/llms.txt-report --type space --sdk static

# upload (or re-upload) the report
hf upload commoncrawl/llms.txt-report ./report . --repo-type=space \
    --commit-message "Update llms.txt content analysis report"
```

Live at <https://huggingface.co/spaces/commoncrawl/llms.txt-report>.

## License

Apache 2.0
