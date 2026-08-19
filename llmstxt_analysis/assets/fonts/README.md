# Vendored fonts

The two typefaces of the Common Crawl style guide,
copied from `cc-web-tools/fonts/`. The report inlines them as base64 data URIs
(`llmstxt_analysis/report.py::font_faces`), so the generated page needs no
network access and the artifact host enforces with a strict CSP.

| File | Family | Upstream | Licence |
|---|---|---|---|
| `LibreFranklin_wght.woff2` | Libre Franklin (variable, 100–900) | [impallari/Libre-Franklin](https://github.com/impallari/Libre-Franklin) | SIL Open Font License 1.1 |
| `IBMPlexMono-Regular.woff2` | IBM Plex Mono | [IBM/plex](https://github.com/IBM/plex) | SIL Open Font License 1.1 |

The OFL permits redistribution of the font files, including bundled inside a
document, provided they are not sold on their own and the licence travels with
them. Neither family is renamed here.

To refresh, copy the `.woff2` files from `cc-web-tools/fonts/` again; nothing
else needs to change.
