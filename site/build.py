"""Builds the GitHub Pages site (docs/) from site/article.html.

article.html is written as a page fragment (no doctype, html, head or body tags) so that the same file can be
published as a private preview. This wraps it in a full document for GitHub Pages: everything before the site
header goes in <head>, the rest in <body>. It also copies site/assets to docs/assets.
usage (project root): python3 -I site/build.py [--draft]   (--draft adds a noindex tag, for review before launch)
"""
import pathlib
import re
import shutil
import sys

DRAFT = "--draft" in sys.argv  # a review build: search engines are asked not to index or follow it

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "site", ROOT / "docs"
frag = (SRC / "article.html").read_text(encoding="utf-8")
cut = frag.index('<header class="site">')
head, body = frag[:cut], frag[cut:]
# namespaced viewer hints (<meta name="prefix:...">) belong to the private preview, not to GitHub Pages
head = "".join(l for l in head.splitlines(keepends=True) if not re.search(r'<meta name="[a-z]+:', l))
page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        + ("<meta name=\"robots\" content=\"noindex, nofollow\">\n" if DRAFT else "")
        + "<meta property=\"og:type\" content=\"article\">\n"
        "<meta property=\"og:title\" content=\"Can an AI scientist learn to discover?\">\n"
        "<meta property=\"og:description\" content=\"An AI scientist designs learners for a 3D brick world with hidden rules, then rewrites its own research procedure, under tests fixed before each run. All of it runs locally on one Apple Silicon laptop.\">\n"
        + head + "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
OUT.mkdir(exist_ok=True)
(OUT / "index.html").write_text(page, encoding="utf-8")
(OUT / ".nojekyll").write_text("")
if (OUT / "assets").exists():
    shutil.rmtree(OUT / "assets")
shutil.copytree(SRC / "assets", OUT / "assets")
print("wrote", OUT / "index.html", len(page), "bytes", "(draft: noindex)" if DRAFT else "")
