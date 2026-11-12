#!/usr/bin/env python3
"""Static checks for the site, run from the repository root:

    python tools/check_site.py            # local checks only
    python tools/check_site.py --external # also request every external link

The script serves the repository with a simple local HTTP server, fetches every
HTML page, and checks that:

  * every local asset and link (href, src, srcset, use, og:image on this domain,
    posts.json entries) returns 200, and every fragment points to an existing id;
  * every external link is well formed (scheme, host, no spaces);
  * the redesigned pages (index.html and posts/*.html) have balanced tags, unique
    ids, alt text and sizes on images, rel="noopener" on new-tab links, valid
    ARIA references, one h1, no skipped heading levels, valid JSON-LD, and no
    em dash, en dash or semicolon in the visible text;
  * the home page, its stylesheet and its script stay under the 300 KB budget;
  * no excluded topic appears on the redesigned pages.

The demo pages are fetched and their local assets are checked, but their markup
is left as it is. Exit status is 1 when any error is found.
"""

from __future__ import annotations

import argparse
import functools
import html.parser
import http.server
import json
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE_ORIGIN = "https://vcnoel.github.io"
REDESIGNED = ["index.html", "posts/index.html", "posts/site-relaunch.html", "posts/template.html"]
ALLOWED_ASSET_HOSTS = {"fonts.googleapis.com", "fonts.gstatic.com", "cdnjs.cloudflare.com", "cdn.jsdelivr.net"}
BUDGET_FILES = ["index.html", "assets/css/site.css", "assets/js/site.js"]
BUDGET_BYTES = 300 * 1024
FORBIDDEN_TEXT = ["\u2014", "\u2013", ";"]  # em dash, en dash, semicolon
# Phrases that must never appear on the site, one per line, kept in an untracked
# file so that the list itself is not published (see .gitignore).
EXCLUDED_FILE = ROOT / "tools" / "excluded-topics.txt"
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
URL_ATTRS = {"href", "src", "action", "poster", "data"}
# Content model checks (a subset of the HTML standard that catches the usual mistakes)
BLOCK = {"address", "article", "aside", "blockquote", "details", "dialog", "div", "dl", "fieldset", "figcaption",
         "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "main", "nav", "ol", "p",
         "pre", "section", "table", "ul", "li", "dt", "dd", "summary"}
PHRASING_ONLY = {"p", "span", "button", "label", "em", "strong", "code", "time", "small", "b", "i", "cite",
                 "h1", "h2", "h3", "h4", "h5", "h6", "summary", "dt", "pre"}
INTERACTIVE = {"a", "button", "details", "input", "select", "textarea", "label", "iframe"}
REQUIRED_PARENT = {"li": {"ul", "ol", "menu"}, "dt": {"dl", "div"}, "dd": {"dl", "div"},
                   "figcaption": {"figure"}, "summary": {"details"}, "source": {"picture", "video", "audio"}}
TIME_RE = re.compile(r"\d{4}(-\d{2}(-\d{2})?)?")


class Page(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.urls: list[tuple[str, str, str]] = []  # (tag, attr, value)
        self.ids: list[str] = []
        self.stack: list[tuple[str, int]] = []
        self.balance_errors: list[str] = []
        self.model_errors: list[str] = []
        self.doctype = False
        self.text_chunks: list[tuple[str, int]] = []
        self.attr_text: list[tuple[str, str, int]] = []
        self.headings: list[tuple[int, int]] = []
        self.imgs: list[dict] = []
        self.blank_targets: list[tuple[dict, int]] = []
        self.aria_refs: list[tuple[str, str, int]] = []
        self.jsonld: list[str] = []
        self.meta: dict[str, str] = {}
        self.links: list[dict] = []
        self.lang = None
        self.has_title = False
        self._in = {"script": 0, "style": 0, "title": 0}
        self._script_type = None
        self._script_buf: list[str] = []
        self.data_posts: list[str] = []

    def content_model(self, tag, attrs, line):
        names = [t for t, _ in self.stack]
        in_svg = "svg" in names
        if in_svg:
            return
        seen = set()
        for k, _ in attrs:
            if k in seen:
                self.model_errors.append(f"line {line}: duplicate attribute {k!r} on <{tag}>")
            seen.add(k)
        parent = names[-1] if names else None
        if tag in BLOCK:
            for anc in reversed(names):
                if anc in PHRASING_ONLY and not (anc == "summary" and tag in {"h1", "h2", "h3", "h4", "h5", "h6"}):
                    self.model_errors.append(f"line {line}: <{tag}> inside <{anc}>, which allows phrasing content only")
                    break
                if anc in {"div", "section", "article", "li", "main", "header", "footer", "nav", "figure",
                           "details", "dd", "blockquote", "aside", "body", "a", "ul", "ol", "dl"}:
                    if anc != "a":
                        break
        if tag in INTERACTIVE and any(t in {"a", "button"} for t in names):
            self.model_errors.append(f"line {line}: interactive <{tag}> nested inside <a> or <button>")
        if tag in REQUIRED_PARENT and parent not in REQUIRED_PARENT[tag]:
            self.model_errors.append(f"line {line}: <{tag}> must be a child of {sorted(REQUIRED_PARENT[tag])}, not <{parent}>")
        if parent in {"ul", "ol"} and tag not in {"li", "script", "template"}:
            self.model_errors.append(f"line {line}: <{tag}> directly inside <{parent}>")
        if tag == "time":
            dt = dict(attrs).get("datetime")
            if dt is not None and not TIME_RE.fullmatch(dt):
                self.model_errors.append(f"line {line}: <time datetime={dt!r}> is not a valid date")

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        line = self.getpos()[0]
        self.content_model(tag, attrs, line)
        if tag == "html":
            self.lang = a.get("lang")
        if tag == "title":
            self.has_title = True
        if tag in self._in:
            self._in[tag] += 1
        if tag == "script":
            self._script_type = a.get("type")
            self._script_buf = []
        if "id" in a:
            self.ids.append(a["id"])
        for k, v in attrs:
            if v is None:
                continue
            if k in URL_ATTRS:
                self.urls.append((tag, k, v))
            if k == "srcset":
                for part in v.split(","):
                    u = part.strip().split(" ")[0]
                    if u:
                        self.urls.append((tag, k, u))
            if k in {"alt", "title", "aria-label", "placeholder"}:
                self.attr_text.append((tag, f"{k}={v}", line))
            if k in {"aria-controls", "aria-labelledby", "aria-describedby", "for", "data-filter-for",
                     "data-filter-status", "data-filter-toggle", "data-copy-status"}:
                for ref in v.split():
                    self.aria_refs.append((k, ref, line))
            if k == "data-posts":
                self.data_posts.append(v)
        if tag == "meta":
            key = a.get("name") or a.get("property")
            if key and a.get("content") is not None:
                self.meta[key] = a["content"]
                if key in {"description", "og:title", "og:description", "og:image:alt", "twitter:title",
                           "twitter:description", "twitter:image:alt"}:
                    self.attr_text.append(("meta", f"{key}={a['content']}", line))
                if key in {"og:image", "twitter:image", "og:url"}:
                    self.urls.append(("meta", key, a["content"]))
        if tag == "link":
            self.links.append(a)
        if tag == "img":
            self.imgs.append(dict(a, _line=line))
        if tag == "a" and a.get("target") == "_blank":
            self.blank_targets.append((a, line))
        if re.fullmatch(r"h[1-6]", tag):
            self.headings.append((int(tag[1]), line))
        if tag not in VOID:
            self.stack.append((tag, line))

    def handle_startendtag(self, tag, attrs):
        # Self-closing syntax: record attributes but do not push on the stack.
        self.handle_starttag(tag, attrs)
        if tag not in VOID and self.stack and self.stack[-1][0] == tag:
            self.stack.pop()
            if tag in self._in:
                self._in[tag] -= 1

    def handle_endtag(self, tag):
        line = self.getpos()[0]
        if tag in self._in and self._in[tag] > 0:
            self._in[tag] -= 1
        if tag == "script" and self._script_type == "application/ld+json":
            self.jsonld.append("".join(self._script_buf))
        if tag in VOID:
            return
        if not self.stack:
            self.balance_errors.append(f"line {line}: stray </{tag}>")
            return
        if self.stack[-1][0] == tag:
            self.stack.pop()
            return
        names = [t for t, _ in self.stack]
        if tag in names:
            while self.stack and self.stack[-1][0] != tag:
                t, l = self.stack.pop()
                if t not in {"p", "li", "dt", "dd", "option", "tr", "td", "th"}:
                    self.balance_errors.append(f"line {l}: <{t}> not closed before </{tag}> at line {line}")
            self.stack.pop()
        else:
            self.balance_errors.append(f"line {line}: </{tag}> without matching start tag")

    def handle_decl(self, decl):
        if decl.lower() == "doctype html":
            self.doctype = True

    def handle_data(self, data):
        if self._in["script"]:
            self._script_buf.append(data)
            return
        if self.stack and self.stack[-1][0] in {"ul", "ol"} and data.strip():
            self.model_errors.append(f"line {self.getpos()[0]}: text directly inside <{self.stack[-1][0]}>")
        if self._in["style"]:
            return
        if data.strip():
            self.text_chunks.append((data, self.getpos()[0]))


def serve(root: Path):
    handler = functools.partial(QuietHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def fetch(url: str, method: str = "GET", timeout: float = 15.0):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "site-check/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read() if method == "GET" else b""
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:  # noqa: BLE001
        return None, str(e).encode()


def well_formed_external(u: str) -> str | None:
    if any(c.isspace() for c in u):
        return "contains whitespace"
    p = urllib.parse.urlsplit(u)
    if p.scheme == "mailto":
        addr = p.path
        return None if re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", addr) else "malformed mailto"
    if p.scheme not in {"http", "https"}:
        return f"unexpected scheme {p.scheme!r}"
    if not p.netloc or "." not in p.netloc:
        return "missing host"
    if re.search(r"[<>\"{}|\\^`\[\]]", u):
        return "illegal character"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--external", action="store_true", help="also request every external link")
    args = ap.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    excluded: list[tuple[int, str]] = []
    if EXCLUDED_FILE.is_file():
        excluded = [(n, ln.strip()) for n, ln in enumerate(EXCLUDED_FILE.read_text(encoding="utf-8").splitlines(), 1)
                    if ln.strip() and not ln.startswith("#")]
    else:
        warnings.append("tools/excluded-topics.txt not found, excluded-topic check skipped")
    httpd, base = serve(ROOT)

    pages = sorted(
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*.html")
        if ".git" not in p.parts and "node_modules" not in p.parts
    )
    parsed: dict[str, Page] = {}
    for rel in pages:
        status, body = fetch(f"{base}/{rel}")
        if status != 200:
            errors.append(f"{rel}: page returned {status}")
            continue
        pg = Page()
        pg.feed(body.decode("utf-8", errors="replace"))
        pg.close()
        parsed[rel] = pg

    external: set[str] = set()
    local_checked: dict[str, int | None] = {}

    def check_local(page_rel: str, raw: str, where: str) -> None:
        page_url = f"{base}/{page_rel}"
        target = urllib.parse.urljoin(page_url, raw)
        tp = urllib.parse.urlsplit(target)
        path_url = urllib.parse.urlunsplit((tp.scheme, tp.netloc, tp.path, tp.query, ""))
        if path_url not in local_checked:
            local_checked[path_url] = fetch(path_url, "GET")[0]
        if local_checked[path_url] != 200:
            errors.append(f"{page_rel}: {where} -> {raw} returned {local_checked[path_url]}")
            return
        if tp.fragment:
            target_rel = urllib.parse.unquote(tp.path.lstrip("/")) or "index.html"
            if target_rel.endswith("/"):
                target_rel += "index.html"
            tpage = parsed.get(target_rel)
            if tpage is not None and tp.fragment not in tpage.ids:
                errors.append(f"{page_rel}: {where} -> {raw}: no element with id {tp.fragment!r}")

    for rel, pg in parsed.items():
        for tag, attr, raw in pg.urls:
            where = f"<{tag} {attr}>"
            raw = raw.strip()
            if not raw:
                errors.append(f"{rel}: {where} is empty")
                continue
            if raw == "#":
                errors.append(f"{rel}: {where} is a dead '#' link")
                continue
            if raw.startswith("#"):
                if raw[1:] not in pg.ids:
                    errors.append(f"{rel}: {where} -> {raw}: no element with that id")
                continue
            if raw.startswith(("data:", "javascript:", "blob:")):
                continue
            sp = urllib.parse.urlsplit(raw)
            if sp.scheme in {"http", "https", "mailto"}:
                problem = well_formed_external(raw)
                if problem:
                    errors.append(f"{rel}: {where} -> {raw}: {problem}")
                    continue
                if sp.netloc == "vcnoel.github.io":
                    local = sp.path.lstrip("/")
                    if attr == "og:url" or not local or local.endswith("/"):
                        local = (local or "") + ("index.html" if not local or local.endswith("/") else "")
                    if not (ROOT / urllib.parse.unquote(local)).is_file():
                        if "your-post-id" not in local:
                            errors.append(f"{rel}: {where} -> {raw}: no such file in the repository")
                    continue
                if sp.scheme in {"http", "https"}:
                    external.add(raw)
                    if rel in REDESIGNED and tag in {"link", "script"} and attr in {"href", "src"}:
                        if sp.netloc not in ALLOWED_ASSET_HOSTS and tag == "script" or (
                                tag == "link" and attr == "href" and raw.endswith(".css") and sp.netloc not in ALLOWED_ASSET_HOSTS):
                            errors.append(f"{rel}: asset from a host outside the allowed list: {raw}")
                    elif tag in {"script"} and sp.netloc not in ALLOWED_ASSET_HOSTS:
                        warnings.append(f"{rel}: demo page loads a script from {sp.netloc} (page left unchanged)")
                continue
            if sp.scheme:
                errors.append(f"{rel}: {where} -> {raw}: unexpected scheme")
                continue
            check_local(rel, raw, where)

        for src in pg.data_posts:
            check_local(rel, src, "data-posts")
            json_url = urllib.parse.urljoin(f"{base}/{rel}", src)
            status, body = fetch(json_url)
            try:
                posts = json.loads(body.decode("utf-8"))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{rel}: {src} is not valid JSON ({e})")
                continue
            for post in posts:
                url = post.get("url", "")
                if not (ROOT / url).is_file():
                    errors.append(f"{src}: post url {url!r} does not exist")
                for field in ("title", "excerpt"):
                    for ch in FORBIDDEN_TEXT:
                        if ch in str(post.get(field, "")):
                            errors.append(f"{src}: {field} of {post.get('id')} contains {ch!r}")

    # Checks on the redesigned pages
    for rel in REDESIGNED:
        pg = parsed.get(rel)
        if pg is None:
            errors.append(f"{rel}: missing")
            continue
        for e in pg.balance_errors + pg.model_errors:
            errors.append(f"{rel}: {e}")
        if not pg.doctype:
            errors.append(f"{rel}: no <!DOCTYPE html>")
        head_bytes = (ROOT / rel).read_bytes()[:1024].lower()
        if b'<meta charset="utf-8">' not in head_bytes:
            errors.append(f"{rel}: <meta charset=\"utf-8\"> is not in the first 1024 bytes")
        for t, l in pg.stack:
            if t not in {"html", "body", "head"}:
                errors.append(f"{rel}: <{t}> opened at line {l} is never closed")
        dup = sorted({i for i in pg.ids if pg.ids.count(i) > 1})
        if dup:
            errors.append(f"{rel}: duplicate ids {dup}")
        for kind, ref, line in pg.aria_refs:
            if ref not in pg.ids:
                errors.append(f"{rel}: line {line}: {kind} refers to missing id {ref!r}")
        for img in pg.imgs:
            if "alt" not in img:
                errors.append(f"{rel}: line {img['_line']}: <img> without alt")
            if "width" not in img or "height" not in img:
                warnings.append(f"{rel}: line {img['_line']}: <img> without width and height")
        for a, line in pg.blank_targets:
            if "noopener" not in (a.get("rel") or ""):
                errors.append(f"{rel}: line {line}: target=_blank without rel=noopener")
        if not pg.lang:
            errors.append(f"{rel}: <html> has no lang")
        if not pg.has_title:
            errors.append(f"{rel}: no <title>")
        for key in ("description", "viewport"):
            if key not in pg.meta:
                errors.append(f"{rel}: no meta {key}")
        if not any(l.get("rel") == "canonical" for l in pg.links):
            errors.append(f"{rel}: no canonical link")
        h1 = [h for h in pg.headings if h[0] == 1]
        if len(h1) != 1:
            errors.append(f"{rel}: {len(h1)} h1 elements")
        prev = 0
        for level, line in pg.headings:
            if prev and level > prev + 1:
                errors.append(f"{rel}: line {line}: heading jumps from h{prev} to h{level}")
            prev = level
        for block in pg.jsonld:
            try:
                data = json.loads(block)
            except json.JSONDecodeError as e:
                errors.append(f"{rel}: JSON-LD does not parse ({e})")
                continue
            if data.get("@type") == "Person":
                for u in data.get("sameAs", []):
                    problem = well_formed_external(u)
                    if problem:
                        errors.append(f"{rel}: JSON-LD sameAs {u}: {problem}")
        for text, line in pg.text_chunks:
            for ch in FORBIDDEN_TEXT:
                if ch in text:
                    errors.append(f"{rel}: line {line}: visible text contains {ch!r}: {text.strip()[:80]!r}")
        for tag, text, line in pg.attr_text:
            for ch in FORBIDDEN_TEXT:
                if ch in text:
                    errors.append(f"{rel}: line {line}: <{tag}> {text[:80]!r} contains {ch!r}")
        raw_html = (ROOT / rel).read_text(encoding="utf-8")
        for n, topic in excluded:
            if topic.lower() in raw_html.lower():
                errors.append(f"{rel}: contains the excluded phrase on line {n} of tools/excluded-topics.txt")

    # Budget
    total = sum((ROOT / f).stat().st_size for f in BUDGET_FILES)
    sizes = ", ".join(f"{f} {(ROOT / f).stat().st_size / 1024:.1f} KB" for f in BUDGET_FILES)
    if total > BUDGET_BYTES:
        errors.append(f"budget: {total / 1024:.1f} KB exceeds 300 KB ({sizes})")

    # The CV is published as it is: warn if it lists an excluded item
    for pdf in sorted((ROOT / "assets").rglob("*.pdf")):
        try:
            import pymupdf  # optional
            text = "".join(page.get_text() for page in pymupdf.open(pdf))
        except Exception:  # noqa: BLE001
            try:
                from pypdf import PdfReader  # optional
                text = "".join((p.extract_text() or "") for p in PdfReader(str(pdf)).pages)
            except Exception:  # noqa: BLE001
                warnings.append(f"{pdf.relative_to(ROOT).as_posix()}: no PDF reader installed, contents not checked")
                continue
        flat = " ".join(text.split()).lower()
        hits = [n for n, t in excluded if t.lower() in flat]
        if hits:
            warnings.append(f"{pdf.relative_to(ROOT).as_posix()}: the PDF contains excluded phrases "
                            f"(lines {hits} of tools/excluded-topics.txt)")

    if args.external:
        for u in sorted(external):
            status, _ = fetch(u, "HEAD", timeout=20)
            if status in (405, 403, None):
                status, _ = fetch(u, "GET", timeout=20)
            if status is None or status >= 400:
                warnings.append(f"external {u} returned {status}")

    httpd.shutdown()

    print(f"Pages fetched: {len(parsed)} ({', '.join(parsed)})")
    print(f"Local URLs checked: {len(local_checked)}, external links well formed: {len(external)}")
    print(f"Budget: {total / 1024:.1f} KB of 300 KB ({sizes})")
    for w in warnings:
        print(f"WARNING  {w}")
    for e in errors:
        print(f"ERROR    {e}")
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
