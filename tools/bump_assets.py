"""Stamp the site's CSS and JS links with a hash of each file's content.

Browsers keep site.css and site.js for a while (GitHub Pages sends max-age=600, and some keep them longer),
so a deploy can show new HTML with an old stylesheet. Each link gets ?v=<first 8 hex of the file's
sha256>: an unchanged file keeps its URL and its cache, a changed file gets a new URL.

    python tools/bump_assets.py          # rewrite the links in every HTML page
    python tools/bump_assets.py --check  # exit 1 if any link is stale (used by check_site.py)
"""
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ["assets/css/site.css", "assets/js/site.js", "assets/js/announcements.js"]
SKIP_DIRS = {"tools", "node_modules", ".git"}


def versions():
    # hash with LF line endings, as committed and served, so a Windows checkout gives the same tag
    return {a: hashlib.sha256((ROOT / a).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:8]
            for a in ASSETS}


def pages():
    for p in ROOT.rglob("*.html"):
        if not SKIP_DIRS.intersection(p.relative_to(ROOT).parts):
            yield p


def stamp(text, vers):
    for asset, v in vers.items():
        text = re.sub(re.escape(asset) + r"(\?v=[0-9a-f]*)?(?=\")", f"{asset}?v={v}", text)
    return text


def main():
    check = "--check" in sys.argv
    vers = versions()
    stale = []
    for p in pages():
        raw = p.read_bytes()
        text = raw.decode("utf-8")
        new = stamp(text, vers)
        if new != text:
            stale.append(p.relative_to(ROOT).as_posix())
            if not check:
                p.write_bytes(new.encode("utf-8"))
    if check:
        if stale:
            print("stale asset versions in: " + ", ".join(stale) + " (run python tools/bump_assets.py)")
            return 1
        return 0
    print("versions:", vers)
    print("updated:", ", ".join(stale) if stale else "nothing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
