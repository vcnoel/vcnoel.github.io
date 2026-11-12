#!/usr/bin/env python3
"""Publish a draft from posts/drafts/, run from the repository root:

    python tools/publish_draft.py peer-notes --date 2026-12-01

It does the whole change in one step:

  1. moves posts/drafts/SLUG.html to posts/SLUG.html (with git mv when the
     file is tracked), which needs no other edit because drafts use
     root-relative paths and already name their final address as canonical;
  2. deletes the robots noindex line and the draft banner;
  3. writes the date into the <time> element;
  4. adds the post to posts/posts.json, newest first, with the page's title,
     its meta description as the excerpt (or --excerpt) and its tags.

Then run python tools/check_site.py, which fails if a published page still
carries noindex or if a draft is listed.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug", help="file name of the draft without .html, for example peer-notes")
    ap.add_argument("--date", required=True, help="publication date, YYYY-MM-DD")
    ap.add_argument("--excerpt", help="one sentence for the listing (default: the meta description)")
    args = ap.parse_args()

    date = datetime.date.fromisoformat(args.date)
    src = ROOT / "posts" / "drafts" / f"{args.slug}.html"
    dst = ROOT / "posts" / f"{args.slug}.html"
    if not src.is_file():
        print(f"no draft at {src.relative_to(ROOT)}")
        return 2
    if dst.exists():
        print(f"{dst.relative_to(ROOT)} already exists")
        return 2

    s = src.read_text(encoding="utf-8")
    s = re.sub(r'[ \t]*<meta name="robots" content="noindex">\n', "", s)
    s = re.sub(r'[ \t]*<p class="draft-banner">.*?</p>\n', "", s, flags=re.S)
    shown = f"{date.day} {date.strftime('%b')} {date.year}"
    s, n = re.subn(r'<time datetime="[^"]*">\[publication date\]</time>',
                   f'<time datetime="{date.isoformat()}">{shown}</time>', s)
    if n != 1:
        print("warning: the [publication date] placeholder was not found, the date is unchanged")
    if 'name="robots"' in s:
        print("warning: a robots meta tag is still in the page")

    title = html.unescape(re.search(r"<h1>(.*?)</h1>", s, re.S).group(1).strip())
    excerpt = args.excerpt or html.unescape(re.search(r'<meta name="description" content="([^"]*)"', s).group(1))
    meta = s[s.index('class="article-meta"'):] if 'class="article-meta"' in s else ""
    tags = [html.unescape(t) for t in re.findall(r'<li class="tag">([^<]+)</li>', meta[:meta.find("</div>")])]

    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(src)], cwd=ROOT,
                             capture_output=True).returncode == 0
    if tracked:
        subprocess.run(["git", "mv", str(src), str(dst)], cwd=ROOT, check=True)
    else:
        shutil.move(str(src), str(dst))
    dst.write_text(s, encoding="utf-8", newline="\n")

    listing = ROOT / "posts" / "posts.json"
    posts = json.loads(listing.read_text(encoding="utf-8"))
    posts = [p for p in posts if p.get("id") != args.slug]
    posts.append({"id": args.slug, "title": title, "date": date.isoformat(), "type": "original",
                  "excerpt": excerpt, "url": f"posts/{args.slug}.html", "tags": tags})
    posts.sort(key=lambda p: p.get("date", ""), reverse=True)
    listing.write_text(json.dumps(posts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"published {dst.relative_to(ROOT).as_posix()} on {date.isoformat()}, listed in posts/posts.json")
    print("next: python tools/check_site.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
