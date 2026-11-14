#!/usr/bin/env python3
"""Draw the three animated icons of the What I work on cards.

    python tools/make_card_icons.py            # from the repository root

Each icon is a small SVG on a 60 x 60 grid, written into index.html between
the markers <!-- card-art:NAME:start --> and <!-- card-art:NAME:end -->. Its
styles and keyframes go into assets/css/site.css between /* card-art:start */
and /* card-art:end */.

Every icon tells one short story on a 6 s loop: the finished picture fades out
(0 to 0.4 s), the scene builds, the event happens, and the finished picture
holds for about 2.5 s. The finished picture is also the static style, so it is
what shows with reduced motion, without the script, before the icon first
comes into view and between plays. The page runs the loop only while the icon
is in view (assets/js/site.js adds is-anim and is-live).

  * rsn, reasoning: attention edges draw and light up, one step of the chain
    flares orange, and the graph splits into a blue and an orange group, the
    way the Fiedler vector splits the attention graph of an invalid proof.
  * agt, agents: a tool call travels from the agent, attention edges converge
    on it, one flares orange, a shield closes over the call and the tool dims.
  * lgl, legal: each claim links to its source and turns blue with a check,
    and the claim with no source is flagged orange.

Colours come from the page's tokens, so the icons follow the theme.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "index.html"
CSS = ROOT / "assets" / "css" / "site.css"

LOOP = 6.0           # seconds
FADE_OUT = 0.35      # the finished picture fades out by here
BLANK = 0.40         # every element jumps to its start state while the icon is blank
HOLD_FROM = 3.6      # from here to the end of the loop the finished picture holds (site.js reads this)
STAGGER = {"rsn": 0.0, "agt": 0.5, "lgl": 1.0}  # the three icons start half a second apart

OUT = "cubic-bezier(.2,.7,.2,1)"
INOUT = "cubic-bezier(.45,0,.25,1)"
DEFAULTS = {"opacity": "1", "stroke-dashoffset": "0", "fill-opacity": "1"}

BLUE, ORANGE, MUTED, SURFACE = "var(--blue)", "var(--orange)", "var(--muted)", "var(--surface)"


class Track:
    """The values of a few properties of one element over the loop.

    start: the values just after the reset. steps: (t0, t1, values, easing), in order,
    each moving the element from its current values to new ones between t0 and t1.
    The values after the last step are the finished picture."""

    def __init__(self, start: dict, steps: list):
        self.start = start
        self.steps = steps

    def final(self) -> dict:
        v = dict(self.start)
        for _, _, p, _ in self.steps:
            v.update(p)
        return v

    def frames(self):
        final = self.final()
        frames = [[0.0, dict(final), None], [BLANK, dict(final), None], [BLANK + 0.01, dict(self.start), None]]
        cur = dict(self.start)
        for t0, t1, props, ease in self.steps:
            assert t0 >= frames[-1][0] - 1e-9 and t1 > t0, (t0, t1, frames[-1][0])
            if abs(frames[-1][0] - t0) < 1e-9:
                frames[-1][2] = ease
            else:
                frames.append([t0, dict(cur), ease])
            cur.update(props)
            frames.append([t1, dict(cur), None])
        if frames[-1][0] < LOOP:
            frames.append([LOOP, dict(final), None])
        keys = list(final)
        for f in frames:  # every frame carries every property, so nothing interpolates across a gap
            for k in keys:
                f[1].setdefault(k, final[k])
        return frames


def pct(t: float) -> str:
    s = f"{t / LOOP * 100:.2f}".rstrip("0").rstrip(".")
    return s + "%"


def decl(values: dict) -> str:
    return ";".join(f"{k}:{v}" for k, v in values.items())


# ---- The icons. Each element: (tag, class, attributes, track or None). Order is paint order. ----

def node(cls, x, y, r=4.6):
    return ("circle", cls, f'cx="{x}" cy="{y}" r="{r}"')


def rsn():
    L = [(15, 15), (10, 30), (15, 45)]
    R = [(45, 15), (50, 30), (45, 45)]
    els, tracks = [], {}

    def line(cls, a, b):
        els.append(("line", cls, f'pathLength="1" x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}"'))

    def edge_track(k, end):
        t = 0.95 + 0.075 * k
        light = 1.7 + 0.04 * k
        return Track(
            {"opacity": "0", "stroke-dashoffset": "1", "stroke": MUTED, "stroke-width": "1.4"},
            [(t, t + 0.35, {"opacity": ".55", "stroke-dashoffset": "0"}, OUT),
             (light, light + 0.3, {"stroke": BLUE, "opacity": ".9", "stroke-width": "1.8"}, None),
             (2.6, 3.0, end, INOUT)])

    order = {"rs-a1": 0, "rs-x1": 1, "rs-b1": 2, "rs-a2": 3, "rs-x2": 4, "rs-b2": 5, "rs-x3": 6, "rs-x4": 7}
    cut = {"stroke": MUTED, "opacity": ".35", "stroke-width": "1.3"}
    els.append(("g", "rs-cross", None))
    for cls, a, b in [("rs-x1", L[0], R[0]), ("rs-x2", L[1], R[1]), ("rs-x3", L[2], R[2]), ("rs-x4", L[1], R[0])]:
        line(cls, a, b)
        tracks[cls] = edge_track(order[cls], cut)
    els.append(("/g", None, None))
    tracks["rs-cross"] = Track({"transform": "scaleX(.83)"}, [(2.6, 3.0, {"transform": "scaleX(1)"}, INOUT)])

    def pop(i, colour, at):
        t = 0.45 + 0.08 * i
        return Track({"opacity": "0", "transform": "scale(.3)", "fill": SURFACE},
                     [(t, t + 0.3, {"opacity": "1", "transform": "scale(1)"}, OUT),
                      (at, at + 0.3, {"fill": colour}, None)])

    els.append(("g", "rs-left", None))
    for cls, a, b in [("rs-a1", L[0], L[1]), ("rs-a2", L[1], L[2])]:
        line(cls, a, b)
        tracks[cls] = edge_track(order[cls], {"stroke": BLUE})
    for i, (x, y) in enumerate(L):
        els.append(node(f"rs-n{i + 1}", x, y))
        tracks[f"rs-n{i + 1}"] = pop(i, BLUE, 2.65 + 0.05 * i)
    els.append(("/g", None, None))
    tracks["rs-left"] = Track({"transform": "translateX(3px)"}, [(2.6, 3.0, {"transform": "translateX(0)"}, INOUT)])

    els.append(("g", "rs-right", None))
    for cls, a, b in [("rs-b1", R[0], R[1]), ("rs-b2", R[1], R[2])]:
        line(cls, a, b)
        tracks[cls] = edge_track(order[cls], {"stroke": ORANGE})
    els.append(("circle", "rs-halo", f'cx="{R[1][0]}" cy="{R[1][1]}" r="8.6"'))
    tracks["rs-halo"] = Track({"opacity": "0", "transform": "scale(.5)"},
                              [(2.95, 3.3, {"opacity": ".95", "transform": "scale(1)"}, OUT)])
    els.append(("circle", "rs-flare", f'cx="{R[1][0]}" cy="{R[1][1]}" r="4.6"'))
    tracks["rs-flare"] = Track({"opacity": "0", "transform": "scale(1)"},
                               [(2.2, 2.26, {"opacity": ".8"}, None),
                                (2.26, 2.9, {"opacity": "0", "transform": "scale(2.7)"}, OUT)])
    for i, (x, y) in enumerate(R):
        els.append(node(f"rs-n{i + 4}", x, y))
        tracks[f"rs-n{i + 4}"] = pop(i + 3, ORANGE, 2.2 if i == 1 else 2.75 + 0.05 * i)
    els.append(("/g", None, None))
    tracks["rs-right"] = Track({"transform": "translateX(-3px)"}, [(2.6, 3.0, {"transform": "translateX(0)"}, INOUT)])

    css = """
.ca-rsn line { stroke-linecap: round; }
.ca-rsn circle { stroke: var(--ink-2); stroke-width: 1.2; transform-box: fill-box; transform-origin: center; }
.ca-rsn .rs-cross { transform-origin: 30px 30px; }
.ca-rsn .rs-halo { fill: none; stroke: var(--orange); stroke-width: 1.6; }
.ca-rsn .rs-flare { fill: var(--orange); stroke: none; }
"""
    return els, tracks, css


def agt():
    els, tracks = [], {}
    els.append(("line", "ag-track", 'x1="16" y1="40" x2="44" y2="40"'))
    tracks["ag-track"] = Track({"opacity": "0"}, [(0.45, 0.8, {"opacity": ".7"}, None)])
    ctx = [(12, 11), (30, 8), (48, 11)]
    ends = [(27.6, 36.4), (29.5, 36), (31.4, 36.4)]
    for k, ((x, y), (ex, ey)) in enumerate(zip(ctx, ends)):
        cls = f"ag-e{k + 1}"
        els.append(("line", cls, f'pathLength="1" x1="{x}" y1="{y}" x2="{ex}" y2="{ey}"'))
        t = 1.7 + 0.15 * k
        steps = [(t, t + 0.4, {"opacity": ".85", "stroke-dashoffset": "0", "stroke": BLUE}, OUT)]
        if k == 2:
            steps += [(2.55, 2.7, {"stroke": ORANGE, "stroke-width": "3.2", "opacity": "1"}, None),
                      (2.7, 3.05, {"stroke-width": "2"}, OUT)]
        tracks[cls] = Track({"opacity": "0", "stroke-dashoffset": "1", "stroke": MUTED, "stroke-width": "1.5"}, steps)
    for k, (x, y) in enumerate(ctx):
        cls = f"ag-c{k + 1}"
        els.append(("circle", cls, f'cx="{x}" cy="{y}" r="3.4"'))
        t = 0.5 + 0.08 * k
        steps = [(t, t + 0.3, {"opacity": "1", "transform": "scale(1)"}, OUT)]
        if k == 2:
            steps.append((2.55, 2.7, {"fill": ORANGE}, None))
        tracks[cls] = Track({"opacity": "0", "transform": "scale(.3)", "fill": SURFACE}, steps)
    els.append(("circle", "ag-agent", 'cx="9.5" cy="40" r="5.6"'))
    tracks["ag-agent"] = Track({"opacity": "0", "transform": "scale(.3)"},
                               [(0.45, 0.75, {"opacity": "1", "transform": "scale(1)"}, OUT)])
    els.append(("g", "ag-tool", None))
    els.append(("rect", None, 'x="45" y="33" width="12" height="14" rx="3"'))
    els.append(("path", None, 'd="M48.6 37.6l2.6 2.4-2.6 2.4M52.6 43.2h1.8"'))
    els.append(("/g", None, None))
    tracks["ag-tool"] = Track({"opacity": "0"}, [(0.55, 0.9, {"opacity": "1"}, None),
                                                 (3.25, 3.6, {"opacity": ".32"}, None)])
    els.append(("rect", "ag-call", 'x="25" y="37" width="9" height="6" rx="2"'))
    tracks["ag-call"] = Track({"opacity": "0", "transform": "translateX(-15px)"},
                              [(0.85, 0.95, {"opacity": "1"}, None),
                               (0.95, 1.7, {"transform": "translateX(0)"}, INOUT)])
    els.append(("path", "ag-shield", 'pathLength="1" d="M29.5 29l9 3.2v7.3c0 5.7-3.7 9.8-9 12-5.3-2.2-9-6.3-9-12v-7.3z"'))
    tracks["ag-shield"] = Track({"opacity": "0", "stroke-dashoffset": "1", "fill-opacity": "0", "transform": "scale(1.3)"},
                                [(2.85, 3.35, {"opacity": "1", "stroke-dashoffset": "0", "transform": "scale(1)"}, OUT),
                                 (3.35, 3.6, {"fill-opacity": ".18"}, None)])
    css = """
.ca-agt line, .ca-agt path { fill: none; stroke-linecap: round; stroke-linejoin: round; }
.ca-agt .ag-track { stroke: var(--muted); stroke-width: 1.4; stroke-dasharray: 1.5 3; }
.ca-agt circle { stroke: var(--ink-2); stroke-width: 1.2; transform-box: fill-box; transform-origin: center; }
.ca-agt .ag-agent { fill: var(--blue); }
.ca-agt .ag-tool rect { fill: var(--surface); stroke: var(--ink-2); stroke-width: 1.5; }
.ca-agt .ag-tool path { stroke: var(--ink-2); stroke-width: 1.5; }
.ca-agt .ag-call { fill: var(--surface); stroke: var(--blue); stroke-width: 1.6; transform-box: fill-box; }
.ca-agt .ag-shield { fill: var(--blue); stroke: var(--blue); stroke-width: 1.8; transform-box: fill-box; transform-origin: center; }
"""
    return els, tracks, css


def lgl():
    els, tracks = [], {}
    ys = [14, 30, 46]
    links = {"lg-l1": "M30.5 14C35 14 36 13 40.5 13", "lg-l2": "M30.5 30C35.5 30 35.5 35 40.5 35", "lg-l3": "M30.5 46H38"}
    timing = {"lg-l1": 1.45, "lg-l2": 2.15}
    for cls, d in links.items():
        els.append(("path", cls, f'pathLength="1" d="{d}"'))
        if cls == "lg-l3":
            tracks[cls] = Track({"opacity": "0", "stroke-dashoffset": "1"},
                                [(2.85, 3.1, {"opacity": ".8", "stroke-dashoffset": "0"}, OUT),
                                 (3.15, 3.4, {"opacity": "0"}, None)])
        else:
            t = timing[cls]
            tracks[cls] = Track({"opacity": "0", "stroke-dashoffset": "1"},
                                [(t, t + 0.4, {"opacity": "1", "stroke-dashoffset": "0"}, OUT)])
    for j, top in enumerate([4, 26]):
        cls = f"lg-d{j + 1}"
        els.append(("g", cls, None))
        els.append(("path", None, f'd="M41 {top}h9l5 5v13H41z"'))
        els.append(("path", "lg-fold", f'd="M50 {top}v5h5"'))
        els.append(("path", "lg-text", f'd="M44 {top + 9}h8M44 {top + 12.5}h8M44 {top + 16}h5"'))
        els.append(("/g", None, None))
        t = 0.75 + 0.15 * j
        tracks[cls] = Track({"opacity": "0", "transform": "translateY(3px)"},
                            [(t, t + 0.35, {"opacity": "1", "transform": "translateY(0)"}, OUT)])
    linked = {0: 1.85, 1: 2.55}
    for i, y in enumerate(ys):
        b = f"lg-b{i + 1}"
        els.append(("line", b, f'pathLength="1" x1="15" y1="{y}" x2="28" y2="{y}"'))
        t = 0.5 + 0.15 * i
        steps = [(t, t + 0.35, {"opacity": ".7", "stroke-dashoffset": "0"}, OUT)]
        at = linked.get(i, 3.2)
        steps.append((at, at + 0.2, {"stroke": BLUE if i in linked else ORANGE, "opacity": "1"}, None))
        tracks[b] = Track({"opacity": "0", "stroke-dashoffset": "1", "stroke": MUTED}, steps)
    for i, y in enumerate(ys):
        s = f"lg-s{i + 1}"
        els.append(("circle", s, f'cx="7.5" cy="{y}" r="4.8"'))
        t = 0.45 + 0.15 * i
        if i in linked:
            tracks[s] = Track({"opacity": "0", "transform": "scale(.4)", "fill": SURFACE},
                              [(t, t + 0.25, {"opacity": "1", "transform": "scale(1)"}, OUT),
                               (linked[i], linked[i] + 0.15, {"fill": BLUE}, None)])
        else:
            tracks[s] = Track({"opacity": "0", "transform": "translateX(0) scale(.4)", "fill": SURFACE},
                              [(t, t + 0.25, {"opacity": "1", "transform": "translateX(0) scale(1)"}, OUT),
                               (3.2, 3.35, {"fill": ORANGE}, None),
                               (3.35, 3.42, {"transform": "translateX(-1.2px) scale(1)"}, None),
                               (3.42, 3.52, {"transform": "translateX(1.2px) scale(1)"}, None),
                               (3.52, 3.6, {"transform": "translateX(0) scale(1)"}, None)])
    for i in linked:
        k = f"lg-k{i + 1}"
        y = ys[i]
        els.append(("path", k, f'pathLength="1" d="M5.2 {y + 0.2}l1.6 1.7 3.1-3.5"'))
        t = linked[i] + 0.1
        tracks[k] = Track({"opacity": "0", "stroke-dashoffset": "1"},
                          [(t, t + 0.25, {"opacity": "1", "stroke-dashoffset": "0"}, OUT)])
    els.append(("path", "lg-f3", 'd="M7.5 43.6v2.9M7.5 48.7v.01"'))
    tracks["lg-f3"] = Track({"opacity": "0"}, [(3.3, 3.45, {"opacity": "1"}, None)])
    css = """
.ca-lgl line, .ca-lgl path { fill: none; stroke-linecap: round; stroke-linejoin: round; }
.ca-lgl line { stroke-width: 3; }
.ca-lgl .lg-l1, .ca-lgl .lg-l2 { stroke: var(--blue); stroke-width: 1.6; }
.ca-lgl .lg-l3 { stroke: var(--muted); stroke-width: 1.4; }
.ca-lgl .lg-d1 path:first-child, .ca-lgl .lg-d2 path:first-child { fill: var(--surface); stroke: var(--ink-2); stroke-width: 1.4; }
.ca-lgl .lg-fold { stroke: var(--ink-2); stroke-width: 1.4; }
.ca-lgl .lg-text { stroke: var(--muted); stroke-width: 1.3; }
.ca-lgl circle { stroke: var(--ink-2); stroke-width: 1.2; transform-box: fill-box; transform-origin: center; }
.ca-lgl .lg-k1, .ca-lgl .lg-k2 { stroke: var(--surface); stroke-width: 1.7; }
.ca-lgl .lg-f3 { stroke: var(--strip-node-line); stroke-width: 1.7; }
"""
    return els, tracks, css


ICONS = {"rsn": rsn, "agt": agt, "lgl": lgl}


def svg(name, els) -> str:
    out = [f'<svg class="card-art ca-{name}" viewBox="0 0 60 60" width="60" height="60" aria-hidden="true" focusable="false">'
           f'<g class="ca-all">']
    for tag, cls, attrs in els:
        if tag == "/g":
            out.append("</g>")
            continue
        c = f' class="{cls}"' if cls else ""
        a = f" {attrs}" if attrs else ""
        out.append(f"<{tag}{c}{a}>" if tag == "g" else f"<{tag}{c}{a}/>")
    out.append("</g></svg>")
    return "".join(out)


def css_block(parts) -> str:
    static, motion, keyframes, dashed = [], [], [], []
    for name, tracks, base in parts:
        static.append(base.strip())
        for cls, tr in tracks.items():
            fin = {k: v for k, v in tr.final().items() if DEFAULTS.get(k) != v and v not in ("scale(1)", "translateX(0)", "translateY(0)", "scaleX(1)", "translateX(0) scale(1)")}
            if fin:
                static.append(f".ca-{name} .{cls} {{ {decl(fin)}; }}")
            if "stroke-dashoffset" in tr.start:
                dashed.append(f".ca-{name}.is-anim .{cls}")
            kf = f"ca-{name}-{cls.split('-', 1)[1]}"
            motion.append(f".ca-{name}.is-anim .{cls} {{ animation-name: {kf}; }}")
            # A property is written only where its value starts or ends a run: between two written
            # values it then interpolates exactly as if every frame carried it.
            fr = tr.frames()
            frames = []
            for i, (t, vals, ease) in enumerate(fr):
                keep = {k: v for k, v in vals.items()
                        if i in (0, len(fr) - 1) or fr[i - 1][1][k] != v or fr[i + 1][1][k] != v}
                if not keep:
                    continue
                body = decl(keep) + (f";animation-timing-function:{ease}" if ease else "")
                frames.append(f"{pct(t)}{{{body}}}")
            keyframes.append(f"@keyframes {kf} {{ {' '.join(frames)} }}")
    fade = (f"@keyframes ca-fade {{ 0%{{opacity:1;animation-timing-function:ease-in}} {pct(FADE_OUT)}{{opacity:0}} "
            f"{pct(BLANK + 0.02)}{{opacity:0}} {pct(BLANK + 0.03)}{{opacity:1}} 100%{{opacity:1}} }}")
    delays = "\n".join(f"    .ca-{n}.is-anim * {{ animation-delay: {d:g}s; }}" for n, d in STAGGER.items() if d)
    lines = [
        "/* card-art:start (written by tools/make_card_icons.py, edit the script and run it again) */",
        *static,
        "@media (prefers-reduced-motion: no-preference) {",
        f"    .card-art.is-anim * {{ animation-duration: {LOOP:g}s; animation-timing-function: linear; animation-iteration-count: infinite;"
        " animation-fill-mode: both; animation-play-state: paused; }",
        "    .card-art.is-anim.is-live * { animation-play-state: running; }",
        delays,
        "    .card-art.is-anim .ca-all { animation-name: ca-fade; }",
        "    " + ",\n    ".join(dashed) + " { stroke-dasharray: 1; }",
        *("    " + m for m in motion),
        "    " + fade,
        *("    " + k for k in keyframes),
        "}",
        "/* card-art:end */",
    ]
    return "\n".join(lines)


def replace_between(text: str, start: str, end: str, body: str, where: str) -> str:
    if start not in text or end not in text:
        raise SystemExit(f"markers {start} ... {end} not found in {where}")
    a, b = text.index(start), text.index(end) + len(end)
    return text[:a] + body + text[b:]


def main() -> int:
    page = PAGE.read_text(encoding="utf-8")
    parts = []
    for name, make in ICONS.items():
        els, tracks, base = make()
        start, end = f"<!-- card-art:{name}:start -->", f"<!-- card-art:{name}:end -->"
        page = replace_between(page, start, end, f"{start}{svg(name, els)}{end}", "index.html")
        parts.append((name, tracks, base))
    css = CSS.read_text(encoding="utf-8")
    css = replace_between(css, "/* card-art:start", "/* card-art:end */", css_block(parts), "site.css")
    PAGE.write_text(page, encoding="utf-8", newline="\n")
    CSS.write_text(css, encoding="utf-8", newline="\n")
    print(f"wrote {len(parts)} icons; loop {LOOP:g} s, finished picture held from {HOLD_FROM:g} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
