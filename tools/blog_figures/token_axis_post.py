#!/usr/bin/env python3
"""Figures for the post "Steering along the token axis" (spectral-trust 0.5.0).

    python tools/blog_figures/token_axis_post.py SOURCE_REPO [--out DIR] [--only NAME ...]

SOURCE_REPO is a clone of https://github.com/vcnoel/spectral-trust at v0.5.0. Every number in
the figures is read from a file, nothing is typed in by hand:

  token-axis        tools/blog_figures/token_axis_signal.json, written by token_axis_signal.py
                    (GPT-2 small, one IOI sentence, the attention graph of layer 5 and one
                    column of the residual stream after block 5, low-pass and high-pass)
  sink-reversal     SOURCE_REPO/examples/use_cases/outputs/sink_ablation.json
                    (sets.results["names/keep_sink" | "names/plain" | "names/zero"]
                    .name_movers.effect, name_mover_rows, clean_ld)
  steering-tradeoff SOURCE_REPO/examples/use_cases/outputs/graph_filter_steering.json
                    (cells at layer 5 low-pass and layer 11 high-pass: dld and dloss of the
                    attention, path, shuffled and mean conditions)

Each figure is written twice into posts/figures/steering-along-the-token-axis/: NAME.png at
1440 px wide and NAME-720.png at 720 px, the same drawing at half the resolution, for srcset.
The figures are drawn on white, as the site shows every post figure on a white card in both
themes. The script prints the numbers the captions and the text use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
SITE = HERE.parents[1]
SLUG = "steering-along-the-token-axis"
SIGNAL = HERE / "token_axis_signal.json"

W_IN = 7.2                       # 1440 px at 200 dpi, 720 px at 100 dpi
DPIS = {"": 200, "-720": 100}

WHITE, INK, MUTED, LIGHT = "#FFFFFF", "#16140F", "#635C52", "#CFC6B7"
ORANGE, BLUE, GREEN, GREY = "#E69F00", "#0072B2", "#009E73", "#8F887D"
MONO = "DejaVu Sans Mono"

# Point sizes. At 200 dpi one point is 2.78 px, so 11 pt is 31 px in the 1440 px frame.
SMALL, BASE, BIG = 11, 12.5, 13.5

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": BASE,
    "axes.unicode_minus": True,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "axes.labelsize": BASE,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
})


def load(path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"result file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def signed(v: float, nd: int = 2) -> str:
    s = f"{v:+.{nd}f}"
    return s.replace("-", "−")


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in DPIS.items():
        path = out / f"{name}{suffix}.png"
        fig.savefig(path, dpi=dpi)
        print(f"  wrote {path.relative_to(SITE).as_posix()}")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 1. A graph frequency along the token axis
# ----------------------------------------------------------------------------

def token_axis(_repo: Path, out: Path) -> None:
    d = load(SIGNAL)
    toks = [t.strip() for t in d["tokens"]]
    s0 = d["span_start"]
    W = np.array(d["adjacency"], dtype=float)
    sig, lo, hi = (np.array(d[k], dtype=float) for k in ("signal", "lowpass", "highpass"))
    n = len(sig)
    assert W.shape == (n, n) and len(toks) == n + s0
    assert np.max(np.abs(lo + hi - sig)) < 1e-3
    print(f"token-axis: {d['model']}, layer {d['layer']}, cutoff {d['cutoff']}, "
          f"{n} tokens in the span, spectral-trust {d['spectral_trust']}")

    fig = plt.figure(figsize=(W_IN, W_IN * 810 / 1440))
    left, right = 0.235, 0.985
    xs = np.arange(-s0, n)                    # token 0 sits at x = -1, outside the span

    # the attention graph as arcs over the tokens
    g = fig.add_axes([left, 0.555, right - left, 0.425])
    g.set_xlim(-1.6, n - 0.4)
    g.set_ylim(-0.75, 3.05)
    g.axis("off")
    off = W.copy()
    np.fill_diagonal(off, 0.0)
    wmax = off.max()
    pairs = sorted(((off[i, j], i, j) for i in range(n) for j in range(i + 1, n)), key=lambda t: t[0])
    shown = 0
    for w, i, j in pairs:
        r = w / wmax
        if r < 0.15:                          # the weakest edges are left out of the drawing only
            continue
        shown += 1
        span = j - i
        h = 0.3 + 0.22 * span
        g.add_patch(Arc(((i + j) / 2, 0.0), span, 2 * h, theta1=0, theta2=180,
                        lw=0.5 + 4.5 * r, color=BLUE, alpha=0.22 + 0.7 * r, zorder=1))
    print(f"  arcs drawn: {shown} of {n * (n - 1) // 2} pairs (weight at least 0.15 of the largest)")
    for x, t in zip(xs, toks):
        inside = x >= 0
        g.text(x - (0.25 if not inside else 0), -0.4, t, ha="center", va="center", fontsize=SMALL - 1.5,
               color=INK if inside else LIGHT, zorder=3)
        if inside:
            g.plot([x], [0.0], "o", ms=5.5, color=BLUE, mec=WHITE, mew=1.0, zorder=2)
    fig.text(0.03, 0.80, f"attention graph\nof layer {d['layer']}", ha="left", va="center", fontsize=BASE,
             color=INK, linespacing=1.35)
    fig.text(0.03, 0.665, "thicker arcs: more\nattention between\nthe two tokens", ha="left",
             va="center", fontsize=SMALL - 1, color=MUTED, linespacing=1.3)

    # one residual signal, its low-pass part and its high-pass part, on one scale
    rows = [("residual signal", f"one column, after block {d['layer']}", sig, GREY),
            ("= low-pass", "smooth on the graph", lo, BLUE),
            ("+ high-pass", "what sets a token apart", hi, ORANGE)]
    lim = 1.08 * max(np.abs(a).max() for _, _, a, _ in rows)
    top, h, gap = 0.505, 0.135, 0.028
    for k, (name, sub, a, colour) in enumerate(rows):
        y0 = top - (k + 1) * h - k * gap
        ax = fig.add_axes([left, y0, right - left, h])
        ax.set_xlim(-1.6, n - 0.4)
        ax.set_ylim(-lim, lim)
        ax.axis("off")
        ax.axhline(0, color=LIGHT, lw=1.0, zorder=1)
        ax.bar(np.arange(n), a, width=0.62, color=colour, zorder=2)
        fig.text(0.03, y0 + h * 0.62, name, ha="left", va="center", fontsize=BASE, color=INK)
        fig.text(0.03, y0 + h * 0.22, sub, ha="left", va="center", fontsize=SMALL - 1, color=MUTED)
    fig.text(left + (right - left) * (0.6 / (n + 1.2)) - 0.005, 0.035,
             "“When”, the attention sink, is left out of the graph and of the edit",
             ha="left", va="center", fontsize=SMALL - 1.5, color=MUTED)
    save(fig, out, "token-axis")


# ----------------------------------------------------------------------------
# 2. Three ways to remove the same edges
# ----------------------------------------------------------------------------

def sink_reversal(repo: Path, out: Path) -> None:
    d = load(repo / "examples" / "use_cases" / "outputs" / "sink_ablation.json")
    res = d["sets"]["results"]
    keys = ["keep_sink", "plain", "zero"]
    eff = {k: res[f"names/{k}"]["name_movers"]["effect"] for k in keys}
    rows = d["name_mover_rows"]
    clean_ld = d["clean_ld"][0]
    print(f"sink-reversal: clean LD {clean_ld:.2f}, n = {d['n_prompts']}")
    for k in keys:
        e = eff[k]
        print(f"  {k:9s} dLD {signed(e[0])} [{signed(e[1])}, {signed(e[2])}]  row {np.round(rows[k], 2).tolist()}")
    print(f"  clean row {np.round(rows['clean'], 2).tolist()}  (names, first token, other tokens)")

    labels = [("before the edit", None), ("keep_sink=True", "(default)"),
              ("keep_sink=False", None), ("renormalize=False", None)]
    comp = [rows["clean"], rows["keep_sink"], rows["plain"], rows["zero"]]
    parts = [("the three names", ORANGE), ("first token (sink)", GREY), ("other tokens", BLUE)]
    bar_colour = {"keep_sink": BLUE, "plain": GREY, "zero": LIGHT}

    fig = plt.figure(figsize=(W_IN, W_IN * 700 / 1440))
    ys = np.arange(4)[::-1].astype(float)
    a = fig.add_axes([0.285, 0.15, 0.335, 0.64])
    b = fig.add_axes([0.665, 0.15, 0.31, 0.64])

    for y, (name, note) in zip(ys, labels):
        is_code = name != "before the edit"
        a.text(-0.045, y + (0.13 if note else 0), name, transform=a.get_yaxis_transform(), ha="right",
               va="center", fontsize=SMALL, color=INK, family=MONO if is_code else None)
        if note:
            a.text(-0.045, y - 0.24, note, transform=a.get_yaxis_transform(), ha="right", va="center",
                   fontsize=SMALL - 1, color=MUTED)

    # panel a: where the attention of the final position goes
    for y, c in zip(ys, comp):
        x = 0.0
        for v, (_, colour) in zip(c, parts):
            if v > 1e-6:
                a.barh(y, v, left=x, height=0.58, color=colour, edgecolor=WHITE, lw=1.0)
            x += v
        if x < 0.999:
            a.barh(y, 1 - x, left=x, height=0.58, color="none", edgecolor=LIGHT, lw=1.0, ls=(0, (3, 2)))
            a.text(x + (1 - x) / 2, y, f"{1 - x:.2f} missing", ha="center", va="center",
                   fontsize=SMALL - 1, color=MUTED)
    a.set_xlim(0, 1)
    a.set_ylim(-0.6, 3.6)
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.set_xticks([0, 0.5, 1])
    a.set_xticklabels(["0", "0.5", "1"])
    a.set_xlabel("share of the row", labelpad=4)
    a.set_title("attention of the final position", loc="left", fontsize=BASE, color=INK, pad=8)
    lx = 0.285
    for name, colour in parts:
        fig.patches.append(Rectangle((lx, 0.905), 0.016, 0.032, transform=fig.transFigure,
                                     color=colour, figure=fig))
        t = fig.text(lx + 0.022, 0.921, name, ha="left", va="center", fontsize=SMALL - 1, color=INK)
        fig.canvas.draw()
        lx = t.get_window_extent().x1 / fig.bbox.width + 0.03

    # panel b: change in logit difference, with 95% intervals over prompts
    for y, k in zip(ys[1:], keys):
        e = eff[k]
        b.barh(y, e[0], height=0.58, color=bar_colour[k], edgecolor=MUTED if k == "zero" else bar_colour[k],
               lw=1.0)
        b.plot([e[1], e[2]], [y, y], color=INK, lw=1.6, solid_capstyle="butt")
        for xe in (e[1], e[2]):
            b.plot([xe, xe], [y - 0.1, y + 0.1], color=INK, lw=1.6)
        tx = min(e[1], 0) - 0.04 if e[0] < 0 else max(e[2], 0) + 0.04
        b.text(tx, y, signed(e[0]), ha="right" if e[0] < 0 else "left", va="center",
               fontsize=BASE, color=INK, fontweight="bold")
    b.axvline(0, color=MUTED, lw=1.2)
    b.set_xlim(-1.25, 0.62)
    b.set_ylim(-0.6, 3.6)
    b.set_yticks([])
    b.spines["left"].set_visible(False)
    b.set_xticks([-1.0, -0.5, 0.0])
    b.set_xlabel("change in logit difference", labelpad=4)
    b.set_title(f"logit difference, clean {clean_ld:.2f}", loc="left", fontsize=BASE, color=INK, pad=8)
    b.text(-1.2, ys[0], "same edges,\nthree answers", ha="left", va="center", fontsize=SMALL,
           color=MUTED, linespacing=1.3)
    save(fig, out, "sink-reversal")


# ----------------------------------------------------------------------------
# 3. Effect against side effect, against controls of the same size
# ----------------------------------------------------------------------------

def steering_tradeoff(repo: Path, out: Path) -> None:
    d = load(repo / "examples" / "use_cases" / "outputs" / "graph_filter_steering.json")
    cells = {(c["layer"], c["kind"], c["strength"]): c for c in d["cells"]}
    print(f"steering-tradeoff: {d['n_live_cells']} live cells, differs from both controls on LD in "
          f"{len(d['beyond_both_ld'])} (above both {len(d['above_both_ld'])}, below both "
          f"{len(d['below_both_ld'])}), on loss in {len(d['beyond_both_loss'])} (above both "
          f"{len(d['above_both_loss'])}, below both {len(d['below_both_loss'])})")

    series = [("attention", "attention graph", ORANGE, "o"), ("shuffled", "shuffled graph", GREY, "D"),
              ("path", "path graph", BLUE, "s"), ("mean", "prefix mean", GREEN, "^")]
    panels = [(5, "lowpass", [0.25, 0.5, 1.0], {"mean": [0.25, 0.5]},
               "after block 5, low-pass"),
              (11, "highpass", [0.25, 0.5], {}, "after block 11, high-pass")]

    fig = plt.figure(figsize=(W_IN, W_IN * 760 / 1440))
    axes = [fig.add_axes([0.105, 0.155, 0.37, 0.70]), fig.add_axes([0.6, 0.155, 0.37, 0.70])]
    for ax, (layer, kind, strengths, only, title) in zip(axes, panels):
        print(f"  layer {layer} {kind}")
        ends = {}
        for cond, name, colour, mark in series:
            ss = only.get(cond, strengths)
            pts = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
            for s in ss:
                c = cells[(layer, kind, s)].get(cond)
                if c is None:
                    break
                pts.append((*c["dloss"], *c["dld"]))
                print(f"    {cond:9s} s={s:<4} size {c['rel_size']:.3f}  dLD {signed(c['dld'][0])} "
                      f"[{signed(c['dld'][1])}, {signed(c['dld'][2])}]  dloss {c['dloss'][0]:+.3f} "
                      f"[{c['dloss'][1]:+.3f}, {c['dloss'][2]:+.3f}]")
            if len(pts) == 1:
                continue
            p = np.array(pts)
            ax.plot(p[:, 0], p[:, 3], color=colour, lw=2.0, zorder=3)
            ax.errorbar(p[1:, 0], p[1:, 3], xerr=[p[1:, 0] - p[1:, 1], p[1:, 2] - p[1:, 0]],
                        yerr=[p[1:, 3] - p[1:, 4], p[1:, 5] - p[1:, 3]], fmt="none", ecolor=colour,
                        elinewidth=1.0, alpha=0.55, zorder=2)
            ax.plot(p[1:, 0], p[1:, 3], mark, color=colour, ms=6.5, mec=WHITE, mew=1.0, zorder=4)
            ends[cond] = (p[-1, 0], p[-1, 3], name, colour)
        ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
        ax.axvline(0, color=MUTED, lw=1.0, zorder=1)
        ax.set_title(title, loc="left", fontsize=BASE, color=INK, pad=10)
        ax.set_xlabel("side effect: Δ loss (nats)", labelpad=5)
        if kind == "lowpass":
            ax.set_ylabel("effect: Δ logit difference", labelpad=6)
            ax.set_xlim(-0.01, 0.34)
            ax.set_ylim(-3.95, 0.25)
            place = {"attention": (-2, -24, "right", "center"), "shuffled": (9, 0, "left", "center"),
                     "path": (9, 0, "left", "center"), "mean": (-10, -4, "right", "center")}
        else:
            ax.set_xlim(-0.02, 0.66)
            ax.set_ylim(-0.08, 1.12)
            place = {"attention": (8, 2, "left", "center"), "shuffled": (8, 0, "left", "center"),
                     "path": (-6, -16, "right", "center")}
        for cond, (x, y, name, colour) in ends.items():
            dx, dy, ha, va = place[cond]
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                        fontsize=SMALL, color=colour if cond != "shuffled" else MUTED, fontweight="bold")
    save(fig, out, "steering-tradeoff")


FIGURES = {"token-axis": token_axis, "sink-reversal": sink_reversal, "steering-tradeoff": steering_tradeoff}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_repo", type=Path, help="a clone of vcnoel/spectral-trust at v0.5.0")
    ap.add_argument("--out", type=Path, default=SITE / "posts" / "figures" / SLUG)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), help="draw only these figures")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # the minus signs print on a Windows console too
    for name in args.only or FIGURES:
        FIGURES[name](args.source_repo.resolve(), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
