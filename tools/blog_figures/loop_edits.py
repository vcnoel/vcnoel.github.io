#!/usr/bin/env python3
"""Animated figures for the post on loop edits in Qwen3.5-0.8B (posts/loop-edits.html).

    python tools/blog_figures/loop_edits.py SOURCE_REPO [--frames DIR] [--only NAME ...]

SOURCE_REPO is the path to the pilot repository. Every figure except the first is drawn from its result
files at run time (paths relative to SOURCE_REPO/exp8):

  summary    hero figure: accuracy at 8k, the cost gap with Astra, three headline tiles. summary.json
  summary-fr the same in French on the light page colour, for email
  landscape  schematic of the two terms on a potential landscape. No data.
  anatomy    the write of the MLP along the loop direction, in loop and productive windows, before and
             after Astra's edit. astra_anatomy.json
  threshold  escape per loop period against the dose alpha of our escape term, with the double exponential
             through alpha = 3 and 4. deep_loops.json
  outcomes   loop, cut and finished shares and accuracy at an 8,192-token budget. edit_eval.json
  match      accuracy at a 32,768-token budget on 32 fresh problems, with the paired counts. eval32k.json

The frames are written to DIR/NAME (default: a folder in the system temporary directory) with a transparent
background, so the figures sit on the page in light and dark themes. Each folder is turned into
posts/figures/loop-edits/NAME.gif (720 px, one-bit transparency) and NAME.png (the last frame, full alpha),
plus NAME-720.png for the post card. Frames are 1440 px wide at 12 frames per second, and each loop holds its
final state for two seconds.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

SITE = Path(__file__).resolve().parents[2]
OUT = SITE / "posts" / "figures" / "loop-edits"

FPS, DPI, W_IN, H_IN = 12, 200, 7.2, 4.05
HOLD = 2 * FPS
GIF_WIDTH = 1440                     # full frame width: sharp on high-density screens
WEBP_LOSSLESS, WEBP_QUALITY = True, 100
# mid-tone colours, legible on the light (#fbfaf7) and the dark (#0f1115) page
INK, MUTED, GRID = "#8C857B", "#8C857B", "#8C857B"
BASE, ASTRA, OURS, ESC = "#A59D92", "#E69F00", "#3D9BE0", "#2BA884"
LOOP, CUT, FIN = "#E1652B", "#A59D92", "#2BA884"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})


def new_fig():
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    fig.patch.set_alpha(0)
    return fig


def style(ax):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(width=1.2, labelsize=10)


def save(fig, d: Path, i: int):
    fig.savefig(d / f"{i:04d}.png", dpi=DPI, transparent=True)
    plt.close(fig)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ encoding
def to_gif_frame(im: Image.Image) -> Image.Image:
    """RGBA -> palette image with index 255 transparent (alpha below one half)."""
    h = round(im.height * GIF_WIDTH / im.width)
    im = im.convert("RGBa").resize((GIF_WIDTH, h), Image.LANCZOS).convert("RGBA")
    alpha = im.getchannel("A")
    p = im.convert("RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = p.getpalette()[: 255 * 3]
    p.putpalette(pal + [0, 0, 0] * (256 - len(pal) // 3))
    p.paste(255, mask=alpha.point(lambda v: 255 if v < 128 else 0))
    return p


def on_solid(im: Image.Image, bg: str) -> Image.Image:
    base = Image.new("RGBA", im.size, bg)
    return Image.alpha_composite(base, im)


def encode(d: Path, name: str, bg: str | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    frames = sorted(d.glob("*.png"))
    last = Image.open(frames[-1]).convert("RGBA")
    if bg:
        last = on_solid(last, bg)
    last.save(OUT / f"{name}.png", optimize=True)
    last.convert("RGBa").resize((720, round(last.height * 720 / last.width)), Image.LANCZOS) \
        .convert("RGBA").save(OUT / f"{name}-720.png", optimize=True)
    on_solid(last, "#fbfaf7").convert("RGB").save(OUT / f"{name}-og.png", optimize=True)   # link previews
    gif = [to_gif_frame(on_solid(Image.open(f).convert("RGBA"), bg) if bg else Image.open(f).convert("RGBA"))
           for f in frames]
    gif[0].save(OUT / f"{name}.gif", save_all=True, append_images=gif[1:], duration=round(1000 / FPS), loop=0,
                transparency=255, disposal=2, optimize=False)
    # the page serves an animated WebP: full 8-bit alpha, so anti-aliased edges stay smooth on both themes
    webp = [on_solid(Image.open(f).convert("RGBA"), bg) if bg else Image.open(f).convert("RGBA") for f in frames]
    webp[0].save(OUT / f"{name}.webp", save_all=True, append_images=webp[1:], duration=round(1000 / FPS), loop=0,
                 lossless=WEBP_LOSSLESS, quality=WEBP_QUALITY, method=6, alpha_quality=100)
    print(f"{name}: {len(frames)} frames, gif {(OUT / f'{name}.gif').stat().st_size / 1e6:.2f} MB, "
          f"webp {(OUT / f'{name}.webp').stat().st_size / 1e6:.2f} MB")


# ------------------------------------------------------------------ figures
def landscape(src, d):
    """Ball on a potential: a loop well, a plateau, an absorbing exit. Three acts."""
    x = np.linspace(-2.2, 3.0, 600)

    def V(x, lift, tilt):
        well = -1.6 * np.exp(-((x + 1.1) ** 2) / 0.12) * (1 - lift)       # the loop well, lifted when gated
        rough = 0.12 * np.sin(5.5 * x) * np.exp(-((x - 0.6) ** 2) / 1.2)   # dithering plateau
        return well + rough - tilt * (x + 2.2) * 0.45

    acts = [("Base model", "the loop is a deep well and the state stays in it", 0, 0),
            ("+ escape term (layers 9 and 11)", "the well is lifted, only where the model loops", 1, 0),
            ("+ commit term (layers 15 and 19)", "the landscape tilts toward concluding", 1, 1)]
    n_act, i = 48, 0
    pos = -1.1
    rng = np.random.default_rng(0)
    for a, (title, sub, lift_to, tilt_to) in enumerate(acts):
        lift_from = acts[a - 1][2] if a else 0
        tilt_from = acts[a - 1][3] if a else 0
        for k in range(n_act + (HOLD if a == 2 else 12)):
            u = ease(k / 18)
            lift = lift_from + (lift_to - lift_from) * u
            tilt = tilt_from + (tilt_to - tilt_from) * u
            if a == 0:
                pos = -1.1 + 0.08 * np.sin(k / 2.2)
            elif a == 1:
                target = 0.55 + 0.35 * np.sin(k / 6.0)
                pos += 0.12 * (target - pos) + 0.02 * rng.standard_normal()
            else:
                pos = min(2.55, pos + 0.02 + 0.08 * u)
            fig = new_fig()
            ax = fig.add_axes([0.04, 0.12, 0.92, 0.66])
            ax.set_facecolor("none")
            ax.axis("off")
            y = V(x, lift, tilt)
            ax.plot(x[x < 2.6], y[x < 2.6], color=INK, lw=2.6, solid_capstyle="round")
            ax.plot([2.6, 2.6], [V(np.array([2.6]), lift, tilt)[0], -3.4], color=INK, lw=2.6)
            ax.text(2.62, -3.0, " </think>\n concluded", color=FIN, fontsize=11, va="center", fontweight="bold")
            ax.text(-1.1, -2.05, "loop", color=LOOP, fontsize=11, ha="center", fontweight="bold")
            ax.text(0.6, 0.62 - tilt * 0.65, "dithering", color=MUTED, fontsize=10, ha="center")
            yb = V(np.array([pos]), lift, tilt)[0] + 0.17
            if pos >= 2.55:
                yb = -3.0 + 0.17
            ax.plot([pos], [yb], "o", ms=13, color=OURS, mec=OURS)
            ax.set_xlim(-2.3, 3.3)
            ax.set_ylim(-3.5, 1.1)
            fig.text(0.04, 0.92, title, fontsize=15, fontweight="bold", color=INK)
            fig.text(0.04, 0.85, sub, fontsize=11.5, color=MUTED)
            fig.text(0.96, 0.04, "schematic", fontsize=9, color=MUTED, ha="right")
            save(fig, d, i)
            i += 1


def anatomy(src, d):
    rep = json.load(open(src / "astra_anatomy.json"))
    rows = []
    for l in ("15", "19"):
        r = rep[l]
        rows.append((l, r["all-neuron loop write along u"], r["chosen loop write along u"],
                     r["all-neuron productive write along u"], r["chosen productive write along u"],
                     r["beta_per_column"][0]))
    beta_final = rows[0][5]
    n = 60
    lo = min(min(a - beta_final * c, p - beta_final * q) for _, a, c, p, q, _ in rows) * 1.15
    for i in range(n + HOLD):
        b = beta_final * ease((i - 6) / (n - 16))
        fig = new_fig()
        fig.text(0.04, 0.92, "Astra's edit reverses the loop writers", fontsize=15, fontweight="bold")
        fig.text(0.04, 0.85, f"MLP write along the loop direction u.  β = {b:.1f}, gain of the 64 writers "
                             f"{1 - b:+.1f}.  Dashed: before", fontsize=10.5,
                 color=MUTED)
        for j, (l, a, c, p, q, _) in enumerate(rows):
            ax = fig.add_axes([0.10 + 0.46 * j, 0.14, 0.38, 0.6])
            style(ax)
            vals = [a - b * c, p - b * q]
            ax.bar([0, 1], [a, p], width=0.62, fill=False, edgecolor=INK, lw=1.4, ls=(0, (3, 2)))
            ax.bar([0, 1], vals, width=0.62, color=[LOOP, FIN])
            ax.axhline(0, color=INK, lw=1)
            ax.set_xticks([0, 1], ["loop windows", "productive"])
            ax.set_ylim(lo, 1.0)
            ax.set_title(f"layer {l}", color=INK, fontsize=12)
            for k, v in enumerate(vals):
                ax.text(k, v + (0.08 if v >= 0 else -0.12), f"{v:+.2f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=11, color=INK)
            if j == 0:
                ax.set_ylabel("write along u")
        save(fig, d, i)


def threshold(src, d):
    rep = json.load(open(src / "deep_loops.json"))
    dose = {int(a): v for a, v in rep["dose"].items()}
    al = sorted(dose)
    esc = np.array([dose[a]["escape_median"] for a in al]) * 100
    cost = np.array([dose[a]["prod_cost"] for a in al])
    base = rep["base_fit_escape"][0] * 100
    e3, e4 = dose[3]["escape_median"], dose[4]["escape_median"]
    g3, g4 = np.log(-np.log(1 - e3)), np.log(-np.log(1 - e4))
    s = g4 - g3
    ac = 3 - g3 / s
    xs = np.linspace(0, max(al), 300)
    fit = (1 - np.exp(-np.exp(s * (xs - ac)))) * 100
    n = 84
    for i in range(n + HOLD):
        t = min(1.0, i / (n - 12))
        amax = t * max(al)
        fig = new_fig()
        ax = fig.add_axes([0.11, 0.15, 0.84, 0.62])
        style(ax)
        ax.axhline(base, color=BASE, lw=1.5, ls=(0, (4, 3)))
        ax.text(9.0, base + 4, f"base model {base:.1f} %", color=BASE, ha="left", fontsize=10)
        m = xs <= amax
        ax.plot(xs[m], fit[m], color=OURS, lw=2.4, alpha=1)
        shown = [k for k, a in enumerate(al) if a <= amax]
        ax.plot(np.array(al)[shown], esc[shown], "o", ms=9, color=OURS)
        ax.axvline(amax, color=MUTED, lw=1, ls=":")
        ax.set_xlim(0, max(al) + 0.5)
        ax.set_ylim(-4, 108)
        ax.set_xlabel("dose α of the escape term")
        ax.set_ylabel("escape per loop period (%)")
        if amax >= ac:
            ax.text(ac + 0.3, 50, f"αc = {ac:.1f}", color=OURS, fontsize=11, va="center")
        if shown:
            k = shown[-1]
            ax.text(0.92, 0.36, f"α = {al[k]}:  escape {esc[k]:.0f} %\nproductive cost {cost[k]:+.3f} nats/token",
                    transform=ax.transAxes, ha="right", fontsize=10, color=MUTED)
        fig.text(0.04, 0.92, "A threshold, as for escape over a barrier", fontsize=15, fontweight="bold")
        fig.text(0.04, 0.85, "Established loops of the base model, teacher forced. Curve: 1 − exp(−exp(s(α − αc)))",
                 fontsize=11, color=MUTED)
        save(fig, d, i)


def outcomes(src, d):
    rep = json.load(open(src / "edit_eval.json"))
    order = [("base", "base model"), ("astra", "Astra (released edit)"), ("deep", "escape term only"),
             ("cascade", "escape + commit")]
    n_each, i = 22, 0
    for step in range(len(order) + 1):
        frames = n_each if step < len(order) else HOLD
        for k in range(frames):
            fig = new_fig()
            ax = fig.add_axes([0.30, 0.13, 0.62, 0.64])
            style(ax)
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False)
            for j, (key, lab) in enumerate(order):
                if j > step or (j == step and step == len(order)):
                    pass
                grow = 1.0 if j < step else (ease(k / 12) if j == step else 0.0)
                if step == len(order):
                    grow = 1.0
                r = rep[key]
                y = len(order) - 1 - j
                left = 0
                for share, col in ((r["loop"], LOOP), (r["cut"], CUT), (r["finished"], FIN)):
                    ax.barh(y, share * 100 * grow, left=left, height=0.62, color=col)
                    left += share * 100 * grow
                if grow > 0.95:
                    ax.text(101, y, f"{r['acc'] * 100:.1f} %", va="center", fontsize=12, fontweight="bold",
                            color=OURS if key == "cascade" else INK)
                fig.text(0.29, 0.13 + 0.64 * (y + 0.5) / len(order), lab, ha="right", va="center", fontsize=11.5,
                         color=OURS if key == "cascade" else INK)
            ax.set_xlim(0, 112)
            ax.set_ylim(-0.6, len(order) - 0.4)
            ax.set_yticks([])
            ax.text(101, len(order) - 0.45, "accuracy", fontsize=10, color=MUTED)
            fig.text(0.04, 0.92, "Escaping is not concluding", fontsize=15, fontweight="bold")
            fig.text(0.04, 0.85, "8,192-token budget, share of 64 practice problems (%)", fontsize=11, color=MUTED)
            fig.text(0.30, 0.03, "loop", fontsize=11, color=LOOP, fontweight="bold")
            fig.text(0.37, 0.03, "cut at the budget", fontsize=11, color=CUT, fontweight="bold")
            fig.text(0.62, 0.03, "finished", fontsize=11, color=FIN, fontweight="bold")
            save(fig, d, i)
            i += 1


def match(src, d):
    rep = json.load(open(src / "eval32k.json"))
    A, C = rep["astra"], rep["cascade"]
    ia = {x["id"]: x for x in A["items"]}
    ic = {x["id"]: x for x in C["items"]}
    only_c = sum(ic[k]["correct"] and not ia[k]["correct"] for k in ic)
    only_a = sum(ia[k]["correct"] and not ic[k]["correct"] for k in ic)
    m = only_a + only_c
    p = min(1.0, 2 * sum(comb(m, j) for j in range(min(only_a, only_c) + 1)) / 2 ** m)
    groups = [("GSM8K (16)", "acc_gsm8k"), ("MATH (16)", "acc_math"), ("all 32", "acc")]
    n = 48
    for i in range(n + HOLD):
        g = ease(i / (n - 12))
        fig = new_fig()
        ax = fig.add_axes([0.10, 0.17, 0.56, 0.58])
        style(ax)
        for j, (lab, key) in enumerate(groups):
            for o, (r, col) in enumerate(((A, ASTRA), (C, OURS))):
                v = r[key] * 100 * g
                ax.bar(j + (o - 0.5) * 0.36, v, width=0.34, color=col)
                if g > 0.95:
                    ax.text(j + (o - 0.5) * 0.36, v + 2, f"{v:.0f}", ha="center", fontsize=10.5, color=col,
                            fontweight="bold")
        ax.set_xticks(range(len(groups)), [x[0] for x in groups])
        ax.set_ylim(0, 100)
        ax.set_ylabel("accuracy (%)")
        fig.text(0.70, 0.66, "Astra", color=ASTRA, fontsize=12, fontweight="bold")
        fig.text(0.70, 0.60, "escape + commit", color=OURS, fontsize=12, fontweight="bold")
        if g > 0.95:
            fig.text(0.70, 0.48, f"paired: {only_c} only ours,\n{only_a} only Astra (p = {p:.2f})", fontsize=10.5,
                     color=MUTED)
            fig.text(0.70, 0.34, f"loops {C['loop'] * 100:.0f} % vs {A['loop'] * 100:.0f} %", fontsize=10.5,
                     color=MUTED)
            fig.text(0.70, 0.27, f"tokens {C['mean_tokens'] / 1e3:.1f}k vs {A['mean_tokens'] / 1e3:.1f}k",
                     fontsize=10.5, color=MUTED)
        fig.text(0.04, 0.92, "Astra-level at 32k tokens", fontsize=15, fontweight="bold")
        fig.text(0.04, 0.85, "32 fresh practice problems, 32,768-token budget, math-verify grading", fontsize=11,
                 color=MUTED)
        save(fig, d, i)


TEXT = {
    "en": {"t1": "Ending reasoning loops in Qwen3.5-0.8B", "s1": "accuracy on 64 practice problems, 8,192-token budget",
           "rows": ["base model", "Astra (GPT-6, 20 h)", "escape + commit (ours)"],
           "t2": "What it took to build the edit", "s2": "tokens generated before the edit was frozen",
           "fewer": "× fewer tokens", "gpu": "{m:.0f} GPU-minutes on one RTX 4080, no frontier model",
           "t3": "Two terms, read from the model",
           "tiles": [("{a:.0f}%", "Astra {b:.0f}%", "8k tokens, p = {p:.3f}"),
                     ("{a:.0f}%", "Astra {b:.0f}%", "32k tokens, on par"),
                     ("{r:.0f}×", "fewer tokens", "{m:.0f} GPU-minutes")]},
    "fr": {"t1": "Mettre fin aux boucles de raisonnement", "s1": "précision sur 64 problèmes, budget de 8 192 tokens",
           "rows": ["modèle de base", "Astra (GPT-6, 20 h)", "fuite + conclusion (nous)"],
           "t2": "Ce qu'il a fallu pour construire l'édition", "s2": "tokens générés avant de figer l'édition",
           "fewer": "× moins de tokens", "gpu": "{m:.0f} minutes de GPU sur une RTX 4080, sans modèle de pointe",
           "t3": "Deux termes, lus dans le modèle",
           "tiles": [("{a:.0f} %", "Astra {b:.0f} %", "8k tokens, p = {p:.3f}"),
                     ("{a:.0f} %", "Astra {b:.0f} %", "32k tokens, à égalité"),
                     ("{r:.0f}×", "moins de tokens", "{m:.0f} minutes de GPU")]},
}


def nf(x, d, lang, unit=""):
    s = f"{x:.{d}f}"
    return s + unit if lang == "en" else s.replace(".", ",") + (" " + unit if unit else "")


def summary(src, d, lang="en"):
    """Hero figure: accuracy race at 8k, the cost gap, then three headline tiles. summary.json"""
    s = json.load(open(src / "summary.json", encoding="utf-8"))
    T = TEXT[lang]
    a8 = [s["acc8k"][k]["acc"] * 100 for k in ("base", "astra", "cascade")]
    cols = [BASE, ASTRA, OURS]
    c = s["cost"]
    i = 0
    # act 1: the race
    n1 = 44
    for k in range(n1 + 20):
        g = ease(k / n1)
        fig = new_fig()
        fig.text(0.05, 0.89, T["t1"], fontsize=18, fontweight="bold")
        fig.text(0.05, 0.81, T["s1"], fontsize=11.5, color=MUTED)
        ax = fig.add_axes([0.05, 0.10, 0.80, 0.62])
        ax.set_facecolor("none")
        ax.axis("off")
        for j, (v, col, lab) in enumerate(zip(a8, cols, T["rows"])):
            y = 2 - j
            gj = ease((k - 6 * j) / (n1 - 12))
            ax.barh(y, v * gj, height=0.5, color=col)
            ax.text(0, y + 0.42, lab, fontsize=12.5, color=col if j else INK, fontweight="bold", va="center")
            ax.text(v * gj + 1.2, y, nf(v * gj, 1, lang, "%"), fontsize=17 if j == 2 else 14, fontweight="bold",
                    color=col if j else INK, va="center")
        ax.set_xlim(0, 80)
        ax.set_ylim(-0.5, 2.8)
        save(fig, d, i)
        i += 1
    # act 2: the cost gap
    n2 = 44
    for k in range(n2 + 18):
        g = ease(k / n2)
        fig = new_fig()
        fig.text(0.05, 0.89, T["t2"], fontsize=18, fontweight="bold")
        fig.text(0.05, 0.81, T["s2"], fontsize=11.5, color=MUTED)
        ax = fig.add_axes([0.05, 0.36, 0.80, 0.36])
        ax.set_facecolor("none")
        ax.axis("off")
        for j, (v, col, lab) in enumerate(((c["astra_tokens_frozen"], ASTRA, T["rows"][1]),
                                           (c["two_term_tokens"], OURS, T["rows"][2]))):
            y = 1 - j
            ax.barh(y, v / 1e6 * min(1, g * 1.4), height=0.42, color=col)
            ax.text(0, y + 0.38, lab, fontsize=12.5, color=col, fontweight="bold", va="center")
            ax.text(v / 1e6 * min(1, g * 1.4) + 1.2, y, nf(v / 1e6 * min(1, g * 1.4), 2 if j else 1, lang, "M"),
                    fontsize=14, fontweight="bold", color=col, va="center")
        ax.set_xlim(0, 112)
        ax.set_ylim(-0.4, 1.7)
        r = 1 + (c["ratio"] - 1) * ease((k - 14) / (n2 - 14))
        if k >= 14:
            fig.text(0.05, 0.13, f"{r:.0f}{T['fewer']}", fontsize=26, fontweight="bold", color=OURS)
            fig.text(0.05, 0.05, T["gpu"].format(m=c["gpu_minutes"]), fontsize=11.5, color=MUTED)
        save(fig, d, i)
        i += 1
    # act 3: the tiles
    p8, a32 = s["paired8k"], s["acc32k"]
    vals = [dict(a=a8[2], b=a8[1], p=p8["p"]), dict(a=a32["cascade"]["acc"] * 100, b=a32["astra"]["acc"] * 100),
            dict(r=c["ratio"], m=c["gpu_minutes"])]
    n3 = 30
    for k in range(n3 + HOLD + 12):
        fig = new_fig()
        fig.text(0.05, 0.89, T["t3"], fontsize=18, fontweight="bold")
        for j, ((big, mid, small), v) in enumerate(zip(T["tiles"], vals)):
            gj = ease((k - 7 * j) / 12)
            if gj <= 0:
                continue
            x0 = 0.05 + 0.31 * j
            yo = 0.06 * (1 - gj)
            fig.patches.append(matplotlib.patches.FancyBboxPatch(
                (x0, 0.20 - yo), 0.28, 0.50, boxstyle="round,pad=0.0,rounding_size=0.025", transform=fig.transFigure,
                facecolor="none", edgecolor=OURS if j != 1 else ASTRA, lw=2.4))
            fig.text(x0 + 0.14, 0.55 - yo, big.format(**v), fontsize=32, fontweight="bold",
                     ha="center", va="center", color=OURS)
            fig.text(x0 + 0.14, 0.38 - yo, mid.format(**v), fontsize=13, fontweight="bold", ha="center",
                     va="center", color=ASTRA if j < 2 else OURS)
            small = small.format(**v)
            fig.text(x0 + 0.14, 0.29 - yo, small.replace("0.", "0,") if lang == "fr" else small, fontsize=10.5, ha="center", va="center", color=MUTED)
        save(fig, d, i)
        i += 1


FIGS = {"summary": summary, "summary-fr": lambda src, d: summary(src, d, "fr"), "landscape": landscape,
        "anatomy": anatomy, "threshold": threshold, "outcomes": outcomes, "match": match}
SOLID = {"summary-fr": "#fbfaf7"}          # the French copy goes into an email, where transparency is not reliable


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_repo", type=Path)
    ap.add_argument("--frames", type=Path, default=Path(tempfile.gettempdir()) / "loop-edits-frames")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    src = args.source_repo / "exp8"
    for name, fn in FIGS.items():
        if args.only and name not in args.only:
            continue
        d = args.frames / name
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.png"):
            f.unlink()
        fn(src, d)
        encode(d, name, SOLID.get(name))


if __name__ == "__main__":
    main()
