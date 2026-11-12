#!/usr/bin/env python3
"""Animated figures for the post "A probe direction is a property of its prompt".

    python tools/blog_figures/probe_direction.py SOURCE_REPO [--frames DIR] [--out DIR] [--only NAME ...]

SOURCE_REPO is a clone of https://github.com/vcnoel/probe-direction (the code and
result files of arXiv:2608.13329). Every figure is drawn from a result file in
SOURCE_REPO/results, nothing is typed in by hand:

    slope-flip        results/wrapper_curves.json
                      (params_b, selected: the three wrappers of the paper's Figure 1)
    sign-matrix       results/wrapper_curves.json
                      (pairs, correlations: all 36 wrappers, Appendix A of the paper)
    direction-angle   results/direction_geometry_v11_validated.json
                      (models.*.interior_within and interior_across, Section 5)
    variance-share    results/gstudy_gtheory_deploy_peak_lf.json
                      (share: the three-way crossed design, Figure 2 panel a)
    reliability       results/gstudy_gtheory_deploy_peak_lf.json
                      (components, n_blocks, g_k1, asymptote_k_inf, asymptote_n_inf:
                      Equation 1 and Figure 3)

The frames (1440 x 810 px PNG, 12 per second) go to DIR/NAME, by default a folder
in the system temporary directory, never inside the website repository. Each
figure is then encoded with tools/blog_figures/encode.py into
posts/figures/probe-direction/NAME.{webm,mp4,gif,png}. Without ffmpeg the helper
still writes the GIF and the poster and this script carries on.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle  # noqa: E402

SITE = Path(__file__).resolve().parents[2]
ENCODE = Path(__file__).resolve().with_name("encode.py")

FPS = 12
W_IN, H_IN, DPI = 7.2, 4.05, 200          # 1440 x 810 px

WHITE, INK, MUTED, LIGHT = "#FFFFFF", "#16140F", "#635C52", "#CFC6B7"
ORANGE, BLUE = "#E69F00", "#0072B2"

# Point sizes. At 200 dpi one point is 2.78 px, so 10 pt is 28 px in the 1440 px frame.
SMALL, BASE, BIG = 10.5, 12, 13.5

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": BASE,
    "axes.unicode_minus": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "axes.labelsize": BASE,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
})


def load(repo: Path, name: str) -> dict:
    path = repo / "results" / name
    if not path.is_file():
        sys.exit(f"result file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ramp(i: int, start: int, length: int) -> float:
    """0 before frame `start`, 1 from frame `start + length`, eased in between."""
    return ease((i - start + 1) / length)


def signed(v: float) -> str:
    return f"{v:+.2f}"


# ----------------------------------------------------------------------------
# 1. The trend with model size follows the wrapper
# ----------------------------------------------------------------------------

def slope_flip(repo: Path):
    d = load(repo, "wrapper_curves.json")
    x = np.array(d["params_b"], dtype=float)
    lx = np.log10(x)
    states = [d["selected"][k] for k in ("most positive", "flattest", "most negative")]
    ys = [np.array(s["values"], dtype=float) for s in states]
    rs = [float(s["r"]) for s in states]
    names = [s["deploy_label"].split(":", 1)[1].strip() for s in states]
    for y, r in zip(ys, rs):                       # the file's r is the correlation drawn here
        assert abs(np.corrcoef(lx, y)[0, 1] - r) < 1e-6

    hold, morph, last = 16, 8, 24
    starts = [0, hold + morph, 2 * (hold + morph)]  # first frame of each state
    n_frames = 2 * (hold + morph) + last

    def draw(fig, i):
        ax = fig.add_axes([0.125, 0.175, 0.845, 0.79])
        ax.set_xscale("log")
        ax.set_xlim(0.4, 19)
        ax.set_ylim(0.10, 0.315)
        ax.set_xticks(x)
        ax.set_xticklabels(["0.5B", "1.5B", "3B", "7B"])
        ax.minorticks_off()
        ax.set_yticks([0.10, 0.15, 0.20, 0.25, 0.30])
        ax.set_xlabel("model size (parameters)", labelpad=6)
        ax.set_ylabel("probe score", labelpad=8)
        for g in (0.15, 0.20, 0.25, 0.30):
            ax.axhline(g, color=LIGHT, lw=0.8, alpha=0.6, zorder=0)

        # which state, and how far into the move to the next one
        s = max(k for k in range(3) if i >= starts[k])
        t = 0.0
        if s < 2 and i >= starts[s] + hold:
            t = ramp(i, starts[s] + hold, morph)
        y = ys[s] if t == 0 else (1 - t) * ys[s] + t * ys[s + 1]
        shown = s if t < 0.5 else s + 1
        fade = abs(1 - 2 * t)                      # labels fade out and back in during a move

        # earlier wrappers stay as grey traces
        for k in range(3):
            a = 1.0 if k < s else (t if k == s and t > 0 else 0.0)
            if a <= 0:
                continue
            ax.plot(x, ys[k], color=LIGHT, lw=3, marker="o", ms=7, alpha=a, zorder=2)
            ax.text(x[-1] * 1.13, ys[k][-1], f"r = {signed(rs[k])}", color=MUTED, fontsize=BASE,
                    va="center", ha="left", alpha=a, zorder=2)

        slope, icpt = np.polyfit(lx, y, 1)
        xx = np.array([x[0] * 0.9, x[-1] * 1.08])
        ax.plot(xx, slope * np.log10(xx) + icpt, color=INK, lw=1.8, ls=(0, (4, 3)), zorder=3)
        ax.plot(x, y, color=ORANGE, lw=4, marker="o", ms=12, mec=WHITE, mew=1.5, zorder=4,
                solid_capstyle="round")
        ax.text(x[-1] * 1.13, y[-1], f"r = {signed(rs[shown])}", color=INK, fontsize=BIG,
                fontweight="bold", va="center", ha="left", zorder=5, alpha=fade)

        ax.text(0.025, 0.975, "only the deployment framing changes", transform=ax.transAxes,
                color=MUTED, fontsize=SMALL, va="top", ha="left")
        ax.text(0.025, 0.885, names[shown], transform=ax.transAxes, color=INK, fontsize=BIG,
                fontweight="bold", va="top", ha="left", alpha=fade)

    return n_frames, draw


# ----------------------------------------------------------------------------
# 2. All 36 wrappers: the sign of the trend, cell by cell
# ----------------------------------------------------------------------------

EVAL_NAMES = {"eval_benchmark": "benchmark", "eval_certification": "certification",
              "eval_leaderboard": "leaderboard", "eval_oversight": "oversight",
              "eval_redteam": "red team", "eval_rubric": "rubric"}
DEPLOY_NAMES = {"deploy_a": "chat", "deploy_advice": "advice", "deploy_agent": "agent",
                "deploy_api": "API", "deploy_ide": "IDE", "deploy_support": "support"}


def sign_matrix(repo: Path):
    d = load(repo, "wrapper_curves.json")
    pairs = [p.split("|") for p in d["pairs"]]
    r = dict(zip(map(tuple, pairs), map(float, d["correlations"])))
    evals = list(dict.fromkeys(p[0] for p in pairs))
    deploys = list(dict.fromkeys(p[1] for p in pairs))
    cells = [(a, b) for a in evals for b in deploys]
    assert len(cells) == len(r) == 36

    lead, last = 4, 24
    n_frames = lead + len(cells) + last

    def draw(fig, i):
        ax = fig.add_axes([0.255, 0.035, 0.52, 0.75])
        ax.set_xlim(0, len(deploys))
        ax.set_ylim(len(evals), 0)
        ax.axis("off")
        done = min(max(i - lead + 1, 0), len(cells))

        for row, a in enumerate(evals):
            ax.text(-0.12, row + 0.5, EVAL_NAMES.get(a, a), color=INK, fontsize=SMALL,
                    ha="right", va="center")
        for col, b in enumerate(deploys):
            ax.text(col + 0.5, -0.14, DEPLOY_NAMES.get(b, b), color=INK, fontsize=SMALL,
                    ha="center", va="bottom")
        fig.text(0.515, 0.94, "deployment framing", color=MUTED, fontsize=BASE, ha="center", va="center")
        fig.text(0.035, 0.41, "evaluation framing", color=MUTED, fontsize=BASE, ha="center",
                 va="center", rotation=90)

        for n, (a, b) in enumerate(cells):
            row, col = evals.index(a), deploys.index(b)
            box = dict(xy=(col + 0.04, row + 0.05), width=0.92, height=0.90)
            if n >= done:
                ax.add_patch(Rectangle(**box, facecolor=WHITE, edgecolor=LIGHT, lw=1.2))
                continue
            v = r[(a, b)]
            colour, alpha = (BLUE, 0.14 + 0.46 * abs(v)) if v > 0 else (ORANGE, 0.20 + 0.60 * abs(v))
            ax.add_patch(Rectangle(**box, facecolor=colour, alpha=alpha, edgecolor="none"))
            if n == done - 1 and done < len(cells):
                ax.add_patch(Rectangle(**box, facecolor="none", edgecolor=INK, lw=2.2))
            ax.text(col + 0.5, row + 0.5, signed(v), color=INK, fontsize=SMALL, ha="center", va="center")

        vals = [r[c] for c in cells[:done]]
        pos, neg = sum(v > 0 for v in vals), sum(v < 0 for v in vals)
        fig.text(0.895, 0.80, "trend with\nmodel size", color=MUTED, fontsize=SMALL, ha="center",
                 va="center", linespacing=1.25)
        fig.text(0.895, 0.585, str(pos), color=BLUE, fontsize=38, fontweight="bold", ha="center", va="center")
        fig.text(0.895, 0.475, "positive", color=INK, fontsize=BASE, ha="center", va="center")
        fig.text(0.895, 0.285, str(neg), color=ORANGE, fontsize=38, fontweight="bold", ha="center", va="center")
        fig.text(0.895, 0.175, "negative", color=INK, fontsize=BASE, ha="center", va="center")

    return n_frames, draw


# ----------------------------------------------------------------------------
# 3. The direction itself: same wrapper against a different wrapper
# ----------------------------------------------------------------------------

def direction_angle(repo: Path):
    d = load(repo, "direction_geometry_v11_validated.json")
    models = d["models"]
    within = float(np.mean([m["interior_within"] for m in models.values()]))
    across = float(np.mean([m["interior_across"] for m in models.values()]))
    a_within = math.degrees(math.acos(within))
    a_across = math.degrees(math.acos(across))

    t_ink, t_hold, t_sweep, t_label, last = 8, 20, 28, 50, 24
    n_frames = t_label + 4 + last

    def arrow(ax, angle, length, colour, lw, scale, z):
        a = math.radians(angle)
        ax.add_patch(FancyArrowPatch((0, 0), (length * math.cos(a), length * math.sin(a)),
                                     arrowstyle="-|>", mutation_scale=scale, lw=lw, color=colour,
                                     shrinkA=0, shrinkB=0, zorder=z, joinstyle="miter", capstyle="butt"))

    def draw(fig, i):
        ax = fig.add_axes([0, 0, 1, 1])
        x0, x1 = -0.09, 2.06
        h = (x1 - x0) * H_IN / W_IN
        ax.set_xlim(x0, x1)
        ax.set_ylim(-0.145, -0.145 + h)
        ax.axis("off")

        arrow(ax, 0, 1.0, BLUE, 7, 34, 3)
        ax.text(1.05, -0.035, "one wrapper, half of the items", color=BLUE, fontsize=BIG,
                fontweight="bold", va="top", ha="left")

        g = ramp(i, t_ink, 10)
        if g > 0:
            arrow(ax, a_within, max(g, 0.06), INK, 2.6, 22, 5)
        a_txt = ramp(i, t_ink + 8, 4)
        if a_txt > 0:
            ax.text(1.05, 0.065, f"same wrapper, the other half\ncosine {within:.4f}", color=INK,
                    fontsize=BIG, va="bottom", ha="left", alpha=a_txt, linespacing=1.3)

        s = ramp(i, t_sweep, t_label - t_sweep)
        if i >= t_sweep:
            ang = a_within + s * (a_across - a_within)
            ax.add_patch(Arc((0, 0), 0.62, 0.62, theta1=0, theta2=ang, color=MUTED, lw=2.2, zorder=2))
            arrow(ax, ang, 1.0, ORANGE, 7, 34, 4)
        a_end = ramp(i, t_label, 4)
        if a_end > 0:
            tip = math.radians(a_across)
            ax.text(math.cos(tip) + 0.09, math.sin(tip) - 0.03,
                    f"a different wrapper\ncosine {across:.4f}", color=INK, fontsize=BIG,
                    fontweight="bold", va="center", ha="left", alpha=a_end, linespacing=1.3)
            mid = math.radians(a_across / 2)
            ax.text(0.40 * math.cos(mid), 0.40 * math.sin(mid), f"about {a_across:.0f}°", color=MUTED,
                    fontsize=BASE, va="center", ha="left", alpha=a_end)

    return n_frames, draw


# ----------------------------------------------------------------------------
# 4. Where the variance of the score sits
# ----------------------------------------------------------------------------

def variance_share(repo: Path):
    d = load(repo, "gstudy_gtheory_deploy_peak_lf.json")
    share = {k: 100 * float(v) for k, v in d["share"].items()}
    assert abs(sum(share.values()) - 100) < 1e-6
    labels = {"s_m": "model", "s_mt": "model × wrapper", "s_mth": "residual", "s_h": "items",
              "s_t": "wrapper", "s_th": "other interactions", "s_mh": "model × items"}
    assert set(labels) == set(share)
    rest = sorted((k for k in share if k != "s_m"), key=lambda k: -share[k])
    rows = ["s_m"] + rest
    colours = {"s_m": BLUE, "s_mt": ORANGE}

    lead, per, last = 3, 5, 24
    n_frames = lead + per * len(rows) + last

    def draw(fig, i):
        ax = fig.add_axes([0.305, 0.175, 0.665, 0.80])
        ax.set_xlim(0, 46)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("share of the variance in the score (%)", labelpad=6)
        for g in (10, 20, 30, 40):
            ax.axvline(g, color=LIGHT, lw=0.8, alpha=0.6, zorder=0)
        ax.axvline(0, color=MUTED, lw=1.2, zorder=1)
        for n, k in enumerate(rows):
            strong = k in colours
            ax.text(-1.2, n, labels[k], color=INK if strong else MUTED, fontsize=BASE,
                    fontweight="bold" if strong else "normal", ha="right", va="center")
            g = ramp(i, lead + per * n, per)
            if g <= 0:
                continue
            ax.barh(n, share[k] * g, height=0.62, color=colours.get(k, LIGHT), zorder=3)
            if g >= 1:                             # the value is printed once the bar is whole
                ax.text(share[k] + 0.9, n, f"{share[k]:.1f}%", color=INK if strong else MUTED,
                        fontsize=BASE, fontweight="bold" if strong else "normal", ha="left", va="center")

    return n_frames, draw


# ----------------------------------------------------------------------------
# 5. Averaging over wrappers against averaging over items
# ----------------------------------------------------------------------------

def reliability(repo: Path):
    d = load(repo, "gstudy_gtheory_deploy_peak_lf.json")
    c, n0 = d["components"], int(d["n_blocks"])

    def erho(k, n):                                # Equation 1 of the paper
        return c["s_m"] / (c["s_m"] + c["s_mt"] / k + c["s_mh"] / n + c["s_mth"] / (k * n))

    start = erho(1, n0)
    lim_w, lim_i = float(d["asymptote_k_inf"]), float(d["asymptote_n_inf"])
    assert abs(start - d["g_k1"]) < 1e-9
    assert abs(erho(1e12, n0) - lim_w) < 1e-6 and abs(erho(1, 1e12) - lim_i) < 1e-6

    top = 60
    g = np.linspace(1, top, 300)
    over_wrappers = np.array([erho(v, n0) for v in g])       # v wrappers, the design's items
    over_items = np.array([erho(1, n0 * v) for v in g])      # one wrapper, v times the items

    t_draw, n_draw, last = 10, 32, 24
    t_end = t_draw + n_draw
    n_frames = t_end + 4 + last

    def draw(fig, i):
        ax = fig.add_axes([0.125, 0.175, 0.845, 0.79])
        ax.set_xlim(0, top + 1.5)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("how many are averaged over", labelpad=6)
        ax.set_ylabel("reliability of the model ranking", labelpad=8)
        ax.axhline(0.80, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
        ax.text(1, 0.825, "0.80, the conventional bar", color=MUTED, fontsize=SMALL, va="bottom", ha="left")

        p = ramp(i, t_draw, n_draw)
        if p > 0:
            m = max(2, int(round(p * len(g))))
            ax.plot(g[:m], over_items[:m], color=MUTED, lw=4, zorder=3, solid_capstyle="round", clip_on=False)
            ax.plot(g[:m], over_wrappers[:m], color=BLUE, lw=4, zorder=4, solid_capstyle="round")

        ax.plot([1], [start], marker="o", ms=13, color=ORANGE, mec=WHITE, mew=1.5, zorder=6, clip_on=False)
        ax.plot([1.5, 4.4], [start + 0.035, 0.30], color=MUTED, lw=1.2, zorder=2)
        ax.text(4.8, 0.30, f"current practice\none wrapper: {start:.3f}", color=INK, fontsize=BASE,
                va="bottom", ha="left", linespacing=1.3)

        a = ramp(i, t_end, 4)
        if a > 0:
            ax.text(top, over_wrappers[-1] + 0.04, f"more wrappers, limit {lim_w:.3f}", color=BLUE,
                    fontsize=BASE, fontweight="bold", va="bottom", ha="right", alpha=a)
            ax.text(top, over_items[-1] + 0.035, f"more items, limit {lim_i:.3f}", color=MUTED,
                    fontsize=BASE, fontweight="bold", va="bottom", ha="right", alpha=a)

    return n_frames, draw


FIGURES = {
    "slope-flip": slope_flip,
    "sign-matrix": sign_matrix,
    "direction-angle": direction_angle,
    "variance-share": variance_share,
    "reliability": reliability,
}


def render(name: str, repo: Path, frames_root: Path) -> Path:
    n_frames, draw = FIGURES[name](repo)
    folder = frames_root / name
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("[0-9][0-9][0-9][0-9].png"):
        old.unlink()
    for i in range(n_frames):
        fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
        draw(fig, i)
        fig.savefig(folder / f"{i:04d}.png", dpi=DPI)
        plt.close(fig)
    print(f"{name}: {n_frames} frames ({n_frames / FPS:.1f} s) in {folder}")
    return folder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_repo", type=Path, help="clone of the paper's code repository")
    ap.add_argument("--frames", type=Path, default=None,
                    help="folder for the PNG frames (default: frames_probe in the system temporary directory)")
    ap.add_argument("--out", type=Path, default=SITE / "posts" / "figures" / "probe-direction",
                    help="folder for the encoded figures")
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    ap.add_argument("--no-encode", action="store_true", help="render the frames and stop")
    args = ap.parse_args()

    repo = args.source_repo.resolve()
    if not (repo / "results").is_dir():
        print(f"no results folder in {repo}")
        return 2
    frames_root = (args.frames or Path(tempfile.gettempdir()) / "frames_probe").resolve()
    if frames_root == SITE or SITE in frames_root.parents:
        print("the frames folder must be outside the website repository")
        return 2

    status = 0
    for name in args.only:
        folder = render(name, repo, frames_root)
        if args.no_encode:
            continue
        r = subprocess.run([sys.executable, str(ENCODE), str(folder), str(args.out / name), "--fps", str(FPS)],
                           cwd=SITE)
        if r.returncode == 3:
            print(f"{name}: no ffmpeg, the videos are still to be made from {folder}")
        elif r.returncode:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
