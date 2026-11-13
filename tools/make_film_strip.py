#!/usr/bin/env python3
"""Draw the attention graphs of the spectral-trust band and page from real model output.

    python tools/make_film_strip.py            # from the repository root

Needs spectral-trust 0.4.0 or later, torch and the Qwen/Qwen3-0.6B weights in
the local Hugging Face cache. It runs the model once on the CPU (2 threads) on
one sentence and writes two pieces of inline SVG into every page that carries
their markers:

  * film-strip: six token graphs (layers 0, 5, 10, 15, 20 and 27), with the
    head-averaged, symmetrised attention as edges and the Fiedler vector as
    node colour, and one line with the Fiedler value of all 28 layers, the six
    layers marked;
  * st-graph: the graph of layer 10 drawn larger, with its tokens as labels.
    The page animates it once (edges draw in, nodes take their colour) when
    motion is allowed.

The markup takes its ink colours from the page, so it serves both themes.
"""

from __future__ import annotations

import html
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
import spectral_trust as st  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGES = [ROOT / "index.html", ROOT / "software" / "spectral-trust.html"]
MODEL = "Qwen/Qwen3-0.6B"
SENTENCE = "The cat sat on the mat because it was warm."
LAYERS = [0, 5, 10, 15, 20, 27]
FEATURE_LAYER = 10
BLUE, ORANGE, WHITE = (0, 114, 178), (230, 159, 0), (255, 255, 255)


def colour(t: float) -> str:
    """Blue for negative entries, white at zero, orange for positive ones."""
    end = ORANGE if t >= 0 else BLUE
    a = min(1.0, abs(t))
    return "#" + "".join(f"{round(w + a * (e - w)):02x}" for w, e in zip(WHITE, end))


def ring(n: int, cx: float, cy: float, radius: float):
    return [(cx + radius * math.sin(2 * math.pi * i / n), cy - radius * math.cos(2 * math.pi * i / n)) for i in range(n)]


def strongest_edges(W: np.ndarray, keep: int = 30):
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    top = W.max()
    pairs = sorted(((W[i, j], i, j) for i in range(W.shape[0]) for j in range(i + 1, W.shape[0])), reverse=True)
    return [p for p in pairs if p[0] >= 0.08 * top][:keep], top


def graph_svg(W: np.ndarray, v: np.ndarray, layer: int) -> str:
    n = W.shape[0]
    pts = ring(n, 60, 60, 46)
    pairs, top = strongest_edges(W)
    out = [f'<svg viewBox="0 0 120 120" role="img" aria-label="Attention graph over the {n} tokens at layer {layer}">',
           '<g class="strip-edges">']
    for w, i, j in reversed(pairs):
        r = w / top
        out.append(f'<line x1="{pts[i][0]:.1f}" y1="{pts[i][1]:.1f}" x2="{pts[j][0]:.1f}" y2="{pts[j][1]:.1f}" '
                   f'stroke-width="{0.5 + 4.5 * r:.2f}" opacity="{0.25 + 0.75 * r:.2f}"/>')
    out.append('</g><g class="strip-nodes">')
    scale = np.abs(v).max() or 1.0
    for (x, y), val in zip(pts, v):
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7.5" fill="{colour(float(val / scale))}"/>')
    out.append("</g></svg>")
    return "".join(out)


def feature_svg(W: np.ndarray, v: np.ndarray, tokens: list[str], layer: int, value: float) -> str:
    """The larger graph with token labels. Edges carry pathLength=1 so they can draw in."""
    n = W.shape[0]
    cx = cy = 200
    pts = ring(n, cx, cy, 128)
    labels = ring(n, cx, cy, 168)
    pairs, top = strongest_edges(W, keep=34)
    scale = np.abs(v).max() or 1.0
    out = [f'<svg class="st-graph" viewBox="0 0 400 400" role="img" aria-label="The attention graph of layer {layer} of '
           f'{MODEL} over the {n} tokens of one sentence, with nodes split into two groups by the Fiedler vector. '
           f'Its Fiedler value is {value:.3f}.">',
           '<g class="g-edges">']
    for k, (w, i, j) in enumerate(reversed(pairs)):
        r = w / top
        out.append(f'<line pathLength="1" style="--k:{k}" x1="{pts[i][0]:.1f}" y1="{pts[i][1]:.1f}" '
                   f'x2="{pts[j][0]:.1f}" y2="{pts[j][1]:.1f}" stroke-width="{0.8 + 7 * r:.2f}" '
                   f'opacity="{0.22 + 0.78 * r:.2f}"/>')
    out.append('</g><g class="g-nodes">')
    for k, ((x, y), val) in enumerate(zip(pts, v)):
        out.append(f'<circle style="--c:{colour(float(val / scale))};--k:{k}" cx="{x:.1f}" cy="{y:.1f}" r="15"/>')
    out.append('</g><g class="g-labels" aria-hidden="true">')
    for (x, y), tok in zip(labels, tokens):
        anchor = "middle" if abs(x - cx) < 20 else ("start" if x > cx else "end")
        out.append(f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="{anchor}">{html.escape(tok.strip() or tok)}</text>')
    out.append("</g></svg>")
    return "".join(out)


def line_svg(values: list[float]) -> str:
    w, h, left, right, top, bottom = 600, 64, 8, 592, 8, 56
    xs = [left + (right - left) * i / (len(values) - 1) for i in range(len(values))]
    ys = [bottom - (bottom - top) * v for v in values]  # the Fiedler value lies in [0, 1] here
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle style="--i:{n}" cx="{xs[l]:.1f}" cy="{ys[l]:.1f}" r="4.5"/>' for n, l in enumerate(LAYERS))
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Fiedler value of each of the 28 layers, between '
            f'{min(values):.2f} and {max(values):.2f}, with the six layers above marked">'
            f'<line class="strip-base" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>'
            f'<polyline class="strip-curve" points="{pts}"/><g class="strip-dots">{dots}</g></svg>')


def fill(page: Path, name: str, markup: str) -> bool:
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if not page.is_file():
        return False
    text = page.read_text(encoding="utf-8")
    if start not in text:
        return False
    a, b = text.index(start) + len(start), text.index(end)
    page.write_text(text[:a] + "\n" + markup + "\n" + text[b:], encoding="utf-8", newline="\n")
    return True


def main() -> int:
    model = st.SpectralModel.from_pretrained(MODEL)
    cache = model.run_with_spectra(SENTENCE, keep_patterns=True)
    layers = cache.layers
    tokens = list(cache.tokens)
    values = [float(cache[f"blocks.{l}.graph.fiedler_value"]) for l in layers]

    def graph_of(l):
        pattern = np.asarray(cache[f"blocks.{l}.attn.pattern"], dtype=np.float64)   # [heads, S, S]
        W = (0.5 * (pattern + pattern.transpose(0, 2, 1))).mean(axis=0)            # symmetrise, then average the heads
        v = np.asarray(cache[f"blocks.{l}.graph.fiedler_vector"], dtype=np.float64)
        if v[0] > 0:                                                                # an eigenvector's sign is arbitrary
            v = -v
        return W, v

    cells = []
    for i, l in enumerate(LAYERS):
        W, v = graph_of(l)
        cells.append(f'<figure class="strip-cell" style="--i:{i}">{graph_svg(W, v, l)}<figcaption>Layer {l}</figcaption></figure>')
    strip = ('<div class="strip" data-strip>\n<div class="strip-cells">\n' + "\n".join(cells) + "\n</div>\n"
             '<div class="strip-line">\n' + line_svg(values) + "\n"
             f'<p class="strip-axis"><span>Layer 0</span><span><code>fiedler_value</code>, all {len(layers)} layers</span>'
             f"<span>Layer {layers[-1]}</span></p>\n</div>\n</div>")
    W, v = graph_of(FEATURE_LAYER)
    feature = feature_svg(W, v, tokens, FEATURE_LAYER, values[FEATURE_LAYER])

    for page in PAGES:
        done = [name for name, markup in (("film-strip", strip), ("st-graph", feature)) if fill(page, name, markup)]
        print(f"{page.relative_to(ROOT).as_posix()}: {', '.join(done) or 'no markers'}")
    print(f"spectral-trust {st.__version__}, {MODEL}, {len(tokens)} tokens, {len(layers)} layers")
    print("snippet value, blocks.10.graph.fiedler_value:", values[FEATURE_LAYER])
    print("fiedler_value at the six layers:", [round(values[l], 3) for l in LAYERS])
    print(f"film strip {len(strip.encode('utf-8')) / 1024:.1f} KB, feature graph {len(feature.encode('utf-8')) / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
