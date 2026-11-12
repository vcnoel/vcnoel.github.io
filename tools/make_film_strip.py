#!/usr/bin/env python3
"""Build the attention-graph film strip of the spectral-trust band.

    python tools/make_film_strip.py            # from the repository root

Needs spectral-trust 0.4.0 or later, torch and the Qwen/Qwen3-0.6B weights in
the local Hugging Face cache. It runs the model once on the CPU (2 threads),
and writes the strip into index.html between the two film-strip markers:

  * six token graphs (layers 0, 5, 10, 15, 20 and 27): the head-averaged,
    symmetrised attention as edges, and the Fiedler vector as node colour;
  * one line with the Fiedler value of all 28 layers, the six layers marked.

The markup has no text inside the drawings and takes its ink colours from the
page, so the same strip serves the light and the dark theme.
"""

from __future__ import annotations

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
MODEL = "Qwen/Qwen3-0.6B"
SENTENCE = "The cat sat on the mat because it was warm."
LAYERS = [0, 5, 10, 15, 20, 27]
START, END = "<!-- film-strip:start -->", "<!-- film-strip:end -->"
BLUE, ORANGE, WHITE = (0, 114, 178), (230, 159, 0), (255, 255, 255)


def colour(t: float) -> str:
    """Blue for negative entries, white at zero, orange for positive ones."""
    end = ORANGE if t >= 0 else BLUE
    a = min(1.0, abs(t))
    return "#" + "".join(f"{round(w + a * (e - w)):02x}" for w, e in zip(WHITE, end))


def graph_svg(W: np.ndarray, v: np.ndarray, layer: int) -> str:
    n = W.shape[0]
    size, cx, cy, radius = 120, 60, 60, 46
    pts = [(cx + radius * math.sin(2 * math.pi * i / n), cy - radius * math.cos(2 * math.pi * i / n)) for i in range(n)]
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    top = W.max()
    pairs = sorted(((W[i, j], i, j) for i in range(n) for j in range(i + 1, n)), reverse=True)
    pairs = [p for p in pairs if p[0] >= 0.08 * top][:30]
    out = [f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="Attention graph over the 11 tokens at layer {layer}">',
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


def line_svg(values: list[float]) -> str:
    w, h, left, right, top, bottom = 600, 64, 8, 592, 8, 56
    xs = [left + (right - left) * i / (len(values) - 1) for i in range(len(values))]
    ys = [bottom - (bottom - top) * v for v in values]  # the Fiedler value lies in [0, 1] here
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle cx="{xs[l]:.1f}" cy="{ys[l]:.1f}" r="4.5"/>' for l in LAYERS)
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Fiedler value of each of the 28 layers, between '
            f'{min(values):.2f} and {max(values):.2f}, with the six layers above marked">'
            f'<line class="strip-base" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>'
            f'<polyline class="strip-curve" points="{pts}"/><g class="strip-dots">{dots}</g></svg>')


def main() -> int:
    model = st.SpectralModel.from_pretrained(MODEL)
    cache = model.run_with_spectra(SENTENCE, keep_patterns=True)
    layers = cache.layers
    values = [float(cache[f"blocks.{l}.graph.fiedler_value"]) for l in layers]
    cells = []
    for i, l in enumerate(LAYERS):
        pattern = np.asarray(cache[f"blocks.{l}.attn.pattern"], dtype=np.float64)   # [heads, S, S]
        W = (0.5 * (pattern + pattern.transpose(0, 2, 1))).mean(axis=0)            # symmetrise, then average the heads
        v = np.asarray(cache[f"blocks.{l}.graph.fiedler_vector"], dtype=np.float64)
        if v[0] > 0:                                                                # an eigenvector's sign is arbitrary
            v = -v
        cells.append(f'<figure class="strip-cell" style="--i:{i}">{graph_svg(W, v, l)}<figcaption>Layer {l}</figcaption></figure>')
    html = ('<div class="strip" data-strip>\n<div class="strip-cells">\n' + "\n".join(cells) + "\n</div>\n"
            '<div class="strip-line">\n' + line_svg(values) + "\n"
            f'<p class="strip-axis"><span>Layer 0</span><span><code>fiedler_value</code>, all {len(layers)} layers</span>'
            f"<span>Layer {layers[-1]}</span></p>\n</div>\n</div>")

    index = ROOT / "index.html"
    text = index.read_text(encoding="utf-8")
    a, b = text.index(START) + len(START), text.index(END)
    index.write_text(text[:a] + "\n" + html + "\n" + text[b:], encoding="utf-8", newline="\n")
    print(f"spectral-trust {st.__version__}, {MODEL}, {len(cache.tokens)} tokens, {len(layers)} layers")
    print("snippet value, blocks.10.graph.fiedler_value:", float(cache["blocks.10.graph.fiedler_value"]))
    print("fiedler_value at the six layers:", [round(values[l], 3) for l in LAYERS])
    print(f"strip markup: {len(html.encode('utf-8')) / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
