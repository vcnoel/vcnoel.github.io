#!/usr/bin/env python3
"""Animated figures for the release post of spectral-trust 0.4.0 (posts/drafts/spectral-trust-0-4-0.html).

    python tools/blog_figures/release_post.py [--frames DIR] [--only NAME ...]

Both figures are drawn from files next to this script, and nothing in them is typed in by hand:

  race             two terminal cards run the same job, 0.3.0 on the left and 0.4.0 on the right, with a
                   progress bar over the 28 layers and a running clock. Read from race_timing.json, written by
                   race_timing.py: the median repetition of the record labelled "defaults" for each version.
                   The cards replay the logged timestamps. The clocks start when the forward pass is done
                   (the same transformers call in both versions), or before it with --with-forward. When
                   the slower side takes more than 30 s, both sides are sped up by the same whole factor
                   (the smallest that brings the slower side to 20 s or less) and the frame says so. The
                   clock always shows measured seconds.
  transformer-lens the two lines that hand a TransformerLens cache to spectral-trust are typed in, then the
                   output of tl_snippet.py appears. Read from tl_snippet_output.txt.

The frames are written to DIR/NAME (default: a folder in the system temporary directory, outside this
repository) and each folder is then encoded with tools/blog_figures/encode.py into
posts/figures/spectral-trust-0-4-0/NAME.{webm,mp4,gif,png}. The script also prints the caption numbers.

Frames are 1440 px wide at 12 frames per second, and each loop holds its final state for three seconds.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SITE = HERE.parents[1]
ENCODE = HERE / "encode.py"
OUT = SITE / "posts" / "figures" / "spectral-trust-0-4-0"
TIMING = HERE / "race_timing.json"
SNIPPET = HERE / "tl_snippet_output.txt"

FPS, WIDTH, SS = 12, 1440, 2  # frames are drawn at twice the size and scaled down
HOLD = 3 * FPS
LEAD = FPS // 2
WHITE, INK, MUTED = "#FFFFFF", "#16140F", "#635C52"
CARD, TEXT, DIM, TRACK = "#14171C", "#E6E4DE", "#9C978D", "#2B3038"
ORANGE, BLUE, SKY = "#E69F00", "#0072B2", "#6CC0EF"

TYPED = [
    "_, tl_cache = model.run_with_cache(tokens)",
    "cache = st.from_transformer_lens(tl_cache, model, tokens)",
]


@lru_cache(maxsize=None)
def font(family: str, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = font_manager.findfont(font_manager.FontProperties(family=family, weight="bold" if bold else "normal"),
                                 fallback_to_default=False)
    return ImageFont.truetype(path, size * SS)


def sans(size: int, bold: bool = False):
    return font("DejaVu Sans", size, bold)


def mono(size: int, bold: bool = False):
    return font("DejaVu Sans Mono", size, bold)


def mix(a: str, b: str, t: float) -> tuple[int, int, int]:
    """Colour a blended towards colour b by t."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return tuple(round(x + (y - x) * t) for x, y in zip(ca, cb))


class Canvas:
    """A white frame, drawn in final pixels (everything is scaled by SS inside)."""

    def __init__(self, height: int):
        self.height = height
        self.im = Image.new("RGB", (WIDTH * SS, height * SS), WHITE)
        self.d = ImageDraw.Draw(self.im)

    def text(self, xy, s, f, fill, anchor="la"):
        self.d.text((xy[0] * SS, xy[1] * SS), s, font=f, fill=fill, anchor=anchor)

    def width(self, s, f) -> float:
        return self.d.textlength(s, font=f) / SS

    def box(self, x0, y0, x1, y1, fill, radius=0):
        self.d.rounded_rectangle((x0 * SS, y0 * SS, x1 * SS, y1 * SS), radius=radius * SS, fill=fill)

    def dot(self, x, y, r, fill):
        self.d.ellipse(((x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS), fill=fill)

    def card(self, x0, y0, x1, y1, label: str, tag: str = "", tag_fill=TEXT):
        """A terminal card with its title row. Returns the y of the first line of content."""
        self.box(x0, y0, x1, y1, CARD, radius=18)
        for k in range(3):
            self.dot(x0 + 34 + 26 * k, y0 + 36, 8, TRACK)
        f = mono(24)
        self.text((x0 + 118, y0 + 36), label, f, DIM, "lm")
        if tag:
            self.text((x0 + 118 + self.width(label + " ", f), y0 + 36), tag, mono(24, True), tag_fill, "lm")
        return y0 + 78

    def save(self, path: Path):
        self.im.resize((WIDTH, self.height), Image.LANCZOS).save(path)


def fresh(folder: Path) -> Path:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    return folder


# ---------------------------------------------------------------------------------------------------------
# 1. the race
# ---------------------------------------------------------------------------------------------------------
def pick(runs, version: str, label: str):
    """The latest record of race_timing.json for a version and a label."""
    hits = [r for r in runs if r["spectral_trust"].startswith(version) and r["label"] == label]
    if not hits:
        raise SystemExit(f"race_timing.json has no record labelled {label!r} for spectral-trust {version}")
    return hits[-1]


def clocked(rec, with_forward: bool):
    """(median time, the repetition that has it) of a record, with or without the forward pass."""
    key = "total_s" if with_forward else "spectral_s"
    times = sorted((rep[key], k) for k, rep in enumerate(rec["reps"]))
    assert len(times) % 2 == 1, "an odd number of repetitions has a median run"
    value, k = times[len(times) // 2]
    return value, rec["reps"][k]


def race_numbers():
    runs = json.loads(TIMING.read_text(encoding="utf-8"))["runs"]
    old, new, dense = pick(runs, "0.3", "defaults"), pick(runs, "0.4", "defaults"), pick(runs, "0.3", "dense")
    for a, b in ((old, new), (old, dense)):  # same job, same environment
        for key in ("model", "n_tokens", "n_layers", "n_heads", "threads", "device", "torch", "transformers",
                    "numpy", "scipy", "python", "clock"):
            assert a[key] == b[key], (key, a[key], b[key])
        assert len(a["reps"]) == len(b["reps"])
    assert old["eigen_solver"] == "sparse" and new["eigen_solver"] == "dense" and dense["eigen_solver"] == "dense"
    return old, new, dense


def race(frames: Path, with_forward: bool = False) -> int:
    old, new, dense = race_numbers()
    n_layers, n_reps = old["n_layers"], len(old["reps"])
    sides = []
    for rec, accent, light, command in (
            (old, ORANGE, None, "analyze_layer, layer by layer"),
            (new, SKY, mix(CARD, SKY, 0.32), "model.run_with_spectra(prompt)")):
        total, rep = clocked(rec, with_forward)
        start = 0.0 if with_forward else rep["forward_s"]  # where the clock of the card starts
        assert len(rep["layer_done_s"]) == n_layers
        batches, covered = [], 0
        for n, t in rep.get("attn_batches", []):  # per-head spectra, several layers per call
            covered += n
            batches.append((t - start, covered))
        sides.append({"version": rec["spectral_trust"].replace("rc1", ""), "accent": accent, "light": light,
                      "command": command, "forward": rep["forward_s"] - start,
                      "layers": [t - start for t in rep["layer_done_s"]], "batches": batches, "total": total})
    slow = max(s["total"] for s in sides)
    speed = 1 if slow <= 30 else math.ceil(slow / 20)
    ratio = sides[0]["total"] / sides[1]["total"]
    n_run = math.ceil(slow / speed * FPS)
    n_frames = LEAD + n_run + HOLD
    height = 620
    card_w, gap, top = 672, 32, 132
    bottom = top + 372
    scope = "forward pass included" if with_forward else "forward pass not timed"

    for i in range(n_frames):
        t = max(i - LEAD, 0) / FPS * speed  # measured seconds
        c = Canvas(height)
        c.text((32, 44), f"Hidden state metrics for the {n_layers} layers of {old['model'].split('/')[-1]}",
               sans(32, True), INK, "lm")
        c.text((32, 92), f"one prompt of {old['n_tokens']} tokens, CPU, {old['threads']} threads, {scope}",
               sans(24), MUTED, "lm")
        if speed > 1:
            c.text((WIDTH - 32, 92), f"replayed at {speed}x speed", sans(24), MUTED, "rm")

        for k, s in enumerate(sides):
            x0 = 32 + k * (card_w + gap)
            y = c.card(x0, top, x0 + card_w, bottom, "spectral-trust", s["version"], s["accent"])
            x, w = x0 + 34, card_w - 68
            c.text((x, y + 22), s["command"], mono(26), TEXT, "lm")

            done = sum(1 for v in s["layers"] if v <= t)
            covered = max([n for when, n in s["batches"] if when <= t], default=0)
            seg = w / n_layers
            for j in range(n_layers):
                fill = s["accent"] if j < done else (s["light"] if j < covered else TRACK)
                c.box(x + j * seg, y + 66, x + (j + 1) * seg - 4, y + 114, fill, radius=3)

            finished = t >= s["total"]
            if finished:
                status = f"done, {n_layers} of {n_layers} layers"
            elif t < s["forward"]:
                status = "forward pass"
            elif s["batches"] and t < s["batches"][-1][0]:
                status = "spectra of every head"
            else:
                status = f"layer {done + 1} of {n_layers}"
            c.text((x, y + 150), status, mono(24), TEXT if finished else DIM, "lm")
            c.text((x, y + 232), f"{min(t, s['total']):.1f} s", mono(68, True), s["accent"] if finished else TEXT, "lm")

        if t >= slow:
            f = sans(38, True)
            c.text((32, bottom + 58), f"{ratio:.1f}x faster", f, BLUE, "lm")
            c.text((32 + c.width(f"{ratio:.1f}x faster", f) + 24, bottom + 60),
                   f"medians of {n_reps} runs, each version on its defaults", sans(24), MUTED, "lm")
        c.save(frames / f"{i:04d}.png")

    dense_s = clocked(dense, with_forward)[0]
    forward = sorted(rep["forward_s"] for rec in (old, new, dense) for rep in rec["reps"])
    print(f"race ({scope}): 0.3.0: {sides[0]['total']:.1f} s, 0.4.0: {sides[1]['total']:.1f} s, {ratio:.1f}x. "
          f"0.3.0 with the dense solver: {dense_s:.1f} s, {dense_s / sides[1]['total']:.1f}x. "
          f"Forward pass {forward[0]:.1f} to {forward[-1]:.1f} s. Replayed at {speed}x, torch {old['torch']}, "
          f"transformers {old['transformers']}, {n_frames} frames, height {height}")
    return n_frames


# ---------------------------------------------------------------------------------------------------------
# 2. the TransformerLens snippet
# ---------------------------------------------------------------------------------------------------------
def snippet_output() -> list[tuple[str, str]]:
    """(kind, line) pairs of tl_snippet_output.txt: "in" for an expression, "out" for what it printed."""
    rows = []
    for line in SNIPPET.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        rows.append(("in", line[4:]) if line.startswith(">>> ") else ("out", line))
    return rows


def transformer_lens(frames: Path) -> int:
    rows = snippet_output()
    size, pitch = 24, 38
    f = mono(size)
    probe = Canvas(2)
    pad = 26
    room = WIDTH - 2 * 32 - 2 * pad
    longest = max(probe.width(("" if kind == "out" else ">>> ") + line, f) for kind, line in
                  rows + [("in", s) for s in TYPED])
    assert longest <= room, f"a line is {longest:.0f} px wide, the card has {room} px"

    per_frame = 3  # characters typed per frame
    script = []  # (first frame, kind, line) of everything that appears
    at = LEAD
    for line in TYPED:
        script.append((at, "type", line))
        at += math.ceil(len(line) / per_frame) + 4
    at += 6  # the call runs
    for kind, line in rows:
        script.append((at, kind, line))
        at += 5 if kind == "in" else 1
    n_frames = at + HOLD
    top = 104
    height = top + 78 + len(script) * pitch + 34 + 32
    height += height % 2

    for i in range(n_frames):
        c = Canvas(height)
        c.text((32, 44), "Already using TransformerLens?", sans(32, True), INK, "lm")
        c.text((WIDTH - 32, 46), "gpt2, no second forward pass", sans(24), MUTED, "rm")
        y = c.card(32, top, WIDTH - 32, height - 32, "python")
        x = 32 + pad
        for row, (start, kind, line) in enumerate(script):
            if i < start:
                break
            yy = y + 14 + row * pitch
            if kind == "out":
                c.text((x, yy), line, f, TEXT, "lm")
                continue
            shown = line if kind == "in" else line[:(i - start + 1) * per_frame]
            c.text((x, yy), ">>> ", f, SKY, "lm")
            c.text((x + c.width(">>> ", f), yy), shown, f, TEXT, "lm")
            if kind == "type" and len(shown) < len(line):  # the cursor
                cx = x + c.width(">>> " + shown, f)
                c.box(cx + 2, yy - size * 0.6, cx + 2 + size * 0.55, yy + size * 0.6, TEXT)
        c.save(frames / f"{i:04d}.png")
    print(f"transformer-lens: {len(rows)} output lines, {n_frames} frames, height {height}")
    return n_frames


FIGURES = {"race": race, "transformer-lens": transformer_lens}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", type=Path, default=Path(tempfile.gettempdir()) / "frames_release")
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=sorted(FIGURES))
    ap.add_argument("--no-encode", action="store_true", help="write the frames only")
    ap.add_argument("--with-forward", action="store_true",
                    help="race: start the clocks before the forward pass (default: when it is done)")
    args = ap.parse_args()
    status = 0
    for name in args.only:
        folder = fresh(args.frames / name)
        if name == "race":
            race(folder, with_forward=args.with_forward)
        else:
            FIGURES[name](folder)
        if not args.no_encode:
            status |= subprocess.run([sys.executable, str(ENCODE), str(folder), str(OUT / name), "--fps", str(FPS)],
                                     cwd=SITE).returncode
    return status


if __name__ == "__main__":
    sys.exit(main())
