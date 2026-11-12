#!/usr/bin/env python3
"""Turn a folder of PNG frames into the four files a post figure needs:

    python tools/blog_figures/encode.py FRAMES_DIR OUT_STEM [--fps 12] [--poster -1]

writes OUT_STEM.webm (VP9), OUT_STEM.mp4 (H.264, faststart), OUT_STEM.gif
(720 px wide, kept under 1.5 MB) and OUT_STEM.png (the poster, one frame).
Frames are read in name order and must all have the same even width and height
(1440 px wide is the house size, shown at 720 CSS px).

The videos need ffmpeg: either on the PATH or from the imageio-ffmpeg package
(pip install imageio-ffmpeg). Without it the GIF and the poster are still
written and the exit status is 3.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

GIF_WIDTH = 720
GIF_LIMIT = 1_500_000


def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def write_gif(frames: list[Path], out: Path, fps: int) -> int:
    """Palette GIF at 720 px. Drops frames, then colours, until it fits the limit."""
    step, colours = 1, 96
    while True:
        picked = frames[::step]
        images = []
        for f in picked:
            im = Image.open(f).convert("RGB")
            h = round(im.height * GIF_WIDTH / im.width)
            images.append(im.resize((GIF_WIDTH, h), Image.LANCZOS).quantize(colors=colours, method=Image.MEDIANCUT, dither=Image.Dither.NONE))
        images[0].save(out, save_all=True, append_images=images[1:], duration=round(1000 * step / fps),
                       loop=0, optimize=True, disposal=1)
        size = out.stat().st_size
        if size <= GIF_LIMIT or (step >= 4 and colours <= 32):
            return size
        if colours > 32:
            colours //= 2
        else:
            step += 1
            colours = 64


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir", type=Path)
    ap.add_argument("out_stem", type=Path)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--poster", type=int, default=-1, help="index of the frame used as the poster")
    args = ap.parse_args()

    frames = sorted(args.frames_dir.glob("*.png"))
    if not frames:
        print(f"no PNG frames in {args.frames_dir}")
        return 2
    sizes = {Image.open(f).size for f in frames}
    if len(sizes) != 1:
        print(f"frames differ in size: {sorted(sizes)}")
        return 2
    w, h = sizes.pop()
    if w % 2 or h % 2:
        print(f"frame size {w}x{h} must be even")
        return 2
    args.out_stem.parent.mkdir(parents=True, exist_ok=True)

    poster = args.out_stem.with_suffix(".png")
    Image.open(frames[args.poster]).convert("RGB").save(poster, optimize=True)
    gif_size = write_gif(frames, args.out_stem.with_suffix(".gif"), args.fps)
    print(f"{poster.name} {poster.stat().st_size / 1024:.0f} KB, gif {gif_size / 1024:.0f} KB, "
          f"{len(frames)} frames at {args.fps} fps, {w}x{h}")

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("ffmpeg not found: the webm and mp4 files were not written")
        return 3

    # Number the frames through a list file so that any file names work.
    listing = args.frames_dir / "frames.txt"
    listing.write_text("".join(f"file '{f.name}'\nduration {1 / args.fps:.6f}\n" for f in frames) + f"file '{frames[-1].name}'\n",
                       encoding="utf-8")
    common = [ffmpeg, "-y", "-loglevel", "error", "-threads", "2", "-f", "concat", "-safe", "0", "-i", listing.name,
              "-an", "-fps_mode", "cfr", "-r", str(args.fps), "-pix_fmt", "yuv420p"]
    jobs = {
        ".webm": ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "36", "-row-mt", "1"],
        ".mp4": ["-c:v", "libx264", "-crf", "26", "-preset", "slow", "-profile:v", "high", "-movflags", "+faststart"],
    }
    status = 0
    for suffix, codec in jobs.items():
        out = args.out_stem.with_suffix(suffix).resolve()
        r = subprocess.run(common + codec + [str(out)], cwd=args.frames_dir)
        if r.returncode:
            print(f"ffmpeg failed for {out.name}")
            status = 1
        else:
            print(f"{out.name} {out.stat().st_size / 1024:.0f} KB")
    listing.unlink(missing_ok=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
