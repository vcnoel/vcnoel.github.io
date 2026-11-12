#!/usr/bin/env python3
"""The TransformerLens snippet of the release post, run for real.

    python tools/blog_figures/tl_snippet.py

Run it with the interpreter of an environment that has spectral-trust 0.4.0 and transformer_lens installed.
It follows the section "Use it with TransformerLens or nnsight" of the library's README: gpt2 through
TransformerBridge, one run_with_cache, then st.from_transformer_lens on that cache (no second forward pass).
What it prints is also written to tl_snippet_output.txt next to this file, after a few comment lines that
record the versions. The animation of the release post (release_post.py) shows that output.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import contextlib  # noqa: E402
import importlib.metadata  # noqa: E402
import io  # noqa: E402
import logging  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import transformers  # noqa: E402
from transformer_lens.model_bridge import TransformerBridge  # noqa: E402

import spectral_trust as st  # noqa: E402

OUT = Path(__file__).resolve().parent / "tl_snippet_output.txt"
PROMPT = "The cat sat on the mat because it was warm."
LAYER, HEAD = 5, 1

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    model = TransformerBridge.boot_transformers("gpt2")  # loading messages are not part of the output
tokens = model.to_tokens(PROMPT)

# the two lines of the post
_, tl_cache = model.run_with_cache(tokens)
cache = st.from_transformer_lens(tl_cache, model, tokens)

shown = [
    ("cache", repr(cache)),
    (f"cache.layer({LAYER}).head({HEAD})", repr(cache.layer(LAYER).head(HEAD))),
]
lines = [
    f"# spectral-trust {st.__version__}, transformer_lens {importlib.metadata.version('transformer_lens')}, "
    f"torch {torch.__version__}, transformers {transformers.__version__}",
    f"# model gpt2 through TransformerBridge, prompt: {PROMPT}",
]
for expr, text in shown:
    lines.append(f">>> {expr}")
    lines.append(text)
text = "\n".join(lines) + "\n"
print(text, end="")
OUT.write_bytes(text.encode("utf-8"))
