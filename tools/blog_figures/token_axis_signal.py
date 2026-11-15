#!/usr/bin/env python3
"""Data for the first figure of the post "Steering along the token axis".

    python tools/blog_figures/token_axis_signal.py

Runs GPT-2 small once on the CPU, on one IOI sentence, with spectral-trust 0.5.0, and writes
token_axis_signal.json next to this script. token_axis_post.py draws the figure from that file.

What it records, for the residual stream after block 8 and the attention of layer 8:

  tokens      the tokens of the sentence (token 0, the attention sink, is left out of the
              graph and of the edit, as with token_span=(1, None) in the use cases)
  adjacency   the attention graph of tokens 1.., built as graph_filter builds it
              (spectral_trust.interventions.attention_adjacency on the sub-block, default
              GSPConfig: heads averaged, symmetrised)
  signal      one column of the residual stream: its projection on W_U[" Mary"] - W_U[" John"]
  lowpass     the same column after st.graph_filter(kind="lowpass", cutoff=1.0)
  highpass    the same column after st.graph_filter(kind="highpass", cutoff=1.0)

The filter acts on every column of the residual in the same way, so filtering first and
projecting after gives the same numbers as the other way round. lowpass + highpass equals
signal up to the Chebyshev approximation of the bands (max_abs_residual).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch  # noqa: E402

import spectral_trust as st  # noqa: E402
from spectral_trust import GSPConfig  # noqa: E402
from spectral_trust.interventions import attention_adjacency  # noqa: E402

TEXT = "When Mary and John went to the store, John gave a drink to"
LAYER = 8
CUTOFF = 1.0
OUT = Path(__file__).resolve().with_name("token_axis_signal.json")


def main() -> None:
    torch.set_num_threads(2)
    model = st.SpectralModel.from_pretrained("gpt2", device="cpu")
    tok = model.tokenizer
    ids = tok(TEXT, return_tensors="pt").input_ids
    hf = model.model
    with torch.no_grad():
        out = hf(ids, output_attentions=True, output_hidden_states=True)
        x = out.hidden_states[LAYER + 1][0]          # residual stream after block LAYER
        attn = out.attentions[LAYER][0]               # [H, S, S]
        lo = st.graph_filter(x, attn, kind="lowpass", cutoff=CUTOFF, token_span=(1, None))
        hi = st.graph_filter(x, attn, kind="highpass", cutoff=CUTOFF, token_span=(1, None))
        w_u = hf.lm_head.weight
        mary, john = tok(" Mary").input_ids[0], tok(" John").input_ids[0]
        d = w_u[mary] - w_u[john]
        sig, sig_lo, sig_hi = x @ d, lo @ d, hi @ d
        adj = attention_adjacency(attn[None, :, 1:, 1:], GSPConfig())[0]
    data = {
        "model": "gpt2",
        "spectral_trust": st.__version__,
        "text": TEXT,
        "layer": LAYER,
        "cutoff": CUTOFF,
        "tokens": [tok.decode([t]) for t in ids[0].tolist()],
        "span_start": 1,
        "adjacency": [[round(float(v), 6) for v in row] for row in adj.tolist()],
        "signal": [round(float(v), 5) for v in sig[1:].tolist()],
        "lowpass": [round(float(v), 5) for v in sig_lo[1:].tolist()],
        "highpass": [round(float(v), 5) for v in sig_hi[1:].tolist()],
        "max_abs_residual": float((sig_lo + sig_hi - sig)[1:].abs().max()),
    }
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.name}: {len(data['signal'])} tokens, spectral-trust {data['spectral_trust']}, "
          f"max |low + high - signal| = {data['max_abs_residual']:.2e}")


if __name__ == "__main__":
    main()
