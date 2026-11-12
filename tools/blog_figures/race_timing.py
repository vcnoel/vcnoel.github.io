#!/usr/bin/env python3
"""Time the hidden-state metrics of spectral-trust on one 512-token prompt, layer by layer.

    python tools/blog_figures/race_timing.py --label defaults
    python tools/blog_figures/race_timing.py --label dense --solver dense
    python tools/blog_figures/race_timing.py --label matched --metrics resid,graph

Run it with the interpreter of an environment that has spectral-trust installed (0.3.0 or 0.4.0), torch and
transformers. Each call appends one record to race_timing.json next to this file. The animation of the
release post (release_post.py) is drawn from that file.

The job, the same in both versions: Qwen/Qwen3-0.6B (28 layers, 16 heads) from the local Hugging Face cache,
one fixed prompt of exactly 512 tokens, and for every layer the hidden-state metrics (energy, smoothness,
spectral entropy, HFER) of the head-averaged graph and of each head's graph. The method follows
benchmarks/end_to_end.py of the library: CPU only, two threads, each version with GSPConfig(device="cpu"),
that is its own defaults.

  0.3.0  analyze_layer on the head-averaged Laplacian of each layer, then a loop of analyze_layer over the
         Laplacians of its heads (0.3.0 has no per-head hybrid function). A timestamp is taken after each layer.
  0.4.0  SpectralModel.from_pretrained(...).run_with_spectra(prompt) with its defaults. That call also returns
         the per-head attention-only metrics, which the 0.3.0 loop does not compute. run_with_spectra has no
         progress callback, so timestamps are taken when the public functions it calls return:
         per_head_metrics (one batched call covers the heads of several layers), per_head_hybrid_metrics and
         SpectralAnalyzer.analyze_layer (one call each per layer). The wrappers only read the clock. With
         --plain the call runs without them, as a check that they cost nothing.

The clock starts before tokenisation and stops when the last metric is back. The model's forward pass is the
same transformers call in both versions (eager attention, float32), and the time at which it is done is
logged as forward_s, so every repetition has two totals: total_s with the forward pass and spectral_s
without it (total_s minus forward_s), the time spent in the library. Loading the model is outside the clock.
--reps repetitions run one after the other in one process, and the record gives the median of each total.
Before each repetition the script waits until the CPU load of the machine is below --max-load percent.
"""

from __future__ import annotations

import os

THREADS = 2
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = str(THREADS)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

import spectral_trust as st  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
N_TOKENS = 512
LOG = Path(__file__).resolve().parent / "race_timing.json"

# The prompt: these sentences in order, repeated, then cut to the first 512 tokens.
SENTENCES = [
    "The night train left the coast a little after nine and climbed slowly towards the hills.",
    "In the second carriage a woman was sorting photographs into three uneven piles.",
    "She had promised her brother that the album would be ready before the wedding.",
    "Outside, the fields were dark, and only the signal lamps gave the window any colour.",
    "A conductor walked through and asked each passenger where they meant to get off.",
    "Most of them answered with the name of the last station, the one beside the lake.",
    "The woman said she did not know yet, which made him check her ticket twice.",
    "Her ticket was valid for the whole line, so he shrugged and moved on.",
    "Near midnight the train stopped at a junction where nobody boarded and nobody left.",
    "A boy across the aisle counted the freight wagons on the other track and lost count at forty.",
    "His mother told him that numbers are easier to keep if you say them aloud.",
    "He tried again, aloud this time, and the whole carriage learned that there were sixty two.",
    "When the train moved off, the photographs slid together and the three piles became one.",
    "The woman laughed, gathered them up, and began once more from the oldest picture.",
    "It showed a harbour, two fishing boats, and a dog that belonged to no one in the family.",
    "Nobody remembered who had taken it, but everybody agreed that it had to come first.",
    "By the time the hills gave way to the plateau, half the album was in order.",
    "The conductor came back with tea and said the lake would be visible at dawn.",
    "She thanked him and asked whether the last station had a post office.",
    "It did, he said, and it opened early, because the fishermen sent their letters before work.",
]


def build_prompt(tokenizer) -> tuple[str, list[int]]:
    """The fixed prompt and its token ids, exactly N_TOKENS of them."""
    text = " ".join(SENTENCES * 4)
    ids = list(tokenizer(text)["input_ids"])
    if len(ids) < N_TOKENS:
        raise SystemExit(f"the source text has only {len(ids)} tokens")
    prompt = tokenizer.decode(ids[:N_TOKENS])
    check = list(tokenizer(prompt)["input_ids"])
    if len(check) != N_TOKENS or check != ids[:N_TOKENS]:
        raise SystemExit(f"the prompt does not tokenise back to {N_TOKENS} tokens ({len(check)})")
    return prompt, check


def cpu_load() -> float | None:
    """CPU load of the machine in percent, or None when it cannot be read."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"],
                capture_output=True, text=True, timeout=60).stdout.strip()
            return float(out.replace(",", "."))
        return 100.0 * os.getloadavg()[0] / (os.cpu_count() or 1)
    except Exception:  # noqa: BLE001
        return None


def wait_until_quiet(limit: float, patience_s: float = 900.0) -> float | None:
    """Wait until two readings in a row are below the limit. Returns the last reading."""
    t0, calm, load = time.time(), 0, None
    while time.time() - t0 < patience_s:
        load = cpu_load()
        if load is None:
            return None
        calm = calm + 1 if load < limit else 0
        if calm >= 2:
            return load
        time.sleep(3.0)
    raise SystemExit(f"the machine stayed above {limit}% CPU load for {patience_s:.0f} s (last reading {load})")


def dtype_kwarg() -> str:
    major = int(transformers.__version__.split(".")[0])
    minor = int(transformers.__version__.split(".")[1])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


# ---------------------------------------------------------------------------------------------------------
# 0.3.0: analyze_layer per layer and per head
# ---------------------------------------------------------------------------------------------------------
def run_old(args, wait):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from spectral_trust import GraphConstructor, GSPConfig, SpectralAnalyzer

    logging.disable(logging.WARNING)  # the truncation warning is logged at every call
    cfg = GSPConfig(device="cpu")  # the defaults of this version
    if args.solver:
        cfg.eigen_solver = args.solver
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager",
                                                 **{dtype_kwarg(): torch.float32})
    model.eval()
    prompt, ids = build_prompt(tokenizer)
    gc, analyzer = GraphConstructor(cfg), SpectralAnalyzer(cfg)

    reps = []
    for _ in range(args.reps):
        load = wait()
        t0 = time.perf_counter()
        enc = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        output_attentions=True, output_hidden_states=True, use_cache=False, return_dict=True)
        forward_s = time.perf_counter() - t0
        layer_done, graph_hfer, modes = [], [], []
        for lay, attn in enumerate(out.attentions):
            a = attn[0].detach().to(device="cpu", dtype=torch.float32)  # [H, S, S]
            x = out.hidden_states[lay + 1][0].detach().to(device="cpu", dtype=torch.float32)  # [S, d]
            sym = gc.symmetrize_attention(a.unsqueeze(0))  # [1, H, S, S]
            lap = gc.construct_laplacian(gc.aggregate_heads(sym))[0]
            diag = analyzer.analyze_layer(x, lap, lay)  # the head-averaged graph
            lap_heads = gc.construct_laplacian(sym[0])  # [H, S, S]
            for h in range(lap_heads.shape[0]):  # each head's graph
                analyzer.analyze_layer(x, lap_heads[h], lay)
            layer_done.append(time.perf_counter() - t0)
            graph_hfer.append(float(diag.hfer))
            modes.append(int(len(diag.eigenvalues)))
        reps.append({"total_s": layer_done[-1], "forward_s": forward_s, "layer_done_s": layer_done,
                     "graph_hfer": graph_hfer, "modes": modes[0], "cpu_load_before": load})
        assert int(enc["input_ids"].shape[1]) == N_TOKENS
        del out
    info = {"path": "analyze_layer per layer, then analyze_layer per head", "eigen_solver": cfg.eigen_solver,
            "n_layers": int(model.config.num_hidden_layers), "n_heads": int(model.config.num_attention_heads),
            "hidden_size": int(model.config.hidden_size), "progress": "a timestamp after each layer of the loop"}
    return info, reps


# ---------------------------------------------------------------------------------------------------------
# 0.4.0: run_with_spectra
# ---------------------------------------------------------------------------------------------------------
def run_new(args, wait):
    import spectral_trust.model as stm

    logging.disable(logging.WARNING)
    if args.solver:
        cfg = stm.default_config()
        cfg.eigen_solver = args.solver
        model = st.SpectralModel.from_pretrained(MODEL, config=cfg)
    else:
        model = st.SpectralModel.from_pretrained(MODEL)  # the defaults of this version
    prompt, ids = build_prompt(model.tokenizer)
    kwargs = {"metrics": args.metrics.split(",")} if args.metrics else {}
    n_heads = model.n_heads

    events: list = []
    clock = {"t0": 0.0}

    def now() -> float:
        return time.perf_counter() - clock["t0"]

    if not args.plain:
        compute_spectra, per_head, per_head_hybrid = stm.compute_spectra, stm.per_head_metrics, stm.per_head_hybrid_metrics
        analyze_layer = st.SpectralAnalyzer.analyze_layer

        def timed_compute_spectra(*a, **k):
            events.append(["forward", None, now()])  # tokenisation and the forward pass are done
            return compute_spectra(*a, **k)

        def timed_per_head(attention, *a, **k):
            out = per_head(attention, *a, **k)
            events.append(["attn", int(attention.shape[0]) // n_heads, now()])  # layers covered by this call
            return out

        def timed_per_head_hybrid(*a, **k):
            out = per_head_hybrid(*a, **k)
            events.append(["resid", int(k.get("layer", -1)), now()])
            return out

        def timed_analyze_layer(self, signals, laplacian, layer_idx):
            out = analyze_layer(self, signals, laplacian, layer_idx)
            events.append(["graph", int(layer_idx), now()])
            return out

        stm.compute_spectra, stm.per_head_metrics = timed_compute_spectra, timed_per_head
        stm.per_head_hybrid_metrics = timed_per_head_hybrid
        st.SpectralAnalyzer.analyze_layer = timed_analyze_layer

    reps = []
    for _ in range(args.reps):
        load = wait()
        events.clear()
        clock["t0"] = time.perf_counter()
        cache = model.run_with_spectra(prompt, **kwargs)
        total = now()
        assert len(cache.tokens) == N_TOKENS, len(cache.tokens)
        rep = {"total_s": total, "cpu_load_before": load}
        if not args.plain:
            rep["forward_s"] = next(t for kind, _, t in events if kind == "forward")
            rep["attn_batches"] = [[n, t] for kind, n, t in events if kind == "attn"]
            done = {}
            for kind, lay, t in events:
                if kind in ("resid", "graph"):
                    done[lay] = max(done.get(lay, 0.0), t)
            rep["layer_done_s"] = [done[lay] for lay in sorted(done)]
        if "blocks.0.graph.hfer" in cache:
            rep["graph_hfer"] = [float(cache[f"blocks.{lay}.graph.hfer"]) for lay in cache.layers]
            rep["modes"] = int(len(cache["blocks.0.graph.eigenvalues"]))
        reps.append(rep)
        del cache
    info = {"path": "SpectralModel.run_with_spectra" + (f"(metrics={kwargs['metrics']})" if kwargs else ""),
            "eigen_solver": model.config.eigen_solver, "n_layers": model.n_layers, "n_heads": n_heads,
            "hidden_size": int(model.model.config.hidden_size),
            "progress": "no timestamps (plain call)" if args.plain else
            "timestamps when per_head_metrics, per_head_hybrid_metrics and analyze_layer return inside the call"}
    return info, reps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", required=True, help="name of this configuration in the log, for example defaults")
    ap.add_argument("--solver", choices=["dense", "sparse"], help="override the version's default eigen_solver")
    ap.add_argument("--metrics", help="0.4.0 only: comma-separated metrics= of run_with_spectra (default: all)")
    ap.add_argument("--plain", action="store_true", help="0.4.0 only: no timestamps inside the call")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-load", type=float, default=20.0, help="wait until the CPU load is below this percentage")
    ap.add_argument("--log", type=Path, default=LOG)
    args = ap.parse_args()

    torch.set_num_threads(THREADS)
    assert not torch.cuda.is_available()

    def wait():
        return wait_until_quiet(args.max_load)

    new_api = hasattr(st, "SpectralModel")
    info, reps = (run_new if new_api else run_old)(args, wait)
    totals = [r["total_s"] for r in reps]
    median = statistics.median(totals)
    for r in reps:
        if "forward_s" in r:
            r["spectral_s"] = r["total_s"] - r["forward_s"]
    spectral = [r["spectral_s"] for r in reps if "spectral_s" in r]
    record = {
        "label": args.label,
        "spectral_trust": st.__version__,
        "model": MODEL,
        "n_tokens": N_TOKENS,
        **info,
        "clock": "total_s starts before tokenisation, spectral_s when the forward pass is done",
        "threads": THREADS,
        "device": "cpu",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "system": f"{platform.system()} {platform.machine()}",
        "logical_cpus": os.cpu_count(),
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "totals_s": totals,
        "median_total_s": median,
        "median_rep": min(range(len(totals)), key=lambda k: abs(totals[k] - median)),
        "spectral_s": spectral,
        "median_spectral_s": statistics.median(spectral) if spectral else None,
        "reps": reps,
    }
    data = json.loads(args.log.read_text(encoding="utf-8")) if args.log.exists() else {"runs": []}
    data["runs"].append(record)
    args.log.write_bytes((json.dumps(data, indent=1) + "\n").encode("utf-8"))
    print(f"spectral-trust {st.__version__} [{args.label}] solver={info['eigen_solver']} "
          f"totals={[round(t, 2) for t in totals]} median={median:.2f} s "
          f"without the forward pass={[round(t, 2) for t in spectral]} "
          f"(torch {torch.__version__}, transformers {transformers.__version__})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
