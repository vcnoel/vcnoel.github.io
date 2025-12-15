"""
ReGA vs SOTA BASELINES (GMN, DGMC)
==================================

Implements simplified versions of:
1. GMN (Graph Matching Networks) - Li et al., ICML 2019
2. DGMC (Deep Graph Matching Consensus) - Fey et al., ICLR 2020

Then compares fairly on synthetic graph matching benchmarks.

Run:
    python rega_enhanced_evaluation_19.py --device cuda
"""

import argparse
import json
import math
import random
import os
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import time

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx required")


@dataclass
class BaselineConfig:
    hidden_dim: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    output_dir: str = "rega_sota_comparison_results"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ===========================================================================
# Sinkhorn Operator (shared)
# ===========================================================================

def sinkhorn(log_alpha, n_iters=10, temp=1.0):
    log_alpha = log_alpha / (temp + 1e-10)
    log_alpha = log_alpha - log_alpha.max(dim=-1, keepdim=True).values
    P = torch.exp(log_alpha).clamp(1e-10, 1e10)
    for _ in range(n_iters):
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-8)
        P = P / (P.sum(dim=-2, keepdim=True) + 1e-8)
    return P


# ===========================================================================
# 1. ReGA (Our Method)
# ===========================================================================

class GNNLayer(nn.Module):
    """Standard GNN layer."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_neigh = nn.Linear(d_in, d_out)

    def forward(self, h, A):
        return F.relu(self.lin_self(h) + self.lin_neigh(torch.matmul(A, h)))


class ReGA(nn.Module):
    """
    ReGA: Our method.
    - GNN encoder
    - Sinkhorn normalization
    - Cross-entropy loss on permutation
    """
    def __init__(self, d_node, hidden_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = d_node
        for _ in range(num_layers):
            self.layers.append(GNNLayer(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def encode(self, H, A):
        h = H
        for layer in self.layers:
            h = layer(h, A)
        return self.norm(h)

    def forward(self, A_s, A_h, H_s, H_h, perm):
        B, N, _ = A_s.shape
        Z_s = self.encode(H_s, A_s)
        Z_h = self.encode(H_h, A_h)
        
        S = torch.matmul(Z_h, Z_s.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        P = sinkhorn(S, 10)
        
        log_P = torch.log(P + 1e-8)
        row_idx = torch.arange(N, device=P.device).unsqueeze(0).expand(B, N)
        picked = log_P[torch.arange(B, device=P.device).unsqueeze(-1), row_idx, perm]
        
        loss = -picked.mean()
        acc = (P.argmax(dim=-1) == perm).float().mean().item()
        
        return loss, acc, P


# ===========================================================================
# 2. GMN (Graph Matching Networks) - Simplified
# ===========================================================================

class GMN(nn.Module):
    """
    Graph Matching Networks (Li et al., ICML 2019) - Simplified.
    
    Key ideas:
    - Cross-graph attention for node correspondence
    - Iterative message passing with cross-graph aggregation
    - Matching score based on aggregated graph representations
    
    Paper: "Graph Matching Networks for Learning the Similarity of Graph Structured Objects"
    """
    def __init__(self, d_node, hidden_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Node embedding
        self.node_embed = nn.Linear(d_node, hidden_dim)
        
        # GNN layers (intra-graph)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Cross-graph attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Final projection for matching
        self.match_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, A_s, A_h, H_s, H_h, perm):
        B, N, _ = A_s.shape
        
        # Initial node embeddings
        h_s = self.node_embed(H_s)
        h_h = self.node_embed(H_h)
        
        # Iterative cross-graph message passing
        for gnn_layer in self.gnn_layers:
            # Intra-graph aggregation
            h_s = gnn_layer(h_s, A_s)
            h_h = gnn_layer(h_h, A_h)
            
            # Cross-graph attention (key GMN idea)
            h_s_cross, _ = self.cross_attn(h_s, h_h, h_h)
            h_h_cross, _ = self.cross_attn(h_h, h_s, h_s)
            
            # Combine
            h_s = h_s + 0.5 * h_s_cross
            h_h = h_h + 0.5 * h_h_cross
        
        # Normalize
        h_s = self.norm(h_s)
        h_h = self.norm(h_h)
        
        # Matching scores
        h_s = self.match_proj(h_s)
        h_h = self.match_proj(h_h)
        
        S = torch.matmul(h_h, h_s.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        P = sinkhorn(S, 10)
        
        # Loss
        log_P = torch.log(P + 1e-8)
        row_idx = torch.arange(N, device=P.device).unsqueeze(0).expand(B, N)
        picked = log_P[torch.arange(B, device=P.device).unsqueeze(-1), row_idx, perm]
        
        loss = -picked.mean()
        acc = (P.argmax(dim=-1) == perm).float().mean().item()
        
        return loss, acc, P


# ===========================================================================
# 3. DGMC (Deep Graph Matching Consensus) - Simplified
# ===========================================================================

class DGMC(nn.Module):
    """
    Deep Graph Matching Consensus (Fey et al., ICLR 2020) - Simplified.
    
    Key ideas:
    - Initial soft assignment via cross-similarity
    - Iterative refinement using consensus (neighborhood agreement)
    - Sparse matching via top-k selection
    
    Paper: "Deep Graph Matching Consensus"
    """
    def __init__(self, d_node, hidden_dim, num_iters=3, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_iters = num_iters
        
        # Node embedding
        self.node_embed = nn.Linear(d_node, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Consensus refinement
        self.consensus_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, A_s, A_h, H_s, H_h, perm):
        B, N, _ = A_s.shape
        
        # Initial node embeddings
        h_s = self.node_embed(H_s)
        h_h = self.node_embed(H_h)
        
        # GNN encoding
        for gnn_layer in self.gnn_layers:
            h_s = gnn_layer(h_s, A_s)
            h_h = gnn_layer(h_h, A_h)
        
        h_s = self.norm(h_s)
        h_h = self.norm(h_h)
        
        # Initial soft assignment
        S = torch.matmul(h_h, h_s.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        P = sinkhorn(S, 5)
        
        # Iterative consensus refinement (key DGMC idea)
        for _ in range(self.num_iters):
            # For each node in h, aggregate matched node features from s
            h_s_matched = torch.matmul(P, h_s)  # [B, N, D]
            
            # Consensus: neighborhood agreement
            # If neighbors of node i match to neighbors of matched(i), increase confidence
            neighbor_s = torch.matmul(A_s, h_s)  # Neighbor features in source
            neighbor_h = torch.matmul(A_h, h_h)  # Neighbor features in hypothesis
            neighbor_s_matched = torch.matmul(P, neighbor_s)
            
            # Combine for consensus
            consensus_input = torch.cat([h_h, neighbor_s_matched - neighbor_h], dim=-1)
            consensus_score = self.consensus_mlp(consensus_input)
            
            # Update embeddings
            h_h = h_h + 0.3 * consensus_score
            h_h = self.norm(h_h)
            
            # Recompute assignment
            S = torch.matmul(h_h, h_s.transpose(1, 2)) / math.sqrt(self.hidden_dim)
            P = sinkhorn(S, 5)
        
        # Final loss
        log_P = torch.log(P + 1e-8)
        row_idx = torch.arange(N, device=P.device).unsqueeze(0).expand(B, N)
        picked = log_P[torch.arange(B, device=P.device).unsqueeze(-1), row_idx, perm]
        
        loss = -picked.mean()
        acc = (P.argmax(dim=-1) == perm).float().mean().item()
        
        return loss, acc, P


# ===========================================================================
# Training and Evaluation
# ===========================================================================

def generate_random_graph(n, p=0.4):
    G = nx.gnp_random_graph(n, p, directed=True)
    return torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)


def train_model(model, d_node, n, epochs, device):
    """Train any model."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for _ in range(epochs):
        A = generate_random_graph(n).to(device)
        H = torch.randn(n, d_node, device=device)
        perm = torch.randperm(n, device=device)
        
        opt.zero_grad()
        loss, _, _ = model(A.unsqueeze(0), A[perm][:, perm].unsqueeze(0),
                          H.unsqueeze(0), H[perm].unsqueeze(0),
                          perm.unsqueeze(0))
        
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    
    return model


def test_model(model, d_node, n, device, n_trials=30):
    """Test any model."""
    model.eval()
    accs = []
    
    for _ in range(n_trials):
        A = generate_random_graph(n).to(device)
        H = torch.randn(n, d_node, device=device)
        perm = torch.randperm(n, device=device)
        
        with torch.no_grad():
            _, _, P = model(A.unsqueeze(0), A[perm][:, perm].unsqueeze(0),
                           H.unsqueeze(0), H[perm].unsqueeze(0),
                           perm.unsqueeze(0))
        
        pred = P[0].argmax(dim=-1)
        accs.append((pred == perm).float().mean().item())
    
    return np.mean(accs), np.std(accs)


# ===========================================================================
# Fair Comparison
# ===========================================================================

def run_fair_comparison(config):
    """
    Fair comparison between ReGA, GMN, and DGMC.
    Same training data, same hyperparameters, same test conditions.
    """
    print("\n" + "="*70)
    print("FAIR COMPARISON: ReGA vs GMN vs DGMC")
    print("="*70)
    
    device = torch.device(config.device)
    d_node = 64
    hidden_dim = config.hidden_dim
    epochs = 100
    
    results = {"comparisons": []}
    
    node_sizes = [10, 15, 20, 25, 30, 40]
    
    models_info = {
        "ReGA": lambda: ReGA(d_node, hidden_dim),
        "GMN": lambda: GMN(d_node, hidden_dim),
        "DGMC": lambda: DGMC(d_node, hidden_dim),
    }
    
    print("\n   METHOD DETAILS:")
    print("   " + "-"*50)
    print("   ReGA:  GNN encoder + Sinkhorn (Our method)")
    print("   GMN:   Cross-graph attention + Sinkhorn (Li et al., ICML 2019)")
    print("   DGMC:  Iterative consensus refinement (Fey et al., ICLR 2020)")
    print()
    
    # Count parameters for each
    print("   PARAMETER COUNT:")
    print("   " + "-"*50)
    for name, model_fn in models_info.items():
        model = model_fn()
        params = sum(p.numel() for p in model.parameters())
        print(f"   {name}: {params:,} parameters")
        del model
    print()
    
    # Header
    print("   ACCURACY BY GRAPH SIZE:")
    print("   " + "-"*50)
    print("   n   |   ReGA   |   GMN    |   DGMC   | Best")
    print("   " + "-"*50)
    
    for n in node_sizes:
        row = {"n": n}
        best_method = None
        best_acc = 0
        
        for name, model_fn in models_info.items():
            # Train
            model = model_fn().to(device)
            model = train_model(model, d_node, n, epochs, device)
            
            # Test
            acc, std = test_model(model, d_node, n, device)
            row[name] = {"accuracy": float(acc), "std": float(std)}
            
            if acc > best_acc:
                best_acc = acc
                best_method = name
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        row["best"] = best_method
        results["comparisons"].append(row)
        
        # Print row
        print(f"   {n:3d} |", end="")
        for name in models_info.keys():
            acc = row[name]["accuracy"]
            marker = "🏆" if name == best_method else "  "
            print(f"  {acc:5.1%} {marker}|", end="")
        print(f" {best_method}")
    
    # Summary statistics
    print("\n" + "-"*50)
    print("   SUMMARY:")
    print("   " + "-"*50)
    
    for name in models_info.keys():
        accs = [r[name]["accuracy"] for r in results["comparisons"]]
        wins = sum(1 for r in results["comparisons"] if r["best"] == name)
        print(f"   {name}: avg={np.mean(accs):.1%}, wins={wins}/{len(node_sizes)}")
    
    # Determine overall winner
    overall_accs = {name: np.mean([r[name]["accuracy"] for r in results["comparisons"]]) 
                   for name in models_info.keys()}
    winner = max(overall_accs, key=overall_accs.get)
    
    print(f"\n   🏆 OVERALL WINNER: {winner} ({overall_accs[winner]:.1%})")
    results["winner"] = winner
    
    return results


def run_timing_comparison(config):
    """Compare training and inference time."""
    print("\n" + "="*70)
    print("TIMING COMPARISON")
    print("="*70)
    
    device = torch.device(config.device)
    d_node = 64
    hidden_dim = config.hidden_dim
    n = 20
    
    models_info = {
        "ReGA": lambda: ReGA(d_node, hidden_dim),
        "GMN": lambda: GMN(d_node, hidden_dim),
        "DGMC": lambda: DGMC(d_node, hidden_dim),
    }
    
    results = {}
    
    print("\n   METHOD    | Train (s) | Inference (ms)")
    print("   " + "-"*40)
    
    for name, model_fn in models_info.items():
        model = model_fn().to(device)
        
        # Training time
        start = time.time()
        model = train_model(model, d_node, n, 50, device)
        train_time = time.time() - start
        
        # Inference time
        model.eval()
        A = generate_random_graph(n).to(device)
        H = torch.randn(n, d_node, device=device)
        perm = torch.randperm(n, device=device)
        
        # Warmup
        with torch.no_grad():
            model(A.unsqueeze(0), A[perm][:, perm].unsqueeze(0),
                 H.unsqueeze(0), H[perm].unsqueeze(0),
                 perm.unsqueeze(0))
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                model(A.unsqueeze(0), A[perm][:, perm].unsqueeze(0),
                     H.unsqueeze(0), H[perm].unsqueeze(0),
                     perm.unsqueeze(0))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_time = (time.time() - start) / 100 * 1000  # ms
        
        results[name] = {
            "train_time": float(train_time),
            "inference_ms": float(inference_time)
        }
        
        print(f"   {name:8s} |   {train_time:5.2f}   |    {inference_time:6.2f}")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Find fastest
    fastest = min(results, key=lambda x: results[x]["inference_ms"])
    print(f"\n   🏆 Fastest inference: {fastest} ({results[fastest]['inference_ms']:.2f} ms)")
    
    return results


def run_scalability_test(config):
    """Test how each method scales with graph size."""
    print("\n" + "="*70)
    print("SCALABILITY TEST")
    print("="*70)
    
    device = torch.device(config.device)
    d_node = 64
    hidden_dim = config.hidden_dim
    
    models_info = {
        "ReGA": lambda: ReGA(d_node, hidden_dim),
        "GMN": lambda: GMN(d_node, hidden_dim),
        "DGMC": lambda: DGMC(d_node, hidden_dim),
    }
    
    # Test up to larger sizes
    node_sizes = [10, 20, 30, 40, 50, 60]
    
    results = {"scalability": []}
    
    print("\n   n   |   ReGA   |   GMN    |   DGMC")
    print("   " + "-"*45)
    
    for n in node_sizes:
        row = {"n": n}
        
        for name, model_fn in models_info.items():
            try:
                model = model_fn().to(device)
                model = train_model(model, d_node, n, 50, device)
                acc, _ = test_model(model, d_node, n, device, n_trials=10)
                row[name] = float(acc)
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:
                row[name] = "OOM"
        
        results["scalability"].append(row)
        
        print(f"   {n:3d} |", end="")
        for name in models_info.keys():
            val = row[name]
            if isinstance(val, float):
                print(f"  {val:5.1%}   |", end="")
            else:
                print(f"   {val}   |", end="")
        print()
    
    return results


# ===========================================================================
# HYPERPARAMETER SENSITIVITY TEST
# ===========================================================================

def run_hyperparameter_sensitivity(config):
    """
    Grid search to ensure baselines aren't handicapped by bad hyperparameters.
    Tests hidden_dim, learning rate, and num_layers for each method.
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER SENSITIVITY TEST")
    print("="*70)
    print("\nGrid search to ensure fair comparison...\n")
    
    device = torch.device(config.device)
    d_node = 64
    n = 20  # Fixed test size
    
    # Hyperparameter grid
    hidden_dims = [128, 256, 512]
    learning_rates = [1e-3, 5e-4, 1e-4]
    num_layers_options = [1, 2, 3]
    
    results = {"tuning": {}}
    
    def create_model_with_params(name, hidden_dim, num_layers):
        if name == "ReGA":
            return ReGA(d_node, hidden_dim, num_layers)
        elif name == "GMN":
            return GMN(d_node, hidden_dim, num_layers)
        elif name == "DGMC":
            return DGMC(d_node, hidden_dim, num_layers=num_layers)
    
    def train_with_lr(model, lr, epochs=50):
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        
        for _ in range(epochs):
            A = generate_random_graph(n).to(device)
            H = torch.randn(n, d_node, device=device)
            perm = torch.randperm(n, device=device)
            
            opt.zero_grad()
            loss, _, _ = model(A.unsqueeze(0), A[perm][:, perm].unsqueeze(0),
                              H.unsqueeze(0), H[perm].unsqueeze(0),
                              perm.unsqueeze(0))
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        
        return model
    
    for method_name in ["ReGA", "GMN", "DGMC"]:
        print(f"\n   Tuning {method_name}...")
        print(f"   {'hidden':>8} | {'lr':>8} | {'layers':>6} | {'accuracy':>8}")
        print("   " + "-"*45)
        
        best_acc = 0
        best_params = {}
        all_configs = []
        
        for hidden_dim in hidden_dims:
            for lr in learning_rates:
                for num_layers in num_layers_options:
                    try:
                        model = create_model_with_params(method_name, hidden_dim, num_layers).to(device)
                        model = train_with_lr(model, lr, epochs=50)
                        acc, _ = test_model(model, d_node, n, device, n_trials=10)
                        
                        all_configs.append({
                            "hidden_dim": hidden_dim,
                            "lr": lr,
                            "num_layers": num_layers,
                            "accuracy": float(acc)
                        })
                        
                        marker = "🏆" if acc > best_acc else "  "
                        print(f"   {hidden_dim:8d} | {lr:8.0e} | {num_layers:6d} | {acc:7.1%} {marker}")
                        
                        if acc > best_acc:
                            best_acc = acc
                            best_params = {"hidden_dim": hidden_dim, "lr": lr, "num_layers": num_layers}
                        
                        del model
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    except Exception as e:
                        print(f"   {hidden_dim:8d} | {lr:8.0e} | {num_layers:6d} |   ERROR")
        
        results["tuning"][method_name] = {
            "best_accuracy": float(best_acc),
            "best_params": best_params,
            "all_configs": all_configs
        }
        
        print(f"\n   Best {method_name}: {best_acc:.1%} with {best_params}")
    
    # Summary comparison with best hyperparameters
    print("\n" + "-"*50)
    print("   BEST HYPERPARAMETERS SUMMARY:")
    print("   " + "-"*50)
    
    for method, data in results["tuning"].items():
        params = data["best_params"]
        acc = data["best_accuracy"]
        print(f"   {method}: {acc:.1%} (d={params.get('hidden_dim')}, lr={params.get('lr')}, L={params.get('num_layers')})")
    
    # Final fair comparison with best params
    print("\n   FINAL COMPARISON (Best Hyperparams for Each):")
    print("   " + "-"*50)
    
    best_method = max(results["tuning"], key=lambda x: results["tuning"][x]["best_accuracy"])
    print(f"   🏆 Best overall: {best_method} ({results['tuning'][best_method]['best_accuracy']:.1%})")
    
    return results


def print_implementation_table():
    """Print implementation details table for paper."""
    print("\n" + "="*70)
    print("IMPLEMENTATION DETAILS (For Paper)")
    print("="*70)
    print("""
    +----------+----------------------------+------------------+
    | Method   | Key Component              | Preserved?       |
    +----------+----------------------------+------------------+
    | GMN      | Cross-graph attention      | ✓                |
    | GMN      | Iterative message passing  | ✓                |
    | GMN      | Graph-level aggregation    | Simplified       |
    +----------+----------------------------+------------------+
    | DGMC     | Initial soft assignment    | ✓                |
    | DGMC     | Consensus refinement       | ✓                |
    | DGMC     | Sparse top-k selection     | Dense Sinkhorn   |
    +----------+----------------------------+------------------+
    
    Note: Our reimplementations preserve core algorithmic components.
    For official implementations, see:
    - GMN: github.com/deepmind/deepmind-research/tree/master/graph_matching_networks
    - DGMC: github.com/rusty1s/deep-graph-matching-consensus
    """)

def main(config):
    set_seed(config.seed)
    ensure_dir(config.output_dir)
    
    print("="*70)
    print("ReGA vs SOTA BASELINES (GMN, DGMC)")
    print("Fair comparison on synthetic graph matching")
    print("="*70)
    
    all_results = {}
    
    # Fair comparison
    all_results["accuracy"] = run_fair_comparison(config)
    
    # Timing
    all_results["timing"] = run_timing_comparison(config)
    
    # Scalability
    all_results["scalability"] = run_scalability_test(config)
    
    # Hyperparameter sensitivity (ensures fair comparison)
    all_results["hyperparameters"] = run_hyperparameter_sensitivity(config)
    
    # Print implementation details for paper
    print_implementation_table()
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    winner = all_results["accuracy"]["winner"]
    
    print(f"\n   📊 Accuracy winner: {winner}")
    
    timing = all_results["timing"]
    fastest = min(timing, key=lambda x: timing[x]["inference_ms"])
    print(f"   ⚡ Speed winner: {fastest}")
    
    # Save
    results_file = f"{config.output_dir}/sota_comparison.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="rega_sota_comparison_results")
    args = parser.parse_args()
    
    config = BaselineConfig(device=args.device, output_dir=args.output_dir)
    main(config)
